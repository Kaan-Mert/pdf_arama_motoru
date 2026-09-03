import os
from pathlib import Path
import glob
import re
import html
import gradio as gr
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import pickle
from rank_bm25 import BM25Okapi
from TurkishStemmer import TurkishStemmer
from sentence_transformers import CrossEncoder

# --- YAPILANDIRMA ---
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "chroma_db"
BM25_PERSIST_FILE = "bm25_index.pkl"
TEMP_IMAGE_DIR = "temp_images"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU varsa CUDA kullan, yoksa CPU

def tr_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()

def normalize_tr(text):
    """Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) ve büyük harfleri ASCII dengiyle normalleştirir."""
    text = tr_lower(text)
    mapping = str.maketrans("çğıöşü", "cgiosu")
    return text.translate(mapping)

stemmer = TurkishStemmer()

def preprocess_text(text):
    """BM25 için metni küçük harfe çevirir, noktalama işaretlerini siler ve kök ayrıştırır."""
    text = tr_lower(text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [stemmer.stem(token) for token in tokens]

# Temp klasörünü oluştur
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Global modeller ve indeksler
vector_store = None
bm25_index = None
bm25_documents = []
cross_encoder = None

# Embedding modelini başlat
print("Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': DEVICE}
)

def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_manifest(ids, metadatas):
    manifest = {}
    temp = {}
    if ids is None or metadatas is None or len(ids) != len(metadatas):
        return None

    for _id, meta in zip(ids, metadatas):
        if not meta or not isinstance(meta, dict):
            return None

        raw_source = meta.get("source") or ""
        source_key = meta.get("source_key")

        if not source_key and raw_source:
            source_key = os.path.normpath(raw_source).replace('\\', '/')

        file_hash = meta.get("file_hash")

        if not source_key or source_key == "." or not file_hash:
            return None

        if source_key not in temp:
            temp[source_key] = {"ids": [], "hashes": set()}
        temp[source_key]["ids"].append(_id)
        temp[source_key]["hashes"].add(file_hash)

    for sk, data in temp.items():
        if len(data["hashes"]) != 1 or None in data["hashes"]:
            return None
        f_hash = list(data["hashes"])[0]
        sorted_ids = sorted(data["ids"])
        digest = hashlib.sha256("".join(sorted_ids).encode('utf-8')).hexdigest()
        manifest[sk] = {
            "file_hash": f_hash,
            "actual_count": len(sorted_ids),
            "ids_digest": digest
        }
    return manifest

def sync_bm25():
    """Chroma'daki kayıtlardan BM25'i yeniden inşa eder."""
    global vector_store, bm25_index, bm25_documents
    print("Chroma'dan BM25'e senkronize ediliyor...")

    all_data = vector_store.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])

    if not docs or not metadatas or len(docs) != len(metadatas) or len(ids) != len(docs):
        print("Uyarı: Senkronizasyon için geçerli/tutarlı doküman yok.")
        bm25_index = None
        bm25_documents = []
        if os.path.exists(BM25_PERSIST_FILE):
            os.remove(BM25_PERSIST_FILE)
        return False

    manifest = generate_manifest(ids, metadatas)
    if manifest is None:
        print("Uyarı: Manifest oluşturulamadı (eksik/çelişen hash).")
        bm25_index = None
        bm25_documents = []
        if os.path.exists(BM25_PERSIST_FILE):
            os.remove(BM25_PERSIST_FILE)
        return False

    bm25_corpus = []
    bm25_documents = []

    for doc_text, meta in zip(docs, metadatas):
        d = Document(page_content=doc_text, metadata=meta)
        bm25_documents.append(d)
        tokens = preprocess_text(doc_text)
        bm25_corpus.append(tokens)

    bm25_index = BM25Okapi(bm25_corpus)

    import tempfile
    temp_dir = os.path.dirname(os.path.abspath(BM25_PERSIST_FILE))
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump({
                "schema_version": 1,
                "manifest": manifest,
                "bm25": bm25_index,
                "documents": bm25_documents
            }, f)
        os.replace(temp_path, BM25_PERSIST_FILE)
        return True
    except Exception as e:
        print(f"BM25 dosyası kaydedilemedi: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        bm25_index = None
        bm25_documents = []
        return False

def init_vector_store():
    """Var olan veritabanını ve BM25 indeksini yükler, eksikse uyarı döndürür."""
    global vector_store, bm25_index, bm25_documents, cross_encoder

    # CrossEncoder modelini başlat (sadece bir kere yüklenir)
    if cross_encoder is None:
        print("CrossEncoder modeli yükleniyor...")
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512, device=DEVICE)

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    if vector_store is None:
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    has_bm25 = os.path.exists(BM25_PERSIST_FILE)

    bm25_valid = False
    if has_bm25:
        try:
            with open(BM25_PERSIST_FILE, "rb") as f:
                data = pickle.load(f)

            if data.get("schema_version") == 1 and "manifest" in data:
                all_data = vector_store.get(include=["metadatas"])
                current_manifest = generate_manifest(all_data.get("ids", []), all_data.get("metadatas", []))

                chroma_count = vector_store._collection.count()
                bm25_docs = data.get("documents", [])
                bm25_obj = data.get("bm25")
                corpus_size = getattr(bm25_obj, "corpus_size", -1) if hasattr(bm25_obj, "corpus_size") else len(getattr(bm25_obj, "corpus", []))

                if current_manifest and current_manifest == data["manifest"]:
                    if len(bm25_docs) == chroma_count and corpus_size == chroma_count:
                        bm25_index = bm25_obj
                        bm25_documents = bm25_docs
                        bm25_valid = True
                    else:
                        print("BM25 belge sayısı Chroma ile eşleşmiyor. Yeniden oluşturulmalı.")
                else:
                    print("BM25 manifesti Chroma ile eşleşmiyor. Yeniden oluşturulmalı.")
        except Exception as e:
            print(f"BM25 dosyası bozuk veya okunamadı: {e}")

    if not bm25_valid:
        bm25_index = None
        bm25_documents = []

    chroma_count = vector_store._collection.count()
    if chroma_count > 0 and bm25_valid:
        return "⚡ Veritabanı hazır! Aramaya başlayabilirsiniz."
    elif chroma_count > 0 and not bm25_valid:
        return "⚠️ Chroma hazır ancak BM25 eksik/bozuk. Lütfen 'Verileri İndeksle' butonuna basınız."
    else:
        return "ℹ️ Henüz indekslenmiş veri yok. Lütfen 'Verileri İndeksle' butonuna basınız."

def index_documents():
    """data/ klasöründeki tüm PDF'leri okur, parçalar ve ChromaDB'ye indeksler."""
    global vector_store, bm25_index, bm25_documents

    if vector_store is None:
        init_vector_store()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return "⚠️ 'data/' klasörü oluşturuldu. Lütfen içine PDF dosyaları ekleyip tekrar deneyin."

    disk_files_info = {}
    seen_disk_source_keys = set()
    added = 0
    updated = 0
    deleted = 0
    skipped = 0
    errors = 0
    chroma_changed = False
    critical_integrity_error = False
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            source_key = os.path.normpath(filepath).replace('\\', '/')
            seen_disk_source_keys.add(source_key)
            try:
                f_hash = get_file_hash(filepath)
                disk_files_info[source_key] = {"path": filepath, "hash": f_hash}
            except Exception as e:
                print(f"Hata: {source_key} okunurken/hash hesaplanırken hata oluştu. Dosya atlanıyor: {e}")
                errors += 1

    existing_data = vector_store.get(include=["metadatas"])
    existing_metas = existing_data.get("metadatas", [])
    existing_ids = existing_data.get("ids", [])

    chroma_sources = {}
    for meta, _id in zip(existing_metas, existing_ids):
        raw_source = meta.get("source", "")
        source_key = meta.get("source_key")
        if not source_key:
            source_key = os.path.normpath(raw_source).replace('\\', '/')

        if source_key not in chroma_sources:
            chroma_sources[source_key] = {"ids": [], "hash": meta.get("file_hash"), "chunk_count": meta.get("chunk_count")}
        chroma_sources[source_key]["ids"].append(_id)

    for source_key, data in chroma_sources.items():
        if source_key not in seen_disk_source_keys:
            print(f"Siliniyor (orphan): {source_key}")
            vector_store.delete(ids=data["ids"])
            deleted += 1
            chroma_changed = True

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for source_key, info in disk_files_info.items():
        needs_indexing = False
        action_stat = ""

        if source_key not in chroma_sources:
            needs_indexing = True
            action_stat = "added"
        else:
            c_data = chroma_sources[source_key]
            if c_data["hash"] != info["hash"] or c_data["chunk_count"] != len(c_data["ids"]):
                needs_indexing = True
                action_stat = "updated"
            else:
                skipped += 1

        if needs_indexing:
            try:
                loader = PyMuPDFLoader(info["path"])
                docs = loader.load()

                ref_start_page = None
                for i, doc in enumerate(docs, 1):
                    lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]
                    for l in lines:
                        if tr_lower(l) in ["kaynakça", "notlar", "dış bağlantılar", "bibliyografya"]:
                            if ref_start_page is None:
                                ref_start_page = i
                            break

                for i, doc in enumerate(docs, 1):
                    doc.metadata["page"] = i
                    doc.metadata["source"] = info["path"]
                    doc.metadata["source_key"] = source_key

                    text = doc.page_content
                    text = re.sub(r'-\s*\n\s*', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    doc.page_content = text.strip()

                    is_ref = (ref_start_page is not None and i >= ref_start_page)
                    doc.metadata["is_reference"] = is_ref
                    doc.metadata["section"] = "kaynakça" if is_ref else "içerik"

                chunks = text_splitter.split_documents(docs)
                chunk_count = len(chunks)
                if chunk_count == 0:
                    raise ValueError("PDF'ten anlamlı metin çıkarılamadı (0 chunk).")

                for i, chunk in enumerate(chunks):
                    chunk.metadata["file_hash"] = info["hash"]
                    chunk.metadata["chunk_count"] = chunk_count

                chunk_ids = [f"{source_key}_{info['hash']}_{i}" for i in range(chunk_count)]

                existing_ids = (
                    set(chroma_sources[source_key]["ids"])
                    if source_key in chroma_sources
                    else set()
                )
                desired_ids = set(chunk_ids)
                missing_ids = desired_ids - existing_ids
                stale_ids = existing_ids - desired_ids

                try:
                    vector_store.add_documents(documents=chunks, ids=chunk_ids)

                    added_data = vector_store.get(ids=chunk_ids, include=["metadatas"])
                    found_ids = set(added_data.get("ids", []))
                    if found_ids != desired_ids or len(found_ids) != chunk_count:
                        raise RuntimeError("ChromaDB'ye yazılan chunk sayısı veya ID'leri doğrulanamadı.")

                    retrieved_metas = added_data.get("metadatas", [])
                    if len(retrieved_metas) != chunk_count:
                        raise RuntimeError("ChromaDB'den dönen metadata sayısı beklenen ile uyuşmuyor.")

                    for m in retrieved_metas:
                        if not m or m.get("file_hash") != info["hash"] or m.get("chunk_count") != chunk_count:
                            raise RuntimeError("Yazılan chunk metadata'ları (file_hash/chunk_count) doğrulanamadı.")

                    if stale_ids:
                        vector_store.delete(ids=list(stale_ids))

                    chroma_changed = True

                    if action_stat == "added":
                        added += 1
                    else:
                        updated += 1

                except Exception as op_error:
                    if missing_ids:
                        try:
                            vector_store.delete(ids=list(missing_ids))
                        except Exception as comp_error:
                            critical_integrity_error = True
                            bm25_index = None
                            bm25_documents = []
                            raise RuntimeError(
                                f"KRİTİK: İşlem başarısız oldu ({op_error}) ve eklenen yeni ID'ler ({len(missing_ids)}) geri alınamadı: {comp_error}"
                            ) from op_error

                    raise RuntimeError(
                        f"İşlem iptal edildi; yeni ID'ler başarıyla geri alındı. Orijinal hata: {op_error}"
                    ) from op_error

            except Exception as e:
                print(f"Hata: {source_key} okunamadı. Detay: {e}")
                errors += 1

    if critical_integrity_error:
        bm25_index = None
        bm25_documents = []
        return f"🚨 KRİTİK BÜTÜNLÜK HATASI! Bazı kayıtlar kısmi eklendi ve geri alınamadı. BM25 devre dışı bırakıldı. Hatalı: {errors}."

    bm25_sync_success = True
    should_sync_bm25 = False
    if chroma_changed:
        should_sync_bm25 = True
    elif errors == 0 and (not bm25_index or len(bm25_documents) != vector_store._collection.count()):
        should_sync_bm25 = True

    if should_sync_bm25:
        bm25_sync_success = sync_bm25()

    if not bm25_sync_success:
        return f"⚠️ Kısmen tamamlandı! Chroma güncellendi ancak BM25 senkronizasyonu başarısız oldu. Eklenen: {added}, Güncellenen: {updated}, Silinen: {deleted}, Atlanan: {skipped}, Hatalı: {errors}."

    if errors > 0:
        return f"⚠️ Bazı dosyalar işlenemedi! Eklenen: {added}, Güncellenen: {updated}, Silinen: {deleted}, Atlanan: {skipped}, Hatalı: {errors}."

    return f"✅ Başarılı! Eklenen: {added}, Güncellenen: {updated}, Silinen: {deleted}, Atlanan: {skipped}, Hatalı: {errors}."

def render_highlighted_pdf_page(pdf_path, page_num, query_words):
    """Verilen PDF'in ilgili sayfasını alır, sorgudaki kelimeleri sarıyla vurgular ve yüksek kaliteli resme çevirir."""
    if not os.path.exists(pdf_path):
        return None

    try:
        doc = fitz.open(pdf_path)
        fitz_page_num = int(page_num) - 1

        if fitz_page_num < 0 or fitz_page_num >= len(doc):
            return None

        page = doc.load_page(fitz_page_num)

        # Metin vurgulama (Highlighting)
        for word in query_words:
            rects = page.search_for(word)
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=[1, 1, 0]) # Sarı renk
                annot.update()

        # Net ve keskin önizleme için 2x zoom matrisi
        zoom_matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=zoom_matrix)

        safe_name = os.path.basename(pdf_path).replace(".pdf", "")
        img_filename = f"{safe_name}_page_{page_num}.png"
        img_path = os.path.join(TEMP_IMAGE_DIR, img_filename)

        pix.save(img_path)
        doc.close()

        return img_path
    except Exception as e:
        print(f"Resim oluşturma hatası: {e}")
        return None

def search(query):
    """Doğal dil sorgusunu semantik arama ile işler, Neumorphic Glass kartları ve galeri görsellerini üretir."""
    global vector_store, bm25_index, bm25_documents, cross_encoder

    query = query.strip()

    if not query:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <p class="empty-text">Lütfen aramak istediğiniz konuyu veya soruyu yukarıdaki kutuya yazın.</p>
        </div>
        """
        return empty_html, []

    if vector_store is None or bm25_index is None or cross_encoder is None:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <p class="empty-text">Önce sol taraftaki 'Verileri İndeksle' butonuna basarak belgeleri indekslemelisiniz.</p>
        </div>
        """
        return empty_html, []

    query_lower = tr_lower(query)
    query_tokens = preprocess_text(query)

    candidate_map = {}

    raw_vec = []
    top_n_indices = []

    # 1. Vektör adaylarını al (k=50)
    try:
        raw_vec = vector_store.similarity_search(query, k=50)
        for doc in raw_vec:
            cid = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}_{doc.page_content[:40]}"
            candidate_map[cid] = doc
    except Exception as e:
        print(f"Vektör arama uyarısı: {e}")

    # 2. BM25 adaylarını al (k=50)
    try:
        if query_tokens:
            import numpy as np
            bm25_scores = bm25_index.get_scores(query_tokens)
            top_n_indices = np.argsort(bm25_scores)[::-1][:50]
            for idx in top_n_indices:
                if bm25_scores[idx] > 0:
                    doc = bm25_documents[idx]
                    cid = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}_{doc.page_content[:40]}"
                    if cid not in candidate_map:
                        candidate_map[cid] = doc
    except Exception as e:
        print(f"BM25 arama uyarısı: {e}")

    unique_candidates = list(candidate_map.values())

    scored_results = []
    if unique_candidates:
        pairs = [(query_lower, tr_lower(doc.page_content)) for doc in unique_candidates]
        scores = cross_encoder.predict(pairs)

        for doc, score in zip(unique_candidates, scores):
            match_type = "⭐ Hybrid Match"
            if doc.metadata.get("is_reference"):
                match_type += " • 📚 Kaynakça"

            # Distance yerine sigmoid benzeri veya direkt skoru kullanıyoruz.
            pseudo_distance = float(score)

            # query tokens for highlighting
            extra_terms = set(query_tokens)
            scored_results.append((score, doc, pseudo_distance, match_type, extra_terms))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        scored_results = scored_results[:15]

    results = [(item[1], item[2], item[3], item[4]) for item in scored_results]

    print("\n=== HYBRID SEARCH DEBUG ===")
    print(f"Query: '{query}'")
    print(f"Vector Candidates: {len(raw_vec)}")
    print(f"BM25 Candidates: {len([i for i in top_n_indices if bm25_scores[i] > 0]) if 'bm25_scores' in locals() else 0}")
    print(f"Unique Pool Size: {len(unique_candidates)}")
    print(f"Top Final Results: {len(results)}")
    print("===========================\n")

    if not results:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <p class="empty-text">Aranan kelime veya kavram belgelerde bulunamadı.</p>
        </div>
        """
        return empty_html, []

    cards_html = []
    gallery_images = []

    for i, (res, distance, match_type, extra_terms) in enumerate(results, 1):
        file_path = res.metadata.get('source', '')
        file_name = os.path.basename(file_path)
        page_num = res.metadata.get('page', 1)

        snippet = html.escape(res.page_content.strip())

        raw_terms = [query] + list(extra_terms) + [w for w in re.split(r'\s+', query) if len(w) > 2]
        highlight_terms = sorted(set([t.strip() for t in raw_terms if t and t.strip()]), key=lambda x: len(x), reverse=True)


        for term in highlight_terms:
            pattern = re.compile(rf'(?<![<a-zA-ZçğıöşüÇĞIŞÖÜ])({re.escape(term)})(?![>a-zA-ZçğıöşüÇĞIŞÖÜ])', re.IGNORECASE)
            if not pattern.search(snippet):
                pattern = re.compile(rf'({re.escape(term)})', re.IGNORECASE)
            snippet = pattern.sub(r'<mark style="background-color: #fef08a; color: #0f172a; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\1</mark>', snippet)

        card = f"""
        <div class="result-card cursor-pointer" id="card-result-{i-1}" data-index="{i-1}" onclick="window.selectPdfResult && window.selectPdfResult({i-1})" title="Sağ tarafta bu sayfayı açmak için tıklayın">
            <div class="result-header">
                <div class="badges-group">
                    <span class="badge-index">#{i}</span>
                    <span class="badge-file">📄 {file_name}</span>
                    <span class="badge-page">Sayfa {page_num}</span>
                    <span class="badge-page" style="background: rgba(167, 243, 208, 0.8); color: #065f46; border-color: rgba(167, 243, 208, 1);">{match_type}</span>
                    <span class="badge-page" style="background: rgba(147, 197, 253, 0.8);">Skor: {distance:.4f}</span>
                    <button type="button" class="btn-jump-page" onclick="window.selectPdfResult && window.selectPdfResult({i-1}); event.stopPropagation();">👁️ Sayfayı Gör</button>
                </div>
            </div>
            <div class="result-snippet">
                <div class="quote-bar"></div>
                <p class="snippet-text">"{snippet}"</p>
            </div>
        </div>
        """
        cards_html.append(card)

        if file_path:
            img_path = render_highlighted_pdf_page(file_path, page_num, highlight_terms)
            if img_path:
                caption = f"Sonuç #{i} • {file_name} (Sayfa {page_num} - Skor: {distance:.4f})"
                gallery_images.append((img_path, caption))

    full_html = f"""
    <div class="results-container">
        {''.join(cards_html)}
        <img src="data:image/svg+xml;utf8,<svg></svg>" style="display:none;" onerror="setTimeout(function(){{ if(window.selectPdfResult) window.selectPdfResult(0); }}, 150);" />
    </div>
    """
    return full_html, gallery_images

# Başlangıç durumu
init_message = init_vector_store()

BASE_DIR = Path(__file__).resolve().parent
PHASE3_CSS = (BASE_DIR / "static" / "phase3.css").read_text(encoding="utf-8")
PHASE3_JS = (BASE_DIR / "static" / "phase3.js").read_text(encoding="utf-8")

TOP_APP_BAR_HTML = """
<div id="top-app-bar">
    <div class="top-bar-inner">
        <div class="brand-section">
            <div class="brand-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>
            </div>
            <div>
                <h1 class="brand-title">Akıllı PDF Arama & Görsel Keşif</h1>
                <p class="brand-subtitle">Anlamsal arama ve orijinal sayfa önizleme platformu</p>
            </div>
        </div>
    </div>
</div>
"""

FOOTER_HTML = """
<div id="app-footer">
    <p>© 2026 Akıllı PDF Arama Platformu</p>
</div>
"""

# --- GRADIO ARAYÜZÜ ---
with gr.Blocks(title="Akıllı PDF Arama & Görsel Keşif") as demo:
    # 1. Üst Başlık Çubuğu
    gr.HTML(TOP_APP_BAR_HTML)

    # 2. Belge Kütüphanesi Akordiyonu (Varsayılan olarak kapalı)
    with gr.Accordion("Belge Kütüphanesi", open=False, elem_id="library-accordion"):
        with gr.Row():
            index_btn = gr.Button("Belgeleri İndeksle", elem_id="index-button", scale=1)
            index_output = gr.Textbox(
                label="İndeksleme Durumu",
                value=init_message,
                interactive=False,
                elem_id="index-status",
                scale=3,
                lines=1
            )

    # 3. İki Sütunlu Ana Alan
    with gr.Row(elem_classes=["main-grid-row"]):
        # Sol Panel: Arama ve Metin Kesitleri
        with gr.Column(scale=1, elem_id="search-panel", elem_classes=["saas-card", "scrollable-panel"]):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-header-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EA580C" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    <span>Akıllı Semantik Arama</span>
                </div>
            </div>
            """)
            with gr.Row(elem_id="search-bar-row"):
                search_input = gr.Textbox(
                    label="",
                    placeholder="PDF belgelerinde ara...",
                    scale=4,
                    elem_id="search-input",
                    container=False
                )
                search_btn = gr.Button("Ara", scale=1, elem_id="search-button")

            gr.HTML("""
            <div class="panel-sub-header" id="results-header">
                <div class="panel-sub-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    <span>Metin Kesitleri</span>
                </div>
            </div>
            """)
            search_output = gr.HTML(
                value="""
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    </div>
                    <p class="empty-text">Aramak istediğiniz konuyu veya soruyu yukarıdaki kutucuğa yazıp <strong>'Ara'</strong> butonuna basın.</p>
                </div>
                """,
                elem_id="search-results"
            )

        # Sağ Panel: Orijinal Sayfa Önizlemeleri
        with gr.Column(scale=1, elem_id="preview-panel", elem_classes=["saas-card", "scrollable-panel"]):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-header-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                    <span>Orijinal Sayfa Önizlemeleri</span>
                </div>
            </div>
            """)
            gallery_output = gr.Gallery(
                label="Orijinal PDF Sayfaları",
                show_label=False,
                elem_id="pdf-preview-gallery",
                columns=[1],
                rows=[1],
                object_fit="contain",
                height="auto"
            )

    # 4. Alt Bilgi (Footer)
    gr.HTML(FOOTER_HTML)

    # Olay Bağlantıları (Event Listeners)
    index_btn.click(fn=index_documents, inputs=[], outputs=[index_output])
    search_btn.click(fn=search, inputs=[search_input], outputs=[search_output, gallery_output])
    search_input.submit(fn=search, inputs=[search_input], outputs=[search_output, gallery_output])

    # İstemci tarafı senkronizasyon betiği - Tek noktadan
    demo.load(js=PHASE3_JS)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, css=PHASE3_CSS, theme=gr.themes.Base())
