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
import urllib.parse
import time
import shutil
from rank_bm25 import BM25Okapi
from TurkishStemmer import TurkishStemmer
from sentence_transformers import CrossEncoder

# --- YAPILANDIRMA ---
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "chroma_db"
BM25_PERSIST_FILE = "bm25_index.pkl"
TEMP_IMAGE_DIR = "temp_images"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
last_index_time = None
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

def sanitize_filename(filename):
    """Zararlı karakterleri ve path traversal girişimlerini temizler."""
    if not filename:
        return "belge.pdf"
    clean_name = os.path.basename(str(filename)).strip()
    clean_name = clean_name.replace("\\", "_").replace("/", "_").replace("\0", "")
    clean_name = re.sub(r'[\r\n\t]', '', clean_name)
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"
    return clean_name

def get_library_stats():
    """data/ klasöründeki dosya sayısı, sistem sağlık durumu ve son güncelleme bilgisini döner."""
    pdf_count = 0
    if os.path.exists(DATA_DIR):
        pdf_count = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])

    is_healthy = (vector_store is not None and bm25_index is not None)
    status_str = "Sağlıklı" if is_healthy else "İndeks Bekliyor"

    global last_index_time
    check_time = last_index_time
    if check_time is None and os.path.exists(BM25_PERSIST_FILE):
        try:
            check_time = os.path.getmtime(BM25_PERSIST_FILE)
        except Exception:
            check_time = None

    if check_time is not None:
        elapsed_sec = max(0, time.time() - check_time)
        elapsed_min = int(elapsed_sec / 60)
        if elapsed_min < 1:
            time_str = "Az önce"
        elif elapsed_min < 60:
            time_str = f"{elapsed_min} dakika önce"
        else:
            hours = elapsed_min // 60
            if hours < 24:
                time_str = f"{hours} saat önce"
            else:
                days = hours // 24
                time_str = f"{days} gün önce"
    else:
        time_str = "Henüz yapılmadı"

    return {
        "pdf_count": pdf_count,
        "status": status_str,
        "is_healthy": is_healthy,
        "last_updated": time_str
    }

def handle_pdf_upload(files):
    """Yüklenen PDF dosyalarını doğrular ve data/ klasörüne kopyalar.
    Gradio file_upload bileşenini otomatik sıfırlamak için (None, mesaj) demeti döndürür.
    """
    if not files:
        return None, "Dosya seçilmedi."

    if not isinstance(files, list):
        files = [files]

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    saved_files = []
    rejected_files = []

    for f in files:
        temp_path = getattr(f, 'name', str(f))
        orig_name = getattr(f, 'orig_name', None) or os.path.basename(temp_path)

        if not orig_name.lower().endswith(".pdf"):
            rejected_files.append(f"{orig_name} (Yalnızca PDF kabul edilir)")
            continue

        try:
            file_size = os.path.getsize(temp_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                rejected_files.append(f"{orig_name} (50 MB sınırını aşıyor: {file_size / (1024*1024):.1f} MB)")
                continue
        except Exception as e:
            rejected_files.append(f"{orig_name} (Boyut okunamadı: {e})")
            continue

        clean_name = sanitize_filename(orig_name)
        dest_path = os.path.join(DATA_DIR, clean_name)
        try:
            shutil.copy2(temp_path, dest_path)
            saved_files.append(clean_name)
        except Exception as e:
            rejected_files.append(f"{orig_name} (Kayıt hatası: {e})")

    msg_parts = []
    if saved_files:
        count = len(saved_files)
        preview_names = ", ".join(saved_files[:3])
        if count > 3:
            preview_names += f" ve {count - 3} diğer dosya"
        msg_parts.append(f"📁 {count} yeni PDF 'data/' klasörüne yüklendi ({preview_names}). Arama motoruna dahil etmek için 'Belgeleri İndeksle' butonuna basınız.")
    if rejected_files:
        msg_parts.append(f"⚠️ Reddedilenler: {'; '.join(rejected_files)}.")

    final_msg = " ".join(msg_parts) if msg_parts else "Dosya işlenemedi."
    return None, final_msg

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

    global last_index_time
    last_index_time = time.time()
    return f"✅ Başarılı! Eklenen: {added}, Güncellenen: {updated}, Silinen: {deleted}, Atlanan: {skipped}, Hatalı: {errors}."

def cleanup_temp_images(max_age_hours=24, max_files=200):
    """Geçici önizleme görsellerini best-effort temizler; Windows kilitlerinde hata vermez."""
    if not os.path.exists(TEMP_IMAGE_DIR):
        return
    try:
        now = time.time()
        files = []
        for fn in os.listdir(TEMP_IMAGE_DIR):
            if fn.endswith((".png", ".jpg")):
                fp = os.path.join(TEMP_IMAGE_DIR, fn)
                try:
                    stat = os.stat(fp)
                    files.append((fp, stat.st_mtime))
                except Exception:
                    pass

        # 1. 24 saatten eski dosyaları sil (her dosya izole try-except)
        for fp, mtime in files:
            if now - mtime > max_age_hours * 3600:
                try:
                    os.remove(fp)
                except Exception:
                    pass

        # 2. Dosya sayısı limiti aşıyorsa en eskileri temizle
        remaining = [f for f in files if os.path.exists(f[0])]
        if len(remaining) > max_files:
            remaining.sort(key=lambda x: x[1])
            excess = len(remaining) - max_files
            for fp, _ in remaining[:excess]:
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception as e:
        print(f"Geçici dosya temizleme uyarısı: {e}")

def calculate_focus_box(page_w, page_h, query, query_words, rects_map):
    """
    Hedef eşleşmenin sayfa içindeki odak kutusunu hesaplar ve normalize (0..1) koordinatlar döner.
    Öncelik: Tam sorgu eşleşmesi -> En anlamlı sorgu terimi -> Güvenli üst/merkez fallback.
    """
    target_rect = None

    # 1. Tam sorgu eşleşmesi
    if query and query in rects_map and rects_map[query]:
        target_rect = rects_map[query][0]

    # 2. Anlamlı kelime eşleşmesi
    if target_rect is None:
        stopwords = {"ve", "ile", "de", "da", "bir", "bu", "şu", "için", "olan", "olarak", "gibi", "daha", "en"}
        meaningful_words = [w for w in query_words if len(w) >= 3 and tr_lower(w) not in stopwords]
        meaningful_words.sort(key=lambda x: len(x), reverse=True)
        for w in meaningful_words:
            if w in rects_map and rects_map[w]:
                target_rect = rects_map[w][0]
                break

    # Hedef en az %60 genişlik, %28 yükseklik bağlam payı
    min_w = page_w * 0.60
    min_h = page_h * 0.28

    if target_rect is None:
        # Fallback: sayfanın üst-orta bölümü
        cx = page_w * 0.5
        cy = page_h * 0.25
        bw = min_w
        bh = min_h
    else:
        cx = (target_rect.x0 + target_rect.x1) / 2.0
        cy = (target_rect.y0 + target_rect.y1) / 2.0
        bw = max(target_rect.width + 80.0, min_w)
        bh = max(target_rect.height + 100.0, min_h)

    bw = min(bw, page_w)
    bh = min(bh, page_h)

    # page.rect sınırları içine sıkıştırma (clamped)
    x0 = max(0.0, min(cx - bw / 2.0, page_w - bw))
    y0 = max(0.0, min(cy - bh / 2.0, page_h - bh))

    return (
        round(x0 / page_w, 4),
        round(y0 / page_h, 4),
        round(bw / page_w, 4),
        round(bh / page_h, 4)
    )

def render_highlighted_pdf_page(pdf_path, page_num, query_words, query=""):
    """
    Verilen PDF'in ilgili sayfasını tek seferlik 2.0x rasterize eder,
    odak kutusunu hesaplar ve dosya tanıtıcısının kapatılmasını garanti eder.
    Mükerrer highlight annotation'larını koordinat toleransıyla engeller.
    """
    if not os.path.exists(pdf_path):
        return None

    doc = None
    try:
        doc = fitz.open(pdf_path)
        fitz_page_num = int(page_num) - 1

        if fitz_page_num < 0 or fitz_page_num >= len(doc):
            return None

        page = doc.load_page(fitz_page_num)
        rects_map = {}
        annotated_rects = []
        q_normalized = tr_lower(query.strip()) if query else ""

        # 1. Tam ifade araması (Öncelikli)
        if query and query.strip():
            q_clean = query.strip()
            exact_rects = page.search_for(q_clean)
            if exact_rects:
                rects_map[q_clean] = list(exact_rects)
                for rect in exact_rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=[1, 1, 0])
                    annot.update()
                    annotated_rects.append(rect)

        # 2. Bireysel kelime aramaları (Mükerrerleri atla)
        for word in query_words:
            if not word:
                continue
            clean_word = word.strip()
            if not clean_word:
                continue
            if q_normalized and tr_lower(clean_word) == q_normalized:
                continue

            w_rects = page.search_for(clean_word)
            if w_rects:
                if clean_word not in rects_map:
                    rects_map[clean_word] = []
                for rect in w_rects:
                    # Aynı x0, y0, x1 ve y1 koordinatlarına tolerans (0.01 pt) içinde sahip rect'i tekrar annotate etme
                    is_dup = any(
                        abs(rect.x0 - ar.x0) < 0.01 and
                        abs(rect.y0 - ar.y0) < 0.01 and
                        abs(rect.x1 - ar.x1) < 0.01 and
                        abs(rect.y1 - ar.y1) < 0.01
                        for ar in annotated_rects
                    )
                    if not is_dup:
                        annot = page.add_highlight_annot(rect)
                        annot.set_colors(stroke=[1, 1, 0])
                        annot.update()
                        annotated_rects.append(rect)
                    rects_map[clean_word].append(rect)

        # Odak koordinatlarını hesapla (normalize 0..1)
        focus_box = calculate_focus_box(page.rect.width, page.rect.height, query, query_words, rects_map)

        # Net ve keskin önizleme için tek seferlik 2x zoom matrisi
        zoom_matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=zoom_matrix)

        norm_src = os.path.normpath(pdf_path).replace('\\', '/')
        src_hash = hashlib.sha256(norm_src.encode('utf-8')).hexdigest()[:10]
        img_filename = f"page_{src_hash}_p{page_num}.png"
        img_path = os.path.join(TEMP_IMAGE_DIR, img_filename)

        pix.save(img_path)
        total_pages = len(doc)

        return img_path, focus_box, total_pages
    except Exception as e:
        print(f"Resim oluşturma hatası: {e}")
        return None
    finally:
        if doc is not None:
            doc.close()

def search(query):
    """Doğal dil sorgusunu semantik arama ile işler, Neumorphic Glass kartları ve galeri görsellerini üretir."""
    global vector_store, bm25_index, bm25_documents, cross_encoder

    query = query.strip()

    if not query:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
            </div>
            <p class="empty-title">Lütfen aramak istediğiniz konuyu yazın</p>
            <p class="empty-desc">Doğal dilde soru veya anahtar kelime girerek arama yapabilirsiniz.</p>
        </div>
        """
        return empty_html, EMPTY_PREVIEW_HTML

    if vector_store is None or bm25_index is None or cross_encoder is None:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
            </div>
            <p class="empty-title">Belgeler henüz indekslenmedi</p>
            <p class="empty-desc">Önce 'Verileri İndeksle' butonuna basarak belgeleri indekslemelisiniz.</p>
        </div>
        """
        return empty_html, EMPTY_PREVIEW_HTML

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

    if not unique_candidates:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
            </div>
            <p class="empty-title">Sonuç bulunamadı</p>
            <p class="empty-desc">Aranan kelime veya kavram belgelerde yer almıyor olabilir. Farklı bir ifade deneyin.</p>
        </div>
        """
        return empty_html, EMPTY_PREVIEW_HTML

    scored_results = []
    if unique_candidates:
        pairs = [(query_lower, tr_lower(doc.page_content)) for doc in unique_candidates]
        scores = cross_encoder.predict(pairs)

        for doc, score in zip(unique_candidates, scores):
            match_type = "⭐ Hybrid Match"
            pseudo_distance = float(score)
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

    normal_items = []
    ref_items = []
    for original_rank, (res, distance, match_type, extra_terms) in enumerate(results, 1):
        if res.metadata.get("is_reference"):
            ref_items.append((original_rank, res, distance, match_type, extra_terms))
        else:
            normal_items.append((original_rank, res, distance, match_type, extra_terms))

    display_items = [(item, False) for item in normal_items] + [(item, True) for item in ref_items]

    normal_cards_html = []
    ref_cards_html = []
    initial_preview_html = EMPTY_PREVIEW_HTML

    for display_idx, ((original_rank, res, distance, match_type, extra_terms), is_ref_group) in enumerate(display_items):
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

        full_url = ""
        fx, fy, fw, fh = 0.0, 0.0, 1.0, 1.0
        total_pages = page_num

        if file_path:
            render_res = render_highlighted_pdf_page(file_path, page_num, highlight_terms, query)
            if render_res:
                img_path, focus_coords, total_pages = render_res
                fx, fy, fw, fh = focus_coords
                qhash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
                full_url = f"/gradio_api/file={urllib.parse.quote(os.path.abspath(img_path))}?v={qhash}"

        esc_file_name = html.escape(file_name, quote=True)
        esc_full_url = html.escape(full_url, quote=True)
        is_first = (display_idx == 0)
        active_class = " active-result-card" if is_first else ""

        # Reference status & uncalibrated logit score formatting (calibrated accurately)
        has_reference = bool(res.metadata.get("is_reference"))
        ref_badge_html = '<span class="badge-reference">📚 Kaynakça</span>' if has_reference else ''
        score_badge_html = f'<span class="badge-score">Skor: {distance:.4f}</span>' if distance is not None else ''

        card = f"""
        <div class="result-card cursor-pointer{active_class}" id="card-result-{display_idx}" data-index="{display_idx}"
             data-full-url="{esc_full_url}"
             data-file-name="{esc_file_name}"
             data-page-num="{page_num}"
             data-total-pages="{total_pages}"
             data-focus-x="{fx}"
             data-focus-y="{fy}"
             data-focus-w="{fw}"
             data-focus-h="{fh}"
             title="Sağ tarafta bu sayfayı odaklamak için tıklayın">
            <div class="result-header">
                <div class="result-primary-row">
                    <span class="badge-rank">#{original_rank}</span>
                    <span class="badge-file" title="{esc_file_name}">📄 {esc_file_name}</span>
                    <span class="badge-page">Sayfa {page_num}</span>
                </div>
                <div class="result-secondary-row">
                    <div class="badges-meta">
                        <span class="badge-match">{html.escape(match_type)}</span>
                        {ref_badge_html}
                        {score_badge_html}
                    </div>
                    <button type="button" class="btn-jump-page">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                        <span>Sayfayı Gör</span>
                    </button>
                </div>
            </div>
            <div class="result-snippet">
                <div class="quote-bar"></div>
                <p class="snippet-text">"{snippet}"</p>
            </div>
        </div>
        """
        if is_ref_group:
            ref_cards_html.append(card)
        else:
            normal_cards_html.append(card)

        if is_first and full_url:
            initial_preview_html = f"""
            <div class="focused-preview-card" id="focused-preview-card">
                <div class="preview-toolbar">
                    <div class="preview-toolbar-meta">
                        <span class="preview-file-badge" title="{esc_file_name}">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                            <span id="preview-file-name">{esc_file_name}</span>
                        </span>
                        <span class="preview-page-badge" id="preview-page-badge">Sayfa {page_num} / {total_pages}</span>
                    </div>
                    <button type="button" class="btn-open-full" id="btn-open-modal" onclick="window.openPdfModal && window.openPdfModal()" title="Tam sayfayı büyük modalda görüntüle">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
                        <span>Tam Sayfayı Aç</span>
                    </button>
                </div>
                <div class="document-viewer-frame" id="preview-viewport-trigger" onclick="window.openPdfModal && window.openPdfModal()" title="Tam sayfayı açmak için tıklayın">
                    <div class="preview-viewport" id="preview-viewport">
                        <img id="preview-viewport-img" src="{esc_full_url}" alt="Focused PDF Preview" draggable="false"
                             data-focus-x="{fx}" data-focus-y="{fy}" data-focus-w="{fw}" data-focus-h="{fh}" />
                    </div>
                    <div class="preview-viewport-overlay">
                        <span class="preview-hint">
                            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                            <span>Büyütmek için tıklayın</span>
                        </span>
                    </div>
                </div>
            </div>
            """

    details_html = ""
    if ref_cards_html:
        open_attr = " open" if len(normal_cards_html) == 0 else ""
        details_html = f"""
        <details class="references-accordion" id="references-accordion"{open_attr}>
            <summary class="references-summary" id="references-summary">
                <div class="references-summary-content">
                    <span class="references-summary-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>
                    </span>
                    <span class="references-summary-title">Kaynakça Kesitleri ({len(ref_cards_html)})</span>
                </div>
                <span class="references-summary-chevron">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                </span>
            </summary>
            <div class="references-cards-container">
                {''.join(ref_cards_html)}
            </div>
        </details>
        """

    full_html = f"""
    <div class="results-container">
        {''.join(normal_cards_html)}
        {details_html}
    </div>
    """
    return full_html, initial_preview_html

EMPTY_PREVIEW_HTML = """
<div class="empty-state">
    <div class="empty-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
    </div>
    <p class="empty-title">Önizleme için bir kesit seçin</p>
    <p class="empty-desc">Arama sonuçlarından bir kart seçtiğinizde ilgili sayfanın odaklanmış önizlemesi burada gösterilir.</p>
</div>
"""

MODAL_VIEWER_HTML = """
<div id="pdf-viewer-modal" class="pdf-modal-overlay" role="dialog" aria-modal="true" aria-hidden="true" tabindex="-1">
    <div class="pdf-modal-backdrop" id="modal-backdrop"></div>
    <div class="pdf-modal-container">
        <div class="pdf-modal-header">
            <div class="modal-meta">
                <span class="modal-filename" id="modal-filename">Belge.pdf</span>
                <span class="modal-pagebadge" id="modal-pagebadge">Sayfa 1 / 1</span>
            </div>
            <div class="modal-controls">
                <button type="button" class="modal-btn-tool" id="modal-zoom-out" title="Uzaklaştır (−)" aria-label="Uzaklaştır">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
                </button>
                <button type="button" class="modal-btn-tool modal-zoom-reset" id="modal-zoom-reset" title="Sıfırla (%100)" aria-label="Sıfırla">
                    <span id="modal-zoom-label">100%</span>
                </button>
                <button type="button" class="modal-btn-tool" id="modal-zoom-in" title="Yakınlaştır (+)" aria-label="Yakınlaştır">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                </button>
                <button type="button" class="modal-btn-tool modal-btn-close" id="modal-close" title="Kapat (Esc)" aria-label="Kapat">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </div>
        </div>
        <div class="pdf-modal-body" id="modal-viewport">
            <div class="pdf-modal-canvas" id="modal-canvas">
                <img id="modal-full-image" src="data:image/svg+xml;utf8,<svg></svg>" alt="Full PDF Page" draggable="false" />
            </div>
        </div>
    </div>
</div>
"""

# Başlangıç durumu ve geçici dosya temizliği
cleanup_temp_images()
init_message = init_vector_store()
lib_stats = get_library_stats()

BASE_DIR = Path(__file__).resolve().parent
PHASE3_CSS = (BASE_DIR / "static" / "phase3.css").read_text(encoding="utf-8")
PHASE3_JS = (BASE_DIR / "static" / "phase3.js").read_text(encoding="utf-8")

# --- GRADIO ARAYÜZÜ ---
with gr.Blocks(title="Akıllı PDF Arama & Görsel Keşif") as demo:
    # 1. Bütünleşik Belge Kütüphanesi ve Operasyon Kabuğu
    with gr.Accordion("Belge Kütüphanesi", open=False, elem_id="library-accordion"):
        gr.HTML(
            f'<div id="library-stats-meta" style="display:none;" data-doc-count="{lib_stats["pdf_count"]}" data-health="{lib_stats["status"]}" data-is-healthy="{str(lib_stats["is_healthy"]).lower()}" data-time="{lib_stats["last_updated"]}"></div>',
            elem_id="library-stats-meta-container"
        )
        with gr.Row(elem_id="accordion-content-row"):
            with gr.Column(scale=1, elem_id="upload-col"):
                file_upload = gr.File(
                    file_count="multiple",
                    file_types=[".pdf"],
                    elem_id="pdf-upload-zone",
                    label="",
                    container=False
                )
            with gr.Column(scale=1, elem_id="index-action-col"):
                index_btn = gr.Button("Belgeleri İndeksle", elem_id="index-button")
                gr.HTML("""
                <div class="index-status-header">
                    <span class="index-status-label">İNDEKSLEME DURUMU</span>
                </div>
                """)
                index_output = gr.Textbox(
                    label="",
                    value=init_message,
                    interactive=False,
                    elem_id="index-status",
                    lines=3,
                    container=False
                )

    # 3. İki Sütunlu Ana Alan
    with gr.Row(elem_classes=["main-grid-row"]):
        # Sol Panel: Arama ve Metin Kesitleri
        with gr.Column(scale=1, elem_id="search-panel", elem_classes=["saas-card", "scrollable-panel"]):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-header-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
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
                <span class="results-count-badge" id="results-count">0 SONUÇ</span>
            </div>
            """)
            search_output = gr.HTML(
                value="""
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    </div>
                    <p class="empty-title">Aramak istediğiniz konuyu veya soruyu yazın</p>
                    <p class="empty-desc">PDF belgeleriniz içindeki ilgili pasajlar ve alıntılar burada listelenecektir.</p>
                </div>
                """,
                elem_id="search-results"
            )

        # Sağ Panel: Orijinal Sayfa Önizlemeleri (Odaklanmış PDF Önizleme)
        with gr.Column(scale=1, elem_id="preview-panel", elem_classes=["saas-card", "scrollable-panel"]):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-header-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                    <span>Orijinal Sayfa Önizlemeleri</span>
                </div>
            </div>
            """)
            preview_output = gr.HTML(
                value=EMPTY_PREVIEW_HTML,
                elem_id="pdf-preview-container"
            )

    # 4. Kök Modal Görüntüleyici (Blocks Kökünde, Paneller Dışında)
    gr.HTML(MODAL_VIEWER_HTML)

    # Olay Bağlantıları (Event Listeners)
    file_upload.upload(fn=handle_pdf_upload, inputs=[file_upload], outputs=[file_upload, index_output])
    index_btn.click(fn=index_documents, inputs=[], outputs=[index_output])
    search_btn.click(fn=search, inputs=[search_input], outputs=[search_output, preview_output])
    search_input.submit(fn=search, inputs=[search_input], outputs=[search_output, preview_output])

    # İstemci tarafı senkronizasyon betiği - Tek noktadan
    demo.load(js=PHASE3_JS)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        css=PHASE3_CSS,
        theme=gr.themes.Base(),
        allowed_paths=[TEMP_IMAGE_DIR]
    )
