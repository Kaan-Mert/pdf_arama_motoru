import os
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

def sync_bm25():
    """Chroma'daki kayıtlardan BM25'i yeniden inşa eder."""
    global vector_store, bm25_index, bm25_documents
    print("Chroma'dan BM25'e senkronize ediliyor...")
    
    all_data = vector_store.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    
    if not docs or not metadatas or len(docs) != len(metadatas):
        print("Uyarı: Senkronizasyon için geçerli doküman yok.")
        bm25_index = None
        bm25_documents = []
        if os.path.exists(BM25_PERSIST_FILE):
            os.remove(BM25_PERSIST_FILE)
        return
        
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
    fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump({
            "bm25": bm25_index,
            "documents": bm25_documents
        }, f)
    os.replace(temp_path, BM25_PERSIST_FILE)

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
                bm25_index = data["bm25"]
                bm25_documents = data["documents"]
            
            # Senkronizasyon kontrolü
            chroma_count = vector_store._collection.count()
            if chroma_count == len(bm25_documents):
                bm25_valid = True
            else:
                print("BM25 belge sayısı Chroma ile eşleşmiyor. Yeniden oluşturulmalı.")
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
    global vector_store
    
    if vector_store is None:
        init_vector_store()
        
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return "⚠️ 'data/' klasörü oluşturuldu. Lütfen içine PDF dosyaları ekleyip tekrar deneyin."
        
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf")) + glob.glob(os.path.join(DATA_DIR, "*.PDF"))
    pdf_files = list(set(pdf_files))
    
    disk_files_info = {}
    for f in pdf_files:
        source_key = os.path.normpath(f).replace('\\', '/')
        file_hash = get_file_hash(f)
        disk_files_info[source_key] = {"path": f, "hash": file_hash}
        
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
        
    added = 0
    updated = 0
    deleted = 0
    skipped = 0
    errors = 0
    chroma_changed = False
    
    for source_key, data in chroma_sources.items():
        if source_key not in disk_files_info:
            print(f"Siliniyor (orphan): {source_key}")
            vector_store.delete(ids=data["ids"])
            deleted += 1
            chroma_changed = True
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    
    for source_key, info in disk_files_info.items():
        needs_indexing = False
        reason = ""
        action_stat = ""
        
        if source_key not in chroma_sources:
            needs_indexing = True
            reason = "Yeni dosya"
            action_stat = "added"
        else:
            c_data = chroma_sources[source_key]
            if c_data["hash"] is None:
                needs_indexing = True
                reason = "Legacy migration"
                action_stat = "updated"
            elif c_data["hash"] != info["hash"]:
                needs_indexing = True
                reason = "Dosya değişti"
                action_stat = "updated"
            elif c_data["chunk_count"] != len(c_data["ids"]):
                needs_indexing = True
                reason = "Yarım kalmış indeks"
                action_stat = "updated"
            else:
                skipped += 1
                
        if needs_indexing:
            print(f"İşleniyor: {source_key} - {reason}")
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
                
                for i, chunk in enumerate(chunks):
                    chunk.metadata["file_hash"] = info["hash"]
                    chunk.metadata["chunk_count"] = chunk_count
                    
                chunk_ids = [f"{source_key}_{info['hash']}_{i}" for i in range(chunk_count)]
                
                if source_key in chroma_sources and chroma_sources[source_key]["ids"]:
                    vector_store.delete(ids=chroma_sources[source_key]["ids"])
                        
                vector_store.add_documents(documents=chunks, ids=chunk_ids)
                chroma_changed = True
                
                if action_stat == "added":
                    added += 1
                else:
                    updated += 1
                    
            except Exception as e:
                print(f"Hata: {source_key} okunamadı. Detay: {e}")
                errors += 1
                
    if chroma_changed or not bm25_index or len(bm25_documents) != vector_store._collection.count():
        sync_bm25()
        
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

# --- VINTAGE KAĞIT, KURŞUN KALEM & DOLMA KALEM (GLASSMORPHISM) CUSTOM CSS ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600;1,700&display=swap');

/* 1. Saf CSS Çizgili Defter Yaprağı / Eski Kağıt Arka Planı */
body, .gradio-container {
    font-family: 'Lora', Georgia, serif !important;
    background-color: #f4f1e8 !important;
    background-image: repeating-linear-gradient(
        transparent,
        transparent 31px,
        rgba(110, 115, 120, 0.16) 31px,
        rgba(110, 115, 120, 0.16) 32px
    ) !important;
    background-size: 100% 32px !important;
    color: #3b3b3b !important;
    position: relative;
    min-height: 100vh;
}

/* 2. Başlıklar & Tipografi (Dolma Kalem Laciverti & Kurşun Kalem Grisi) */
h1, h2, h3, .panel-title, .hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #2c3e50 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    border-bottom: 1px solid rgba(44, 62, 80, 0.25) !important;
    padding-bottom: 6px !important;
    margin-bottom: 14px !important;
}

.hero-header {
    text-align: center;
    margin: 15px auto 35px auto;
    position: relative;
    z-index: 1;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 18px;
    background: rgba(255, 253, 248, 0.85) !important;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(44, 62, 80, 0.35) !important;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #2c3e50 !important;
    box-shadow: 2px 3px 8px rgba(60, 50, 40, 0.05);
    margin-bottom: 12px;
}

.hero-title {
    font-size: 2.3rem !important;
    margin: 0 !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
    background: none !important;
    -webkit-text-fill-color: initial !important;
}

.hero-subtitle {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.05rem;
    color: #5c5c5c !important;
    font-style: italic;
    margin-top: 8px;
}

/* 3. Paneller & Glassmorphism */
.glass-panel {
    background: rgba(255, 253, 248, 0.82) !important;
    backdrop-filter: blur(6px) !important;
    -webkit-backdrop-filter: blur(6px) !important;
    border: 1px solid rgba(180, 170, 155, 0.5) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(70, 60, 50, 0.06), 0 1px 3px rgba(70, 60, 50, 0.04) !important;
    padding: 22px !important;
    position: relative;
    z-index: 1;
    transition: all 0.25s ease;
}

.glass-panel:hover {
    box-shadow: 0 6px 20px rgba(70, 60, 50, 0.09) !important;
}

/* 4. Arama Çubuğu & Input Alanları */
.glass-input input, .glass-input textarea {
    background: rgba(255, 253, 248, 0.85) !important;
    backdrop-filter: blur(4px) !important;
    -webkit-backdrop-filter: blur(4px) !important;
    border: 1px solid rgba(135, 130, 120, 0.55) !important;
    border-radius: 8px !important;
    box-shadow: inset 1px 1px 4px rgba(70, 60, 50, 0.08) !important;
    color: #3b3b3b !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.98rem !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

.glass-input input:focus, .glass-input textarea:focus {
    border-color: #2c3e50 !important;
    box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.12), inset 1px 1px 3px rgba(70, 60, 50, 0.06) !important;
    outline: none !important;
}

/* 5. Butonlar (Dolma Kalem Laciverti & Zarif Kağıt) */
.btn-primary {
    background: #2c3e50 !important;
    color: #fbf9f5 !important;
    border: 1px solid #1a252f !important;
    border-radius: 8px !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    padding: 11px 22px !important;
    box-shadow: 2px 3px 8px rgba(44, 62, 80, 0.22) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.btn-primary:hover {
    background: #1b2836 !important;
    transform: translateY(-1px) !important;
    box-shadow: 3px 5px 12px rgba(44, 62, 80, 0.3) !important;
}

.btn-primary:active {
    transform: translateY(1px) !important;
    box-shadow: inset 1px 1px 4px rgba(0, 0, 0, 0.3) !important;
}

.btn-secondary {
    background: rgba(255, 253, 248, 0.9) !important;
    color: #2c3e50 !important;
    border: 1px solid rgba(44, 62, 80, 0.45) !important;
    border-radius: 8px !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 18px !important;
    box-shadow: 1px 2px 6px rgba(70, 60, 50, 0.08) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.btn-secondary:hover {
    background: #fdfbf7 !important;
    border-color: #2c3e50 !important;
    transform: translateY(-1px) !important;
}

/* 6. Sonuç Kartları (Kağıt + Glassmorphism + Kesikli Kenarlar + Kalın Sol Kenar) */
.results-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 10px;
}

.result-card {
    background: rgba(255, 253, 248, 0.85) !important;
    backdrop-filter: blur(4px) !important;
    -webkit-backdrop-filter: blur(4px) !important;
    border-top: 1px dashed rgba(160, 150, 135, 0.55) !important;
    border-right: 1px dashed rgba(160, 150, 135, 0.55) !important;
    border-bottom: 1px dashed rgba(160, 150, 135, 0.55) !important;
    border-left: 5px solid #2c3e50 !important;
    border-radius: 4px 10px 10px 4px !important;
    padding: 16px 20px !important;
    box-shadow: 2px 4px 12px rgba(70, 60, 50, 0.05) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.result-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 3px 6px 16px rgba(70, 60, 50, 0.12) !important;
    background: rgba(255, 254, 250, 0.95) !important;
    border-color: rgba(44, 62, 80, 0.35) !important;
}

.result-card.active-result-card {
    border-left: 6px solid #1a252f !important;
    border-top: 1.5px solid rgba(44, 62, 80, 0.75) !important;
    border-right: 1.5px solid rgba(44, 62, 80, 0.75) !important;
    border-bottom: 1.5px solid rgba(44, 62, 80, 0.75) !important;
    background: rgba(255, 253, 248, 0.98) !important;
    box-shadow: 0 6px 22px rgba(44, 62, 80, 0.16) !important;
}

.btn-jump-page {
    background: rgba(44, 62, 80, 0.08);
    color: #2c3e50;
    border: 1px solid rgba(44, 62, 80, 0.25);
    padding: 3px 10px;
    border-radius: 12px;
    font-family: 'Lora', Georgia, serif;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
}

.btn-jump-page:hover {
    background: #2c3e50;
    color: #fff;
    border-color: #2c3e50;
    box-shadow: 0 2px 6px rgba(44, 62, 80, 0.25);
}

.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.badges-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

/* 7. Vurgular ve Etiketler (Soluk Mürekkep Mavisi & Eskimiş Kağıt Sarısı) */
.badge-index {
    background: #2c3e50 !important;
    color: #fbf9f5 !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 4px 9px !important;
    border-radius: 4px !important;
    border: 1px solid #1a252f !important;
}

.badge-file {
    background: rgba(246, 238, 220, 0.95) !important;
    color: #42382c !important;
    border: 1px solid rgba(185, 168, 140, 0.65) !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    border-radius: 4px !important;
}

.badge-page {
    background: rgba(226, 236, 244, 0.9) !important;
    color: #1e3a5f !important;
    border: 1px solid rgba(145, 175, 205, 0.65) !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    border-radius: 4px !important;
}

.result-snippet {
    position: relative;
    padding-left: 14px;
}

.quote-bar {
    position: absolute;
    left: 0;
    top: 2px;
    bottom: 2px;
    width: 3px;
    border-radius: 2px;
    background: rgba(44, 62, 80, 0.6) !important;
}

.snippet-text {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.94rem !important;
    line-height: 1.65 !important;
    color: #3b3b3b !important;
    margin: 0;
}

/* 8. Kurumuş Fosforlu Kalem Sarısı Metin Vurgulama */
mark, .result-snippet mark {
    background-color: rgba(255, 235, 59, 0.4) !important;
    color: #2c3e50 !important;
    padding: 1px 4px !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    border-bottom: 1px dashed rgba(200, 170, 40, 0.6) !important;
}

/* 9. Galeri ve Boş Durumlar */
.empty-state {
    text-align: center;
    padding: 32px 18px;
    background: rgba(255, 253, 248, 0.5) !important;
    border: 1px dashed rgba(160, 150, 135, 0.6) !important;
    border-radius: 8px;
}

.empty-icon {
    font-size: 2rem;
    margin-bottom: 6px;
    opacity: 0.75;
}

.empty-text {
    color: #5c5c5c !important;
    font-family: 'Lora', Georgia, serif !important;
    font-style: italic;
    font-size: 0.92rem;
    margin: 0;
}

.glass-gallery {
    background: transparent !important;
    border: none !important;
}

.glass-gallery .gallery-item {
    background: rgba(255, 253, 248, 0.85) !important;
    border: 1px solid rgba(180, 170, 155, 0.5) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(70, 60, 50, 0.06) !important;
    overflow: hidden !important;
}

/* Standart Gradio Çerçevelerini Temizleme & Özel Scrollbar */
.gradio-container .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.scrollable-panel {
    max-height: 75vh;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
}

.scrollable-panel > * {
    flex-shrink: 0 !important;
    width: 100% !important;
}

.scrollable-panel::-webkit-scrollbar { width: 6px; }
.scrollable-panel::-webkit-scrollbar-track { background: transparent; }
.scrollable-panel::-webkit-scrollbar-thumb { background: rgba(44, 62, 80, 0.25); border-radius: 4px; }
.scrollable-panel::-webkit-scrollbar-thumb:hover { background: rgba(44, 62, 80, 0.5); }
"""

CLIENT_SYNC_JS = """
() => {
    window.selectPdfResult = function(idx) {
        // 1. Sol paneldeki kartları güncelle
        document.querySelectorAll('.result-card').forEach((c, i) => {
            if (i === idx) {
                c.classList.add('active-result-card');
            } else {
                c.classList.remove('active-result-card');
            }
        });
        // 2. Sağ paneldeki galeri küçük resmini (thumbnail) tıkla
        const thumbs = document.querySelectorAll('.glass-gallery button.thumbnail-item');
        if (thumbs && thumbs[idx]) {
            thumbs[idx].click();
            thumbs[idx].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }
    };

    // Global tıklama dinleyicisi (Event delegation)
    if (!window._pdf_click_listener_bound) {
        window._pdf_click_listener_bound = true;
        document.addEventListener('click', function(e) {
            const card = e.target.closest('.result-card');
            if (card) {
                const idxAttr = card.getAttribute('data-index');
                if (idxAttr !== null) {
                    const idx = parseInt(idxAttr, 10);
                    if (!isNaN(idx)) {
                        window.selectPdfResult(idx);
                    }
                }
            }
        });
    }

    // İlk kartı otomatik seç
    setTimeout(() => {
        const first = document.querySelector('.result-card');
        if (first && !document.querySelector('.active-result-card')) {
            first.classList.add('active-result-card');
        }
    }, 200);
}
"""

# --- HERO BAŞLIK HTML ---
AMBIENT_BG_HTML = """
<div class="hero-header">
    <div class="hero-pill">
        <span>📜</span> Semantik Doküman Kütüphanesi
    </div>
    <h1 class="hero-title">Akıllı PDF Arama & Görsel Keşif</h1>
    <p class="hero-subtitle">Yapay zeka ile PDF sayfalarınızda anlamsal araştırma yapın ve orijinal metinleri keşfedin.</p>
</div>
"""

# --- GRADIO ARAYÜZÜ ---
with gr.Blocks(title="Source Finder - PDF Arama ve Kaynakça Bulucu") as demo:
    # Arka plan küreleri ve Hero Başlık
    gr.HTML(AMBIENT_BG_HTML, js_on_load=CLIENT_SYNC_JS)
    
    with gr.Column(elem_classes=["glass-panel"]):
        gr.HTML('<div class="panel-title">📂 Veri Seti Yönetimi</div>')
        with gr.Row():
            index_btn = gr.Button("🔄 Belgeleri İndeksle", elem_classes=["btn-secondary", "btn-primary"], variant="primary")
            index_output = gr.Textbox(
                label="İndeksleme Durumu",
                value=init_message,
                interactive=False,
                elem_classes=["glass-input"],
                lines=1
            )
        
        gr.HTML('<div class="panel-title" style="margin-top: 16px;">🔍 Akıllı Semantik Arama</div>')
        with gr.Row():
            search_input = gr.Textbox(
                label="",
                placeholder="Örn: Türk bayrağı kanunu nedir veya Cumhuriyet nasıl ilan edildi?",
                scale=4,
                elem_classes=["glass-input"],
                container=False
            )
            search_btn = gr.Button("Ara ⚡", scale=1, elem_classes=["btn-primary"], variant="primary")
            
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["glass-panel", "scrollable-panel"]):
            gr.HTML('<div class="panel-title">📑 Metin Kesitleri</div>')
            search_output = gr.HTML(
                value="""
                <div class="empty-state">
                    <div class="empty-icon">🔎</div>
                    <p class="empty-text">Aramak istediğiniz soruyu yukarıya yazıp 'Ara' butonuna basın.</p>
                </div>
                """,
                js_on_load=CLIENT_SYNC_JS
            )
        with gr.Column(scale=1, elem_classes=["glass-panel", "scrollable-panel"]):
            gr.HTML('<div class="panel-title">🖼️ Orijinal Sayfa Önizlemeleri</div>')
            gallery_output = gr.Gallery(
                label="Orijinal PDF Sayfaları",
                show_label=False,
                elem_classes=["glass-gallery"],
                columns=[1],
                rows=[1],
                object_fit="contain",
                height="auto"
            )
            
    # Olay Bağlantıları (Event Listeners)
    index_btn.click(fn=index_documents, inputs=[], outputs=[index_output])
    
    # Arama tetikleyicileri
    search_btn.click(fn=search, inputs=[search_input], outputs=[search_output, gallery_output])
    search_input.submit(fn=search, inputs=[search_input], outputs=[search_output, gallery_output])
    
    # İstemci tarafı senkronizasyon betiği
    demo.load(js=CLIENT_SYNC_JS)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, css=CUSTOM_CSS, theme=gr.themes.Base())
