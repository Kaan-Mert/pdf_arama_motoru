import os
import glob
import re
import html
import gradio as gr
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- YAPILANDIRMA ---
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "chroma_db"
TEMP_IMAGE_DIR = "temp_images"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cpu"  # CPU veya CUDA kullanımı
MAX_DISTANCE = 0.65  # Maksimum mesafe eşiği

def tr_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()

def normalize_tr(text):
    """Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) ve büyük harfleri ASCII dengiyle normalleştirir."""
    text = tr_lower(text)
    mapping = str.maketrans("çğıöşü", "cgiosu")
    return text.translate(mapping)

def get_tr_stem(word):
    """Türkçe yaygın çekim ve yapım eklerini temizleyerek kelime kökünü bulur."""
    w = word.lower()
    w = re.sub(r"['’].*$", "", w)
    suffixes = [
        "lerini", "larını", "lerine", "larına", "lerinde", "larında", "lerinden", "larından",
        "sinin", "sının", "sünün", "sunun", "sine", "sına", "süne", "suna", "sinde", "sında", "sünde", "sunda",
        "sinden", "sından", "sünden", "sundan", "leri", "ları", "lerin", "ların", "lere", "lara", "lerde", "larda",
        "lerden", "lardan", "deki", "daki", "teki", "taki",
        "den", "dan", "ten", "tan", "nin", "nın", "nün", "nun", "in", "ın", "ün", "un",
        "ler", "lar", "lik", "lık", "lik", "luk", "de", "da", "te", "ta",
        "ye", "ya", "yi", "yı", "yü", "yu", "si", "sı", "sü", "su",
        "e", "a", "i", "ı", "ü", "u"
    ]
    for suf in suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w

# Sert ünsüzler (Fıstıkçı Şahap) - Ses uyumu kontrolü için
UNVOICED_CONSONANTS = set("fstkçşhpFSTKÇŞHP")

# Temel çekim ekleri (İsmin halleri, çoğul, iyelik)
INFLECTION_SUFFIXES = [
    "lerinin", "larının", "lerine", "larına", "lerinde", "larında", "lerinden", "lardan", "leriyle", "larıyla",
    "sinin", "sının", "sünün", "sunun", "sine", "sına", "süne", "suna", "sinde", "sında", "sünde", "sunda",
    "sinden", "sından", "sünden", "sundan", "siyle", "sıyla",
    "leri", "ları", "lerin", "ların", "lere", "lara", "lerde", "larda", "lerden", "lardan", "lerle", "larla",
    "deki", "daki", "teki", "taki",
    "ler", "lar",
    "nin", "nın", "nün", "nun", "in", "ın", "ün", "un", "im", "ım", "üm", "um",
    "den", "dan", "ten", "tan", "de", "da", "te", "ta",
    "ye", "ya", "yi", "yı", "yü", "yu", "e", "a", "i", "ı", "ü", "u",
    "si", "sı", "sü", "su"
]

# Geçerli türetme / ilişkilendirme ekleri (Ülke, dil, meslek, mensubiyet)
DERIVATION_SUFFIXES = [
    "stan", "istan", "ıstan", "üstan", "ustan",
    "menistan", "manistan",
    "ce", "ca", "çe", "ça",
    "li", "lı", "lu", "lü",
    "ci", "cı", "cü", "cu", "çü", "çu", "çi", "çı",
    "lik", "lık", "lük", "luk",
    "sel", "sal", "sız", "siz", "suz", "süz"
]

# Özel bilinen kök-ülke/kavram türemeleri
SPECIAL_DERIVATIONS = {
    "türk": ["türkiye", "türkmen", "türkmenistan", "türki"],
    "turk": ["turkiye", "turkmen", "turkmenistan", "turki"],
    "ermeni": ["ermenistan"],
    "arap": ["arabistan"],
    "arab": ["arabistan"],
    "yunan": ["yunanistan"],
    "bulgar": ["bulgaristan"],
    "gürcü": ["gürcistan"],
    "gurcu": ["gurcistan"],
    "macar": ["macaristan"],
    "rus": ["rusya"],
}

def check_relation(query_word, doc_word):
    """
    Sorgu kelimesi ile metindeki kelime arasındaki dilbilimsel ve anlamsal ilişkiyi kontrol eder.
    Örn: 'ermeni' -> 'ermenistan' (türemiş), 'bal' -> 'balık' (ilişkisiz/None).
    """
    q = query_word.lower()
    w = doc_word.lower()
    w = re.sub(r"['’].*$", "", w)  # Kesme işaretinden sonrasını temizle
    
    if w == q:
        return "tam", w

    # Özel türeme haritası kontrolü (Türk -> Türkiye, Ermeni -> Ermenistan)
    if q in SPECIAL_DERIVATIONS:
        for spec in SPECIAL_DERIVATIONS[q]:
            if w == spec or w.startswith(spec):
                return "türemiş", spec
                
    if not w.startswith(q):
        return None, None
        
    remainder = w[len(q):]
    last_char_of_q = q[-1]
    
    # Sert ünsüz ses uyumu denetimi (te/ta, ten/tan sadece fstkçşhp sonrası gelebilir; balta, balten elenir)
    if remainder in ("te", "ta", "ten", "tan", "teki", "taki"):
        if last_char_of_q not in UNVOICED_CONSONANTS:
            return None, None
            
    # 1. Çekim eki mi? (bal -> balı, balda; ermeni -> ermeniler)
    for inf in INFLECTION_SUFFIXES:
        if remainder == inf:
            return "çekim", w
            
    # 2. Türetme eki mi? (ermeni -> ermenistan, ermenice; bal -> ballı, balcı)
    for der in DERIVATION_SUFFIXES:
        if remainder == der:
            return "türemiş", w
        if remainder.startswith(der):
            sub_rem = remainder[len(der):]
            if sub_rem in ("te", "ta", "ten", "tan"):
                if der[-1] not in UNVOICED_CONSONANTS:
                    continue
            for inf in INFLECTION_SUFFIXES:
                if sub_rem == inf:
                    return "türemiş", w

    return None, None

def analyze_doc_relations(query, doc_text):
    """Metin içindeki kelimeleri sorgudaki kelimelerle karşılaştırarak tam, çekim ve türemiş eşleşmeleri bulur."""
    words = re.findall(r'\b[a-zA-ZçğıöşüÇĞIŞÖÜ]+\b', doc_text)
    q_words = [w for w in re.split(r'\s+', query.lower()) if len(w) > 1]
    
    found_tam = set()
    found_cekim = set()
    found_turemis = set()
    matched_qwords = set()
    
    for qw in q_words:
        for dw in words:
            rel, matched_w = check_relation(qw, dw)
            if rel == "tam":
                found_tam.add(matched_w)
                matched_qwords.add(qw)
            elif rel == "çekim":
                found_cekim.add(matched_w)
                matched_qwords.add(qw)
            elif rel == "türemiş":
                found_turemis.add(matched_w)
                matched_qwords.add(qw)
                
    return found_tam, found_cekim, found_turemis, matched_qwords

# Temp klasörünü oluştur
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Global vektör veritabanı değişkeni
vector_store = None

# Embedding modelini başlat
print("Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': DEVICE}
)

def init_vector_store():
    """Var olan veritabanını yükler, yoksa None döndürür."""
    global vector_store
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("Var olan ChromaDB yükleniyor...")
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return "⚡ Veritabanı hazır! Aramaya başlayabilirsiniz."
    return "ℹ️ Henüz indekslenmiş veri yok. Lütfen 'Verileri İndeksle' butonuna basınız."

def index_documents():
    """data/ klasöründeki tüm PDF'leri okur, parçalar ve ChromaDB'ye indeksler."""
    global vector_store
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return "⚠️ 'data/' klasörü oluşturuldu. Lütfen içine PDF dosyaları ekleyip tekrar deneyin."
        
    # Eski koleksiyonu temizle (Duplicate önleme)
    if vector_store is not None:
        try:
            vector_store.delete_collection()
        except Exception:
            pass
            
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
            print("Mevcut ChromaDB temizlendi (Duplicate önleme).")
        except Exception as e:
            print(f"Uyarı: Eski ChromaDB klasörü silinemedi. {e}")
    
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        return "⚠️ 'data/' klasöründe hiç PDF bulunamadı."
    
    all_documents = []
    
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Kaynakça / Notlar / Dış bağlantılar başlangıç sayfasını tespit et
            ref_start_page = None
            for i, doc in enumerate(docs, 1):
                lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]
                for l in lines:
                    ll = tr_lower(l)
                    if ll in ["kaynakça", "notlar", "dış bağlantılar", "bibliyografya"]:
                        if ref_start_page is None:
                            ref_start_page = i
                        break
            
            # PyPDFLoader sayfa numaralarını 0-tabanlı başlatır. 
            # Kullanıcı için 1-tabanlı olarak düzeltiyoruz ve kaynakça meta verisini ekliyoruz.
            for i, doc in enumerate(docs, 1):
                doc.metadata["page"] = i
                
                # Metni temizle ve tutarlı hale getir
                text = doc.page_content
                # Tire ile bölünen kelimeleri birleştir (örn: "keli-\nme" -> "kelime")
                text = re.sub(r'-\s*\n\s*', '', text)
                # Rastgele satır sonlarını ve fazla boşlukları temizle
                text = re.sub(r'\s+', ' ', text)
                doc.page_content = text.strip()
                
                is_ref = (ref_start_page is not None and i >= ref_start_page)
                doc.metadata["is_reference"] = is_ref
                doc.metadata["section"] = "kaynakça" if is_ref else "içerik"
                    
            all_documents.extend(docs)
        except Exception as e:
            print(f"Hata: {file_path} okunamadı. Detay: {e}")
            
    if not all_documents:
        return "❌ Dosyalar okunamadı veya içerikleri boş."
        
    # Metinleri parçalama (chunking) - Daha anlamlı bütünlük için boyutu artırdık
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    
    chunks = text_splitter.split_documents(all_documents)
    
    # ChromaDB'ye ekleme ve diske kaydetme
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return f"✅ Başarılı! {len(pdf_files)} PDF dosyasından {len(chunks)} parça başarıyla indekslendi."

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
    global vector_store
    
    query = query.strip()
    
    if not query:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <p class="empty-text">Lütfen aramak istediğiniz konuyu veya soruyu yukarıdaki kutuya yazın.</p>
        </div>
        """
        return empty_html, []
        
    if vector_store is None:
        empty_html = """
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <p class="empty-text">Önce sol taraftaki 'Verileri İndeksle' butonuna basarak belgeleri indekslemelisiniz.</p>
        </div>
        """
        return empty_html, []
        
    query_lower = tr_lower(query)
    query_norm = normalize_tr(query)
    exact_pattern = r'(?<![\wçğışöüÇĞIŞÖÜ])' + re.escape(query_lower) + r'(?![\wçğışöüÇĞIŞÖÜ])'
    exact_norm_pattern = r'(?<![a-z0-9])' + re.escape(query_norm) + r'(?![a-z0-9])'
    words = [w for w in re.split(r'\s+', query_lower) if len(w) > 1]
    stems_norm = [normalize_tr(get_tr_stem(w)) for w in words if len(w) >= 3]
    
    def get_scored_candidates(is_ref_mode=False):
        candidate_map = {}
        # 1. Vektör adaylarını al (k=35)
        try:
            raw_vec = vector_store.similarity_search_with_score(query, k=35, filter={"is_reference": is_ref_mode})
            for doc, dist in raw_vec:
                cid = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}_{doc.page_content[:40]}"
                candidate_map[cid] = (doc, dist)
        except Exception as e:
            print(f"Vektör arama uyarısı: {e}")
            
        # 2. Morfolojik / dilbilimsel ilişki taraması (Ermeni -> Ermenistan gibi türemiş/ilişkili kavramlar)
        try:
            all_chunks = vector_store.get(where={"is_reference": is_ref_mode})
            for doc_text, meta in zip(all_chunks['documents'], all_chunks['metadatas']):
                tam, cekim, turemis, matched_q = analyze_doc_relations(query, doc_text)
                if matched_q:
                    cid = f"{meta.get('source')}_{meta.get('page')}_{doc_text[:40]}"
                    if cid not in candidate_map:
                        doc = Document(page_content=doc_text, metadata=meta)
                        candidate_map[cid] = (doc, 0.50)
        except Exception as e:
            print(f"Koleksiyon tarama uyarısı: {e}")
            
        # 3. Skorla ve sırala
        scored = []
        for doc, distance in candidate_map.values():
            doc_lower = tr_lower(doc.page_content)
            doc_norm = normalize_tr(doc.page_content)
            tam, cekim, turemis, matched_q = analyze_doc_relations(query, doc.page_content)
            
            score = 0
            match_type = ""
            
            # A. Birebir Tam İfade Eşleşmesi (tüm ardışık cümle/ifade)
            if re.search(exact_pattern, doc_lower) or re.search(exact_norm_pattern, doc_norm):
                score = 100 - distance
                match_type = "🎯 Tam Eşleşme"
            # B. Tüm sorgu kelimelerinin tam/çekimli geçmesi
            elif len(words) > 0 and len(matched_q) == len(words):
                if turemis and not (tam or cekim):
                    tur_name = list(turemis)[0].capitalize()
                    score = 70 - distance
                    match_type = f"🌿 Türemiş / İlişkili: {tur_name}"
                elif tam:
                    score = 90 - distance
                    match_type = "🎯 Tam Eşleşme"
                else:
                    score = 80 - distance
                    match_type = "🎯 Çekim Eşleşmesi"
            # C. Tüm Köklerin Birlikte Geçmesi (Örn: Hatay + Mesele)
            elif len(stems_norm) > 1 and all(re.search(r'(?<![a-z0-9])' + re.escape(s), doc_norm) for s in stems_norm):
                score = 75 - distance
                match_type = "🎯 Kavram Eşleşmesi"
            # D. Tek kelimelik sorgularda türemiş kelime (Örn: ermeni -> Ermenistan)
            elif len(words) == 1 and turemis:
                tur_name = list(turemis)[0].capitalize()
                score = 65 - distance
                match_type = f"🌿 Türemiş / İlişkili: {tur_name}"
            # E. Kısmi eşleşmeler (Yalnızca çok kelimeli sorgularda geçerli)
            elif len(words) > 1 and matched_q:
                ratio = len(matched_q) / len(words)
                score = (40 * ratio) - distance
                match_type = "🧩 Konu Eşleşmesi"
            elif distance <= 0.35 and len(query.strip()) > 3:
                score = 10 - distance
                match_type = "🧠 Semantik"
                    
            if score > 0:
                if is_ref_mode:
                    match_type += " • 📚 Kaynakça"
                extra_terms = list(tam | cekim | turemis)
                scored.append((score, doc, distance, match_type, extra_terms))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # 1. Aşama: İlk olarak ana içerik parçalarını ara
    scored_main = get_scored_candidates(is_ref_mode=False)
        
    # 2. Aşama: Eğer ana içerikte sonuç varsa, KESİNLİKLE sadece ana içeriği göster (kaynakçayı gizle)
    if scored_main:
        scored_results = scored_main
    else:
        # 3. Aşama: Ana içerikte HİÇBİR sonuç çıkmadıysa, o takdirde kaynakçayı ara ve yedek olarak sun
        scored_results = get_scored_candidates(is_ref_mode=True)

    results = [(item[1], item[2], item[3], item[4]) for item in scored_results]
    
    print("\n=== SEARCH DEBUG ===")
    print(f"Query: '{query}'")
    print(f"Filtered and sorted results: {len(results)}")
    print("==================\n")
    
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
        
        # 1. Metni al
        # 2. HTML Injection riskine karşı escape et (Highlight işleminden ÖNCE!)
        snippet = html.escape(res.page_content.strip())
        
        # 3. Akıllı Vurgulama (Hem tam sorgu, hem türemiş kavramlar, hem kökler)
        # Önce uzun olan kelimeleri vurgulamak için uzunluğa göre azalan sırala (örn: 'Ermenistan', 'Ermeni'den önce vurgulanır)
        raw_terms = [query] + list(extra_terms) + [w for w in re.split(r'\s+', query) if len(w) > 2]
        highlight_terms = sorted(set([t.strip() for t in raw_terms if t and t.strip()]), key=lambda x: len(x), reverse=True)
        for term in highlight_terms:
            pattern = re.compile(rf'(?<![<a-zA-ZçğıöşüÇĞIŞÖÜ])({re.escape(term)})(?![>a-zA-ZçğıöşüÇĞIŞÖÜ])', re.IGNORECASE)
            # Eğer sınır eşleşmediyse genel regex ile dene ama mark taglerini bozma
            if not pattern.search(snippet):
                pattern = re.compile(rf'({re.escape(term)})', re.IGNORECASE)
            snippet = pattern.sub(r'<mark style="background-color: #fef08a; color: #0f172a; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\1</mark>', snippet)
        
        # HTML Kart Oluşturma (Tıklama ile sağ panelde sayfayı açma desteği)
        card = f"""
        <div class="result-card cursor-pointer" id="card-result-{i-1}" data-index="{i-1}" onclick="window.selectPdfResult && window.selectPdfResult({i-1})" title="Sağ tarafta bu sayfayı açmak için tıklayın">
            <div class="result-header">
                <div class="badges-group">
                    <span class="badge-index">#{i}</span>
                    <span class="badge-file">📄 {file_name}</span>
                    <span class="badge-page">Sayfa {page_num}</span>
                    <span class="badge-page" style="background: rgba(167, 243, 208, 0.8); color: #065f46; border-color: rgba(167, 243, 208, 1);">{match_type}</span>
                    <span class="badge-page" style="background: rgba(147, 197, 253, 0.8);">Mesafe: {distance:.4f}</span>
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
                caption = f"Sonuç #{i} • {file_name} (Sayfa {page_num} - Mesafe: {distance:.4f})"
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

/* Eski tema neon ışık kürelerini gizleme */
.ambient-glow-wrapper {
    display: none !important;
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

# --- AMBIENT & HERO BAŞLIK HTML ---
AMBIENT_BG_HTML = """
<div class="ambient-glow-wrapper">
    <div class="glow-orb orb-cyan"></div>
    <div class="glow-orb orb-purple"></div>
    <div class="glow-orb orb-pink"></div>
</div>
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
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS, theme=gr.themes.Base())
