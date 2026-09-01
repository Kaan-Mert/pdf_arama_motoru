# Source Finder: PDF Search & Bibliography Retrieval Engine

Source Finder is a Retrieval-Augmented Generation (RAG) search engine designed for dense, multi-page PDF documents and academic archives. It provides semantic search, rule-based morphological derivation filtering, adaptive bibliography separation, and zero-latency client-side page synchronization.

---

## Key Features

- **Morphological & Relational Derivation Engine**:
  - Differentiates valid linguistic derivations and inflections from coincidental substrings.
  - Automatically identifies and matches related terms (e.g., searching `ermeni` properly matches `ermenistan`, `ermenice`, `ermeniler` with a dedicated derived-concept badge).
  - Strictly prevents false positive substring matches (e.g., searching `bal` will never match unrelated tokens like `balık`, `balkon`, or `balta`).

- **Adaptive Bibliography Filtering**:
  - Distinguishes main narrative content from references and bibliographies during indexing.
  - Prioritizes results from the main document body. Falls back to citation sections only if no primary matches exist, clearly tagging them as references.

- **Concept Multi-Stem Matching**:
  - Scores queries with multiple conceptual roots (e.g., `Hatay Meselesi`) to ensure pages containing all relevant roots in context rank at the top.

- **Bidirectional Zero-Latency Page Preview**:
  - Clicking any result snippet card on the left panel immediately focuses the corresponding high-resolution rendered PDF page on the right panel via client-side JavaScript.
  - Matches and derived keywords are dynamically highlighted with yellow background markers on both the text snippet and the rendered PDF page.

- **Vintage Glassmorphism Interface**:
  - Ruled paper background effect, responsive layout, custom typography (Lora and Playfair Display), and clean visual feedback states.

---

## System Architecture

```
[ User Query ]
      |
      +---> Vector Similarity Search (HuggingFace Embeddings / ChromaDB)
      |
      +---> Morphological & Linguistic Relation Analyzer (check_relation)
      |
      v
[ Hybrid Candidate Pool ]
      |
      +---> Exact Match / Inflection Match
      +---> Multi-Stem Concept Verification
      +---> Derived Concept Identification
      +---> Semantic Threshold & False Positive Elimination
      |
      v
[ Ranked Results & Snippets ] <===> [ Synced PDF Page Previews ]
```

---

## Tech Stack

- **Core / Backend**: Python 3.10+, PyMuPDF (Fitz), LangChain, HuggingFace Embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Store**: ChromaDB
- **User Interface**: Gradio Blocks, Vanilla CSS3, Client-Side JavaScript Synchronization

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Kaan-Mert/pdf_arama_motoru.git
cd pdf_arama_motoru
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:7860`.

---

## Project Structure

```
.
|-- app.py              # Gradio web interface, RAG pipeline & morphology engine
|-- requirements.txt    # Project dependencies
|-- .gitignore          # Git exclusion rules
|-- data/               # Target PDF documents for indexing
`-- README.md           # Project documentation
```
