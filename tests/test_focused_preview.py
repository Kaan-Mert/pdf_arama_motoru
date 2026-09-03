import unittest
from unittest.mock import patch, MagicMock
import html
import fitz
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

class TestFocusedPreview(unittest.TestCase):

    def test_focus_rect_prefers_exact_query_and_clamps_to_page(self):
        """Tam sorgunun ilk eşleşmesini tercih etmeli, bağlam payı eklemeli ve sayfa sınırları içine sıkıştırmalı."""
        pw, ph = 595.0, 842.0
        query = "Hatay meselesi"
        query_words = ["Hatay", "meselesi", "mesele"]

        rects_map = {
            "Hatay meselesi": [fitz.Rect(480, 750, 580, 770)],
            "Hatay": [fitz.Rect(100, 100, 150, 120)],
            "meselesi": [fitz.Rect(200, 300, 260, 320)]
        }

        fx, fy, fw, fh = app.calculate_focus_box(pw, ph, query, query_words, rects_map)

        # 1. Koordinatlar normalize (0..1 aralığında) olmalı
        self.assertGreaterEqual(fx, 0.0)
        self.assertGreaterEqual(fy, 0.0)
        self.assertLessEqual(fx + fw, 1.0)
        self.assertLessEqual(fy + fh, 1.0)

        # 2. Tam sorgu sağ-alt köşede olduğundan fy alt yarıda olmalı
        self.assertGreater(fy, 0.5)

        # 3. Minimum bağlam payı korunmalı (en az %50 genişlik, %20 yükseklik)
        self.assertGreaterEqual(fw, 0.5)
        self.assertGreaterEqual(fh, 0.2)

    def test_focus_rect_uses_safe_fallback_without_match(self):
        """Eşleşme bulunamadığında güvenli fallback alanı dönmeli ve sınırları aşmamalı."""
        pw, ph = 600.0, 800.0
        query = "bulunmayan_kelime_xyz"
        query_words = ["bulunmayan", "kelime", "xyz"]
        rects_map = {}

        fx, fy, fw, fh = app.calculate_focus_box(pw, ph, query, query_words, rects_map)

        # Sınır denetimi
        self.assertGreaterEqual(fx, 0.0)
        self.assertGreaterEqual(fy, 0.0)
        self.assertLessEqual(fx + fw, 1.0)
        self.assertLessEqual(fy + fh, 1.0)

        # Fallback üst-merkez bölgede olmalı
        self.assertLessEqual(fy, 0.3)
        self.assertGreaterEqual(fw, 0.5)

    def test_search_html_escapes_malicious_metadata(self):
        """Arama çıktısında zararlı karakterler ve metadata XSS vektörlerinin güvenle kaçırıldığını kanıtlar."""
        from langchain_core.documents import Document
        import numpy as np

        malicious_filename = 'evil_<img src=x onerror=alert(1)>_"quote.pdf'
        malicious_doc = Document(
            page_content="Zararlı içerik <script>alert(2)</script> metin kesiti.",
            metadata={
                "source": f"C:/data/{malicious_filename}",
                "page": 1,
                "file_hash": "dummyhash"
            }
        )

        orig_vs = app.vector_store
        orig_bm25 = app.bm25_index
        orig_docs = app.bm25_documents
        orig_ce = app.cross_encoder

        try:
            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = [malicious_doc]

            mock_bm25 = MagicMock()
            mock_bm25.get_scores.return_value = np.array([10.0])

            mock_ce = MagicMock()
            mock_ce.predict.return_value = [0.95]

            app.vector_store = mock_vs
            app.bm25_index = mock_bm25
            app.bm25_documents = [malicious_doc]
            app.cross_encoder = mock_ce

            with patch('app.render_highlighted_pdf_page', return_value=('dummy.png', (0.1, 0.1, 0.5, 0.5), 10)):
                cards_html, preview_html = app.search("zararlı")

            # 1. Ham executable etiketler bulunmamalı
            self.assertNotIn("<img src=x onerror=alert(1)>", cards_html)
            self.assertNotIn("<script>alert(2)</script>", cards_html)
            self.assertNotIn("<img src=x onerror=alert(1)>", preview_html)

            # 2. Kaçırılmış güvenli versiyonlar bulunmalı
            escaped_name = html.escape(malicious_filename, quote=True)
            self.assertIn(escaped_name, cards_html)
            self.assertIn(escaped_name, preview_html)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", cards_html)
            self.assertIn("&lt;script&gt;alert(2)&lt;/script&gt;", cards_html)
        finally:
            app.vector_store = orig_vs
            app.bm25_index = orig_bm25
            app.bm25_documents = orig_docs
            app.cross_encoder = orig_ce

    def test_render_closes_document_on_failure(self):
        """Render sırasında hata oluşsa dahi PDF dokümanı kapatılmalı."""
        mock_doc = MagicMock()
        mock_doc.load_page.side_effect = RuntimeError("Mocked rendering crash")
        mock_doc.__len__.return_value = 10

        with patch('fitz.open', return_value=mock_doc):
            with patch('os.path.exists', return_value=True):
                result = app.render_highlighted_pdf_page("fake_path.pdf", 1, ["test"], "test")

        self.assertIsNone(result)
        mock_doc.close.assert_called_once()

    def test_render_does_not_duplicate_exact_query_annotation(self):
        """Tam sorgu ile kelime araması aynı dikdörtgeni bulduğunda çift annotation eklenmemeli."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__len__.return_value = 5
        mock_doc.load_page.return_value = mock_page

        same_rect = fitz.Rect(100, 200, 300, 250)
        mock_page.rect = fitz.Rect(0, 0, 595, 842)

        def mock_search_for(term):
            if term in ("Hatay meselesi", "Hatay"):
                return [same_rect]
            return []

        mock_page.search_for.side_effect = mock_search_for
        mock_pix = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        with patch('fitz.open', return_value=mock_doc):
            with patch('os.path.exists', return_value=True):
                with patch('os.makedirs'):
                    result = app.render_highlighted_pdf_page(
                        pdf_path="C:/dummy/test.pdf",
                        page_num=1,
                        query_words=["Hatay meselesi", "Hatay"],
                        query="Hatay meselesi"
                    )

        # add_highlight_annot aynı rect için sadece 1 kez çağrılmış olmalı
        self.assertEqual(mock_page.add_highlight_annot.call_count, 1)
        mock_doc.close.assert_called_once()
        self.assertIsNotNone(result)

    def test_search_reranker_contract_and_reference_badge(self):
        """CrossEncoder'a normalize edilmiş çiftlerin gittiğini, ilk 15 sıralamasını ve is_reference rozetini kanıtlar."""
        from langchain_core.documents import Document
        import numpy as np

        test_docs = []
        for i in range(20):
            is_ref = (i == 5)
            doc = Document(
                page_content=f"İÇERİK ÖRNEĞİ NO {i} - TÜRKÇE KARAKTERLER ÇÖĞIŞÜ",
                metadata={
                    "source": f"C:/data/doc_{i}.pdf",
                    "page": i + 1,
                    "file_hash": f"hash_{i}",
                    "is_reference": is_ref
                }
            )
            test_docs.append(doc)

        orig_vs = app.vector_store
        orig_bm25 = app.bm25_index
        orig_docs = app.bm25_documents
        orig_ce = app.cross_encoder

        try:
            mock_vs = MagicMock()
            mock_vs.similarity_search.return_value = test_docs[:10]

            mock_bm25 = MagicMock()
            mock_bm25.get_scores.return_value = np.array([float(i + 1) for i in range(20)])

            mock_ce = MagicMock()
            recorded_pairs = []

            def mock_predict(pairs):
                nonlocal recorded_pairs
                recorded_pairs = list(pairs)
                scores = []
                for p in pairs:
                    if "no 5" in p[1]:
                        scores.append(100.0)
                    else:
                        scores.append(10.0)
                return scores

            mock_ce.predict.side_effect = mock_predict

            app.vector_store = mock_vs
            app.bm25_index = mock_bm25
            app.bm25_documents = test_docs
            app.cross_encoder = mock_ce

            with patch('app.render_highlighted_pdf_page', return_value=('dummy.png', (0.1, 0.1, 0.5, 0.5), 10)):
                cards_html, preview_html = app.search("TÜRKÇE ARAMA")

            # 1. CrossEncoder'a normalize edilmiş çiftlerin gittiği
            self.assertTrue(len(recorded_pairs) > 0)
            query_expected = app.tr_lower("TÜRKÇE ARAMA")
            for q_p, c_p in recorded_pairs:
                self.assertEqual(q_p, query_expected)
                self.assertNotIn("İÇERİK", c_p)
                self.assertIn("içerik", c_p)

            # 2. İlk 15 sıralama kuralı
            card_count = cards_html.count('class="result-card')
            self.assertEqual(card_count, 15)

            # 3. is_reference rozetinin korunduğu
            self.assertIn("📚 Kaynakça", cards_html)
            self.assertIn("⭐ Hybrid Match", cards_html)
        finally:
            app.vector_store = orig_vs
            app.bm25_index = orig_bm25
            app.bm25_documents = orig_docs
            app.cross_encoder = orig_ce

if __name__ == '__main__':
    unittest.main()
