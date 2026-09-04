import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import app

class MockUploadFile:
    """Gradio UploadFile nesnesini simüle eden test sınıfı."""
    def __init__(self, path, orig_name=None):
        self.name = path
        self.orig_name = orig_name or os.path.basename(path)

class TestFileUpload(unittest.TestCase):
    def setUp(self):
        self.original_data_dir = app.DATA_DIR
        self.original_vector_store = app.vector_store
        self.original_bm25_index = app.bm25_index
        self.original_last_index_time = app.last_index_time

        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        app.DATA_DIR = self.data_dir

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        finally:
            app.DATA_DIR = self.original_data_dir
            app.vector_store = self.original_vector_store
            app.bm25_index = self.original_bm25_index
            app.last_index_time = self.original_last_index_time

    def test_sanitize_filename(self):
        # 1. Normal dosya adı
        self.assertEqual(app.sanitize_filename("rapor.pdf"), "rapor.pdf")

        # 2. Uzantısız dosya adı
        self.assertEqual(app.sanitize_filename("makale"), "makale.pdf")

        # 3. Path traversal girişimleri
        self.assertEqual(app.sanitize_filename("../../gizli/belge.pdf"), "belge.pdf")
        self.assertEqual(app.sanitize_filename("..\\..\\windows\\sistem.pdf"), "sistem.pdf")

        # 4. Boş veya None girdi
        self.assertEqual(app.sanitize_filename(""), "belge.pdf")
        self.assertEqual(app.sanitize_filename(None), "belge.pdf")

        # 5. Tehlikeli karakterler (newline, null byte)
        self.assertEqual(app.sanitize_filename("tehlikeli\n\rbelge\0.pdf"), "tehlikelibelge.pdf")

    def test_get_library_stats(self):
        # Boş klasör
        stats = app.get_library_stats()
        self.assertEqual(stats["pdf_count"], 0)
        self.assertIn("status", stats)
        self.assertIn("last_updated", stats)

        # 2 PDF ve 1 TXT dosyası ekleyelim
        with open(os.path.join(self.data_dir, "doc1.pdf"), "w") as f:
            f.write("pdf 1")
        with open(os.path.join(self.data_dir, "doc2.PDF"), "w") as f:
            f.write("pdf 2")
        with open(os.path.join(self.data_dir, "doc3.txt"), "w") as f:
            f.write("text file")

        stats = app.get_library_stats()
        self.assertEqual(stats["pdf_count"], 2)

    def test_valid_pdf_upload_saves_to_data_dir(self):
        source_path = os.path.join(self.test_dir, "source_doc.pdf")
        with open(source_path, "wb") as f:
            f.write(b"%PDF-1.4 Mock PDF Content")

        upload_file = MockUploadFile(source_path, orig_name="ornek_belge.pdf")
        reset_val, result = app.handle_pdf_upload([upload_file])

        self.assertIsNone(reset_val)
        self.assertIn("1 yeni PDF 'data/' klasörüne yüklendi", result)
        self.assertIn("ornek_belge.pdf", result)

        dest_path = os.path.join(self.data_dir, "ornek_belge.pdf")
        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 Mock PDF Content")

    def test_invalid_extension_rejected(self):
        source_path = os.path.join(self.test_dir, "zararli.exe")
        with open(source_path, "wb") as f:
            f.write(b"Mock exe binary")

        upload_file = MockUploadFile(source_path, orig_name="zararli.exe")
        reset_val, result = app.handle_pdf_upload([upload_file])

        self.assertIsNone(reset_val)
        self.assertIn("Reddedilenler", result)
        self.assertIn("Yalnızca PDF kabul edilir", result)

        dest_path = os.path.join(self.data_dir, "zararli.exe")
        self.assertFalse(os.path.exists(dest_path))

    def test_file_size_limit(self):
        source_path = os.path.join(self.test_dir, "buyuk_dosya.pdf")
        with open(source_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        # getsize fonksiyonunu 55 MB döndürecek şekilde mock'layalım
        fake_size = 55 * 1024 * 1024
        with patch('os.path.getsize', return_value=fake_size):
            upload_file = MockUploadFile(source_path, orig_name="buyuk_dosya.pdf")
            reset_val, result = app.handle_pdf_upload([upload_file])

            self.assertIsNone(reset_val)
            self.assertIn("Reddedilenler", result)
            self.assertIn("50 MB sınırını aşıyor", result)

            dest_path = os.path.join(self.data_dir, "buyuk_dosya.pdf")
            self.assertFalse(os.path.exists(dest_path))

    def test_path_traversal_sanitization(self):
        source_path = os.path.join(self.test_dir, "temp_upload.pdf")
        with open(source_path, "wb") as f:
            f.write(b"%PDF-1.4 Traversing PDF")

        upload_file = MockUploadFile(source_path, orig_name="../../etc/saldiri.pdf")
        reset_val, result = app.handle_pdf_upload([upload_file])

        self.assertIsNone(reset_val)
        self.assertIn("1 yeni PDF 'data/' klasörüne yüklendi", result)
        self.assertIn("saldiri.pdf", result)

        safe_dest = os.path.join(self.data_dir, "saldiri.pdf")
        self.assertTrue(os.path.exists(safe_dest))

if __name__ == "__main__":
    unittest.main()
