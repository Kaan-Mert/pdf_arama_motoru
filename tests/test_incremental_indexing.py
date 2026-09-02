import os
import shutil
import tempfile
import unittest
import pickle
import hashlib
from langchain_core.documents import Document

import app

class TestIncrementalIndexing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.chroma_dir = os.path.join(self.test_dir, "chroma_db")
        self.bm25_file = os.path.join(self.test_dir, "bm25_index.pkl")
        
        os.makedirs(self.data_dir)
        
        app.DATA_DIR = self.data_dir
        app.CHROMA_PERSIST_DIR = self.chroma_dir
        app.BM25_PERSIST_FILE = self.bm25_file
        
        app.vector_store = None
        app.bm25_index = None
        app.bm25_documents = []
        
        self.pdf1_path = os.path.join(self.data_dir, "test1.pdf")
        self.pdf2_path = os.path.join(self.data_dir, "test2.pdf")
        
        self.create_mock_pdf(self.pdf1_path, "This is the first test document.")
        self.create_mock_pdf(self.pdf2_path, "This is the second test document with kaynakça.")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def create_mock_pdf(self, path, text):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(fitz.Point(50, 50), text)
        doc.save(path)
        doc.close()
        
    def test_incremental_lifecycle(self):
        # 1. İlk indeksleme
        res = app.index_documents()
        self.assertIn("Eklenen: 2", res)
        self.assertEqual(app.vector_store._collection.count(), len(app.bm25_documents))
        self.assertEqual(len(app.bm25_documents), 2)
        
        # 2. No-op
        res = app.index_documents()
        self.assertIn("Atlanan: 2", res)
        
        # 3. Değişiklik
        self.create_mock_pdf(self.pdf1_path, "This is the updated first document.")
        res = app.index_documents()
        self.assertIn("Güncellenen: 1", res)
        self.assertIn("Atlanan: 1", res)
        self.assertEqual(app.vector_store._collection.count(), len(app.bm25_documents))
        
        # 4. Silme
        os.remove(self.pdf2_path)
        res = app.index_documents()
        self.assertIn("Silinen: 1", res)
        self.assertEqual(app.vector_store._collection.count(), 1)
        self.assertEqual(len(app.bm25_documents), 1)

    def test_legacy_migration(self):
        app.init_vector_store()
        skey = os.path.normpath(self.pdf1_path).replace('\\', '/')
        doc = Document(page_content="Legacy text", metadata={"source": self.pdf1_path, "source_key": skey, "page": 1})
        app.vector_store.add_documents([doc], ids=["legacy_id"])
        
        res = app.index_documents()
        self.assertIn("Güncellenen: 1", res)
        self.assertIn("Eklenen: 1", res)
        
    def test_incomplete_write(self):
        app.init_vector_store()
        hash1 = app.get_file_hash(self.pdf1_path)
        skey = os.path.normpath(self.pdf1_path).replace('\\', '/')
        
        meta = {"source": self.pdf1_path, "source_key": skey, "file_hash": hash1, "chunk_count": 3}
        doc1 = Document(page_content="p1", metadata=meta)
        doc2 = Document(page_content="p2", metadata=meta)
        app.vector_store.add_documents([doc1, doc2], ids=[f"{skey}_{hash1}_0", f"{skey}_{hash1}_1"])
        
        res = app.index_documents()
        self.assertIn("Güncellenen: 1", res)
        self.assertIn("Eklenen: 1", res)
        
if __name__ == '__main__':
    unittest.main()
