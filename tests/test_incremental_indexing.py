import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pickle
import hashlib
from langchain_core.documents import Document

import app

class TestIncrementalIndexing(unittest.TestCase):
    def setUp(self):
        self.original_state = {
            "DATA_DIR": app.DATA_DIR,
            "CHROMA_PERSIST_DIR": app.CHROMA_PERSIST_DIR,
            "BM25_PERSIST_FILE": app.BM25_PERSIST_FILE,
            "vector_store": app.vector_store,
            "bm25_index": app.bm25_index,
            "bm25_documents": app.bm25_documents,
            "cross_encoder": app.cross_encoder
        }

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
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        finally:
            app.DATA_DIR = self.original_state["DATA_DIR"]
            app.CHROMA_PERSIST_DIR = self.original_state["CHROMA_PERSIST_DIR"]
            app.BM25_PERSIST_FILE = self.original_state["BM25_PERSIST_FILE"]
            app.vector_store = self.original_state["vector_store"]
            app.bm25_index = self.original_state["bm25_index"]
            app.bm25_documents = self.original_state["bm25_documents"]
            app.cross_encoder = self.original_state["cross_encoder"]

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
        bm25_mtime = os.path.getmtime(self.bm25_file)
        with patch('app.PyMuPDFLoader') as mock_loader, patch('app.sync_bm25') as mock_sync:
            res = app.index_documents()
            self.assertIn("Atlanan: 2", res)
            mock_loader.assert_not_called()
            mock_sync.assert_not_called()
            self.assertEqual(bm25_mtime, os.path.getmtime(self.bm25_file))

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
        doc = Document(page_content="Legacy text", metadata={"source": self.pdf1_path, "page": 1})
        app.vector_store.add_documents([doc], ids=["legacy_id"])

        res = app.index_documents()
        self.assertIn("Güncellenen: 1", res)
        self.assertIn("Eklenen: 1", res)

        # Doğrulama: Eski legacy ID silinmeli, yeni chunk'larda file_hash ve chunk_count olmalı
        data = app.vector_store.get()
        self.assertNotIn("legacy_id", data["ids"])
        for meta in data["metadatas"]:
            self.assertIn("source_key", meta)
            self.assertIn("file_hash", meta)
            self.assertIn("chunk_count", meta)

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

    def test_transaction_rollback(self):
        # Önce sağlıklı bir PDF indeksle
        app.index_documents()
        initial_count = app.vector_store._collection.count()
        bm25_mtime = os.path.getmtime(self.bm25_file)
        old_data = app.vector_store.get()

        # Dosyayı değiştir
        self.create_mock_pdf(self.pdf1_path, "This is a changed document that will fail.")

        # add_documents hata fırlatacak şekilde mockla
        with patch.object(app.vector_store, 'add_documents', side_effect=Exception("Mocked DB error")):
            res = app.index_documents()
            self.assertIn("Hatalı: 1", res)

        # Eski kayıtlar durmalı, BM25 değişmemeli
        self.assertEqual(app.vector_store._collection.count(), initial_count)
        self.assertEqual(bm25_mtime, os.path.getmtime(self.bm25_file))
        new_data = app.vector_store.get()
        self.assertEqual(set(old_data["ids"]), set(new_data["ids"]))

    def test_rollback_flow_logic(self):
        # 1. Sağlıklı bir PDF indeksle
        app.index_documents()
        initial_count = app.vector_store._collection.count()
        old_data = app.vector_store.get()
        bm25_mtime = os.path.getmtime(self.bm25_file)

        # 2. PDF içeriğini değiştir
        self.create_mock_pdf(self.pdf1_path, "This is a completely different document.")

        # 3. Yalnızca eski ID'lerin silinmesi anında delete çağrısını hata fırlatacak şekilde mockla
        original_delete = app.vector_store.delete
        delete_call_count = 0

        def mock_delete(ids, **kwargs):
            nonlocal delete_call_count
            delete_call_count += 1
            if delete_call_count == 1:
                raise Exception("Mocked delete old_ids error")
            return original_delete(ids=ids, **kwargs)

        with patch.object(app.vector_store, 'delete', side_effect=mock_delete):
            with patch('builtins.print') as mock_print:
                res = app.index_documents()

                # Verify that the correct rollback message was printed
                error_printed = False
                for call in mock_print.call_args_list:
                    if "İşlem iptal edildi; yeni ID'ler başarıyla geri alındı" in str(call):
                        error_printed = True
                        break
                self.assertTrue(error_printed, "Rollback message was not printed correctly.")
            self.assertIn("Hatalı: 1", res)

        # 4. Doğrula: new_ids kalmadı, eski ID'ler korundu, toplam sayı aynı, bm25 aynı.
        self.assertEqual(app.vector_store._collection.count(), initial_count)
        self.assertEqual(bm25_mtime, os.path.getmtime(self.bm25_file))
        new_data = app.vector_store.get()
        self.assertEqual(set(old_data["ids"]), set(new_data["ids"]))

    def test_in_memory_purge_on_bm25_failure(self):
        app.index_documents()
        self.create_mock_pdf(self.pdf1_path, "This is a changed document for memory purge test.")

        with patch('os.replace', side_effect=PermissionError("Mocked write error")):
            res = app.index_documents()
            self.assertIn("BM25 senkronizasyonu başarısız oldu", res)

        self.assertIsNone(app.bm25_index)
        self.assertEqual(app.bm25_documents, [])

        # Ayrıca search denendiğinde de indeksleme gerektiğini söylemeli
        search_res = app.search("test")
        self.assertIn("Verileri İndeksle", search_res[0])

    def test_manifest_detects_orphan_chunk(self):
        # 1. Sağlıklı Chroma ve BM25 indeksi oluştur.
        app.index_documents()

        # 2. Silme işlemi öncesinde sunucunun temiz başladığını teyit et
        app.vector_store = None
        app.bm25_index = None
        app.bm25_documents = []
        app.cross_encoder = None

        msg = app.init_vector_store()
        self.assertIn("Veritabanı hazır", msg)
        self.assertIsNotNone(app.bm25_index)
        self.assertEqual(len(app.bm25_documents), app.vector_store._collection.count())

        # 3. Chroma'dan yalnızca bir chunk ID'sini manuel olarak sil; diğer metadata'yı değiştirme.
        all_data = app.vector_store.get()
        id_to_delete = all_data["ids"][0]
        app.vector_store.delete(ids=[id_to_delete])

        # 4. Runtime state'ini sıfırla.
        app.vector_store = None
        app.bm25_index = None
        app.bm25_documents = []
        app.cross_encoder = None

        # 5. init_vector_store() çağır.
        app.init_vector_store()

        # 6. actual_count veya ids_digest farkı nedeniyle BM25'in geçersiz sayıldığını doğrula.
        self.assertIsNone(app.bm25_index)
        self.assertEqual(app.bm25_documents, [])

        # 7. Aramanın engellendiğini doğrula.
        search_res = app.search("test")
        self.assertIn("Verileri İndeksle", search_res[0])

        # 8. Ardından index_documents() çağır.
        app.index_documents()

        # 9. Eksik kaynağın yeniden oluşturulduğunu doğrula.
        # Toplam baştaki haline dönmeli (2 chunk test1 ve test2'den vs.)
        new_data = app.vector_store.get()
        self.assertEqual(len(new_data["ids"]), len(all_data["ids"]))

        # BM25 yüklenmiş olmalı
        self.assertIsNotNone(app.bm25_index)
        self.assertTrue(len(app.bm25_documents) > 0)

    def test_stale_bm25_manifest_rebuild(self):
        # Önce sağlıklı bir ortam kur
        app.index_documents()

        # Aynı chunk sayısını koruyacak biçimde PDF'yi değiştir
        self.create_mock_pdf(self.pdf1_path, "This is a changed document with exactly same length... roughly.")

        # Hata manipülasyonundan hemen önce diskte schema_version: 1 ve geçerli manifestin var olduğunu doğrulayan (assert) ekleme
        with open(self.bm25_file, "rb") as f:
            data = pickle.load(f)
        self.assertEqual(data.get("schema_version"), 1)
        self.assertIn("manifest", data)
        self.assertTrue(len(data["manifest"]) > 0)

        # Chroma güncellendikten sonra BM25 dosya yazımını kontrollü biçimde başarısız kıl (os.replace mocku)
        with patch('os.replace', side_effect=Exception("Mocked write error")):
            res = app.index_documents()
            self.assertIn("BM25 senkronizasyonu başarısız oldu", res)

        # BM25 dosyası hatalı durumda, stale kaldı.
        # State'i sıfırla
        app.vector_store = None
        app.bm25_index = None
        app.bm25_documents = []
        app.cross_encoder = None

        # init_vector_store çalıştır, belge sayısı aynı olsa bile manifest'ten stale olduğunu fark edip uyarı verecek
        res = app.init_vector_store()
        self.assertIn("BM25 eksik/bozuk", res)
        self.assertIsNone(app.bm25_index)

    def test_hash_failure_preserves_existing_source(self):
        # 1. Önce iki sağlıklı PDF indeksle
        app.index_documents()
        initial_count = app.vector_store._collection.count()
        old_data = app.vector_store.get()

        # 2. get_file_hash'i pdf1_path için hata verecek şekilde mockla
        original_hash = app.get_file_hash

        def mock_hash(filepath):
            if "test1.pdf" in filepath:
                raise IOError("Mocked hash read error")
            return original_hash(filepath)

        with patch('app.get_file_hash', side_effect=mock_hash):
            with patch('builtins.print') as mock_print:
                res = app.index_documents()

                # Check error message
                error_printed = False
                for call in mock_print.call_args_list:
                    if "okunurken/hash hesaplanırken hata oluştu. Dosya atlanıyor" in str(call):
                        error_printed = True
                        break
                self.assertTrue(error_printed)

            self.assertIn("Hatalı: 1", res)

        # 3. Doğrula: test1.pdf orphan sayılmamalı, eski kayıtlar silinmemeli
        self.assertEqual(app.vector_store._collection.count(), initial_count)
        new_data = app.vector_store.get()
        self.assertEqual(set(old_data["ids"]), set(new_data["ids"]))

    def test_bm25_manifest_roundtrip_on_restart(self):
        # İzole ortamda ilk indekslemeyi yap.
        app.index_documents()

        # Pickle dosyasını doğrudan aç.
        with open(self.bm25_file, "rb") as f:
            data = pickle.load(f)

        # schema_version == 1 olduğunu doğrula.
        self.assertEqual(data.get("schema_version"), 1)

        # manifest bulunduğunu ve beklenen kaynakları içerdiğini doğrula.
        self.assertIn("manifest", data)
        self.assertTrue(len(data["manifest"]) > 0)

        # documents, BM25 corpus ve Chroma sayılarının eşitliğini doğrula.
        chroma_count = app.vector_store._collection.count()
        self.assertEqual(len(data["documents"]), chroma_count)

        bm25_obj = data["bm25"]
        corpus_size = getattr(bm25_obj, "corpus_size", -1) if hasattr(bm25_obj, "corpus_size") else len(getattr(bm25_obj, "corpus", []))
        self.assertEqual(corpus_size, chroma_count)

        # Global state'i temizle ve init_vector_store() çağır.
        app.vector_store = None
        app.bm25_index = None
        app.bm25_documents = []
        app.cross_encoder = None

        msg = app.init_vector_store()

        # Başlangıç mesajının "Veritabanı hazır" olduğunu doğrula.
        self.assertIn("Veritabanı hazır", msg)

        # bm25_index değerinin None olmadığını doğrula.
        self.assertIsNotNone(app.bm25_index)

    def test_incomplete_repair_add_failure_preserves_existing_ids(self):
        # 1. İki sayfalık bir PDF oluşturup izole ortamda indeksle
        import fitz
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text(fitz.Point(50, 50), "Bu test dokümanı sayfa 1 içeriğidir.")
        p2 = doc.new_page()
        p2.insert_text(fitz.Point(50, 50), "Bu test dokümanı sayfa 2 içeriğidir.")
        doc.save(self.pdf1_path)
        doc.close()

        app.index_documents()

        test1_skey = os.path.normpath(self.pdf1_path).replace('\\', '/')
        initial_data = app.vector_store.get()
        test1_ids = [
            _id for _id, meta in zip(initial_data["ids"], initial_data["metadatas"])
            if meta.get("source_key") == test1_skey or meta.get("source") == self.pdf1_path
        ]
        self.assertTrue(len(test1_ids) >= 2, "test1.pdf en az 2 chunk üretmeli")

        # 2. Kaynağın chunk'larından birini silerek gerçek incomplete-index oluştur
        deleted_chunk_id = test1_ids[0]
        app.vector_store.delete(ids=[deleted_chunk_id])

        # 3. Onarım öncesinde kalan ID kümesini ve BM25 dosyasının hash/mtime değerini kaydet
        before_data = app.vector_store.get()
        before_ids = set(before_data["ids"])
        self.assertNotIn(deleted_chunk_id, before_ids)

        with open(self.bm25_file, "rb") as f:
            bm25_bytes_before = f.read()
        bm25_hash_before = hashlib.sha256(bm25_bytes_before).hexdigest()
        bm25_mtime_before = os.path.getmtime(self.bm25_file)
        ram_bm25_count_before = len(app.bm25_documents)
        ram_bm25_index_before = app.bm25_index

        # 4. add_documents() çağrısına hata enjekte et
        with patch.object(app.vector_store, 'add_documents', side_effect=Exception("Mocked add error on incomplete repair")):
            # 5. index_documents() çalıştır
            res = app.index_documents()

        # 6. Doğrulamalar:
        # - Onarım öncesindeki bütün ID'ler korunmuş
        # - Önceden bulunmayan yeni ID kalmamış
        after_data = app.vector_store.get()
        after_ids = set(after_data["ids"])
        self.assertEqual(before_ids, after_ids)

        # - BM25 dosyası değiştirilmemiş
        with open(self.bm25_file, "rb") as f:
            bm25_bytes_after = f.read()
        bm25_hash_after = hashlib.sha256(bm25_bytes_after).hexdigest()
        bm25_mtime_after = os.path.getmtime(self.bm25_file)
        self.assertEqual(bm25_hash_before, bm25_hash_after)
        self.assertEqual(bm25_mtime_before, bm25_mtime_after)

        # - RAM'de yanlış veya yeni BM25 state oluşmamış
        self.assertEqual(len(app.bm25_documents), ram_bm25_count_before)
        self.assertIs(app.bm25_index, ram_bm25_index_before)

        # - Sonuç başarı değil hata bildiriyor
        self.assertIn("Hatalı: 1", res)
        self.assertNotIn("✅ Başarılı!", res)

    def test_manifest_rejects_missing_identity_metadata(self):
        # 1. Boş source_key ve boş source
        res1 = app.generate_manifest(["id1"], [{"source_key": "", "source": "", "file_hash": "hash1"}])
        self.assertIsNone(res1)

        # 2. source_key=None ve source=None
        res2 = app.generate_manifest(["id1"], [{"source_key": None, "source": None, "file_hash": "hash1"}])
        self.assertIsNone(res2)

        # 3. file_hash=None
        res3 = app.generate_manifest(["id1"], [{"source_key": "src1.pdf", "source": "src1.pdf", "file_hash": None}])
        self.assertIsNone(res3)

        # 4. file_hash=""
        res4 = app.generate_manifest(["id1"], [{"source_key": "src1.pdf", "source": "src1.pdf", "file_hash": ""}])
        self.assertIsNone(res4)

        # 5. Aynı kaynak altında çelişen iki hash
        res5 = app.generate_manifest(
            ["id1", "id2"],
            [
                {"source_key": "src1.pdf", "source": "src1.pdf", "file_hash": "hash_a"},
                {"source_key": "src1.pdf", "source": "src1.pdf", "file_hash": "hash_b"}
            ]
        )
        self.assertIsNone(res5)

        # 6. Geçerli metadata'nın hâlâ doğru manifest üretmesi
        valid_metas = [
            {"source_key": "src1.pdf", "source": "src1.pdf", "file_hash": "hash123"},
            {"source_key": "src1.pdf", "source": "src1.pdf", "file_hash": "hash123"},
        ]
        res6 = app.generate_manifest(["id2", "id1"], valid_metas)
        self.assertIsNotNone(res6)
        self.assertIn("src1.pdf", res6)
        self.assertEqual(res6["src1.pdf"]["file_hash"], "hash123")
        self.assertEqual(res6["src1.pdf"]["actual_count"], 2)
        expected_digest = hashlib.sha256("".join(["id1", "id2"]).encode('utf-8')).hexdigest()
        self.assertEqual(res6["src1.pdf"]["ids_digest"], expected_digest)

if __name__ == '__main__':
    unittest.main()
