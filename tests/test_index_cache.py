import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_contest.index_cache import cached_documents, cached_retriever
from rag_contest.pipeline import Config
from rag_contest.retrieval import WordCodec


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{"document_id": "1", "filename": "001_test.pdf", "sha256": "test",
                      "pages": [{"page": 1, "text": "Alpha has 46 staff. Beta has 20 staff."}]}]

    def test_pdf_change_add_remove_and_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "pdfs"
            folder.mkdir()
            pdf = folder / "001_test.pdf"
            pdf.write_bytes(b"first")
            cache = Path(root) / "cache"
            with patch("rag_contest.index_cache.extract_documents", return_value=self.docs) as extract:
                _, first = cached_documents(folder, cache)
                _, second = cached_documents(folder, cache)
                self.assertFalse(first["hit"])
                self.assertTrue(second["hit"])
                self.assertEqual(extract.call_count, 1)
                pdf.write_bytes(b"changed")
                _, changed = cached_documents(folder, cache)
                self.assertFalse(changed["hit"])
                extra = folder / "002_test.pdf"
                extra.write_bytes(b"extra")
                _, added = cached_documents(folder, cache)
                self.assertFalse(added["hit"])
                extra.unlink()
                _, removed = cached_documents(folder, cache)
                self.assertTrue(removed["hit"])
                self.assertEqual(removed["key"], changed["key"])
                Path(removed["path"]).write_text("broken")
                _, repaired = cached_documents(folder, cache)
                self.assertFalse(repaired["hit"])

    def test_bm25_roundtrip_and_invalidation(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = Config(retrieval="bm25")
            first, report = cached_retriever(self.docs, WordCodec(), None, cfg, root, {"model": "a"})
            with patch("rag_contest.index_cache.Retriever.__init__", side_effect=AssertionError("rebuilt")):
                second, hit = cached_retriever(self.docs, WordCodec(), None, cfg, root, {"model": "a"})
            self.assertTrue(hit["hit"])
            self.assertEqual(first.search("Alpha"), second.search("Alpha"))
            self.assertEqual(first.evidence(first.search("Alpha")[0], WordCodec()),
                             second.evidence(second.search("Alpha")[0], WordCodec()))
            # Evidence budget does not change the stored index.
            _, budget = cached_retriever(self.docs, WordCodec(), None,
                                        Config(retrieval="bm25", evidence_tokens=1200), root, {"model": "a"})
            self.assertTrue(budget["hit"])
            for cfg2, identity in [(Config(retrieval="bm25", chunk_size=300), {"model": "a"}),
                                   (cfg, {"model": "b"})]:
                _, changed = cached_retriever(self.docs, WordCodec(), None, cfg2, root, identity)
                self.assertFalse(changed["hit"])
            path = Path(report["path"])
            state = json.loads(path.read_text())
            state["payload"]["idf"] = {}
            path.write_text(json.dumps(state))
            _, repaired = cached_retriever(self.docs, WordCodec(), None, cfg, root, {"model": "a"})
            self.assertFalse(repaired["hit"])

    def test_hybrid_does_not_reencode_documents_on_hit(self):
        import numpy as np
        class Dense:
            def __init__(self): self.calls = 0
            def encode(self, texts):
                self.calls += 1
                return np.array([[0.6, 0.8] for _ in texts], dtype=np.float32)
        with tempfile.TemporaryDirectory() as root:
            dense = Dense()
            a, _ = cached_retriever(self.docs, WordCodec(), dense, Config(), root, {})
            b, report = cached_retriever(self.docs, WordCodec(), dense, Config(), root, {})
            self.assertTrue(report["hit"])
            self.assertEqual(dense.calls, 1)
            self.assertTrue(np.array_equal(a.vectors, b.vectors))
            self.assertEqual(a.search("Alpha"), b.search("Alpha"))
