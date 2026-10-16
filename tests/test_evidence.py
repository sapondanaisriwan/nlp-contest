import unittest
from rag_contest.evidence import guided_evidence
from rag_contest.retrieval import Retriever, WordCodec


class EvidenceTests(unittest.TestCase):
    def test_table_keeps_material_with_class_and_header(self):
        doc = {'document_id': '1', 'pages': [{'page': 1, 'text':
            'Table 1. Fleet\nName Class Material\nLark 3x cedar\nFinch 2x mahogany\n2. Storage\n' + 'storage ' * 100}]}
        r = Retriever([doc], WordCodec(), chunk_size=20, overlap=5)
        evidence = r.evidence(r.search('storage')[0], WordCodec(), 40, 'chunks',
                              question='What class is made of mahogany?', strategy='tables')
        self.assertIn('Name Class Material', evidence)
        self.assertIn('Finch 2x mahogany', evidence)
        self.assertLessEqual(len(WordCodec().encode(evidence)), 40)

    def test_requested_section_across_pages_before_high_ranked_noise(self):
        doc = {"document_id": "1", "pages": [
            {"page": 1, "text": "1. INTRODUCTION\n" + "irrelevant " * 100 + "\n4. FEES\nCurrent rate is 7.25."},
            {"page": 2, "text": "Discount applies only to students.\n4.1 Exceptions\nNo discount for guests.\n5. OTHER\n" + "noise " * 100}]}
        r = Retriever([doc], WordCodec(), chunk_size=30, overlap=5)
        c = r.search("irrelevant noise")[0]
        text = r.evidence(c, WordCodec(), 70, "chunks", question="Based on Section 4, what fee applies?", strategy="guided")
        self.assertIn("7.25", text)
        self.assertIn("Discount applies only to students", text)
        self.assertIn("No discount for guests", text)
        self.assertLessEqual(len(WordCodec().encode(text)), 70)

    def test_row_preserves_header_units_and_neighbors(self):
        doc = {"document_id": "2", "pages": [{"page": 1, "text":
            "Table 1. Sites\nSite Distance (m) Status\nAB-07 80 Open\nAB-08 90 Closed\nNote: distances measured from inlet.\n2. OTHER\n" + "noise " * 100}]}
        r = Retriever([doc], WordCodec(), chunk_size=25, overlap=5)
        text = r.evidence(r.search("noise")[0], WordCodec(), 65, "chunks",
                          question="How far is AB-07?", strategy="guided")
        for expected in ("Distance (m)", "AB-07 80", "AB-08 90", "distances measured from inlet"):
            self.assertIn(expected, text)
        self.assertEqual(text.count("AB-07 80"), 1)

    def test_missing_section_falls_back_and_duplicate_spans_merge(self):
        text = "Alpha has 20 workers. Beta has 30 workers."
        doc = {"document_id": "1", "pages": [{"page": 1, "text": text}]}
        c = {"chunks": [
            {"page": 1, "order": 0, "text": "Alpha has 20 workers. Beta"},
            {"page": 1, "order": 1, "text": "Beta has 30 workers."}]}
        result = guided_evidence(doc, c, WordCodec(), 80, "Section 99?")
        self.assertIn(text, result)
        self.assertEqual(result.count("Beta"), 1)

    def test_no_budget_overflow_for_oversized_section(self):
        doc = {"document_id": "1", "pages": [{"page": 1, "text": "4. FEES\n" + "rate " * 300}]}
        r = Retriever([doc], WordCodec(), chunk_size=20, overlap=5)
        text = r.evidence(r.search("rate")[0], WordCodec(), 40, "chunks", question="Section 4?", strategy="guided")
        self.assertTrue(text)
        self.assertLessEqual(len(WordCodec().encode(text)), 40)
