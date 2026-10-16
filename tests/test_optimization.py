import unittest
from rag_contest.data import validate_answer
from rag_contest.retrieval import Retriever, WordCodec


class OptimizationTests(unittest.TestCase):
    def test_copied_option_without_punctuation(self):
        q = 'Select a requirement:\nA Level 2\nB Level 7\nC Level 9'
        self.assertEqual(validate_answer('B Level 7', 'choice', q), 'B')
        self.assertIsNone(validate_answer('B Level 9', 'choice', q))
        self.assertIsNone(validate_answer('B or C', 'choice', q))

    def test_lexical_score_is_not_rank_only(self):
        docs = [{'document_id':str(i), 'pages':[{'page':1,'text':t}]} for i,t in
                [(1,'silver bridge repair'),(2,'silver garden planting')]]
        r=Retriever(docs,WordCodec())
        a=r.search('silver bridge repair')
        b=r.search('silver')
        self.assertGreater(a[0]['lexical_score']-a[1]['lexical_score'],0)
        self.assertEqual(b[0]['lexical_score'],b[1]['lexical_score'])
