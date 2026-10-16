import unittest
from rag_contest.data import options,validate_answer
from rag_contest.precision import precise_evidence
from rag_contest.retrieval import WordCodec

class PrecisionTests(unittest.TestCase):
    def test_comma_delimited_choices(self):
        q='Which? [A] Red symbol ,[B] Blue symbol , [C] Green symbol'
        self.assertEqual(options(q)['B'],'Blue symbol')
        self.assertEqual(validate_answer('[B] Blue symbol','choice',q),'B')
        self.assertIsNone(validate_answer('[B] Red symbol','choice',q))

    def test_long_prose_does_not_hide_price(self):
        text='1. Overview\n'+('The plant produces clean water for residents. '*80)+'The project costs 7 million dollars. Work starts next year.'
        doc={'document_id':'1','pages':[{'page':1,'text':text}]}
        evidence=precise_evidence(doc,WordCodec(),100,'How much does the project cost?')
        self.assertIn('7 million dollars',evidence)
        self.assertLessEqual(len(WordCodec().encode(evidence)),100)

    def test_table_preserves_columns_and_other_rows(self):
        doc={'document_id':'1','pages':[{'page':1,'text':'Table 1. Sites\nSite Distance\nZX-01 25\nZX-02 70'}]}
        evidence=precise_evidence(doc,WordCodec(),100,'How far is ZX-01?')
        self.assertIn('Site Distance\nZX-01 25',evidence)
