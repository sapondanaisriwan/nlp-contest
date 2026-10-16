import unittest
from rag_contest.blocks import complete_evidence
from rag_contest.retrieval import WordCodec
from rag_contest.prompts import verification_guidance
from rag_contest.pipeline import Config,answer_one
from test_contest import FakeGenerator,document
from rag_contest.retrieval import Retriever

class FocusedV2Tests(unittest.TestCase):
    def evidence(self,text,q,budget=250):
        return complete_evidence(document(1,text),WordCodec(),budget,q,focused=True,conservative=True)

    def test_explicit_section_keeps_formula_and_excludes_other_topic(self):
        text='4. Fees\nThe permit costs F = c * n. The variable c is 6.25 per hive per month. Discounts require a permit.\n5. Health\nEvery hive is inspected each month. Distance is 8 km.'
        e=self.evidence(text,'Based on Section 4, how much is charged per hive each month?')
        self.assertIn('6.25',e)
        self.assertIn('Discounts require',e)
        self.assertNotIn('8 km',e)

    def test_exact_table_stops_irrelevant_padding(self):
        text='Table 1. Survey\nSite Distance\nZX-03 65\nZX-04 200\n2. History\nThe distance of the site from the old dock was 800.'
        e=self.evidence(text,'What is the distance to ZX-03?')
        self.assertIn('Site Distance\nZX-03 65',e)
        self.assertNotIn('800',e)

    def test_long_opening_is_not_discarded(self):
        text='1. Project\n'+('This plant serves local homes. '*90)+'The renewal project costs 9 million dollars. It starts this year.'
        e=self.evidence(text,'How much does the renewal project cost?',90)
        self.assertIn('9 million',e)
        self.assertLessEqual(len(WordCodec().encode(e)),90)

    def test_whole_exception_is_kept(self):
        e=self.evidence('1. Rules\nAdults must wait. Children under six need not wait.','Must children under six wait?')
        self.assertIn('Children under six need not wait',e)

    def test_project_price_beats_named_survey(self):
        text='1. Renewal\nThe renewal will cost $7 million.\nTable 1. Alpha Project Survey\nAlpha project survey alpha project survey\nSite Distance\nXY-01 70'
        e=self.evidence(text,'What does the Alpha Project cost?',35)
        self.assertIn('$7 million',e)

    def test_verify_prompt_does_not_add_call(self):
        r=Retriever([document(1,'Maximum salary is 56000.')],WordCodec())
        g=FakeGenerator(['no'])
        t=answer_one({'question_id':'new','question_text':'Is the maximum salary 50000?'},r,g,
                     Config(prompt_mode='verify',evidence_strategy='focused_v2'))
        self.assertEqual(g.calls,1)
        self.assertEqual(t['answer_text'],'no')
        self.assertIn('exact claim',t['calls'][0]['messages'][-1]['content'])
