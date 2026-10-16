import unittest
from rag_contest.blocks import complete_evidence
from rag_contest.targets import short_instruction
from rag_contest.retrieval import WordCodec,Retriever
from rag_contest.pipeline import answer_one,Config
from test_contest import document,FakeGenerator

class TargetTests(unittest.TestCase):
    def evidence(self,text,q,exclude=''):
        return complete_evidence(document(1,text),WordCodec(),150,q,focused=True,conservative=True,targeted=True,exclude=exclude)

    def test_salary_beats_repeated_job_heading(self):
        e=self.evidence('1. Ocean Research Analyst\nOcean research analyst vacancy.\n2. Pay\nBase salary ranges from 30000 to 45000.',
                        'Is the maximum base salary of the Ocean Research Analyst 40000?')
        self.assertIn('45000',e)

    def test_material_selects_matching_table(self):
        e=self.evidence('Table 1. Fleet\nName Class Hull\nAster 3x Teak\nTable 2. Class settings\nClass Sling\n3x 5',
                        'What class is the boat made of teak?')
        self.assertIn('Aster 3x Teak',e)

    def test_retry_excludes_old_section(self):
        text='1. Fees\nThe membership fee is 90.\n2. Other fees\nThe joining fee is 15.'
        old='[Document 1, page 1]\n1. Fees\nThe membership fee is 90.'
        e=self.evidence(text,'What fee applies?',old)
        self.assertNotIn('membership fee',e)
        self.assertIn('joining fee',e)

    def test_short_instruction_specific(self):
        self.assertIn('job title',short_instruction("What is the technician's job title?"))
        self.assertNotIn('who =',short_instruction('Which street is it on?'))

    def test_retry_uses_different_passages_once(self):
        r=Retriever([document(1,'1. Fees\nThe membership fee is 90.\n2. Other fees\nThe joining fee is 15.')],WordCodec())
        g=FakeGenerator(['NOT_FOUND','15'])
        prepared={'candidates':r.search('fee'), 'evidence':'[Document 1, page 1]\n1. Fees\nThe membership fee is 90.'}
        t=answer_one({'question_id':'x','question_text':'How much is the joining fee?'},r,g,
                     Config(evidence_strategy='focused_v3'),prepared=prepared)
        self.assertEqual(t['retry_strategy'],'new_passages_same_document')
        self.assertEqual(t['answer_text'],'15')
        self.assertEqual(g.calls,2)
        self.assertNotIn('membership fee',t['calls'][1]['messages'][-1]['content'])
