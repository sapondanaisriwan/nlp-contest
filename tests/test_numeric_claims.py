import json
import unittest
from rag_contest.numeric_claims import claim_spec, inspect_claim
from rag_contest.pipeline import Config,answer_one
from rag_contest.retrieval import Retriever,WordCodec
from rag_contest.lookup_evidence import label_fleet_row,lookup_evidence
from test_contest import document,FakeGenerator


class ClaimTests(unittest.TestCase):
    question='Is maximum base salary of the Research Analyst 40000?'
    evidence='Base salary: 30,000 to 45,000 baht per month.'

    def check_value(self,value,quote=None):
        return inspect_claim(json.dumps({'value':value,'quote':quote or self.evidence}),self.evidence,claim_spec(self.question))

    def test_endpoint_compared_not_membership(self):
        result,error=self.check_value('45000')
        self.assertIsNone(error)
        self.assertEqual(result['answer'],'no')
        self.assertEqual(self.check_value('30000')[1],'wrong_range_endpoint')

    def test_equal_endpoint(self):
        spec=claim_spec(self.question.replace('40000','45000'))
        result,error=inspect_claim(json.dumps({'value':'45000','quote':self.evidence}),self.evidence,spec)
        self.assertEqual(result['answer'],'yes')

    def test_reject_fabricated_quote_and_embedded_number(self):
        self.assertEqual(self.check_value('45000','Base salary: 45,000 to 60,000 baht.')[1],'quote_not_in_evidence')
        self.assertEqual(self.check_value('450')[1],'value_not_in_quote')

    def test_ambiguous_questions_stay_on_original_path(self):
        for q in ['Is the maximum base salary not 40000?',
                  'Is maximum base salary over 40000?',
                  'Is maximum base salary in 2027 40000?',
                  'Is maximum base salary before October 40000?']:
            self.assertIsNone(claim_spec(q))

    def test_warranty_same_units(self):
        spec=claim_spec('Is the warranty for taps valid for 6 years from the transfer date?')
        result,error=inspect_claim(json.dumps({'value':'2','quote':'Warranty: 2 years for taps.'}),
                                   'Warranty: 2 years for taps.',spec)
        self.assertEqual(result['answer'],'no')
        result,error=inspect_claim(json.dumps({'value':'24','quote':'Warranty: 24 months for taps.'}),
                                   'Warranty: 24 months for taps.',spec)
        self.assertIsNone(result)

    def test_pipeline_one_call_and_audited_comparison(self):
        r=Retriever([document(1,self.evidence)],WordCodec())
        g=FakeGenerator([json.dumps({'value':'45000','quote':self.evidence})])
        t=answer_one({'question_id':'random','question_text':self.question},r,g,Config(evidence_strategy='focused_v5'))
        self.assertEqual(t['answer_text'],'no')
        self.assertEqual(t['numeric_comparison']['value'],'45000')
        self.assertEqual(g.calls,1)

    def test_deferred_retry_stays_structured(self):
        r=Retriever([document(1,self.evidence)],WordCodec())
        g=FakeGenerator(['yes',json.dumps({'value':'45000','quote':self.evidence})])
        row={'question_id':'random','question_text':self.question}
        cfg=Config(evidence_strategy='focused_v5')
        first=answer_one(row,r,g,cfg,defer_retry=True)
        self.assertTrue(first['retry_pending'])
        t=answer_one(row,r,g,cfg,previous=first)
        self.assertEqual(t['answer_text'],'no')
        self.assertEqual(g.calls,2)
        self.assertIn('Return JSON',t['calls'][-1]['messages'][-1]['content'])

    def test_invalid_after_retry_not_guessed(self):
        r=Retriever([document(1,self.evidence)],WordCodec())
        t=answer_one({'question_id':'random','question_text':self.question},r,FakeGenerator(['yes','no']),Config(evidence_strategy='focused_v5'))
        self.assertEqual(t['status'],'unresolved')

    def test_table_anchored_columns(self):
        text='Boat Class Built Hull Ref. kg Amber kg Red kg\nAster 3x 1995 Teak 10 11 12'
        self.assertIn('Class: 3x; Built: 1995; Hull: Teak',label_fleet_row(text))
        unknown='Boat Class Hull\nAster 3x Teak'
        self.assertEqual(label_fleet_row(unknown),unknown)

    def test_room_selection_preserves_conditions(self):
        d=document(1,'1. About\nThe laboratory is on floor 3.\n2. Collection\nCollect the repair in Room 405, except on holidays.\n3. Payments\nPay at the ground floor.')
        got=lookup_evidence(d,WordCodec(),250,'Which room do patients collect their repairs in?',structured=True)
        self.assertIn('Room 405',got)
        self.assertIn('except on holidays',got)
        self.assertNotIn('laboratory',got)

    def test_room_section_across_pages(self):
        d=document(1,'1. Collection\nCollect the repair in Room 405.')
        d['pages'].append({'page':2,'text':'Except on holidays, when Room 406 is used.\n2. Fees\nPrice is 10.'})
        got=lookup_evidence(d,WordCodec(),250,'Which room do patients collect their repairs in?',structured=True)
        self.assertIn('Room 405',got)
        self.assertIn('Room 406',got)
        self.assertIn('page 2',got)
