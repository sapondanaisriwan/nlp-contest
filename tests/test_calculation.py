import json
import unittest
from rag_contest.calculation import calculate, needs_calculation
from rag_contest.pipeline import Config, answer_one
from rag_contest.retrieval import Retriever, WordCodec
from test_contest import document, FakeGenerator


class CalculationTests(unittest.TestCase):
    def test_normalize_typography_without_changing_values(self):
        raw=json.dumps(dict(expression='0.7 × 36',quote='Time = 0.7 × m0.',units='kg to hours'))
        self.assertEqual(calculate(raw,'Time = 0.7× m 0.','For 36 kg?')['answer'],'25.2')
        self.assertIsNone(calculate(raw,'Time = 0.8× m 0.','For 36 kg?'))

    def test_started_periods_and_conversion_assumptions(self):
        raw=json.dumps(dict(expression='ceil((61-36)/24)*1200',quote='61 hours; 36 hours free; 1200 per started day.',units='hours to days, 24 hours per day'))
        self.assertEqual(calculate(raw,'61 hours; 36 hours free; 1200 per started day.')['answer'],'2400')
        raw=json.dumps(dict(expression='72/30',quote='Fee is 72 monthly.',units='money'))
        self.assertIsNone(calculate(raw,'Fee is 72 monthly.'))

    def test_scalar_cannot_bypass_calculation_and_retry_keeps_source(self):
        r=Retriever([document(1,'Fee is 72 monthly.')],WordCodec())
        q=dict(question_id='x',question_text='How much does one year cost?')
        raw=json.dumps(dict(expression='72*12',quote='Fee is 72 monthly.',units='monthly to yearly'))
        first=answer_one(q,r,FakeGenerator(['72']),Config(arithmetic=True),defer_retry=True)
        self.assertEqual(first['status'],'unresolved')
        self.assertEqual(first['retry_reason'],'invalid_calculation')
        second=answer_one(q,r,FakeGenerator([raw]),Config(arithmetic=True),previous=first)
        self.assertEqual(second['answer_text'],'864')
        self.assertEqual(second['retry_strategy'],'same_evidence_repair')
        self.assertNotIn('retry_sources',second)
        copied=json.dumps(dict(expression='72',quote='Fee is 72 monthly.',units='monthly to yearly'))
        self.assertIsNone(calculate(copied,'Fee is 72 monthly.',q['question_text']))

    def test_lookup_does_not_pay_json_generation_cost(self):
        for q in ['How many valves are there in total?', 'What is the normal allowance per day?']:
            self.assertFalse(needs_calculation(q))
    def test_question_operand_and_lookup_routing(self):
        raw=json.dumps(dict(expression='0.7*36',quote='Minimum time is 0.7 hours per kg.',units='hours per kg times kg yields hours'))
        self.assertIsNone(calculate(raw,'Minimum time is 0.7 hours per kg.'))
        self.assertEqual(calculate(raw,'Minimum time is 0.7 hours per kg.','What brining time for 36 kg?')['answer'],'25.2')
        self.assertFalse(needs_calculation('How many meters from the diffuser is site AB-02?'))
        self.assertTrue(needs_calculation('What is the daily cost?'))

    def test_quoted_arithmetic(self):
        raw = json.dumps(dict(expression='72/30', quote='Fee is 72 monthly.', units='monthly to daily, assume 30 days'))
        self.assertEqual(calculate(raw, 'Fee is 72 monthly.')['answer'], '2.4')
        self.assertIsNone(calculate(raw, 'Fee is 82 monthly.'))

    def test_reject_code_missing_operands_and_zero_division(self):
        for expr in ['__import__("os")', '9**9', '72/0', '999/30', '72//30', '72%30']:
            raw = json.dumps(dict(expression=expr, quote='72', units='units'))
            self.assertIsNone(calculate(raw, '72'))

    def test_pipeline_computes_and_rejects_truncation(self):
        r = Retriever([document(1, 'Fee is 72 monthly.')], WordCodec())
        raw = json.dumps(dict(expression='72*12', quote='Fee is 72 monthly.', units='monthly to yearly'))
        q = dict(question_id='x', question_text='How much does one year cost?')
        result = answer_one(q, r, FakeGenerator([raw]), Config(arithmetic=True))
        self.assertEqual(result['answer_text'], '864')
        result = answer_one(q, r, FakeGenerator([dict(text=raw, truncated=True), 'NOT_FOUND']), Config(arithmetic=True))
        self.assertEqual(result['status'], 'unresolved')
