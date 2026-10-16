import unittest
from rag_contest.trials import quick_subset


class TrialTests(unittest.TestCase):
    def test_mixed_controls_preserve_order_and_reject_holdout(self):
        qs = [{'question_id': str(i)} for i in range(16)]
        details = [dict(question_id=str(i), kind=('choice', 'number', 'short', 'yesno')[i//4],
                        document_score=1, answer_score=int(i % 4 == 0)) for i in range(16)]
        score = dict(details=details, evaluation_set='dev')
        subset = quick_subset(qs, score)
        self.assertEqual(len(subset), 12)
        self.assertEqual([q['question_id'] for q in subset], [str(i) for i in range(16) if i%4 != 2])
        score['evaluation_set'] = 'holdout'
        with self.assertRaises(ValueError): quick_subset(qs, score)
