import unittest
from rag_contest.data import classify


class QuestionPatternTests(unittest.TestCase):
    def test_request_families(self):
        cases = {
            'What was the narrowest diameter, in millimetres, at the inspection?':'number',
            'What value of the dynamic factor is used?':'number',
            'What is the rated power in kW?':'number',
            'Which Hall number houses the equipment?':'number',
            "What's Alex's job title?":'short',
            'Which cave has the required temperature?':'short',
            'What color is this?':'short',
            'What is the phone number to call?':'short',
            'How much grain can the machine process per batch?':'short',
            'The report was issued before the inspection.':'yesno',
            'If there is smoke, should the team wait until Monday?':'yesno',
            'Policy update: Is it true that the office closes on Friday?':'yesno',
            'What is the final date (answer as a number)?':'number',
            'What year was the rule issued?':'number',
            'Which color? [A] Green [B] Blue':'choice',
            'How long does processing take?':'unknown',
        }
        for q, expected in cases.items():
            with self.subTest(q=q): self.assertEqual(classify(q),expected)
