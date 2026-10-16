import unittest
from rag_contest.data import classify, validate_answer
from rag_contest.prompts import guidance
from rag_contest.pipeline import Config, answer_one
from test_contest import FakeGenerator, document
from rag_contest.retrieval import Retriever, WordCodec


class PromptTests(unittest.TestCase):
    def test_explicit_units_and_format(self):
        for q in ['What is the reduced top speed in kilometers per hour of a bike?',
                  'What is the minimum required brining time in hours for a cheese wheel?',
                  'What is the final date in May (answer as a number)?']:
            self.assertEqual(classify(q),'number')
        self.assertEqual(classify('Which type of worker receives the allowance?'),'short')
        self.assertEqual(classify('Is the speed in kilometers per hour reduced?'),'yesno')

    def test_scalar_and_category_requests_skip_classifier_call(self):
        cases = [("What is the normal on-call allowance in baht per day for technicians?", "number", "1200"),
                 ("What is the approximate daily cost of the subscription?", "number", "2.5"),
                 ("Which category is for damaged parcels?", "short", "Category Z"),
                 ("What class is the wooden boat?", "short", "3x")]
        r = Retriever([document(1, "Allowance, subscription, parcels and boat policy.")], WordCodec())
        for question, kind, answer in cases:
            gen = FakeGenerator([answer])
            trace = answer_one({"question_id": "test", "question_text": question}, r, gen, Config())
            self.assertEqual(trace["kind"], kind)
            self.assertEqual(trace["type_source"], "rules")
            self.assertEqual(gen.calls, 1)
        self.assertEqual(classify("Is the normal on-call allowance paid daily?"), "yesno")

    def test_default_prompt_is_basic(self):
        self.assertEqual(Config().prompt_mode, "basic")

    def test_full_choice_requires_matching_label_and_text(self):
        q = "When?\nA) March 2030\nB) July 2031\nC) March 2031\nD) July 2032"
        for raw in ["D) July 2032", "[D] July 2032", "(d) July   2032", "D: July\n2032"]:
            self.assertEqual(validate_answer(raw, "choice", q), "D")
        for raw in ["D) July 2031", "C) July 2032", "D) July 2032 because it is next",
                    "D) July", "D) July 2032 or C", "E) July 2032", "July 2032",
                    "D) july 2032", "D) July 2032."]:
            self.assertIsNone(validate_answer(raw, "choice", q), raw)

    def test_full_choice_does_not_retry_or_change_document(self):
        r = Retriever([document(1, "Deadline is July 2032.")], WordCodec())
        row = {"question_id": "new", "question_text": "When? A) July 2031 B) July 2032"}
        gen = FakeGenerator(["B) July 2032"])
        trace = answer_one(row, r, gen, Config())
        self.assertEqual(gen.calls, 1)
        self.assertEqual(trace["answer_text"], "B")
        self.assertEqual(trace["document_id"], "1")
        self.assertEqual(trace["calls"][0]["text"], "B) July 2032")

    def test_contextual_questions(self):
        for q in ["According to the notice, does the fee apply?",
                  "If it is raining, can visitors enter?",
                  "On Friday, is a permit required?"]:
            self.assertEqual(classify(q), "yesno")
        for q in ["Which company manages the fund?", "Which street is the office on?",
                  "If the manager is absent, who should be called?"]:
            self.assertEqual(classify(q), "short")
        self.assertEqual(classify("What year was it updated?"), "number")
        self.assertEqual(classify("What is the annual fee?"), "number")

    def test_ambiguous_statements_and_identifiers_not_forced(self):
        self.assertEqual(classify("The office closes at five."), "unknown")
        self.assertEqual(classify("What room number houses the office?"), "number")
        self.assertEqual(classify("How long does it last?"), "unknown")
        self.assertEqual(classify("Which company? A) Aster B) Birch"), "choice")
        self.assertEqual(classify("Which company manages the fund?", "basic"), "unknown")

    def test_prompt_switch_does_not_add_calls_or_change_evidence(self):
        r = Retriever([document(1, "Annual fee is 60. Old fee was 40.")], WordCodec())
        row = {"question_id": "x", "question_text": "How much is the annual fee?"}
        traces = []
        for mode in ["basic", "guided"]:
            g = FakeGenerator(["60"])
            trace = answer_one(row, r, g, Config(prompt_mode=mode))
            self.assertEqual(g.calls, 1)
            self.assertEqual(trace["answer_text"], "60")
            traces.append(trace)
        prompts = [t["calls"][0]["messages"][-1]["content"] for t in traces]
        self.assertEqual(prompts[0].split("Question:")[0], prompts[1].split("Question:")[0])
        self.assertIn("withdrawn rates", prompts[1])
        self.assertNotIn("withdrawn rates", prompts[0])

    def test_guidance_keeps_output_short_and_covers_exceptions(self):
        text = guidance("Is this claim correct?", "yesno")
        self.assertIn("exceptions", text)
        self.assertIn("NOT_FOUND", text)
        self.assertIn("Do not output working", text)
