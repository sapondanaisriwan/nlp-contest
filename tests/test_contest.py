import json
import socket
import tempfile
import unittest
from pathlib import Path

from rag_contest.data import (answer_score, audit_and_split, classify, evaluate, extract_documents,
                              load_questions, options, validate_answer, write_csv)
from rag_contest.models import MODELS, asset_hashes, offline, verify_assets
from rag_contest.pipeline import Config, answer_one, run_questions
from rag_contest.retrieval import Retriever, WordCodec


def document(docid, text):
    return {"document_id": str(docid), "filename": f"{docid:03}_test.pdf", "sha256": "test",
            "pages": [{"page": 1, "text": text}]}


class FakeGenerator(WordCodec):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(self, messages, max_tokens):
        self.calls += 1
        value = next(self.outputs)
        return value if isinstance(value, dict) else {"text": value, "truncated": False}


class DataTests(unittest.TestCase):
    def test_article_before_explicit_options_and_unicode_lines(self):
        q = "A 60-year-old visitor arrives. Which policy applies?\nA. Permit required\u2028B. No permit"
        self.assertEqual(options(q), {"A": "Permit required", "B": "No permit"})
        self.assertEqual(classify(q), "choice")
        self.assertEqual(options("Which?\nA. One\nA. Two\nB. Three"), {})

    def test_contest_scores(self):
        self.assertEqual(answer_score("yes", "YES", "yesno"), 1)
        self.assertEqual(answer_score("A", "A.", "choice"), 0)
        self.assertAlmostEqual(answer_score("10", "12", "number"), 1 / 3)
        self.assertEqual(answer_score("-2.5", "-2.5", "number"), 1)
        self.assertEqual(answer_score("2", "2 kg", "number"), 0)
        self.assertEqual(answer_score("2", "NaN", "number"), 0)
        self.assertAlmostEqual(answer_score("cat", "cut", "short"), .6)
        self.assertLess(answer_score("Dr. A", "dr. a", "short"), 1)

    def test_options(self):
        self.assertEqual(options("Which? [A] Alice [B] Bob"), {"A": "Alice", "B": "Bob"})
        self.assertEqual(options("Which?\nA) Alice\nB) Bob"), {"A": "Alice", "B": "Bob"})
        self.assertEqual(options("Which? A. Alice B. Bob"), {"A": "Alice", "B": "Bob"})
        self.assertEqual(options("Which?\nA Alice\nB Bob"), {"A": "Alice", "B": "Bob"})
        self.assertEqual(options(r"Which?\nA) Alice\nB) Bob"), {"A": "Alice", "B": "Bob"})
        self.assertEqual(options("The A. Smith policy"), {})

    def test_formats(self):
        self.assertEqual(validate_answer(" YES ", "yesno", ""), "yes")
        self.assertIsNone(validate_answer("46\nAssistant: no", "number", ""))
        self.assertIsNone(validate_answer("1,000", "number", ""))
        self.assertEqual(validate_answer("-1.25", "number", ""), "-1.25")
        self.assertEqual(validate_answer("19:30", "short", ""), "19:30")
        self.assertIsNone(validate_answer("F", "choice", "Which? [A] one [B] two"))
        self.assertIsNone(validate_answer("Information not found in the documents.", "short", ""))
        self.assertIsNone(validate_answer("<think>42</think>", "short", ""))

    def test_classifier_does_not_read_gold(self):
        self.assertEqual(classify("Does the policy apply?"), "yesno")
        self.assertEqual(classify("How many staff?"), "number")
        self.assertEqual(classify("What time is the deadline?"), "short")

    def test_information_requests_and_embedded_yesno(self):
        for question in ("Which animal nests here?", "In which month does the project start?",
                         "According to the policy, which team handles returns?", "Who manages the site?"):
            self.assertEqual(classify(question), "short")
        self.assertEqual(classify("At night, the team waits. Is this statement correct?"), "yesno")
        self.assertEqual(classify("Is the team which manages returns available?"), "yesno")
        self.assertEqual(classify("What time? A) 10 B) 11"), "choice")
        self.assertEqual(classify("Which team? Answer only a number."), "number")

    def test_choice_normalization_is_unambiguous(self):
        q = "Choose: [A] red [B] blue"
        for raw in ("[A]", "(A)", "A.", "A)"):
            self.assertEqual(validate_answer(raw, "choice", q), "A")
        for raw in ("A or B", "[A] because red", "[C]", "[A][B]"):
            self.assertIsNone(validate_answer(raw, "choice", q))

    def test_csv_quoting_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.csv"
            rows = [{"question_id": "1", "question_text": 'Who is "A, B"?\nNext line'}]
            write_csv(path, rows, ["question_id", "question_text"])
            self.assertEqual(load_questions(path), rows)
            write_csv(path, rows + rows, ["question_id", "question_text"])
            with self.assertRaises(ValueError):
                load_questions(path)

    def test_score_aligns_ids_not_row_order(self):
        qs = [{"question_id": "1", "question_text": "How many?"}, {"question_id": "2", "question_text": "Is it?"}]
        gold = [{"question_id": "1", "answer_text": "46", "document_id": "69"},
                {"question_id": "2", "answer_text": "yes", "document_id": "1"}]
        self.assertEqual(evaluate(qs, gold, list(reversed(gold)))["score_out_of_200"], 200)
        with self.assertRaises(ValueError):
            evaluate(qs, gold, gold[:1])

    def test_split_grouped_reproducible_and_duplicates_connected(self):
        qs = [{"question_id": str(i), "question_text": f"Question {i}"} for i in range(1, 31)]
        gs = [{"question_id": str(i), "answer_text": "yes", "document_id": str(i)} for i in range(1, 31)]
        qs[1]["question_text"] = qs[0]["question_text"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            audit_and_split(qs, gs, a)
            audit_and_split(qs, gs, b)
            for name in ("dev_questions.csv", "holdout_questions.csv", "audit.json"):
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes())
            held = set(json.loads((Path(a) / "audit.json").read_text())["holdout_document_ids"])
            self.assertEqual("1" in held, "2" in held)
            self.assertEqual(len(load_questions(Path(a) / "dev_questions.csv")) +
                             len(load_questions(Path(a) / "holdout_questions.csv")), 30)


class RetrievalTests(unittest.TestCase):
    def test_unique_document_rank_and_stable_ties(self):
        r = Retriever([document(2, "apple apple"), document(1, "apple apple")], WordCodec())
        self.assertEqual([x["document_id"] for x in r.search("apple")], ["1", "2"])

    def test_source_order_and_budget(self):
        r = Retriever([document(1, "1. OLD\nOld fee 80.\n2. NEW\nNew fee 50.")], WordCodec(), chunk_size=8, overlap=2)
        found = r.search("new fee")[0]
        text = r.evidence(found, WordCodec(), budget=25, mode="chunks")
        self.assertLessEqual(len(WordCodec().encode(text)), 25)
        self.assertIn("page 1", text)

    def test_adding_documents_is_rebuild_not_append(self):
        docs = [document(i, f"Unique topic{i}") for i in range(1, 31)]
        old = Retriever(docs[:10], WordCodec())
        new = Retriever(docs, WordCodec())
        again = Retriever(docs, WordCodec())
        self.assertEqual(len(old.documents), 10)
        self.assertEqual(len(new.documents), 30)
        self.assertEqual(len(new.chunks), len(again.chunks))
        self.assertEqual(new.search("topic29")[0]["document_id"], "29")

    def test_empty_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                extract_documents(folder)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.r = Retriever([document(1, "Alpha has 46 staff."), document(2, "Beta has 20 staff.")], WordCodec())
        self.q = {"question_id": "1", "question_text": "How many staff does Alpha have?"}
        self.config = Config()

    def test_answer(self):
        result = answer_one(self.q, self.r, FakeGenerator(["46"]), self.config)
        self.assertEqual(result["answer_text"], "46")
        self.assertEqual(result["document_id"], "1")

    def test_retry_maps_source_not_model_document_id(self):
        result = answer_one(self.q, self.r, FakeGenerator(["NOT_FOUND", "SOURCE: S2\nANSWER: 20"]), self.config)
        self.assertEqual(result["document_id"], "2")
        self.assertEqual(len(result["calls"]), 2)
        bad = answer_one(self.q, self.r, FakeGenerator(["NOT_FOUND", "SOURCE: 150\nANSWER: 46"]), self.config)
        self.assertEqual(bad["status"], "unresolved")

    def test_format_repair_keeps_same_document(self):
        result = answer_one(self.q, self.r, FakeGenerator(['46 staff','46']), self.config)
        self.assertEqual(result['answer_text'],'46')
        self.assertEqual(result['document_id'],'1')
        self.assertEqual(result['retry_strategy'],'same_evidence_repair')
        self.assertNotIn('retry_sources',result)

    def test_model_cannot_turn_duration_request_into_yesno(self):
        q=dict(question_id='new',question_text='How long does processing take?')
        gen=FakeGenerator(['yesno'])
        result=answer_one(q,self.r,gen,self.config)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['error'],'type_classification_conflicts_with_question')
        self.assertEqual(gen.calls,1)

    def test_retry_opaque_label_maps_to_actual_document(self):
        r = Retriever([document(86, "Alpha has 46 staff.")], WordCodec())
        result = answer_one(self.q, r, FakeGenerator(["NOT_FOUND", "SOURCE: S1\nANSWER: 46"]), self.config)
        self.assertEqual(result["document_id"], "86")
        self.assertEqual(result["retry_sources"], {"S1": "86"})
        for label in ("86", "1", "S2", "S4"):
            bad = answer_one(self.q, r, FakeGenerator(["NOT_FOUND", f"SOURCE: {label}\nANSWER: 46"]), self.config)
            self.assertEqual(bad["status"], "unresolved")

    def test_truncated_number_not_accepted(self):
        result = answer_one(self.q, self.r, FakeGenerator([{"text": "46", "truncated": True}, "NOT_FOUND"]), self.config)
        self.assertEqual(result["status"], "unresolved")

    def test_ambiguous_type_calls_same_generator(self):
        q = {"question_id": "2", "question_text": "Describe the closing schedule for Alpha."}
        gen = FakeGenerator(["short", "19:30"])
        result = answer_one(q, self.r, gen, self.config)
        self.assertEqual(result["answer_text"], "19:30")
        self.assertEqual(gen.calls, 2)

    def test_wrapped_correct_choice_does_not_retry_or_change_source(self):
        q = {"question_id": "new", "question_text": "Alpha staff? [A] 46 [B] 20"}
        gen = FakeGenerator(["[A]"])
        result = answer_one(q, self.r, gen, self.config)
        self.assertEqual(result["answer_text"], "A")
        self.assertEqual(result["document_id"], "1")
        self.assertEqual(gen.calls, 1)

    def test_short_request_skips_model_classification(self):
        gen = FakeGenerator(["Support Desk"])
        result = answer_one({"question_id": "new", "question_text": "Which team handles Alpha?"},
                            self.r, gen, self.config)
        self.assertEqual(result["kind"], "short")
        self.assertEqual(result["type_source"], "rules")
        self.assertEqual(gen.calls, 1)
        self.assertGreaterEqual(result["calls"][0]["elapsed_seconds"], 0)

    def test_failure_does_not_create_submission(self):
        with tempfile.TemporaryDirectory() as out:
            result = run_questions([self.q], self.r, FakeGenerator(["NOT_FOUND", "NOT_FOUND"]), self.config, out, {})
            self.assertFalse(result["complete"])
            self.assertFalse((Path(out) / "answer.csv").exists())
            self.assertTrue((Path(out) / "partial_answers.csv").exists())

    def test_checkpoint_resume_and_changed_inputs(self):
        q2 = {"question_id": "2", "question_text": "How many staff does Beta have?"}
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(StopIteration):
                run_questions([self.q, q2], self.r, FakeGenerator(["46"]), self.config, out, {})
            with self.assertRaises(ValueError):
                run_questions([self.q, q2], self.r, FakeGenerator([]), self.config, out, {"changed": True})
            gen = FakeGenerator(["20"])
            result = run_questions([self.q, q2], self.r, gen, self.config, out, {})
            self.assertTrue(result["complete"])
            self.assertEqual(gen.calls, 1)
            self.assertEqual((Path(out) / "answer.csv").read_text().splitlines()[1:], ["1,46,1", "2,20,2"])

    def test_deferred_retry_runs_after_all_first_answers_and_resumes(self):
        q2 = {"question_id": "2", "question_text": "How many staff does Beta have?"}
        config = Config(question_order="input")
        with tempfile.TemporaryDirectory() as out:
            # Stop as retry begins; both first-pass answers must already be saved.
            with self.assertRaises(StopIteration):
                run_questions([self.q, q2], self.r, FakeGenerator(["NOT_FOUND", "20"]), config, out, {})
            saved = json.loads((Path(out) / "checkpoint.json").read_text())
            self.assertEqual(len(saved["traces"]), 2)
            self.assertTrue(saved["traces"][0]["retry_pending"])
            self.assertIn("2,20,2", (Path(out) / "partial_answers.csv").read_text())
            gen = FakeGenerator(["SOURCE: S1\nANSWER: 46"])
            run_questions([self.q, q2], self.r, gen, config, out, {})
            self.assertEqual(gen.calls, 1)
            traces = json.loads((Path(out) / "diagnostics.json").read_text())
            self.assertEqual([c["purpose"] for c in traces[0]["calls"]], ["answer", "retry"])
            self.assertEqual((Path(out) / "answer.csv").read_text().splitlines()[1:], ["1,46,1", "2,20,2"])

    def test_easy_first_and_submission_stays_in_input_order(self):
        from rag_contest.pipeline import prepare_schedule
        class PreparedRetriever:
            def search(self, q, top):
                return [{"document_id": "1", "score": 1.0, "q": q}]
            def evidence(self, candidate, codec, budget, mode, **kwargs):
                return "word " * (900 if "Long" in candidate["q"] else 10)
        qs = [{"question_id": "9", "question_text": "How many Long items?"},
              {"question_id": "3", "question_text": "How many Short items?"}]
        retriever = PreparedRetriever()
        _, plan = prepare_schedule(qs, retriever, WordCodec(), self.config, lambda: None)
        self.assertEqual([p["question_id"] for p in plan], ["3", "9"])
        for _ in range(2):
            with tempfile.TemporaryDirectory() as out:
                run_questions(qs, retriever, FakeGenerator(["10", "900"]), self.config, out, {})
                self.assertEqual((Path(out) / "answer.csv").read_text().splitlines()[1:], ["9,900,1", "3,10,1"])

    def test_network_block_and_restore(self):
        original = socket.create_connection
        with offline():
            with self.assertRaises(RuntimeError):
                socket.create_connection(("example.com", 443))
        self.assertIs(socket.create_connection, original)

    def test_identical_uncached_fake_runs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for out in (a, b):
                run_questions([self.q], self.r, FakeGenerator(["46"]), self.config, out, {}, resume=False)
            self.assertEqual((Path(a) / "answer.csv").read_bytes(), (Path(b) / "answer.csv").read_bytes())


class IntegrityTests(unittest.TestCase):
    def test_dense_encoder_preserves_tail_beyond_embedding_limit(self):
        import numpy as np
        from rag_contest.retrieval import DenseEncoder
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()
            def decode(self, tokens):
                return " ".join(tokens)
        class Model:
            tokenizer = Tokenizer()
            max_seq_length = 8
            def encode(self, windows, **kwargs):
                return np.array([[1., float("target" in w.split())] for w in windows])
        encoder = object.__new__(DenseEncoder)
        encoder.model = Model()
        vector = encoder.encode(["filler " * 20 + "target"])[0]
        self.assertGreater(vector[1], 0)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1., places=6)

    def test_asset_hash_detects_changed_weights(self):
        from rag_contest.data import write_json
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            (root / "generator").mkdir()
            weight = root / "generator/test.gguf"
            weight.write_bytes(b"fixture only, not a model")
            manifest = {"model_name": "qwen3-1.7b-q8", "spec": MODELS["qwen3-1.7b-q8"], "hashes": asset_hashes(root)}
            write_json(root / "assets.json", manifest)
            self.assertEqual(verify_assets(root), manifest)
            weight.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_assets(root)

    def test_competition_export_compiles_and_rejects_changed_sources(self):
        from rag_contest.competition import export_notebook
        from rag_contest.data import write_json
        from rag_contest.pipeline import source_fingerprint
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            lock = {"sources": source_fingerprint(), "config": {}, "model_name": "qwen3-1.7b-q8"}
            write_json(root / "lock.json", lock)
            export_notebook(root / "lock.json", root / "competition.ipynb")
            notebook = json.loads((root / "competition.ipynb").read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile(cell["source"], "competition-cell", "exec")
            lock["sources"] = {}
            write_json(root / "lock.json", lock)
            with self.assertRaises(ValueError):
                export_notebook(root / "lock.json", root / "bad.ipynb")

    def test_repeat_requires_fresh_runs(self):
        from argparse import Namespace
        from rag_contest.__main__ import verify_repeat
        from rag_contest.data import write_json
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for out in (a, b):
                write_json(Path(out) / "status.json", {"complete": True, "fingerprint": "same"})
                write_json(Path(out) / "metrics.json", {"fresh": False})
            with self.assertRaises(ValueError):
                verify_repeat(Namespace(first=a, second=b, output=Path(a) / "repeat.json"))


if __name__ == "__main__":
    unittest.main()
