"""Model-free integration checks on the supplied corpus and notebook bundle."""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_contest.data import evaluate, extract_documents, load_gold, load_questions, write_json
from rag_contest.retrieval import Retriever, WordCodec

documents = extract_documents(ROOT / "sample_docs")
assert len(documents) == 100
assert {d["document_id"] for d in documents} == {str(i) for i in range(1, 101)}
pages = sum(len(d["pages"]) for d in documents)
doc1 = "\n".join(p["text"] for p in documents[0]["pages"])
assert "−15" in doc1 or "-15" in doc1
assert "053 552 418" in doc1
doc33 = "\n".join(p["text"] for p in documents[32]["pages"])
assert "43.5" in doc33 and "SAT-09" in doc33 and "TABLE 2" in doc33

# Simulate withheld PDFs becoming available; no query/gold from the holdout is read.
initial = Retriever(documents[:80], WordCodec())
added = Retriever(documents, WordCodec())
rebuilt = Retriever(documents, WordCodec())
assert len(initial.documents) == 80 and len(added.documents) == 100
assert len(added.chunks) == len(rebuilt.chunks)
assert len({(c["document_id"], c["order"]) for c in added.chunks}) == len(added.chunks)

devq = load_questions(ROOT / "outputs/data/dev_questions.csv")
devg = load_gold(ROOT / "outputs/data/dev_answers.csv")
assert evaluate(devq, devg, devg)["score_out_of_200"] == 200

notebook = json.loads((ROOT / "RAG_Contest_Colab.ipynb").read_text(encoding="utf-8"))
code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
for i, cell in enumerate(code_cells):
    compile(cell["source"], f"notebook-cell-{i}", "exec")
    assert cell["outputs"] == [] and cell["execution_count"] is None
payload_line = next(line for c in code_cells for line in c["source"].splitlines() if line.startswith("PAYLOAD = "))
payload = json.loads(payload_line.removeprefix("PAYLOAD = "))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
    for path in (ROOT / "rag_contest").glob("*.py"):
        assert z.read("rag_contest/" + path.name) == path.read_bytes(), f"Stale notebook source: {path}"

report = {"pdf_count": len(documents), "page_count": pages, "rebuilt_chunk_count": len(added.chunks),
          "added_documents": 20, "preserves_sample_negative_numbers_and_tables": True,
          "development_gold_self_score": 200, "notebook_code_cells_compile": len(code_cells),
          "embedded_source_matches_workspace": True,
          "limitations": "No model inference, hybrid embeddings, Colab RAM/time or real-model repeatability measured by this check."}
write_json(ROOT / "outputs/local_validation.json", report)
print(json.dumps(report, indent=2))
