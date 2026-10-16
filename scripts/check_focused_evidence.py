"""Inspect a saved dev run without generation. Word counts are not Qwen tokens."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag_contest.data import extract_documents, load_questions, write_json, classify
from rag_contest.blocks import complete_evidence
from rag_contest.retrieval import WordCodec

parser=argparse.ArgumentParser()
parser.add_argument('--diagnostics',required=True)
args=parser.parse_args()
raw=Path(args.diagnostics).read_text(encoding='utf-8-sig')
rows=json.loads(raw[raw.index('['):],strict=False)
dev={q['question_id']:q['question_text'] for q in load_questions('outputs/data/dev_questions.csv')}
assert all(dev.get(r['question_id'])==r['question_text'] for r in rows), 'Dev only'
docs={d['document_id']:d for d in extract_documents('sample_docs')}
report=[]
for row in rows:
    calls=[c for c in row.get('calls',[]) if c['purpose']=='answer']
    if not calls or not row.get('candidates'): continue
    doc=docs[row['candidates'][0]['document_id']]
    text=complete_evidence(doc,WordCodec(),1200,row['question_text'],focused=True)
    old=calls[0]['messages'][-1]['content'].split('Evidence:\n',1)[-1].split('\n\nQuestion:',1)[0]
    report.append(dict(question_id=row['question_id'],evidence=text,
                       old_words=len(WordCodec().encode(old)),new_words=len(WordCodec().encode(text)),
                       classified=classify(row['question_text'])))
summary=dict(count=len(report),empty_ids=[r['question_id'] for r in report if not r['evidence']],
             old_words=sum(r['old_words'] for r in report),new_words=sum(r['new_words'] for r in report),
             note='Same saved top-1 documents. WordCodec only; not an LLM score or actual Qwen token budget.')
write_json('outputs/focused_audit/report.json',dict(summary=summary,details=report))
print(json.dumps(summary,indent=2))
