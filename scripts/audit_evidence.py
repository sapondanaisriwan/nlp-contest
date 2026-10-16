"""Offline dev-only audit. Word budgets are NOT Qwen token counts."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_contest.data import extract_documents, load_questions, load_gold, write_json
from rag_contest.retrieval import Retriever, WordCodec
from rag_contest.blocks import content_query

docs = extract_documents(ROOT / 'sample_docs')
r = Retriever(docs, WordCodec())
questions = load_questions(ROOT / 'outputs/data/dev_questions.csv')
gold = {x['question_id']: x for x in load_gold(ROOT / 'outputs/data/dev_answers.csv')}
results = []
for q in questions:
    raw = r.search(q['question_text'])
    cleaned = r.search(content_query(q['question_text']))
    expected = gold[q['question_id']]['document_id']
    results.append(dict(question_id=q['question_id'], expected=expected,
                        raw=[c['document_id'] for c in raw], cleaned=[c['document_id'] for c in cleaned]))
report = {'count':len(results), 'codec':'WordCodec: not Qwen; diagnostic only', 'results':results}
for mode in ('raw','cleaned'):
    report[mode] = dict(top1=sum(x[mode][0]==x['expected'] for x in results),
                        top3=sum(x['expected'] in x[mode] for x in results))
write_json(ROOT/'outputs/evidence_audit/retrieval.json',report)

# IDs come only from the previously evaluated dev run, never inference rules.
attachment=Path('C:/Users/sapon/.codex/attachments/68fad304-1384-4ef8-8250-b938ea4381be/Pasted text.txt')
traces=json.loads(attachment.read_text(encoding='utf-8'),strict=False)
ids={str(t['question_id']) for t in traces}
assert ids <= set(gold)
audit=[]
for q in questions:
    if q['question_id'] not in ids: continue
    candidate=r.search(q['question_text'])[0]
    text=r.evidence(candidate,WordCodec(),800,'chunks',question=q['question_text'],strategy='complete')
    audit.append(dict(**q,document_id=candidate['document_id'], expected_document=gold[q['question_id']]['document_id'],
                      word_count=len(WordCodec().encode(text)), evidence=text, empty=not text))
write_json(ROOT/'outputs/evidence_audit/dev48.json',audit)
assert len(audit)==48
summary={'retrieval':{k:report[k] for k in ('count','codec','raw','cleaned')},
         'evidence_count':len(audit),'empty_evidence':sum(x['empty'] for x in audit),
         'budget':'800 WordCodec words; actual Qwen budget must be checked in Colab'}
write_json(ROOT/'outputs/evidence_audit/summary.json',summary)
print(json.dumps(summary,indent=2))
