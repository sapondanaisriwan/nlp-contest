"""Cheap full development-set checks; never opens held-out labels or generates answers."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag_contest.data import load_questions,load_gold,classify,gold_type,write_json
from rag_contest.index_cache import cached_documents
from rag_contest.retrieval import Retriever,WordCodec

root=Path(__file__).resolve().parents[1]
questions=load_questions(root/'outputs/data/dev_questions.csv')
gold={g['question_id']:g for g in load_gold(root/'outputs/data/dev_answers.csv')}
docs,_=cached_documents(root/'sample_docs',root/'outputs/local_audit_cache')
r=Retriever(docs,WordCodec())
rows=[]
for q in questions:
    hits=r.search(q['question_text'])
    predicted=classify(q['question_text'])
    g=gold[q['question_id']]
    rows.append({'question_id':q['question_id'],'top1':hits[0]['document_id']==g['document_id'],
                 'top3':g['document_id'] in [h['document_id'] for h in hits],
                 'type_match':predicted==gold_type(q['question_text'],g['answer_text'])})
report={'count':len(rows),'generation_calls':0,
        'top1':sum(x['top1'] for x in rows),'top3':sum(x['top3'] for x in rows),
        'type_matches':sum(x['type_match'] for x in rows),
        'note':'Word-window diagnostic only, not Qwen tokenizer retrieval or answer score. Dev only.',
        'details':rows}
write_json(root/'outputs/all_dev_audit.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
