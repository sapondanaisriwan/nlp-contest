"""Check actual local model tokenizer budgets without loading generator weights."""
import argparse
import json
from pathlib import Path
from .data import load_questions, load_gold, write_json, digest, classify, gold_type
from .blocks import content_query
from .retrieval import Retriever
from .models import offline
from .pipeline import Config, SYSTEM, FORMATS
from .prompts import guidance


def main():
    parser=argparse.ArgumentParser()
    for name in ('docs','questions','assets','config','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gold', help='Offline evaluation only; never passed to retrieval or prompts')
    args=parser.parse_args()
    config=Config(**json.loads(Path(args.config).read_text()))
    if config.retrieval != 'bm25':
        raise ValueError('Tokenizer preflight currently supports BM25 only')
    with offline():
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(Path(args.assets)/'tokenizer'),local_files_only=True,trust_remote_code=False)
        class Codec:
            def encode(self,text):return tokenizer.encode(text,add_special_tokens=False)
            def decode(self,tokens):return tokenizer.decode(tokens,skip_special_tokens=True)
        codec=Codec()
        from .index_cache import cached_documents, cached_retriever
        from .models import verify_assets, environment
        cache_dir=Path(args.assets).parent.parent/'index_cache'
        docs,document_cache=cached_documents(args.docs,cache_dir)
        questions=load_questions(args.questions)
        r,index_cache=cached_retriever(docs,codec,None,config,cache_dir,
            {'model_digest':digest(verify_assets(args.assets)), 'environment':environment()})
        rows=[]
        for row in questions:
            q=row['question_text']
            found=r.search(content_query(q) if config.query_mode=='content' else q)
            evidence=r.evidence(found[0],codec,config.evidence_tokens,config.context_mode,question=q,strategy=config.evidence_strategy)
            count=len(codec.encode(evidence))
            assert count<=config.evidence_tokens
            # Check all possible answer types before spending time generating/classifying.
            sizes=[]
            for kind, instruction in FORMATS.items():
                if config.prompt_mode=='guided':
                    instruction=guidance(q,kind)+'\n'+instruction
                if config.arithmetic and kind=='number':
                    from .calculation import INSTRUCTION
                    instruction=INSTRUCTION
                messages=[{'role':'system','content':SYSTEM},
                          {'role':'user','content':f'Evidence:\n{evidence}\n\nQuestion: {q}\n{instruction}'}]
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                reserve=max(config.answer_tokens,160) if config.arithmetic and kind=='number' else config.answer_tokens
                sizes.append(len(codec.encode(prompt))+reserve)
            rows.append(dict(**row,document_id=found[0]['document_id'],evidence=evidence,evidence_tokens=count,
                             candidates=[c['document_id'] for c in found],
                             predicted_type=classify(q,config.classifier_mode),
                             empty=not evidence,first_answer_context_required=max(sizes)))
        report=dict(question_digest=digest(questions),count=len(rows),empty_ids=[x['question_id'] for x in rows if x['empty']],
                    over_context_ids=[x['question_id'] for x in rows if x['first_answer_context_required']>config.context_tokens],
                    tokenizer=str(Path(args.assets)/'tokenizer'),generation_calls=0,details=rows)
        report['index_cache']={'documents':document_cache,'index':index_cache}
        if args.gold:
            gold={g['question_id']:g for g in load_gold(args.gold)}
            if set(gold)!={q['question_id'] for q in questions}:
                raise ValueError('Question/gold IDs differ')
            # Labels are accessed only AFTER retrieval and prompt construction.
            report['retrieval_top1']=sum(x['document_id']==gold[x['question_id']]['document_id'] for x in rows)/len(rows)
            report['retrieval_top3']=sum(gold[x['question_id']]['document_id'] in x['candidates'] for x in rows)/len(rows)
            report['type_mismatch_ids']=[x['question_id'] for x in rows if x['predicted_type']!=gold_type(x['question_text'],gold[x['question_id']]['answer_text'])]
            report['note']='Retrieval/type checks only; not generated-answer accuracy. Labels never enter retrieval or prompts.'
        write_json(args.output,report)
        print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False))


if __name__=='__main__':main()
