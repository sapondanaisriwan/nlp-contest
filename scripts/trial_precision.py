"""Stratified dev-only experiment against saved baseline answers."""
import argparse,json,random,subprocess,sys,os
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag_contest.data import load_questions,load_gold,write_csv,evaluate
root=Path(__file__).resolve().parents[1]
os.chdir(root)
parser=argparse.ArgumentParser()
parser.add_argument('--strategy',choices=['precise','focused','focused_v2','focused_v3','focused_v4','focused_v5'],default='precise')
parser.add_argument('--all-dev',action='store_true',help='Use all development questions; never holdout')
parser.add_argument('--prompt',choices=['basic','guided','verify','targeted'],default='basic')
parser.add_argument('--label',choices=['precision','improve'],default='precision')
args=parser.parse_args()
# OS-backed lock shared by experimental recipes; released on process exit.
from filelock import FileLock, Timeout
trial_lock=FileLock(str(root/'outputs/local_experiment.lock'))
try: trial_lock.acquire(timeout=0)
except Timeout: raise SystemExit('Another experiment is running. Use just improve-status or just precision-status.')
base=root/'outputs/local-dev277-20261015-013510'
score=json.loads((base/'score.json').read_text())
rng=random.Random(20261015)
ids=[]
for kind in ('choice','number','short','yesno'):
    pool=[d['question_id'] for d in score['details'] if d['kind']==kind]
    ids.extend(pool if args.all_dev else rng.sample(pool,12))
questions=[q for q in load_questions('outputs/data/dev_questions.csv') if q['question_id'] in ids]
gold=[g for g in load_gold('outputs/data/dev_answers.csv') if g['question_id'] in ids]
run=root/'outputs'/(args.label+str(len(questions))+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
run.mkdir()
(root/('outputs/'+args.label+'_latest.txt')).write_text(str(run),encoding='utf-8')
write_csv(run/'eval_questions.csv',questions,['question_id','question_text'])
write_csv(run/'eval_answers.csv',gold,['question_id','answer_text','document_id'])
config=json.loads((base/'config.json').read_text())
config.update(evidence_strategy=args.strategy,prompt_mode=args.prompt)
(run/'config.json').write_text(json.dumps(config))
old=[d for d in score['details'] if d['question_id'] in ids]
summary={'count':len(old),'score_out_of_200':100*sum(d['document_score']+d['answer_score'] for d in old)/len(old),
         'full_correct':sum(d['document_score']==d['answer_score']==1 for d in old),'details':old}
(run/'baseline_subset.json').write_text(json.dumps(summary,indent=2))
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
print('TRIAL',run,flush=True)
print('Baseline score on SAME',len(questions),'questions:',summary['score_out_of_200'],flush=True)
print('Strategy:',args.strategy,'Prompt:',args.prompt,flush=True)
with (run/'run.log').open('w',encoding='utf-8') as log:
 process=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','rag_contest','run','--docs','sample_docs',
      '--questions',str(run/'eval_questions.csv'),'--assets','assets/qwen3-1.7b-q8','--config',str(run/'config.json'),
      '--output',str(run),'--fresh'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
 try:
  for line in process.stdout:
   log.write(line);log.flush();print(line,end='',flush=True)
  rc=process.wait()
 except KeyboardInterrupt:
  process.terminate();process.wait();raise
if rc not in (0,2):raise SystemExit(rc)
with (run/'score.log').open('w',encoding='utf-8') as log:
 subprocess.run([sys.executable,'-X','utf8','-m','rag_contest','score','--questions',str(run/'eval_questions.csv'),
      '--gold',str(run/'eval_answers.csv'),'--run',str(run),'--label','dev'],stdout=log,stderr=subprocess.STDOUT,check=True)
new=json.loads((run/'score.json').read_text())
print('BASELINE',summary['score_out_of_200'],'NEW',new['score_out_of_200'],flush=True)
newmap={d['question_id']:d for d in new['details']}
changes=[]
for d in old:
 n=newmap[d['question_id']]
 delta=n['document_score']+n['answer_score']-d['document_score']-d['answer_score']
 if abs(delta)>1e-9:changes.append({'question_id':d['question_id'],'delta':delta,'old':d['actual'],'new':n['actual']})
(run/'comparison.json').write_text(json.dumps({'baseline_score':summary['score_out_of_200'],
 'new_score':new['score_out_of_200'],'improved':sum(d['delta']>0 for d in changes),
 'regressed':sum(d['delta']<0 for d in changes),'changes':changes},ensure_ascii=False,indent=2),encoding='utf-8')
