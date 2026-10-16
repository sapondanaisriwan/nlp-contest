"""Run all development questions locally; holdout is never loaded."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
from filelock import FileLock, Timeout
run_lock=FileLock(str(ROOT/'outputs/local_experiment.lock'))
try: run_lock.acquire(timeout=0)
except Timeout: raise SystemExit('Another model run is active. Use just status or just improve-status.')
RUN = ROOT / 'outputs' / ('local-dev277-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
RUN.mkdir(parents=True)
config = json.loads((ROOT/'config.json').read_text())
config.update(device='cpu', retrieval='bm25', evidence_strategy='focused', arithmetic=True,
              threads=8, max_seconds=86400, total_seconds=172800)
(RUN/'config.json').write_text(json.dumps(config,indent=2))
(ROOT/'outputs/local_latest_run.txt').write_text(str(RUN),encoding='utf-8')
os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
                  TOKENIZERS_PARALLELISM='false')
q = str(ROOT/'outputs/data/dev_questions.csv')
gold = str(ROOT/'outputs/data/dev_answers.csv')
assets = str(ROOT/'assets/qwen3-1.7b-q8')
def command(args, log):
    with (RUN/log).open('w',encoding='utf-8') as out:
        with subprocess.Popen([sys.executable,'-X','utf8','-u','-m',*args],stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as process:
            for line in process.stdout:
                out.write(line)
                out.flush()
                print(line,end='',flush=True)
            return process.wait()
print('RESULTS:',RUN,flush=True)
rc=command(['rag_contest.preflight','--docs','sample_docs','--questions',q,'--gold',gold,
            '--assets',assets,'--config',str(RUN/'config.json'),'--output',str(RUN/'preflight.json')], 'preflight.log')
if rc: raise SystemExit('Preflight failed; see '+str(RUN/'preflight.log'))
rc=command(['rag_contest','run','--docs','sample_docs','--questions',q,'--assets',assets,
            '--config',str(RUN/'config.json'),'--output',str(RUN),'--fresh'], 'run.log')
if rc not in (0,2): raise SystemExit('Inference failed; see '+str(RUN/'run.log'))
rc=command(['rag_contest','score','--questions',q,'--gold',gold,'--run',str(RUN),'--label','dev'],'score.log')
if rc: raise SystemExit('Scoring failed; see '+str(RUN/'score.log'))
report=json.loads((RUN/'score.json').read_text(encoding='utf-8'))
print(json.dumps({k:v for k,v in report.items() if k not in ('details','metrics')},ensure_ascii=False,indent=2),flush=True)
