"""Small local CLI for just; no generation unless explicitly requested."""
import json
import csv
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
def active_runs():
    found=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if any('run_local_dev.py' in a or 'trial_precision.py' in a for a in cmd) and Path(p.cwd()).resolve()==ROOT:
                found.append(p.info['pid'])
        except (psutil.Error,OSError): pass
    return found

def latest(precision=False, improve=False):
    pointer=ROOT/('outputs/improve_latest.txt' if improve else 'outputs/precision_latest.txt' if precision else 'outputs/local_latest_run.txt')
    if not pointer.exists(): raise SystemExit('No run yet. Use: just improve48' if improve else 'No saved precision trial.' if precision else 'No run yet. Use: just run')
    return Path(pointer.read_text(encoding='utf-8').strip())

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    action=sys.argv[1]
    if action=='run':
        running=active_runs()
        if running: raise SystemExit(f'Already running (PID {running}). Use just status or just logs.')
        pointer=ROOT/'outputs/local_latest_run.txt'
        if pointer.exists():
            previous=latest()
            log=previous/'run.log'
            if log.exists() and not (previous/'score.json').exists() and time.time()-log.stat().st_mtime<600:
                raise SystemExit('Latest run log is still active/recent. Use just status; do not start a duplicate run.')
        raise SystemExit(subprocess.call([sys.executable,'-X','utf8','-u',str(ROOT/'scripts/run_local_dev.py')],cwd=ROOT))
    improve=action.startswith('improve-')
    if improve: action=action.removeprefix('improve-')
    precision=action.startswith('precision-')
    if precision: action=action.removeprefix('precision-')
    run=latest(precision,improve)
    print('Output:',run,flush=True)
    if action=='status':
        print('Visible runner PIDs (may be restricted):',active_runs())
        checkpoint=run/'checkpoint.json'
        if checkpoint.exists():
            rows=read(checkpoint)['traces']
            questions=run/'eval_questions.csv'
            if questions.exists():
                with questions.open(encoding='utf-8-sig',newline='') as f:
                    total=sum(1 for _ in csv.DictReader(f))
            else:
                total=277
            print('Saved:',len(rows),'/',total)
            print('Accepted format:',sum(r['status']=='ok' for r in rows))
            print('Pending retry:',sum(bool(r.get('retry_pending')) for r in rows))
        log=run/'run.log'
        if log.exists(): print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-4:]))
        print('Score ready:',(run/'score.json').exists())
    elif action=='score':
        if not (run/'score.json').exists(): raise SystemExit('Score not ready. Use just '+('improve-status' if improve else 'precision-status' if precision else 'status'))
        score=read(run/'score.json')
        print(json.dumps({k:score[k] for k in ('count','score_out_of_200','document_accuracy',
              'format_valid_fraction','answer_score_by_type','complete')},ensure_ascii=False,indent=2))
        print('Full document AND answer credit:',sum(d['document_score']==1 and d['answer_score']==1 for d in score['details']))
        print('Files to analyze: score.json, diagnostics.json, metrics.json, config.json')
        if (run/'comparison.json').exists():
            comparison=read(run/'comparison.json')
            print('Baseline:',comparison['baseline_score'],'New:',comparison['new_score'])
            print('Improved:',comparison['improved'],'Regressed:',comparison['regressed'])
    elif action=='logs':
        position=0
        while True:
            log=run/'run.log'
            if log.exists():
                with log.open(encoding='utf-8',errors='replace') as f:
                    f.seek(position)
                    print(f.read(),end='',flush=True)
                    position=f.tell()
            if (run/'score.json').exists():
                print('\nFinished. Use just '+('improve-score' if improve else 'precision-score' if precision else 'score'))
                return
            time.sleep(1)

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('\nStopped watching/running this command.')
