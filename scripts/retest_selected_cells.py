# %% 8A
from pathlib import Path
import json, uuid
from rag_contest.data import load_questions, write_csv

RETEST_IDS = ""  # เช่น "65" หรือ "65, 99, 206"; ต้องเป็น question_id ไม่ใช่ลำดับแถว

def prepare_retest(parent_run, selected_ids):
    parent_run = Path(parent_run)
    requested = [x.strip() for x in selected_ids.split(',') if x.strip()]
    if not requested:
        raise ValueError('ใส่ RETEST_IDS ก่อน เช่น "65" แล้วรันข้อ 8A')
    if len(requested) != len(set(requested)):
        raise ValueError('RETEST_IDS มีเลขข้อซ้ำ')
    source = load_questions(parent_run/'questions.csv')
    known = {q['question_id'] for q in source}
    unknown = set(requested)-known
    if unknown:
        raise ValueError(f'ไม่พบ question_id ในรอบหลัก: {sorted(unknown)}')
    selected = [q for q in source if q['question_id'] in set(requested)]
    folder = parent_run/'retests'/('trial-'+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True,exist_ok=False)
    write_csv(folder/'questions.csv',selected,['question_id','question_text'])
    (folder/'config.json').write_bytes((parent_run/'config.json').read_bytes())
    (folder/'retest_info.json').write_text(json.dumps({
        'parent_run':str(parent_run),'question_ids':[q['question_id'] for q in selected],
        'note':'Independent retest; not merged into parent submission; count this time in contest budget.'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return folder

RETEST_RUN = prepare_retest(RUN, RETEST_IDS)
RETEST_QUESTIONS = RETEST_RUN/'questions.csv'
RETEST_CONFIG = RETEST_RUN/'config.json'
print('รอบหลัก:', RUN)
print('รอบทดสอบใหม่:', RETEST_RUN)
print('ขั้นต่อไป: 8B; อย่ากด 8A ซ้ำหากต้องการรันต่อรอบทดสอบนี้')

# %% 8B
print('ทดสอบใหม่:',RETEST_RUN,flush=True)
if (RETEST_RUN/'status.json').exists():
    print('รอบทดสอบนี้จบแล้ว ไป 8C; ถ้าต้องการลองใหม่จริงให้สร้างรอบด้วย 8A')
else:
    fresh = not (RETEST_RUN/'checkpoint.json').exists()
    extra = ['--fresh','--setup-seconds','0'] if fresh else []
    cmd=[PYTHON,'-u','-m','rag_contest','run','--docs',str(DOCS),
         '--questions',str(RETEST_QUESTIONS),'--assets',str(ASSETS),
         '--config',str(RETEST_CONFIG),'--output',str(RETEST_RUN),
         '--expected-count',str(len(load_questions(RETEST_QUESTIONS))),*extra]
    exit_code=stream_command(cmd,log_path=RETEST_RUN/'run.log')
    print('รหัสจบ:',exit_code)
    if exit_code not in (0,2):
        raise RuntimeError('การทดสอบหยุดผิดพลาด ดู run.log ของ RETEST_RUN')

# %% 8C
import csv
traces=[]
if (RETEST_RUN/'checkpoint.json').exists():
    traces=json.loads((RETEST_RUN/'checkpoint.json').read_text(encoding='utf-8')).get('traces',[])
accepted={str(t['question_id']):t for t in traces if t.get('status')=='ok'}
selected=load_questions(RETEST_QUESTIONS)
export_dir=RETEST_RUN/'submission'
export_dir.mkdir(exist_ok=True)
retest_csv=export_dir/'answer.csv'
missing=[]
with retest_csv.open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['question_id','answer_text','document_id'])
    writer.writeheader()
    for q in selected:
        row=accepted.get(q['question_id'],{})
        answer_text=row.get('answer_text','')
        document_id=row.get('document_id','')
        if not answer_text or not document_id:
            answer_text,document_id='',''
            missing.append(q['question_id'])
        writer.writerow({'question_id':q['question_id'],'answer_text':answer_text,'document_id':document_id})
print('ไฟล์เฉพาะข้อที่ทดสอบใหม่:',retest_csv)
print('ยังไม่สำเร็จ:',missing)
print('ไม่ได้รวมกับไฟล์รอบหลัก; ไม่ใช่ไฟล์ส่งครบทั้งหมด')
for t in traces:
    print(t['question_id'],t.get('status'),repr(t.get('answer_text','')),t.get('error',''))
from google.colab import files
files.download(str(retest_csv))
