"""Build a self-contained Colab notebook from the reviewed source files."""
import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    files = list((ROOT / "rag_contest").glob("*.py")) + [ROOT / "config.json", ROOT / "requirements.txt"]
    files += list((ROOT / "tests").glob("*.py"))
    for path in sorted(files):
        info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 9, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, path.read_bytes())
payload = base64.b64encode(buffer.getvalue()).decode()
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n", "outputs": [], "execution_count": None})


md("""
# RAG Contest — ตรวจ dev ทั้งชุดและใช้ cache • 15 ต.ค. 2569

รุ่นนี้เพิ่มกฎจำแนกชื่อ/หน่วย/ช่วงค่า แยกข้ออ่านค่าจากคำนวณ และซ่อมผลผิดรูปแบบด้วยหลักฐานเดิม
คง Qwen3 Q8 CPU และชุดคำถาม 100 ข้อเดิม ไม่ resume checkpoint ข้ามรุ่น

รอบนี้ล็อก Qwen3-1.7B Q8 ใช้ config จาก BASELINE_RUN แล้วเปิด focused evidence และคำนวณเฉพาะข้อที่จำเป็น
BASELINE_RUN ต้องเป็นรอบ 100 ข้อล่าสุด เพื่อใช้คำถามชุดเดิม การตอบทั้งหมดสร้างใหม่
ใส่โฟลเดอร์ผลรอบล่าสุดในข้อ 1 แล้วรัน 1–7 เท่านั้น ค่าเริ่มต้น DEV_LIMIT=100
ส่วน 8–12 เป็นเครื่องมือภายหลัง ไม่ต้องรันในรอบนี้ และอย่ากด Run all
โมเดลและ index cache ทำงานจาก /content ผล/ความคืบหน้าบันทึกใน Drive
เป็นการซ้อมบนชุด dev ไม่ใช่การวัด holdout และการรันครั้งเดียวยังไม่พิสูจน์การรันซ้ำ

ก่อนรัน เลือก Runtime → Change runtime type → CPU หรือ Hardware accelerator=None → Save
ข้อ 1 ตั้ง DEVICE="cpu" ตามค่าเริ่มต้น ใช้ GGUF Q8 เดิมและ uv ติดตั้งแพ็กเกจ
GPU ไม่ใช่ผลยืนยันเวลา/RAM/การรันซ้ำของการแข่งขัน และไม่สามารถ freeze/export เป็นเวอร์ชันแข่งขันได้
การติดตั้ง CUDA ครั้งแรกอาจนาน เพราะสร้างตัวรันจาก source ให้ตรงกับ Python/CUDA ของ runtime

รุ่น prompt/classifier: classifier_mode=enhanced รองรับคำถามหลังคำนำ ค่าเริ่มต้น prompt_mode=basic กลับไปใช้ prompt สั้นเดิม เพราะ guided คะแนนลดในชุดทดลอง
รับตัวเลือกพร้อมข้อความ เช่น D) September 2027 เฉพาะข้อความที่ตรงกับตัวเลือกนั้นทั้งหมดหลังจัดช่องว่าง ไม่รับคำอธิบายหรือข้อความที่ขัดกับตัวอักษร
ไม่เพิ่มรอบคิดหรือเปิด thinking ไม่ใช้เฉลยเป็นตัวอย่างใน prompt คำถามที่ยังคลุมเครือจะส่งให้โมเดลจัดประเภทตามเดิม
ข้อ 8 สามารถเทียบ prompt_mode=basic กับ guided โดยคงหลักฐานและโมเดลเดิม ส่วน classifier_mode=basic ใช้กฎรุ่นก่อน

รุ่นเลือกหลักฐานตามคำถาม: ค่าเริ่มต้น BM25 + 1200 tokens + evidence_strategy=guided
ถ้าคำถามระบุ Section จะให้หัวข้อนั้นมาก่อน ถ้ามีรหัสแถวตารางจะเก็บบริเวณตารางพร้อมหัวตารางและข้อความใกล้เคียง
รวมช่วงข้อความต้นฉบับที่ซ้อนกันโดยไม่เดาตำแหน่ง ถ้าจับตำแหน่งไม่ได้จะเก็บ chunk ตามเดิม
ไม่มีการใส่เฉลยหรือเปลี่ยนโมเดล หากหัวข้อใหญ่เกินงบจะเลือก chunks ที่ทับหัวข้อนั้นก่อน จึงยังต้องตรวจความครบของหลักฐาน
ใช้ evidence_strategy=ranked ใน config เพื่อเทียบวิธีเดิม โมเดลและ prompt ตอบยังเหมือนเดิม

เพิ่ม persistent index cache: ข้อความ PDF, chunks, BM25 และเวกเตอร์ (กรณี Hybrid) เก็บที่ rag_results/index_cache
ครั้งแรกสร้างและบันทึก รอบถัดไปตรวจ hash แล้วโหลดซ้ำ ถ้า PDF/โมเดล/tokenizer/วิธีแบ่งข้อความ/โค้ดที่เกี่ยวข้องเปลี่ยนจะสร้างใหม่
ยังโหลดดัชนีเข้า RAM เพื่อค้นหา และ Hybrid ยังใช้ embedding model แปลงคำถาม แต่ไม่แปลงเอกสารเดิมซ้ำ
ดู metrics.json → index_cache → documents/index → hit ว่าใช้ cache หรือไม่ ค่า fresh หมายถึงตอบใหม่ ไม่ได้หมายถึงดัชนีใหม่
เวลาที่ใช้ cache ต้องแยกจากการซ้อมเริ่มต้นใหม่ ไม่มี cache คำตอบหรือเฉลยในดัชนี

รุ่นนี้ค้นหลักฐานทุกข้อก่อน แล้วเรียงตามความยาวหลักฐาน ความซับซ้อนโดยประมาณ และช่องว่างคะแนนค้นหา
เป็นการประมาณความง่าย ไม่รับประกันว่าข้อแรกจะตอบถูก เริ่มตอบแต่ละข้อครั้งเดียวก่อนค่อยย้อนข้อที่รอแก้รูปแบบ/หลักฐาน
คำถามกำกวมอาจมีการเรียกโมเดลจัดประเภทเพิ่ม รอบแก้ไขเรียกได้อีกหนึ่งครั้งต่อข้อ ไม่มีการวนไม่จำกัด
CSV ส่งออกยังเรียงตามไฟล์คำถามเดิม ดูลำดับทำงานได้ใน schedule.json และผลระหว่างทางใน partial_answers.csv
ต้องเริ่มรอบใหม่ด้วย RESUME_RUN="" หลังเปลี่ยนรุ่น ไม่ใช้ checkpoint รุ่นก่อน

ใช้ **Runtime → Change runtime type → CPU** ก่อนเริ่ม

Notebook นี้มีโค้ดระบบแนบอยู่ภายใน ไม่ต้องอัปโหลดแพ็กเกจ Python แยก
วาง `sample_docs`, `question_samples.csv`, `answer_samples.csv` ใน Google Drive ของคุณ
ข้อมูลและโมเดลประมวลผลบน CPU ภายใน Colab เท่านั้น ไม่มี API สร้างคำตอบภายนอก

**สถานะ:** มีชุดทดสอบโปรแกรม แต่ยังต้องวัดคะแนน LLM เวลา และ RAM บน Colab ของคุณ
ค่า 190/200 และ Rank 1 เป็นเป้าหมาย ไม่ใช่ผลที่ทดสอบได้แล้ว

รันส่วน 1–7 เพื่อทดลองครั้งแรก ส่วน 8 เป็นต้นไปเป็นเครื่องมือเพิ่มเติม ไม่รันอัตโนมัติ
อย่าใช้ Run all ในวันแข่งก่อนอ่านหัวข้อการแข่งขันและล็อกระบบ
ส่วน 5 แบ่งเป็น 5A เตรียมรอบ, 5B รัน/รันต่อ, 5C ดูสถานะ เมื่อหยุดกลางทางให้รันซ้ำเฉพาะ 5B
รุ่นนี้เปลี่ยนโค้ดหลัก จึงไม่ใช้ checkpoint จากรุ่นก่อนปะปนกัน ผลรอบเก่าใน Drive จะยังอยู่
""")
md("""## 1. เชื่อม Drive และตั้งค่า 3 ตำแหน่ง
แก้เฉพาะโฟลเดอร์เอกสาร ไฟล์คำถาม และโฟลเดอร์ผลลัพธ์ ถ้าข้อมูลอยู่คนละตำแหน่ง
ครั้งแรกใช้ไฟล์ตัวอย่าง ระบบจะสร้างชุดพัฒนาแยกเองและเก็บ holdout ไว้
""")
code('''
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import time, sys, subprocess, json, os

DOCS = Path("/content/drive/MyDrive/NLP Project/sample_docs")
DEVICE = "cpu"  # ใช้ CPU ตามเดิม; เปลี่ยน runtime แล้วรันใหม่ 1–4
QUESTIONS = Path("/content/drive/MyDrive/NLP Project/question_samples.csv")
OUTPUT = Path("/content/drive/MyDrive/NLP Project/rag_results")
BASELINE_RUN = ""  # จำเป็น: วางโฟลเดอร์รอบล่าสุดที่มี config.json (ไม่ใช่ชื่อไฟล์)
assert BASELINE_RUN.strip(), "ใส่ BASELINE_RUN ของรอบล่าสุด เพื่อรักษา prompt และหลักฐานเดิม"
assert (Path(BASELINE_RUN) / "config.json").is_file(), "ไม่พบ config.json ใน BASELINE_RUN"
assert DOCS.is_dir(), f"ไม่พบโฟลเดอร์เอกสาร: {DOCS}"
assert QUESTIONS.is_file(), f"ไม่พบไฟล์คำถาม: {QUESTIONS}"
OUTPUT.mkdir(parents=True, exist_ok=True)
''')
md("""## 2. ติดตั้งและนำโค้ดที่แนบมาใช้
ใช้ uv สร้างสภาพแวดล้อมและติดตั้งแพ็กเกจ ไม่ต้องมี pip/ensurepip ใน venv
ขั้นตอนนี้ใช้อินเทอร์เน็ตเพื่อติดตั้งไลบรารีได้ตามกติกา
เวลา setup เริ่มก่อนติดตั้งและสิ้นสุดหลังเตรียมโมเดล โดยรวมเวลาที่หยุดอ่านระหว่างเซลล์ด้วย
หากติดตั้ง GGUF ไม่สำเร็จ จะใช้ Transformers เฉพาะการพัฒนา และแจ้งโมเดลที่ใช้จริง
""")
code('''
SETUP_START = time.perf_counter()
PROJECT = Path("/content/rag_contest_project")
PROJECT.mkdir(exist_ok=True)
import base64, io, zipfile
PAYLOAD = "''' + payload + '''"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    for name in z.namelist():
        target = (PROJECT / name).resolve()
        assert target.is_relative_to(PROJECT.resolve())
    z.extractall(PROJECT)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
# Reload this notebook's bundled code instead of keeping earlier imports in the kernel.
for module_name in list(sys.modules):
    if module_name == "rag_contest" or module_name.startswith("rag_contest."):
        del sys.modules[module_name]

# Isolated environment avoids conflicts with Colab's preinstalled GPU/vision packages.
assert DEVICE in ("cpu", "cuda")
if DEVICE == "cuda":
    subprocess.run(["nvidia-smi"], check=True)
VENV = Path("/content/rag_gpu_venv" if DEVICE == "cuda" else "/content/rag_cpu_venv")
from rag_contest.bootstrap import ensure_environment, ensure_uv
UV = ensure_uv(PROJECT / "uv-bin")
PYTHON = ensure_environment(VENV, uv=UV)
from rag_contest.notebook_io import stream_command

def uv_install(*packages):
    # Capture explicitly: some notebook frontends hide subprocess stderr.
    for flag in ("--index-url", "--extra-index-url"):
        if flag in packages:
            from urllib.parse import urlsplit
            url = packages[packages.index(flag) + 1]
            assert urlsplit(url).scheme == "https" and not any(c in url for c in "[]() "), f"ใช้ URL ธรรมดา ไม่ใช่ Markdown: {url!r}"
    command = [UV, "pip", "install", "--python", PYTHON, *packages]
    print("กำลังติดตั้ง:", " ".join(packages), flush=True)
    status = stream_command(command, log_path=PROJECT / "install.log")
    if status:
        raise subprocess.CalledProcessError(status, command)

uv_install("setuptools", "wheel")
uv_install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
requirements = [line.strip() for line in Path("requirements.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#") and not line.startswith("llama-cpp")]
uv_install(*requirements)
GGUF_AVAILABLE = True
try:
    if DEVICE == "cuda":
        # Build against the runtime CUDA toolkit, including Python versions without prebuilt wheels.
        import shutil
        if not shutil.which("nvcc"):
            raise RuntimeError("ไม่พบ CUDA toolkit: เลือก GPU runtime แล้วเริ่มใหม่")
        old_cmake = os.environ.get("CMAKE_ARGS")
        old_jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=ON"
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"
        try:
            uv_install("llama-cpp-python>=0.3.9,<0.4", "--no-binary", "llama-cpp-python", "--no-cache")
        finally:
            for key, old in (("CMAKE_ARGS", old_cmake), ("CMAKE_BUILD_PARALLEL_LEVEL", old_jobs)):
                if old is None: os.environ.pop(key, None)
                else: os.environ[key] = old
        status = stream_command([PYTHON, "-c", "from rag_contest.gpu import require_cuda; require_cuda()"])
        if status:
            raise RuntimeError("ตัวรันไม่รองรับ GPU: หยุดก่อนตอบ ส่ง install.log มาตรวจ")
    else:
        uv_install("llama-cpp-python>=0.3.9,<0.4", "--extra-index-url",
                    "https://abetlen.github.io/llama-cpp-python/whl/cpu")
    status = stream_command([PYTHON, "-c", "from llama_cpp import Llama"])
    if status:
        raise subprocess.CalledProcessError(status, [PYTHON, "-c", "from llama_cpp import Llama"])
except subprocess.CalledProcessError:
    if DEVICE == "cuda":
        raise RuntimeError("ติดตั้ง GPU ไม่สำเร็จ ไม่สลับไป CPU อัตโนมัติ ดู install.log")
    GGUF_AVAILABLE = False
    print("ติดตั้ง GGUF ไม่สำเร็จ: ใช้ Transformers สำหรับพัฒนา แล้ววัด RAM/เวลาใหม่")

def cli(*args, allow_incomplete=False):
    try:
        status = stream_command([PYTHON, "-u", "-m", "rag_contest", *map(str, args)])
    except KeyboardInterrupt:
        print("หยุดโปรแกรมแล้ว ข้อที่บันทึก checkpoint สำเร็จยังอยู่ใน Drive", flush=True)
        raise
    if status and not (allow_incomplete and status == 2):
        raise RuntimeError(f"ขั้นตอนหยุดด้วยรหัส {status}; อ่านข้อความด้านบน")
    return status

assert stream_command([PYTHON, "-m", "unittest", "discover", "-s", "tests", "-q"]) == 0, "ชุดทดสอบไม่ผ่าน"
print("ผ่านการทดสอบโปรแกรมแล้ว แต่ยังไม่ใช่การยืนยันคะแนน LLM")
''')
md("""## 3. ตรวจชุดข้อมูลและกันข้อสอบปิดท้าย
เซลล์นี้ใช้เฉลยเพื่อแบ่งชุดและตรวจข้อมูลเท่านั้น ขั้นตอนตอบคำถามไม่อ่านเฉลย
ค่าเริ่มต้นหา `answer_samples.csv` ข้างไฟล์คำถาม หากเป็นชุดแข่งขันให้ข้ามส่วนนี้
""")
code('''
GOLD = QUESTIONS.with_name("answer_samples.csv")
assert GOLD.exists(), "ส่วนนี้สำหรับชุดตัวอย่างที่มีเฉลยเท่านั้น"
DATA = OUTPUT / "data"
cli("audit", "--questions", QUESTIONS, "--gold", GOLD, "--output", DATA)
audit = json.loads((DATA / "audit.json").read_text())
print("จำนวนคำถาม:", audit["count"], "รายการควรตรวจ:", len(audit["issues"]))
print("เก็บ holdout ไว้ก่อน อย่าใช้คะแนน holdout เลือกโมเดล")
''')
md("""## 4. เตรียมโมเดลแรก
เริ่ม Qwen3-1.7B Q8 ปิด thinking ในระบบ ใช้โมเดลสร้างข้อความตัวเดียว
การดาวน์โหลดเป็นขั้นตอนออนไลน์ที่อนุญาต จากนั้นตรวจ hash และอ่านไฟล์ local เท่านั้น
""")
code('''
assert GGUF_AVAILABLE, "รอบนี้ใช้ Q8 เท่านั้น: แก้การติดตั้ง GGUF ก่อน ไม่เปลี่ยนเป็น HF อัตโนมัติ"
assert DEVICE == "cpu", "การซ้อมรอบนี้ใช้ CPU"
MODEL = "qwen3-1.7b-q8"
DRIVE_ASSETS = OUTPUT / "assets" / MODEL
cli("prepare", "--model", MODEL, "--assets", DRIVE_ASSETS)
LOCAL_ROOT = Path("/content/rag_contest_local")
ASSETS = LOCAL_ROOT / "assets" / MODEL
print("คัดลอกและตรวจ hash โมเดลไปพื้นที่ local...")
subprocess.run([PYTHON, "-c", "from rag_contest.runtime import stage_assets; import sys; stage_assets(sys.argv[1], sys.argv[2])",
                str(DRIVE_ASSETS), str(ASSETS)], check=True)
import shutil
if (OUTPUT / "index_cache").is_dir():
    shutil.copytree(OUTPUT / "index_cache", LOCAL_ROOT / "index_cache", dirs_exist_ok=True)
BASE_CONFIG = PROJECT / "config.json"
base_settings = json.loads((Path(BASELINE_RUN) / "config.json").read_text())
assert base_settings.get("device", "cpu") == "cpu", "BASELINE_RUN ต้องเป็นรอบ CPU"
from rag_contest.pipeline import Config
Config(**base_settings)
BASE_CONFIG.write_text(json.dumps(base_settings))
# These limits apply to fresh inference subprocesses, before native libraries import.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(base_settings.get("threads", 2))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("คง config จากรอบเดิม:", base_settings)
SETUP_SECONDS = time.perf_counter() - SETUP_START
print("อุปกรณ์:", DEVICE, "— GPU ใช้พัฒนาเท่านั้น" if DEVICE == "cuda" else "CPU")
print("โมเดล:", MODEL, "เก็บไว้ที่:", ASSETS)
''')
md("""## 5A. เตรียมรอบทดลอง — ยังไม่เริ่มตอบ
ค่าเริ่มต้นเลือก 100 ข้อจาก dev ให้ครอบคลุมประเภทคำตอบ โดยไม่ใช้ holdout
รอบนี้ใช้ EVAL_SOURCE_RUN=BASELINE_RUN และ QUICK_TRIAL=False เพื่อซ้อม 100 ข้อชุดเดิม
คะแนนชุดเล็กใช้ตรวจระบบและเปรียบเทียบเบื้องต้น ไม่ใช่คะแนนยืนยันความพร้อมแข่งขัน
ปล่อย `RESUME_RUN` ว่างเพื่อสร้างรอบใหม่ ถ้ากลับมาหลัง runtime หาย ให้ใส่ตำแหน่งรอบเดิมของรุ่นนี้
หากเปรียบเทียบกับรอบเก่า ให้ใส่โฟลเดอร์รอบนั้นใน `EVAL_SOURCE_RUN` เพื่อใช้คำถามเดิมและสร้างคำตอบใหม่ (ไม่ใช้คำตอบเก่า)
การแก้ตัวอ่านตัวเลือกอาจเปลี่ยนการสุ่มชุด dev และชนิดที่ตัววัดคะแนนอนุมาน จึงควรใช้ชุดเดิมและตัววัดคะแนนรุ่นเดียวกันทั้งสองรอบ
อย่ารัน 5A ซ้ำเมื่อแค่ต้องการทำต่อใน runtime เดิม ให้รัน 5B แทน
การเปลี่ยนโค้ด โมเดล ข้อมูล หรือ dependencies อาจทำให้ใช้ checkpoint เดิมไม่ได้ ระบบจะแจ้งแทนการฝืนใช้ผลเก่า
""")
code('''
from rag_contest.data import load_questions, load_gold, gold_type, write_csv
from collections import defaultdict, Counter
import random, uuid

DEV_LIMIT = 100  # ซ้อมครั้งเดียวครบ 100 ข้อจาก dev; ไม่เปิด holdout
RESUME_RUN = ""  # เช่น /content/drive/MyDrive/NLP Project/rag_results/runs/ชื่อรอบเดิม
EVAL_SOURCE_RUN = BASELINE_RUN  # ใช้คำถาม 100 ข้อเดิมเพื่อเทียบผลอย่างตรงกัน
QUICK_TRIAL = False  # True: คัดไม่เกิน 12 ข้อจาก score.json ของ EVAL_SOURCE_RUN มีข้อผิดและข้อถูก
EXPECTED_EVALUATION_DIGEST = ""  # วาง evaluation_digest จาก score.json รอบที่จะเทียบ
TRIAL_CONFIG = {"evidence_strategy": "focused", "arithmetic": True}
def prepare_trial():
    global RUN, EVAL_QUESTIONS, EVAL_GOLD, CONFIG, ASSETS, MODEL
    if RESUME_RUN.strip():
        RUN = Path(RESUME_RUN)
        saved = RUN / "run_setup.json"
        if not saved.exists():
            raise ValueError("รอบนี้ไม่มีข้อมูลตั้งค่าของ notebook รุ่นใหม่ ให้เริ่มรอบใหม่ โดยเก็บรอบเก่าไว้")
        setup = json.loads(saved.read_text())
        MODEL, ASSETS = setup["model"], Path(setup["assets"])
        EVAL_QUESTIONS, EVAL_GOLD = RUN / "eval_questions.csv", RUN / "eval_answers.csv"
        CONFIG = RUN / "config.json"
        if json.loads(CONFIG.read_text()).get("device", "cpu") != DEVICE:
            raise ValueError("อุปกรณ์ของรอบเดิมไม่ตรงกับ DEVICE ให้เริ่มรอบใหม่")
        for path in (EVAL_QUESTIONS, EVAL_GOLD, CONFIG):
            assert path.exists(), f"ไม่พบไฟล์ที่ต้องใช้: {path}"
        print("เลือกรันต่อ:", RUN, flush=True)
        return
    create_trial()

def create_trial():
    global RUN, EVAL_QUESTIONS, EVAL_GOLD, CONFIG
''' + textwrap.indent('''
questions = load_questions(DATA / "dev_questions.csv")
gold = {row["question_id"]: row for row in load_gold(DATA / "dev_answers.csv")}
buckets = defaultdict(list)
for row in questions:
    buckets[gold_type(row["question_text"], gold[row["question_id"]]["answer_text"])].append(row)
rng = random.Random(20261009)
for kind in sorted(buckets):
    rng.shuffle(buckets[kind])
selected_ids = set()
limit = len(questions) if DEV_LIMIT is None else min(DEV_LIMIT, len(questions))
assert limit > 0
if DEV_LIMIT == 100 and len(questions) < 100:
    raise ValueError("dev มีไม่ถึง 100 ข้อ ห้ามเติมจาก holdout")
if limit > 12 and limit != 100 and not EVAL_SOURCE_RUN.strip():
    raise ValueError("รอบเกิน 12 ข้อต้องระบุ EVAL_SOURCE_RUN เพื่อป้องกันสุ่มคนละชุดกับผลก่อนหน้า")
while len(selected_ids) < limit:
    for kind in sorted(buckets):
        if buckets[kind] and len(selected_ids) < limit:
            selected_ids.add(buckets[kind].pop()["question_id"])
selected = [row for row in questions if row["question_id"] in selected_ids]
if EVAL_SOURCE_RUN.strip():
    source = Path(EVAL_SOURCE_RUN)
    selected = load_questions(source / "eval_questions.csv")
    source_gold = {row["question_id"]: row for row in load_gold(source / "eval_answers.csv")}
    dev_questions = {row["question_id"]: row for row in questions}
    if set(source_gold) != {row["question_id"] for row in selected} or any(
        row != dev_questions.get(row["question_id"]) or source_gold.get(row["question_id"]) != gold.get(row["question_id"])
        for row in selected
    ):
        raise ValueError("ชุดเดิมต้องเป็นคำถามและเฉลย dev เดียวกันทั้งหมด ห้ามใช้ holdout")
    limit = len(selected)
    print("ใช้ชุดคำถามเดิมจาก:", source, flush=True)
if QUICK_TRIAL:
    if not EVAL_SOURCE_RUN.strip():
        raise ValueError("QUICK_TRIAL ต้องระบุ EVAL_SOURCE_RUN ของรอบ dev ที่มี score.json")
    from rag_contest.trials import quick_subset
    selected = quick_subset(selected, json.loads((Path(EVAL_SOURCE_RUN) / "score.json").read_text()))
    limit = len(selected)
RUN = OUTPUT / "runs" / (MODEL + "-dev" + str(limit) + "-" + uuid.uuid4().hex[:8])
RUN.mkdir(parents=True, exist_ok=False)
EVAL_QUESTIONS = RUN / "eval_questions.csv"
EVAL_GOLD = RUN / "eval_answers.csv"
from rag_contest.data import digest
evaluation_digest = digest([selected, [gold[row["question_id"]] for row in selected]])
print("ชุดคำถาม:", evaluation_digest)
print("question_id:", [row["question_id"] for row in selected])
if EXPECTED_EVALUATION_DIGEST and evaluation_digest != EXPECTED_EVALUATION_DIGEST.strip():
    raise ValueError("ชุดคำถามไม่ตรงกับรอบที่ต้องการเทียบ หยุดก่อนเรียกโมเดล ตรวจ EVAL_SOURCE_RUN")
if EVAL_SOURCE_RUN.strip() and not QUICK_TRIAL:
    source_score = json.loads((Path(EVAL_SOURCE_RUN) / "score.json").read_text())
    if evaluation_digest != source_score.get("evaluation_digest"):
        raise ValueError("ไฟล์ eval ของรอบต้นทางไม่ตรงกับ score.json หยุดก่อนเรียกโมเดล")
write_csv(EVAL_QUESTIONS, selected, ["question_id", "question_text"])
write_csv(EVAL_GOLD, [gold[row["question_id"]] for row in selected],
          ["question_id", "answer_text", "document_id"])
print("รอบนี้ตอบ", len(selected), "ข้อ — ผลลัพธ์:", RUN, flush=True)
print("ชนิดคำตอบ:", dict(Counter(gold_type(row["question_text"], gold[row["question_id"]]["answer_text"])
                                    for row in selected)), flush=True)
CONFIG = RUN / "config.json"
trial_config = json.loads(BASE_CONFIG.read_text())
trial_config.update(TRIAL_CONFIG)
from rag_contest.pipeline import Config
Config(**trial_config)
CONFIG.write_text(json.dumps(trial_config))
(RUN / "run_setup.json").write_text(json.dumps({"model": MODEL, "assets": str(ASSETS)}, indent=2))
'''.strip(), '    ') + '''

prepare_trial()
print("ตรวจชุด dev ทั้งหมดก่อน โดยไม่สร้างคำตอบและไม่เปิด holdout...")
subprocess.run([PYTHON, "-u", "-m", "rag_contest.preflight", "--docs", str(DOCS),
                "--questions", str(DATA / "dev_questions.csv"), "--gold", str(DATA / "dev_answers.csv"),
                "--assets", str(ASSETS), "--config", str(CONFIG),
                "--output", str(RUN / "all_dev_preflight.json")], check=True)
print("ตรวจหลักฐานด้วย tokenizer ของโมเดลจริง โดยยังไม่โหลดโมเดลตอบ...")
subprocess.run([PYTHON, "-u", "-m", "rag_contest.preflight", "--docs", str(DOCS),
                "--questions", str(EVAL_QUESTIONS), "--assets", str(ASSETS), "--config", str(CONFIG),
                "--output", str(RUN / "evidence_preflight.json")], check=True)
preflight = json.loads((RUN / "evidence_preflight.json").read_text())
assert not preflight["empty_ids"], f'หลักฐานว่าง: {preflight["empty_ids"]} หยุดก่อนเสียเวลาตอบ'
assert not preflight["over_context_ids"], f'บริบทเกิน: {preflight["over_context_ids"]} หยุดก่อนเริ่มตอบ'
if DEV_LIMIT == 100 and not RESUME_RUN.strip():
    assert preflight["count"] == 100, "รอบนี้ต้องมี 100 ข้อ ตรวจ EVAL_SOURCE_RUN/QUICK_TRIAL"
if not RESUME_RUN.strip():
    SETUP_SECONDS = time.perf_counter() - SETUP_START
    (RUN / "preparation.json").write_text(json.dumps({"setup_seconds": SETUP_SECONDS,
        "baseline_run": BASELINE_RUN, "local_assets": str(ASSETS),
        "note": "Includes installation, staging, preflight and pauses between cells; Drive caches may already exist."}))
print("ขั้นต่อไป: รันเซลล์ 5B เพื่อเริ่มตอบ")
''')
md("""## 5B. เริ่มตอบ / รันต่อจาก checkpoint
รันเซลล์นี้ซ้ำหลังหยุดได้โดยไม่สร้างโฟลเดอร์ใหม่ หากตอบครบแล้วจะไม่รันซ้ำ
จะแสดงขั้นตอนเตรียมระบบและแต่ละข้อ ระหว่างช่วงเงียบจะแจ้งเวลาที่รอทุก 15 วินาที
ข้อความแจ้งว่ายังรอไม่ได้ยืนยันว่าโปรแกรมคืบหน้า ต้องดูขั้นตอนและจำนวนข้อด้วย
การรันต่อยังต้องโหลดโมเดล แต่ใช้ดัชนีที่บันทึกไว้ได้ และข้ามข้อที่บันทึกไว้แล้ว
งบเวลานับสะสมข้ามการรันต่อ รุ่นนี้ห้าม resume checkpoint ของโค้ดรุ่นเก่า
""")
code('''
print("รอบที่ใช้:", RUN, flush=True)
if (RUN / "answer.csv").exists():
    print("รอบนี้มีไฟล์คำตอบครบแล้ว ไปส่วน 6 ได้เลย")
else:
    fresh = not (RUN / "checkpoint.json").exists()
    extra = ["--fresh", "--setup-seconds", str(SETUP_SECONDS)] if fresh else []
    try:
        cli("run", "--docs", DOCS, "--questions", EVAL_QUESTIONS, "--assets", ASSETS,
            "--config", CONFIG, "--output", RUN, *extra, allow_incomplete=True)
    finally:
        # Cache persistence is separate from answer checkpoints, which stay on Drive.
        if (LOCAL_ROOT / "index_cache").is_dir():
            shutil.copytree(LOCAL_ROOT / "index_cache", OUTPUT / "index_cache", dirs_exist_ok=True)
''')
md("""## 5C. ดูสถานะหลังจบหรือกดหยุด
เซลล์นี้อ่านไฟล์อย่างเดียว ไม่เริ่มโมเดล ไม่ลบข้อมูล หากส่วน 5B ยังรันอยู่ต้องหยุดก่อนรันเซลล์นี้
""")
code('''
checkpoint = RUN / "checkpoint.json"
traces = json.loads(checkpoint.read_text()).get("traces", []) if checkpoint.exists() else []
print("บันทึกแล้ว:", len(traces), "ข้อ")
print("ผ่านรูปแบบ:", sum(t.get("status") == "ok" for t in traces), "ข้อ (ยังไม่ใช่จำนวนตอบถูก)")
print("ยังตอบไม่สำเร็จ:", [t["question_id"] for t in traces if t.get("status") != "ok"])
print("ไฟล์ส่งครบ:", (RUN / "answer.csv").exists())
print("โฟลเดอร์:", RUN)
if not (RUN / "status.json").exists():
    print("ยังไม่จบรอบ: รัน 5B ต่อก่อนวัดคะแนนในส่วน 6")
''')
md("""## 6. วัดคะแนนและอ่านข้อผิดพลาด
สูตรคะแนนใช้ตามสไลด์ โดย short text เทียบตัวอักษรตรง ๆ และชนิดคำถามอนุมานจากเฉลย
ยังไม่ใช่ตัวตรวจอย่างเป็นทางการของอาจารย์ ข้อที่ระบบตอบไม่ได้คิดเป็นศูนย์
""")
code('''
assert (RUN / "status.json").exists(), "ยังไม่จบรอบ ให้รันส่วน 5B ต่อก่อน"
cli("score", "--questions", EVAL_QUESTIONS, "--gold", EVAL_GOLD,
    "--run", RUN, "--label", "dev")
score = json.loads((RUN / "score.json").read_text())
print("คะแนนเทียบเต็ม 200:", round(score["score_out_of_200"], 2))
print("เอกสารถูก:", f'{score["document_accuracy"]:.1%}')
print("รูปแบบถูก:", f'{score["format_valid_fraction"]:.1%}')
print("ระบบตอบครบ (ไม่ใช่รับรองคำตอบถูก):", score["complete"])
print("รูปแบบตรงกับเฉลยทุกข้อ:", score["gold_format_valid"])
print("ข้อที่จำแนกประเภทไม่ตรงเฉลย:", score["type_mismatch_ids"])
print("ข้อที่ยังไม่มีประเภท:", score.get("unclassified_ids", []))
print("จำนวนสถานะแต่ละแบบ:", score.get("progress_counts", {}))
print("แยกประเภท:", score["answer_score_by_type"])
print("เวลา/RAM:", score["metrics"])
traces = json.loads((RUN / "diagnostics.json").read_text())
for purpose in ("classify", "answer", "retry"):
    calls = [c for t in traces for c in t["calls"] if c["purpose"] == purpose]
    print(purpose, "calls:", len(calls), "seconds:", round(sum(c["elapsed_seconds"] for c in calls), 2),
          "input tokens:", sum(c.get("prompt_tokens", 0) for c in calls),
          "output tokens:", sum(c.get("completion_tokens") or 0 for c in calls))
print("inference_seconds รวมการอ่านบริบทและสร้างคำตอบ ยังไม่ได้แยก prefill/decode")
errors = sorted(score["details"], key=lambda r: r["document_score"] + r["answer_score"])
for row in errors[:10]:
    print(row)
''')
md("""## 7. เก็บเวอร์ชันไลบรารีและดูไฟล์ผลลัพธ์
`answer.csv` จะถูกสร้างเมื่อทุกข้อผ่านการตรวจรูปแบบเท่านั้น แต่ไม่ได้แปลว่าทุกคำตอบถูก
หากไม่ครบจะมี `partial_answers.csv` และ `diagnostics.json` ให้วิเคราะห์ ห้ามส่ง partial แทนไฟล์ครบ
""")
code('''
frozen = subprocess.check_output([UV, "pip", "freeze", "--python", PYTHON], text=True)
(RUN / "uv.version.txt").write_text(subprocess.check_output([UV, "--version"], text=True))
(RUN / "environment.freeze.txt").write_text(frozen)
print("ผลการทดลองอยู่ที่", RUN)
print("ไฟล์ส่งครบ:", (RUN / "answer.csv").exists())
print("สถานะ:", json.loads((RUN / "status.json").read_text()))
print("จบรอบซ้อมครั้งเดียวแล้ว ส่ง score.json, metrics.json และ diagnostics.json จากโฟลเดอร์นี้มาตรวจ")
print("ไม่ต้องรันข้อ 8–12 ในรอบนี้; ไฟล์คำตอบชุด dev ไม่ใช่ไฟล์ส่งแข่งขันจริง")
''')
md("""## 8. ทดลองตัวเลือกถัดไป — กดรันเองเมื่อพร้อม
ทดลองทีละตัว ไม่โหลดโมเดลสร้างคำตอบสองตัวพร้อมกัน
ตัวเลือกโมเดลถัดไปคือ `qwen2.5-1.5b-q8` เมื่อพร้อมทดลอง ไม่ใช้โมเดลตัวอย่างอาจารย์เป็นตัวเปรียบเทียบ
ใช้ Q4 เมื่อ Q8 ไม่ผ่านเวลา: `qwen2.5-1.5b-q4`
ทดลอง `context_mode=chunks` หรือ `retrieval=bm25` ทีละอย่าง โดยใช้ชุดพัฒนาเดิม
ค่าเริ่มต้นปัจจุบันคือ 1200 tokens แล้ว จึงไม่ต้องเรียก SHORT_CONTEXT_RUN ซ้ำเพื่อทดลองค่าเดียวกัน
บริบทสั้นอาจขาดตารางหรือข้อยกเว้น ต้องเทียบคะแนนก่อนเลือกใช้ ไม่ถือว่าเร็วหรือแม่นขึ้นจนกว่าจะวัดจริง
รุ่นนี้แก้โค้ดแล้ว ต้องสร้างรอบใหม่ ห้าม resume checkpoint จากรุ่นเก่า เก็บผลเก่าไว้เปรียบเทียบ
เวลาที่ได้จาก runtime ที่มี cache ให้ถือเป็นการพัฒนา ยังไม่ใช่ cold-start rehearsal
""")
code('''
def experiment(model_name, context_mode=None, retrieval=None, evidence_tokens=None, prompt_mode=None, classifier_mode=None):
    start = time.perf_counter()
    assets = OUTPUT / "assets" / model_name
    cli("prepare", "--model", model_name, "--assets", assets)
    config = json.loads(CONFIG.read_text())
    for key, value in dict(context_mode=context_mode, retrieval=retrieval, evidence_tokens=evidence_tokens, prompt_mode=prompt_mode, classifier_mode=classifier_mode).items():
        if value is not None:
            config[key] = value
    run = OUTPUT / "runs" / (model_name + "-" + config["context_mode"] + "-" + config["retrieval"] + "-" + str(config["evidence_tokens"]) + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    cfg = run / "config.json"
    cfg.write_text(json.dumps(config))
    # Omit setup time here: not a clean runtime benchmark.
    cli("run", "--docs", DOCS, "--questions", EVAL_QUESTIONS, "--assets", assets,
        "--config", cfg, "--output", run, "--fresh", allow_incomplete=True)
    cli("score", "--questions", EVAL_QUESTIONS, "--gold", EVAL_GOLD,
        "--run", run, "--label", "dev")
    return run

# Uncomment one line at a time:
# SECOND_RUN = experiment("qwen2.5-1.5b-q8")
# CHUNKS_RUN = experiment(MODEL, context_mode="chunks")
# SHORT_CONTEXT_RUN = experiment(MODEL, evidence_tokens=1200)
# LEXICAL_RUN = experiment(MODEL, retrieval="bm25")
# BASIC_PROMPT_RUN = experiment(MODEL, prompt_mode="basic")

def compare_runs(*runs):
    cli("compare", "--runs", *runs, "--output", OUTPUT / "comparison.json")
    return json.loads((OUTPUT / "comparison.json").read_text())

# COMPARISON = compare_runs(RUN, SECOND_RUN)
''')
md("""## 9. ซ้อม 100 ข้อและตรวจรันซ้ำ — ยังไม่ใช้ holdout
เลือก 100 ข้อจากชุดพัฒนา ตามลำดับ ID คงที่
สำหรับ cold-start ให้เปิด runtime CPU ใหม่ แล้วจับเวลาติดตั้ง/เตรียมจริงก่อนเรียกฟังก์ชัน
สองรอบใช้โฟลเดอร์ใหม่และ `--fresh` เพื่อพิสูจน์ว่าไม่ได้อ่านคำตอบเดิม
หลังซ้อมใช้คำสั่ง `freeze` ในคู่มือเพื่อล็อกโมเดล โค้ด และ dependencies
""")
code('''
def rehearse(selected_assets, selected_config, measured_setup_seconds):
    from rag_contest.data import load_questions, write_csv
    selected = load_questions(DATA / "dev_questions.csv")[:100]
    assert len(selected) == 100
    rehearsal_questions = OUTPUT / "rehearsal_questions.csv"
    write_csv(rehearsal_questions, selected, ["question_id", "question_text"])
    tag = time.strftime("%Y%m%d-%H%M%S")
    first, second = OUTPUT / ("rehearsal-" + tag), OUTPUT / ("repeat-" + tag)
    for target in (first, second):
        cli("run", "--docs", DOCS, "--questions", rehearsal_questions, "--assets", selected_assets,
            "--config", selected_config, "--output", target, "--fresh",
            "--setup-seconds", measured_setup_seconds)
    repeat_report = OUTPUT / ("repeat-" + tag + ".json")
    cli("verify-repeat", "--first", first, "--second", second, "--output", repeat_report)
    return first, second, repeat_report

# REHEARSAL, REPEAT, REPEAT_REPORT = rehearse(ASSETS, CONFIG, SETUP_SECONDS)
''')
md("""## 10. เปิด holdout ครั้งเดียวหลังเลือกโมเดล
เปิดเมื่อหยุดเลือกโมเดลแล้วเท่านั้น หากแก้ระบบจากผลชุดนี้ ต้องรายงานว่าชุดนี้กลายเป็นข้อมูลพัฒนา
ไม่ใช้ผลนี้เลือกระหว่างโมเดล
""")
code('''
def evaluate_holdout(selected_assets, selected_config):
    run = OUTPUT / ("holdout-" + time.strftime("%Y%m%d-%H%M%S"))
    cli("run", "--docs", DOCS, "--questions", DATA / "holdout_questions.csv", "--assets", selected_assets,
        "--config", selected_config, "--output", run, "--fresh", allow_incomplete=True)
    cli("score", "--questions", DATA / "holdout_questions.csv", "--gold", DATA / "holdout_answers.csv",
        "--run", run, "--label", "holdout")
    return run

# HOLDOUT_RUN = evaluate_holdout(ASSETS, CONFIG)
''')
md("""## 11. วันแข่งขัน
ใช้โค้ดและ lock ที่ส่งอาจารย์ไว้ก่อนแจกข้อสอบเท่านั้น ห้ามเปลี่ยนโมเดลหรือโค้ดหลังเห็นข้อสอบ
ให้เพิ่ม PDF ทั้งหมดลงโฟลเดอร์เอกสาร แล้วใช้คำถามจริงกับฟังก์ชันด้านล่าง
ไม่รันเซลล์แบ่งชุดข้อมูล ไม่อ่านเฉลย และไม่เลือกโมเดลใหม่

เตรียม dependencies ตาม lock ใน runtime ใหม่ล่วงหน้า ตรวจรุ่น Python ให้ตรง
ใช้ `freeze` แล้ว `export-competition` เพื่อสร้าง notebook แข่งขันอัตโนมัติก่อนส่งโค้ด
""")
code('''
def competition_run(question_csv, locked_assets, locked_config, competition_lock):
    run = OUTPUT / ("competition-" + time.strftime("%Y%m%d-%H%M%S"))
    cli("run", "--docs", DOCS, "--questions", question_csv, "--assets", locked_assets,
        "--config", locked_config, "--lock", competition_lock, "--expected-count", 100,
        "--output", run, "--fresh")
    from google.colab import files
    files.download(str(run / "answer.csv"))
    return run

# competition_run(QUESTIONS, ASSETS, CONFIG, OUTPUT / "competition.lock.json")
''')
md("""## 12. สร้าง notebook แข่งขันอัตโนมัติหลังผ่านการซ้อม
ฟังก์ชันนี้ไม่รันอัตโนมัติ ต้องใช้ผล dev ของโมเดลที่เลือก การซ้อม 100 ข้อ และรายงานรันซ้ำที่ผ่านแล้ว
นำ notebook ที่ส่งออกไปทดสอบจาก runtime ใหม่อีกครั้งก่อนส่งอาจารย์
""")
code('''
def export_final(selected_dev_run, selected_assets, rehearsal, repeat_report):
    lock = OUTPUT / "competition.lock.json"
    final_notebook = OUTPUT / "Competition_Colab.ipynb"
    cli("freeze", "--run", selected_dev_run, "--assets", selected_assets, "--repeat", repeat_report,
        "--rehearsal", rehearsal, "--output", lock)
    cli("export-competition", "--lock", lock, "--output", final_notebook)
    from google.colab import files
    files.download(str(final_notebook))
    return final_notebook

# FINAL_NOTEBOOK = export_final(RUN, ASSETS, REHEARSAL, REPEAT_REPORT)
''')

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"name": "RAG_Contest_Colab.ipynb", "provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03}"
path = ROOT / "RAG_Contest_Colab.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(path)
