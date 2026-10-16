"""Export a submission notebook with one verified model and no model selector."""
import base64
import io
import json
import zipfile
from pathlib import Path

from .data import write_json
from .pipeline import source_fingerprint


def export_notebook(lock_path, output):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock['config'].get('device', 'cpu') != 'cpu':
        raise ValueError('Competition export requires CPU')
    if lock["sources"] != source_fingerprint():
        raise ValueError("Code changed since freeze; rerun validation before export")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for source in sorted(Path(__file__).parent.glob("*.py")):
            z.writestr("rag_contest/" + source.name, source.read_bytes())
        z.writestr("competition.lock.json", json.dumps(lock))
        z.writestr("config.json", json.dumps(lock["config"]))
    payload = base64.b64encode(stream.getvalue()).decode()
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": f"cell-{len(cells)}", "metadata": {}, "source": source}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    add("markdown", "# RAG Contest — เวอร์ชันแข่งขันที่ล็อกแล้ว\n\n"
        f"โมเดล: **{lock['model_name']}** ใช้ CPU เท่านั้น ไม่เปลี่ยนโมเดลหรือโค้ดหลังเห็นข้อสอบ\n"
        "เก็บโฟลเดอร์ assets ที่ตรวจสอบแล้วใน Drive ก่อนแข่งขัน ไม่ใช้ notebook พัฒนาเลือกโมเดลวันสอบ\n"
        "แก้เฉพาะตำแหน่งข้อมูล โฟลเดอร์ผลลัพธ์ใหม่สำหรับการรันใหม่; รันจากบนลงล่าง\n")
    add("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"
        "from pathlib import Path\nimport sys, subprocess, json, time, os\n"
        "DOCS = Path('/content/drive/MyDrive/NLP Project/sample_docs')\n"
        "QUESTIONS = Path('/content/drive/MyDrive/NLP Project/question.csv')\n"
        "OUTPUT = Path('/content/drive/MyDrive/NLP Project/competition_result')\n"
        f"ASSETS = Path('/content/drive/MyDrive/NLP Project/rag_results/assets/{lock['model_name']}')\n"
        "START = time.perf_counter()\n")
    add("code", "import base64, io, zipfile, platform\n"
        "PROJECT = Path('/content/rag_competition'); PROJECT.mkdir(exist_ok=True)\n"
        f"PAYLOAD = {payload!r}\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:\n"
        "    assert all((PROJECT / n).resolve().is_relative_to(PROJECT.resolve()) for n in z.namelist())\n"
        "    z.extractall(PROJECT)\n"
        "os.chdir(PROJECT)\n"
        "VENV = Path('/content/rag_cpu_competition_venv')\n"
        "sys.path.insert(0, str(PROJECT))\n"
        "from rag_contest.bootstrap import ensure_environment, ensure_uv\n"
        "UV = ensure_uv(PROJECT / 'uv-bin')\n"
        "PYTHON = ensure_environment(VENV, uv=UV)\n"
        "lock = json.loads(Path('competition.lock.json').read_text())\n"
        "assert platform.python_version() == lock['python'], 'รุ่น Python ต่างจากรอบซ้อม หยุดก่อนรัน'\n"
        "deps = lock['resolved_dependencies']\n"
        "torch_pin = 'torch==' + deps['torch']\n"
        "subprocess.run([UV, 'pip', 'install', '--python', PYTHON, torch_pin, '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)\n"
        "pins = [name + '==' + version for name, version in deps.items() if name != 'torch']\n"
        "subprocess.run([UV, 'pip', 'install', '--python', PYTHON, *pins, '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cpu'], check=True)\n"
        "SETUP_SECONDS = time.perf_counter() - START\n")
    add("code", "result = subprocess.run([PYTHON, '-u', '-m', 'rag_contest', 'run',\n"
        "    '--docs', str(DOCS), '--questions', str(QUESTIONS), '--assets', str(ASSETS),\n"
        "    '--config', 'config.json', '--lock', 'competition.lock.json', '--expected-count', '100',\n"
        "    '--output', str(OUTPUT), '--setup-seconds', str(SETUP_SECONDS)])\n"
        "assert result.returncode == 0, 'ยังไม่พร้อมส่ง อ่าน diagnostics/status และข้อความข้างบน'\n"
        "from google.colab import files\nfiles.download(str(OUTPUT / 'answer.csv'))\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    write_json(output, notebook)
    return {"notebook": str(output), "model": lock["model_name"]}
