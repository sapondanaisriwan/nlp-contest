import csv, json
from pathlib import Path

# อ่านรายการคำถามต้นฉบับตามลำดับ ไม่อ่านเฉลย
question_file = RUN / "questions.csv"
if not question_file.exists():
    question_file = Path(QUESTIONS)
with question_file.open(encoding="utf-8-sig", newline="") as f:
    questions_to_export = list(csv.DictReader(f))
ids = [q["question_id"] for q in questions_to_export]
assert len(ids) == len(set(ids)), "question_id ซ้ำ ต้องตรวจไฟล์คำถาม"

accepted = {}
checkpoint = RUN / "checkpoint.json"
if checkpoint.exists():
    traces = json.loads(checkpoint.read_text(encoding="utf-8")).get("traces", [])
    accepted = {str(t["question_id"]): t for t in traces if t.get("status") == "ok"}
else:
    for name in ("answer.csv", "partial_answers.csv"):
        source = RUN / name
        if source.exists():
            with source.open(encoding="utf-8-sig", newline="") as f:
                accepted = {r["question_id"]: r for r in csv.DictReader(f)}
            break

rows, missing = [], []
for qid in ids:
    item = accepted.get(qid, {})
    answer_text = str(item.get("answer_text") or "")
    document_id = str(item.get("document_id") or "")
    if not answer_text.strip() or not document_id.strip():
        answer_text, document_id = "", ""
        missing.append(qid)
    rows.append({"question_id": qid, "answer_text": answer_text, "document_id": document_id})

# แยกจาก RUN/answer.csv ซึ่งระบบใช้ตัดสินว่ารันครบแล้ว
export_dir = RUN / "submission"
export_dir.mkdir(parents=True, exist_ok=True)
answer = export_dir / "answer.csv"
with answer.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question_id", "answer_text", "document_id"])
    writer.writeheader()
    writer.writerows(rows)
(export_dir / "export_status.json").write_text(json.dumps({
    "count": len(rows), "answered": len(rows)-len(missing),
    "blank_question_ids": missing, "complete": not missing
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("สร้างไฟล์:", answer)
print("ทั้งหมด:", len(rows), "ข้อ; มีคำตอบ:", len(rows)-len(missing), "ข้อ; ว่าง:", missing)
print("ไฟล์มีครบทุกแถว แต่แถวว่างไม่ใช่คำตอบที่สำเร็จ ตรวจว่ากติกายอมรับช่องว่างก่อนส่ง")
from google.colab import files
files.download(str(answer))
