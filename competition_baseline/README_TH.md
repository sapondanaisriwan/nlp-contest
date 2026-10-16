# ชุด baseline สำหรับเตรียมแข่งขัน

เปิด Competition_Baseline_CPU.ipynb บน Colab และเลือก CPU
แก้ DOCS, QUESTIONS, ASSETS_DRIVE และ OUTPUT_ROOT ในเซลล์แรก
ใช้ไฟล์คำถามจริง 100 ข้อและ PDF ทั้งหมดที่กำหนด ไม่ใช้ answer_samples.csv
assets ต้องเป็นโฟลเดอร์ Qwen3-1.7B Q8 เดิมพร้อม manifest/tokenizer/model

โค้ด rag_contest ทั้ง 18 ไฟล์ตรวจ SHA-256 ตรงกับ provenance ของรอบ
outputs/local-dev277-20261015-013510 (161.9032/200)
ไม่มี Fine-tuning และไม่มีโค้ด v3/v4/v5 ผสมในโฟลเดอร์นี้
โค้ดทดลองที่โฟลเดอร์หลักและคำสั่ง just improve* ยังอยู่เหมือนเดิม

config.local.json คือค่าที่ใช้ทดสอบในเครื่องเดิม
config.colab.json เปลี่ยนเฉพาะ threads=2, max_seconds=5400,
total_seconds=6300 เพื่อให้เข้ากับเป้าหมาย CPU/เวลาบน Colab
ค่าเวลาเป็น soft limit ที่ตรวจระหว่างการเรียกโมเดล ไม่รับประกันว่าจะหยุดตรงวินาที
จึงไม่อ้างว่าคะแนนหรือเวลาที่วัดในเครื่องจะเหมือนกันบน Colab

requirements.lock.txt คือ dependency versions จากการทดสอบ Windows
Notebook ติดตั้งรุ่นเหล่านี้ด้วย uv โดยต้องมี CPU wheel ของ llama-cpp-python
ยังไม่ได้ทดสอบการติดตั้งบน Colab จริง หากไม่มี wheel จะหยุด ไม่เปลี่ยนรุ่นเอง
ยังไม่ผ่านการรับรอง freeze/repeat/rehearsal ของระบบ ไม่ได้สร้าง lock ที่อ้างว่าผ่านแล้ว
baseline ทดสอบ 277 ข้อยังมี 10 ข้อ unresolved ต้องตรวจความครบก่อนส่ง

รัน notebook ตามลำดับ การตอบไม่โหลดเฉลยและไม่เรียก API สร้างคำตอบ
ถ้าหยุดกลางทางใน runtime เดิม รันเฉพาะเซลล์ตอบซ้ำเพื่อ resume
resume ใช้ได้เมื่อ code/config/ข้อมูลตรงกัน และไม่ช่วยแก้ข้อที่ unresolved แบบสิ้นสุดแล้ว
ดู status.json และ diagnostics.json หากไม่ครบ ห้ามแก้ answer_text หรือ document_id เอง
เซลล์สุดท้ายดาวน์โหลด answer.csv เฉพาะเมื่อสถานะครบและโครงสร้าง/ลำดับถูกต้อง
อย่าส่ง partial_answers.csv โดยเข้าใจว่าเป็นคำตอบครบทุกข้อ

ก่อนแข่งขันต้องทดสอบบน Colab CPU ถ้ายังมีเวลา ไม่ใช้ผลเวลาในเครื่องแทนผล Colab
