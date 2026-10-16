# RAG Contest — คู่มือเริ่มต้น

ระบบนี้สร้างขึ้นเพื่อทดลองและเลือกวิธีที่ทำคะแนนได้ดีที่สุดภายใต้ CPU, RAM 12.7 GB และโมเดลสร้างข้อความไม่เกิน 2B **ยังไม่ใช่ผลรับรอง Rank 1**

## เริ่มใช้งานบน Colab

1. อัปโหลด `RAG_Contest_Colab.ipynb` เข้า Google Colab แล้วเลือก **CPU**
2. วาง `sample_docs`, `question_samples.csv`, `answer_samples.csv` ใน `MyDrive/NLP Project`
3. แก้ 3 ตำแหน่งในเซลล์แรกถ้าจำเป็น แล้วรันส่วน 1–7 ตามลำดับ
4. ส่งผล `score.json` กับข้อที่ผิดใน `diagnostics.json` กลับมาวิเคราะห์ร่วมกัน

Notebook แนบโค้ดทั้งหมดไว้ภายใน ใช้ได้โดยไม่ต้องอัปโหลดโฟลเดอร์ Python แยก ต้นฉบับไฟล์ของอาจารย์ไม่ได้ถูกแก้ไข

รุ่นปรับปรุง 10 ต.ค. 2569: ซ่อม pip ก่อนติดตั้ง แสดง stdout/stderr สด และแจ้งเวลาที่รอทุก 15 วินาที เริ่มด้วย dev 12 ข้อ ประเภทละ 3 ข้อ เปลี่ยน `DEV_LIMIT` เป็น 48 หรือ `None` เพื่อขยายชุดได้

ส่วน 5A ใช้เตรียมรอบใหม่ ส่วน 5B ใช้รันและรันต่อ ส่วน 5C ใช้อ่านสถานะ เมื่อหยุดใน runtime เดิมให้รันซ้ำเฉพาะ 5B ไม่รัน 5A ใหม่ หาก runtime หายให้เตรียมส่วน 1–4 แล้วใส่ตำแหน่งรอบเดิมใน `RESUME_RUN` ของ 5A โดยระบบยังตรวจว่าโค้ดและ dependencies ตรงกันก่อนใช้ผลเดิม

เมื่อย้ายจาก notebook รุ่นก่อนมารุ่นนี้ ต้องเริ่มรอบใหม่เนื่องจากโค้ดหลักเปลี่ยน ผลเก่าจะยังอยู่ใน Drive ไม่ถูกลบ โมเดลที่ดาวน์โหลดครบแล้วใช้ซ้ำได้

**อธิบายแบบง่าย:** BM25 ค้นจากคำ/ชื่อที่ตรงกัน; embedding ค้นจากความหมาย; RRF รวมอันดับสองวิธี; LLM อ่านหลักฐานแล้วตอบ; ตัวตรวจรูปแบบกันข้อความที่ส่งไม่ได้

## สิ่งที่มีแล้วและสิ่งที่ต้องวัดต่อ

- อ่าน PDF และเก็บเลขเอกสาร/หน้า สร้างดัชนีใหม่ ไม่ใช้ฐาน Chroma ตัวอย่าง
- ค้นแบบผสม BM25 + MiniLM และเลือกบริบททั้งเอกสารหรือเฉพาะส่วน
- รองรับ Qwen3-1.7B Q8, Qwen2.5-1.5B Q8/Q4 และ Transformers รวม baseline 0.5B
- แบ่งข้อมูลตามเอกสาร ตรวจข้อมูล วัดคะแนน และแยก holdout
- บังคับใช้ CPU, ปิด network ของ Python ระหว่าง inference, อ่านโมเดล local พร้อมตรวจ hash
- เก็บ checkpoint, บันทึกหลักฐาน/คำตอบดิบ, ตรวจรันซ้ำ และส่งออก notebook แข่งขันที่ล็อกหนึ่งโมเดล
- **ยังต้องทดสอบบน Colab:** คุณภาพโมเดลจริง, dense retrieval, RAM/เวลาทั้งระบบ, การติดตั้ง wheel ใน runtime จริง และความเหมือนของคำตอบเมื่อรันใหม่

ค่าผลทดสอบ local อยู่ใน `outputs/retrieval_dev.json` และ `outputs/local_validation.json` ตัวเลข retrieval เป็น BM25 แบบไม่ใช้โมเดล ไม่ใช่คะแนนคำตอบจาก LLM

## อ่านผลการทดลอง

| ไฟล์ | ความหมาย |
|---|---|
| `answer.csv` | ไฟล์ครบทุกข้อและผ่านรูปแบบ ยังต้องวัดความถูกต้องกับเฉลย |
| `partial_answers.csv` | มีข้อไม่สำเร็จ ห้ามใช้แทนไฟล์ส่งครบ |
| `status.json` | ข้อที่ยังตอบไม่ได้และลายนิ้วมือของข้อมูล/โมเดล/โค้ด |
| `diagnostics.json` | เอกสารที่ค้นได้ หลักฐาน prompt คำตอบดิบ และการลองใหม่ |
| `score.json` | คะแนนรวมเทียบ 200 คะแนนเอกสาร คะแนนแยกประเภท |
| `metrics.json` | เวลา RAM ที่ใช้สูงสุดจากการสุ่มวัด และเวลาประมาณสำหรับ 100 ข้อ |
| `checkpoint.json` | ผลแต่ละข้อสำหรับรันต่อเมื่อหลุด |
| `audit.json` | ปัญหาในข้อมูลและการแบ่งชุด ไม่แก้เฉลยต้นฉบับ |

ตัวตรวจคะแนนไม่ใช่โค้ดทางการของอาจารย์: yes/no และ choice ไม่สนตัวพิมพ์; scalar ใช้ `1/(1+abs(gold-prediction))`; short text ใช้ `1/(1+2*LD/len(gold))` โดยไม่ลบช่องว่าง/เปลี่ยน case; ข้อที่ยังตอบไม่ได้ให้ศูนย์ ชนิดเฉลยอนุมานจากคำตอบและตัวเลือก เพราะ CSV ไม่มีคอลัมน์ประเภท

ตัวกรองรูปแบบไม่ใช่ตัวตรวจความจริง คำตอบผิดที่ถูกไวยากรณ์ยังผ่านได้ ต้องวัดด้วยเฉลยและอ่านหลักฐาน ตัวเลือกซ้ำ/เฉลยกำกวมควรตรวจด้วยคนก่อนสรุปคุณภาพ

## ลำดับการทดลอง

1. ใช้ชุดพัฒนาเดียวกัน เปรียบเทียบ baseline 0.5B, Qwen3 Q8 และ Qwen2.5 Q8
2. ใช้โมเดลที่คะแนนดีที่สุด เปรียบเทียบ `full` กับ `chunks`
3. ทดลอง `hybrid` เทียบ `bm25` แล้วดูทั้งคะแนนและเวลา
4. หาก Q8 ช้าเกินไป ทดลอง Qwen2.5 Q4 อย่าสมมติว่าเล็กลงแล้วคะแนนจะเท่าเดิม
5. เปลี่ยนครั้งละเรื่อง และตั้งโฟลเดอร์ผลลัพธ์ใหม่ทุกครั้ง
6. หยุดเลือกโมเดลก่อนเปิด holdout; ถ้าปรับตาม holdout ให้ถือว่ามันกลายเป็นชุดพัฒนา

ไม่มีการ fine-tune หรือเก็บเฉลยไว้ค้นตอบ ระบบ inference รับแค่ PDF กับ question CSV ทุกคำถามใช้ generator ตัวเดียวกัน และเรียกไม่เกิน 3 ครั้ง: จัดประเภทเมื่อจำเป็น, ตอบ, ลองใหม่ ถ้ายังไม่ได้จะบันทึก unresolved

## คำสั่งสำหรับใช้นอก notebook

รันจากโฟลเดอร์โปรเจกต์หลังติดตั้ง dependencies; `requirements.txt` เป็นช่วงเวอร์ชันสำหรับพัฒนา รุ่นที่ทดสอบผ่านจะถูกตรึงก่อนส่งจริง

```bash
python -m unittest discover -s tests -v
python -m rag_contest audit --questions question_samples.csv --gold answer_samples.csv --output outputs/data
python -m rag_contest retrieval-check --docs sample_docs --questions outputs/data/dev_questions.csv --gold outputs/data/dev_answers.csv --output outputs/retrieval_dev.json
python -m rag_contest prepare --model qwen3-1.7b-q8 --assets assets/qwen3
python -m rag_contest run --docs sample_docs --questions outputs/data/dev_questions.csv --assets assets/qwen3 --config config.json --output outputs/qwen3-dev --fresh
python -m rag_contest score --questions outputs/data/dev_questions.csv --gold outputs/data/dev_answers.csv --run outputs/qwen3-dev --label dev
python -m rag_contest compare --runs outputs/qwen3-dev outputs/qwen25-dev --output outputs/comparison.json
```

`compare` เลือกผู้ชนะเบื้องต้นจากคะแนน dev โดยกรองเวลา **ประมาณสำหรับ 100 ข้อ ไม่รวม setup** และ RAM ก่อน ตัดสินคะแนนเท่ากันด้วยเวลา การประมาณนี้ไม่แทนการซ้อมเต็มรอบ

หากหลุด ให้ใช้คำสั่ง run เดิมและโฟลเดอร์เดิมโดยเอา `--fresh` ออก ห้ามเปลี่ยน input/config/model/code/environment การตรวจรันซ้ำต้องใช้ `--fresh` และโฟลเดอร์ใหม่ทั้งสองครั้ง

## ซ้อมและล็อกก่อนแข่งขัน

ใช้ฟังก์ชัน `rehearse(...)` ใน notebook เพื่อเลือก 100 ข้อจาก dev และรันสองครั้งในโฟลเดอร์ใหม่ ต้องวัดเวลาใน runtime CPU ใหม่จริง รวมติดตั้ง อ่านโมเดลจาก Drive/ดาวน์โหลด สร้างดัชนี และตอบคำถาม หากเตรียมโมเดลไว้ใน Drive ล่วงหน้าให้ใช้ขั้นตอนเดียวกันตอนซ้อมและวันจริง

การซ้อมใช้ ID เดิมของชุด dev ได้ แต่การรันแข่งขันบังคับ ID 1–100

```bash
python -m rag_contest verify-repeat --first outputs/rehearsal --second outputs/repeat --output outputs/repeat.json
python -m rag_contest freeze --run outputs/qwen3-dev --assets assets/qwen3 --repeat outputs/repeat.json --rehearsal outputs/rehearsal --output outputs/competition.lock.json
python -m rag_contest export-competition --lock outputs/competition.lock.json --output Competition_Colab.ipynb
```

`freeze` ตรวจ config, โมเดล, dependencies, โค้ด และหลักฐานการซ้อมก่อนสร้าง lock; `export-competition` สร้าง notebook อีกไฟล์ที่มีโมเดลเดียว ไม่มีตัวเลือกทดลอง ไม่อ่านเฉลย และติดตั้ง dependency versions ตาม lock รุ่น Python ต้องตรงด้วย

ก่อนส่ง source code ให้อาจารย์ต้องทดสอบ notebook แข่งขันที่ส่งออกอีกครั้งจาก runtime ใหม่ เก็บ assets ที่ตรวจสอบแล้วไว้ใน Drive ไม่ใช้การดาวน์โหลดจาก branch `main` ใหม่หลังล็อก หาก Colab เปลี่ยนรุ่น Python ให้แก้และซ้อมใหม่ก่อนกำหนดส่งโค้ด

เมื่อได้เอกสารเพิ่ม 50 ไฟล์ เพียงใส่ PDF ในโฟลเดอร์เอกสาร ระบบอ่านใหม่เอง ไม่ต้องแก้โค้ด ไม่แก้คำตอบด้วยมือหลังรัน

## ข้อจำกัดที่ตั้งใจแสดงไว้

- RAM วัด process tree และหน่วยความจำที่ host ใช้ทุก 0.2 วินาที อาจไม่เห็น peak ที่สั้นมาก; บน Colab ให้ตรวจกราฟ System RAM เพิ่มด้วย
- ตรวจเวลาและ RAM ระหว่างการเรียกโมเดล ไม่ใช่ตัว kill กลาง native inference จึงต้องเผื่อเวลาจากรอบซ้อมจริง
- การปิด network เป็นการบล็อก Python socket + offline flags ไม่ใช่ OS firewall; ทุก model loader ใช้ไฟล์ local และไม่มี API ภายนอก
- `temperature=0` ไม่รับประกันความเหมือนข้าม CPU/เวอร์ชัน; ใช้ verify-repeat จริงบนสภาพแวดล้อมที่จะส่ง
- หากทุกวิธีตอบไม่สำเร็จ ระบบเก็บ partial และไม่สร้าง answer.csv ต้องแก้ในช่วงพัฒนาก่อนวันแข่ง
- Qwen3 Q4 ยังไม่มี preset ในระบบนี้; Q4 ที่เตรียมไว้คือ Qwen2.5 จาก repository ของผู้พัฒนา

## แหล่งอ้างอิงทางเทคนิค

- [Qwen3-1.7B และการปิด thinking](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3 Q8 GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)
- [Qwen2.5-1.5B GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [llama-cpp-python API](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)
