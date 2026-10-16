# พัฒนาด้วย GPU บน Colab

รุ่นนี้ใช้ uv venv / uv pip install / uv pip freeze ทั้ง notebook พัฒนาและตัวส่งออกแข่งขัน ติดตั้ง uv ด้วย standalone installer ของ Astral เมื่อไม่มีอยู่ ไม่พึ่ง pip/ensurepip ใน Python environment คง cache ของ uv สำหรับแพ็กเกจทั่วไป ส่วน CUDA build ยังใช้ no-cache เพื่อไม่ใช้ binary CPU เก่า การ build CUDA ครั้งแรกอาจยังนาน

1. อัปโหลด RAG_Contest_Colab.ipynb รุ่นล่าสุดแล้วเปิดใน Colab
2. เลือก Runtime → Change runtime type → Hardware accelerator เป็น GPU เช่น T4 หากมี → Save ไม่เลือก TPU
3. รันข้อ 1 โดย DEVICE="cuda" และตั้งตำแหน่ง Drive ตามเดิม
4. รันข้อ 2: ตรวจ nvidia-smi และสร้าง llama-cpp-python พร้อม GGML_CUDA=ON ตาม CUDA toolkit ของ runtime ขั้นตอนติดตั้งครั้งแรกอาจนาน ไม่ใช่เวลาตอบคำถาม ถ้าล้มเหลวหยุดและดู /content/rag_contest_project/install.log ไม่เปลี่ยนไป CPU อัตโนมัติ
5. รันข้อ 3–4 ใช้ Qwen3-1.7B-Q8 ไฟล์เดิม ไม่เปลี่ยนเป็น Transformers อัตโนมัติในโหมด GPU การโหลดโมเดลใน 5B จะพิมพ์รายละเอียดการย้าย layers ไป GPU
6. ข้อ 5A ใส่ RESUME_RUN="", EVAL_SOURCE_RUN เป็นโฟลเดอร์ dev 48 ข้อล่าสุด, QUICK_TRIAL=True, TRIAL_CONFIG={"evidence_strategy":"tables","arithmetic":True}
7. รัน 5B–7 ตรวจ metrics.json ว่า device=cuda และ development_only=true ส่ง score.json, metrics.json, diagnostics.json กลับมาวิเคราะห์

ใช้ virtual environment แยก rag_gpu_venv และ rag_cpu_venv PyTorch ในโหมด GGUF ยังเป็น CPU สำหรับงานประกอบ แต่โมเดลสร้างคำตอบใช้ CUDA ผ่าน llama.cpp ไม่ได้ใช้ torch.cuda

กลับ CPU ภายหลัง: เปลี่ยน runtime เป็น CPU/None ตั้ง DEVICE="cpu" แล้วรัน 1–4 ใหม่ สร้างรอบใหม่ ห้ามใช้ checkpoint GPU ต่อเป็น CPU ต้องวัดคะแนน เวลา RAM และรันซ้ำบน CPU ก่อน freeze/export

ผลบน GPU อาจต่างจาก CPU จึงไม่รับประกันคะแนนหรือเวลาจนกว่าจะทดสอบ การตรวจในเครื่องครอบคลุมตรรกะเลือกอุปกรณ์และการปฏิเสธ fallback แต่ยังไม่ได้ทดสอบติดตั้งและ inference บน Colab GPU จริง

กติกา PDF ระบุพัฒนาได้บนคอมพิวเตอร์ใดก็ได้ และกำหนด CPU ในการทดสอบแข่งขัน โหมดนี้ใช้เพื่อพัฒนาเท่านั้น การตีความนี้ไม่ใช่ประกาศเพิ่มเติมจากอาจารย์

อ้างอิงการติดตั้ง CUDA: https://github.com/abetlen/llama-cpp-python#installation
การตั้งค่า Colab: https://research.google.com/colaboratory/faq.html
