"""Forward child output to notebooks, including stderr and quiet-stage heartbeats."""
import queue
import subprocess
import threading
import time
from pathlib import Path


def stream_command(command, log_path=None, heartbeat_seconds=15):
    started = time.monotonic()
    messages = queue.Queue()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    log = Path(log_path).open("a", encoding="utf-8") if log_path else None
    try:
        while True:
            try:
                line = messages.get(timeout=heartbeat_seconds)
            except queue.Empty:
                print(f"ยังรอโปรแกรมส่งผลลัพธ์ — ผ่านไป {time.monotonic() - started:.0f} วินาที "
                      "(ยังไม่ใช่จำนวนข้อที่ทำเสร็จ)", flush=True)
                continue
            if line is None:
                break
            print(line, end="", flush=True)
            if log:
                log.write(line)
                log.flush()
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        reader.join(timeout=2)
        process.stdout.close()
        if log:
            log.close()
