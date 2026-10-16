"""Bootstrap uv and isolated Python without pip or ensurepip."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


def ensure_uv(folder):
    folder = Path(folder).resolve()
    executable = folder / ('uv.exe' if os.name == 'nt' else 'uv')
    existing = str(executable) if executable.is_file() else shutil.which('uv')
    if existing:
        subprocess.run([existing, '--version'], check=True)
        return existing
    if os.name == 'nt':
        raise RuntimeError('Install uv first: winget install --id=astral-sh.uv -e')
    folder.mkdir(parents=True, exist_ok=True)
    with urlopen('https://astral.sh/uv/install.sh', timeout=60) as response:
        installer = response.read().decode('utf-8')
    env = dict(os.environ, UV_INSTALL_DIR=str(folder), UV_NO_MODIFY_PATH='1')
    subprocess.run(['sh'], input=installer, text=True, env=env, check=True)
    subprocess.run([str(executable), '--version'], check=True)
    return str(executable)


def ensure_environment(folder, host_python=None, uv=None):
    host_python = host_python or sys.executable
    folder = Path(folder)
    uv = uv or ensure_uv(folder.parent / 'uv-bin')
    python = str(folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
    if not Path(python).exists():
        subprocess.run([uv, 'venv', '--python', host_python, '--no-python-downloads', str(folder)], check=True)
    subprocess.run([python, '--version'], check=True)
    return python
