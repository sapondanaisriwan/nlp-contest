"""Local staging and CPU diagnostics; no change to prompts or decoding."""
import json
import os
import platform
import shutil
from pathlib import Path

from .data import file_hash
from .models import verify_assets


def stage_assets(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = verify_assets(source)
    if source.resolve() == destination.resolve():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['hashes'].items():
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError('Unsafe asset path')
        if target.is_file() and file_hash(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.staging')
        shutil.copyfile(source / name, temporary)
        if file_hash(temporary) != expected:
            temporary.unlink()
            raise ValueError('Asset copy checksum mismatch')
        temporary.replace(target)
    (destination / 'assets.json').write_text(json.dumps(manifest), encoding='utf-8')
    verify_assets(destination)
    return destination


def cpu_report(threads):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    model = platform.processor()
    info = Path('/proc/cpuinfo')
    if info.exists():
        model = next((line.split(':', 1)[1].strip() for line in info.read_text().splitlines()
                      if line.startswith('model name')), model)
    return {'cpu_model': model, 'logical_cpus': os.cpu_count(), 'affinity': affinity,
            'configured_threads': threads,
            'thread_environment': {key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}}
