"""Content-addressed, non-executable caches; never stores questions or answers."""
import json
import time
import unicodedata
from importlib.metadata import version
from collections import Counter
from pathlib import Path

from .data import digest, extract_documents, file_hash, write_json
from .retrieval import BM25, Retriever


def read_cache(path, key):
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if envelope["key"] == key and envelope["checksum"] == digest(payload):
            return payload
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def save_cache(path, key, payload):
    write_json(path, {"key": key, "checksum": digest(payload), "payload": payload})


def cached_documents(folder, cache_dir):
    started = time.perf_counter()
    files = [{"filename": p.name, "sha256": file_hash(p)}
             for p in sorted(Path(folder).glob("*.pdf"))]
    key = digest({"version": 1, "files": files,
                  "pypdf": version("pypdf"), "unicode": unicodedata.unidata_version,
                  "extractor": file_hash(Path(__file__).with_name("data.py")),
                  "cache_code": file_hash(Path(__file__))})
    path = Path(cache_dir) / "documents" / (key + ".json")
    docs = read_cache(path, key)
    hit = docs is not None
    if not hit:
        docs = extract_documents(folder)
        # Avoid saving under stale hashes if PDFs changed while being read.
        if [{"filename": p.name, "sha256": file_hash(p)}
            for p in sorted(Path(folder).glob("*.pdf"))] != files:
            raise ValueError("PDFs changed during extraction; retry with stable input files")
        save_cache(path, key, docs)
    return docs, {"hit": hit, "key": key, "path": str(path),
                  "seconds": time.perf_counter() - started}


def cached_retriever(docs, codec, dense, config, cache_dir, identity):
    started = time.perf_counter()
    key = digest({"version": 1, "documents": digest(docs), "identity": identity,
                  "retrieval": config.retrieval, "chunk_size": config.chunk_size,
                  "overlap": config.overlap, "threads": config.threads,
                  "retrieval_code": file_hash(Path(__file__).with_name("retrieval.py")),
                  "cache_code": file_hash(Path(__file__))})
    path = Path(cache_dir) / "indexes" / (key + ".json")
    state = read_cache(path, key)
    hit = False
    if state is not None:
        try:
            retriever = object.__new__(Retriever)
            retriever.documents = {d["document_id"]: d for d in docs}
            retriever.chunks = state["chunks"]
            retriever.dense = dense
            lexical = object.__new__(BM25)
            lexical.counts = [Counter(c) for c in state["counts"]]
            lexical.lengths = state["lengths"]
            lexical.average = state["average"]
            lexical.idf = state["idf"]
            retriever.lexical = lexical
            retriever.vectors = None
            if dense is not None:
                import numpy as np
                retriever.vectors = np.asarray(state["vectors"], dtype=np.float32)
                if retriever.vectors.ndim != 2 or len(retriever.vectors) != len(retriever.chunks):
                    raise ValueError("Invalid cached vector shape")
                if not np.isfinite(retriever.vectors).all():
                    raise ValueError("Invalid cached vectors")
            if not (len(lexical.counts) == len(lexical.lengths) == len(retriever.chunks)):
                raise ValueError("Invalid cached index shape")
            hit = True
        except (KeyError, TypeError, ValueError):
            hit = False
    if not hit:
        retriever = Retriever(docs, codec, dense, config.chunk_size, config.overlap)
        save_cache(path, key, {
            "chunks": retriever.chunks, "counts": retriever.lexical.counts,
            "lengths": retriever.lexical.lengths, "average": retriever.lexical.average,
            "idf": retriever.lexical.idf,
            "vectors": retriever.vectors.tolist() if retriever.vectors is not None else None,
        })
    return retriever, {"hit": hit, "key": key, "path": str(path),
                       "seconds": time.perf_counter() - started}
