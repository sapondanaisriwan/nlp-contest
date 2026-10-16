"""Deterministic lexical/dense retrieval. No labels enter this module."""
import math
import re
from collections import Counter


def words(text):
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", text.casefold(), re.UNICODE)


class BM25:
    def __init__(self, texts):
        self.counts = [Counter(words(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(texts)) or 1
        df = Counter(w for c in self.counts for w in c)
        self.idf = {w: math.log(1 + (len(texts) - n + .5) / (n + .5)) for w, n in df.items()}

    def scores(self, query):
        terms = set(words(query))
        return [sum(self.idf.get(t, 0) * (c[t] * 2.5) /
                    (c[t] + 1.5 * (.25 + .75 * length / self.average))
                    for t in terms if c[t]) for c, length in zip(self.counts, self.lengths)]


def chunk_pages(documents, codec, size=350, overlap=60):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    chunks = []
    for doc in documents:
        for page in doc["pages"]:
            # Prefer paragraph/headings; hard window only for oversized sections.
            sections = re.split(r"\n(?=\d+(?:\.\d+)*[.)]?\s+[A-Z]|TABLE\s+\d|\n)", page["text"])
            buffer = []
            for section in sections:
                tokens = codec.encode(section + "\n")
                if len(buffer) + len(tokens) > size and buffer:
                    chunks.append(dict(document_id=doc["document_id"], page=page["page"],
                                       order=len(chunks), text=codec.decode(buffer)))
                    buffer = buffer[-overlap:] if overlap else []
                buffer.extend(tokens)
                while len(buffer) > size:
                    chunks.append(dict(document_id=doc["document_id"], page=page["page"],
                                       order=len(chunks), text=codec.decode(buffer[:size])))
                    buffer = buffer[size - overlap:]
            if buffer:
                chunks.append(dict(document_id=doc["document_id"], page=page["page"],
                                   order=len(chunks), text=codec.decode(buffer)))
    return chunks


class WordCodec:
    """Only for model-free retrieval diagnostics; never for inference budgets."""
    def encode(self, text):
        return re.findall(r"\S+\s*", text)

    def decode(self, tokens):
        return "".join(tokens)


class DenseEncoder:
    def __init__(self, local_path, threads=2):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(threads)
        self.model = SentenceTransformer(str(local_path), device="cpu", local_files_only=True)

    def encode(self, texts):
        import numpy as np
        # MiniLM truncates long inputs. Pool smaller windows instead of silently losing tails.
        tokenizer = self.model.tokenizer
        cap = min(224, self.model.max_seq_length - 2)
        windows, owners = [], []
        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(1, len(tokens)), cap):
                windows.append(tokenizer.decode(tokens[start:start + cap]))
                owners.append(i)
        vectors = self.model.encode(windows, batch_size=16, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
        pooled = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        for owner, vector in zip(owners, vectors):
            pooled[owner] += vector
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled


class Retriever:
    def __init__(self, documents, codec, dense=None, chunk_size=350, overlap=60):
        self.documents = {d["document_id"]: d for d in documents}
        self.chunks = chunk_pages(documents, codec, chunk_size, overlap)
        self.lexical = BM25([c["text"] for c in self.chunks])
        self.dense = dense
        self.vectors = dense.encode([c["text"] for c in self.chunks]) if dense else None

    def search(self, question, top=3):
        lexical = self.lexical.scores(question)
        sources = [lexical]
        if self.dense:
            sources.append((self.vectors @ self.dense.encode([question])[0]).tolist())
        ids = sorted(self.documents, key=int)
        doc_fusion = dict.fromkeys(ids, 0.0)
        chunk_fusion = [0.0] * len(self.chunks)
        for scores in sources:
            # Rank documents by their best passage so long PDFs do not win by duplication.
            best = {d: max(s for s, c in zip(scores, self.chunks) if c["document_id"] == d) for d in ids}
            for rank, d in enumerate(sorted(ids, key=lambda d: (-best[d], int(d))), 1):
                doc_fusion[d] += 1 / (60 + rank)
            ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], int(self.chunks[i]["document_id"]), i))
            for rank, index in enumerate(ranking, 1):
                chunk_fusion[index] += 1 / (60 + rank)
        ranked = sorted(ids, key=lambda d: (-doc_fusion[d], int(d)))[:top]
        lexical_best = {d: max(s for s,c in zip(lexical,self.chunks) if c['document_id']==d) for d in ids}
        return [{"document_id": d, "score": doc_fusion[d], "lexical_score": lexical_best[d],
                 "chunks": sorted([dict(c, score=chunk_fusion[i]) for i, c in enumerate(self.chunks)
                                   if c["document_id"] == d], key=lambda c: (-c["score"], c["order"]))}
                for d in ranked]

    def evidence(self, candidate, codec, budget=4000, mode="full", question="", strategy="ranked"):
        docid = candidate["document_id"]
        if strategy in ('complete', 'focused'):
            from .blocks import complete_evidence
            return complete_evidence(self.documents[docid], codec, budget, question, focused=strategy == 'focused')
        if mode == "full":
            blocks = [{"page": p["page"], "text": p["text"], "order": i}
                      for i, p in enumerate(self.documents[docid]["pages"])]
            text = "\n\n".join(f"[Document {docid}, page {b['page']}]\n{b['text']}" for b in blocks)
            if len(codec.encode(text)) <= budget:
                return text
        if strategy in ("guided", "tables"):
            from .evidence import guided_evidence
            return guided_evidence(self.documents[docid], candidate, codec, budget, question, tables=strategy == "tables")
        # Rank before packing; restore source order. Do not clip a table midway silently.
        selected = []
        for chunk in candidate["chunks"]:
            trial = sorted(selected + [chunk], key=lambda c: c["order"])
            text = "\n\n".join(f"[Document {docid}, page {c['page']}]\n{c['text']}" for c in trial)
            if len(codec.encode(text)) <= budget:
                selected.append(chunk)
        return "\n\n".join(f"[Document {docid}, page {c['page']}]\n{c['text']}"
                            for c in sorted(selected, key=lambda c: c["order"]))
