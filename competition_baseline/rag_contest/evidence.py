"""Question-guided source spans. No gold answers or question IDs are used."""
import re


HEADING = re.compile(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+([A-Z][^\n]+)$", re.M)


def anchors(pages, question):
    targets = []
    sections = set(re.findall(r"\bsections?\s+(\d+(?:\.\d+)*)\b", question, re.I))
    # Follow a numbered section across pages until the next peer/parent heading.
    active = None
    for page in pages:
        text = page["text"]
        headings = list(HEADING.finditer(text))
        start = 0 if active else None
        for heading in headings:
            number = heading[1]
            if active and not number.startswith(active + "."):
                targets.append((page["page"], start, heading.start()))
                active, start = None, None
            if number in sections:
                active, start = number, heading.start()
        if active:
            targets.append((page["page"], start, len(text)))

    codes = set(re.findall(r"\b[A-Za-z]{1,8}-\d+[A-Za-z]?\b", question))
    for page in pages:
        text = page["text"]
        lines = list(re.finditer(r"[^\n]+", text))
        for i, line in enumerate(lines):
            # Only codes at the start of a line followed by a numeric cell count as rows.
            if not any(re.match(re.escape(code) + r"\s+[+-]?\d", line[0], re.I) for code in codes):
                continue
            # Preserve the preceding table caption/header/rows and nearby notes.
            captions = [j for j in range(i) if re.match(r"\s*table\s+\d", lines[j][0], re.I)]
            start = captions[-1] if captions else max(0, i - 8)
            end = i + 1
            while end < len(lines) and end < i + 14:
                if re.match(r"\s*table\s+\d", lines[end][0], re.I) or HEADING.match(lines[end][0]):
                    break
                end += 1
            targets.append((page["page"], lines[start].start(), lines[end - 1].end()))
    return [t for t in targets if t[2] > t[1]]


def table_spans(pages, question):
    """Keep captions, columns and rows together; rank by specific query terms."""
    stop = set('what which is the a an of for to in on at how many much does it and with from'.split())
    terms = set(re.findall(r"[a-z][a-z0-9-]+", question.lower())) - stop
    ranked = []
    for page in pages:
        text = page['text']
        captions = list(re.finditer(r'^\s*Table\s+\d+[^\n]*', text, re.M | re.I))
        for caption in captions:
            ends = [m.start() for m in HEADING.finditer(text, caption.end())]
            ends += [m.start() for m in captions if m.start() > caption.start()]
            end = min(ends) if ends else len(text)
            block = text[caption.start():end]
            found = set(re.findall(r'[a-z][a-z0-9-]+', block.lower())) & terms
            if found:
                ranked.append((len(found), page['page'], caption.start(), end))
    return [(p, start, end) for _, p, start, end in sorted(ranked, key=lambda x: (-x[0], x[1], x[2]))]


def guided_evidence(document, candidate, codec, budget, question, tables=False):
    pages = {p["page"]: p["text"] for p in document["pages"]}
    target_spans = anchors(document["pages"], question)
    if tables:
        target_spans += table_spans(document['pages'], question)
    chunks = []
    for chunk in candidate["chunks"]:
        text = chunk["text"].strip()
        start = pages[chunk["page"]].find(text)
        chunks.append({"page": chunk["page"], "start": start, "end": start + len(text),
                       "text": text, "order": chunk["order"]})
    def targeted(chunk):
        return chunk["start"] >= 0 and any(
            p == chunk["page"] and chunk["start"] < end and chunk["end"] > start
            for p, start, end in target_spans)
    # Exact requested spans first, then their chunks if the whole section is too large.
    preferred = [{"page": p, "start": start, "end": end, "text": pages[p][start:end], "order": -1}
                 for p, start, end in target_spans]
    preferred += sorted(chunks, key=lambda c: not targeted(c))  # stable retrieval order

    def render(items):
        blocks = []
        for page in sorted(pages):
            intervals = sorted((c["start"], c["end"]) for c in items if c["page"] == page and c["start"] >= 0)
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            known = [pages[page][start:end] for start, end in merged]
            blocks.extend((page, text) for text in known)
            # Tokenizer decoding can change spacing. Keep those chunks without guessing offsets.
            for c in sorted(items, key=lambda c: c["order"]):
                if c["page"] == page and c["start"] < 0 and not any(c["text"] in t for t in known):
                    blocks.append((page, c["text"]))
                    known.append(c["text"])
        return "\n\n".join(f"[Document {document['document_id']}, page {page}]\n{text}" for page, text in blocks)

    selected = []
    for item in preferred:
        trial = render(selected + [item])
        if len(codec.encode(trial)) <= budget:
            selected.append(item)
    return render(selected)
