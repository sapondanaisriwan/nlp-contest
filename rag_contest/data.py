"""Input validation, source extraction, data audit, and contest scoring.

This module deliberately does not import an inference engine.
"""
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def integer_id(value, name, maximum=None):
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text) or int(text) < 1:
        raise ValueError(f"Invalid {name}: {value!r}")
    result = str(int(text))
    if maximum and int(result) > maximum:
        raise ValueError(f"{name} exceeds {maximum}: {value}")
    return result


def read_csv(path, fields):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(x not in reader.fieldnames for x in fields):
            raise ValueError(f"{path}: required columns: {fields}")
        rows = []
        seen = set()
        for line, row in enumerate(reader, 2):
            if not any(v for v in row.values()):
                continue
            if None in row or any(row.get(x) is None for x in fields):
                raise ValueError(f"Malformed CSV row {line}: {path}")
            row = {key: row[key] for key in fields}
            qid = integer_id(row["question_id"], "question_id")
            if qid in seen:
                raise ValueError(f"Duplicate question_id {qid}: {path}")
            row["question_id"] = qid
            seen.add(qid)
            if "document_id" in row:
                row["document_id"] = integer_id(row["document_id"], "document_id", 150)
            if "question_text" in row and not row["question_text"].strip():
                raise ValueError(f"Empty question: {qid}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def load_questions(path):
    return read_csv(path, ["question_id", "question_text"])


def load_gold(path):
    return read_csv(path, ["question_id", "answer_text", "document_id"])


def options(question):
    # Prefer explicit labels so an opening article ('A patient ...') is not
    # mistaken for an option. Normalize Unicode line separators as well.
    question = "\n".join(question.replace("\\n", "\n").splitlines())
    pattern = r"(?<![\w])(?:\[([A-F])\]|([A-F])[).:])[ \t]*"
    matches = list(re.finditer(pattern, question, re.M))
    if len(matches) < 2:
        matches = list(re.finditer(r"^([A-F])[ \t]+", question, re.M))
    if len(matches) < 2:
        return {}
    labels = [next(group for group in m.groups() if group) for m in matches]
    if len(set(labels)) != len(labels) or labels[0] != "A":
        return {}
    return {label: question[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(question)].strip().rstrip(',;').rstrip()
            for i, (label, m) in enumerate(zip(labels, matches))}


def classify_basic(question):
    q = question.strip().lower()
    if options(question):
        return "choice"
    if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|must|may)\b", q):
        return "yesno"
    if re.search(r"\b(how many|how much|answer only (?:with )?(?:a |the )?number|numbers? only)\b", q):
        # Durations and ranges can still be ambiguous: the caller may supply overrides.
        return "number"
    # Recognize explicit information requests, including contextual preambles.
    # Do not classify every occurrence of 'which': it can be a relative clause.
    if re.search(r"\bis (?:this|that|the) statement (?:correct|true|false)\s*\?", q):
        return "yesno"
    if re.search(r"(?:^|[,;:.!?]\s+)(?:who|whom|whose|when|where)\b", q) or re.search(
        r"\b(?:which|what)\s+(?:animal|team|person|department|organisation|organization|"
        r"month|date|day|time|name|location|city|country)\b", q
    ):
        return "short"
    return "unknown"


def classify(question, mode="enhanced"):
    if mode == "basic":
        return classify_basic(question)
    q = question.replace("\\n", "\n").replace("’", "'").strip().lower()
    q = re.sub(r"\b(what|who|where|when)'s\b", r"\1 is", q)
    if options(question):
        return "choice"
    # A main yes/no clause after a contextual preamble; do not infer from a
    # bare assertion or an auxiliary embedded in a relative clause.
    if re.search(r"(?:^|[,;:.!?]\s+)(?:is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|must|may)\b[^?]*\?", q) or (not re.search(r'\b(?:what|which|who|where|when|how|why)\b', q) and re.search(r'\bis (?:this|that|it)\b[^?]*\?\s*$', q)):
        return "yesno"
    # Explicit scalar format takes precedence over date/time nouns.
    if re.search(r'\banswer (?:as|with) (?:a |one )?number\b', q):
        return 'number'
    # Identifiers and ranges are text even when they contain digits.
    if re.search(r'\b(?:what|which|how much)\b[^?]{0,70}\b(?:range|operating limits|phone number|telephone number|contact number|cut-?off time|full name|job title|light pattern)\b', q):
        return 'short'
    if re.search(r'\bhow much\b[^?]+\b(?:process|hold|handle)\b[^?]*\bper batch\b', q):
        return 'short'
    if re.search(r'\b(?:what|which)\s+(?:hall|room)\s+number\b', q):
        return 'number'
    # Measure words need not use the exact phrase "what is the cost".
    if re.search(r'\bwhat\b[^?]{0,100}\b(?:diameter|dynamic factor|detention charge|minimum age|order value|bottom do|final digit|lead time|callout charge|amount of cash|room number|hall number|wind speed|maintenance fee|monthly fee|volume|rated power|mileage|survival percentage|battery percentage|credit amount)\b', q):
        return 'number'
    if re.search(r'\b(?:what|which)\s+(?:is\s+)?(?:the\s+)?(?:full\s+)?(?:name|colou?r|position|role|plant|cave|application|resort|bank|part|component)\b', q):
        return 'short'
    if re.search(r'\b(?:what|which)\b[^?]{0,90}\b(?:job title|position|staff member|candidate|physician)\b', q):
        return 'short'
    if re.search(r'\b(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:name|title|category|type|date|location)\b', q):
        return 'short'
    if re.search(r'\bwhat\s+is\s+(?:the\s+)?[^?]*(?:speed\s+in\s+(?:kilometers|kilometres|miles)|time\s+in\s+(?:hours|minutes|seconds))\b', q):
        return 'number'
    if re.search(r'\b(?:which|what)\s+type\s+of\b', q):
        return 'short'
    if re.search(r'\b(?:which|what)\s+[^?]+\s+(?:is|are)\s+required\b', q):
        return 'short'
    result = classify_basic(question)
    if result != "unknown":
        return result
    if re.search(r"\b(?:which|what)\s+(?:company|street|road|province|office|role|job title|"
                 r"category|class|document|form|species|building|room(?!\s+number)|phone number|telephone number|contact number)\b", q):
        return "short"
    if re.search(r"\b(?:who|whom|where|when)\s+(?:is|are|was|were|does|do|should|will|must|can)\b", q):
        return "short"
    if re.search(r"\b(?:what|which)\s+year\b", q):
        return "number"
    # Explicit scalar quantities; leave ranges, schedules and 'how long' ambiguous.
    if re.search(r"\bwhat\s+is\s+the\s+(?:(?:total|monthly|annual|daily|approximate|normal|on-call|minimum|maximum|current|new|old)\s+)*"
                 r"(?:number of|percentage|cost|fee|price|amount|allowance)\b", q):
        return "number"
    # Explicit information questions left over are usually phrases, not truth claims.
    # Leave ambiguous quantity/duration questions for the model.
    if re.search(r'\b(?:what|which|who|whom|whose|where|when|why)\b', q):
        return 'short'
    if '?' not in q and not re.search(r'\b(?:how|what|which|who|where|when|why)\b', q) and re.search(
            r'\b(?:is|are|was|were|has|have|includes?|must|should|occurred|affected)\b', q):
        return 'yesno'
    return "unknown"


NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def numeric(value):
    if not NUMBER.fullmatch(str(value)):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def gold_type(question, answer):
    if options(question):
        return "choice"
    if answer.strip().lower() in ("yes", "no"):
        return "yesno"
    if numeric(answer.strip()) is not None:
        return "number"
    return "short"


def validate_answer(raw, kind, question):
    text = raw.strip()
    if not text or re.search(r"information not found|not in (?:the )?(?:context|document)|insufficient|no evidence|not_found|<think>|</think>", text, re.I):
        return None
    if kind == "yesno":
        return text.lower() if text.lower() in ("yes", "no") else None
    if kind == "choice":
        choices = options(question)
        # Accept a single wrapped label, never extract one from explanatory prose.
        label = text.upper()
        if re.fullmatch(r"\[[A-F]\]|\([A-F]\)|[A-F][.)]", label):
            label = label.strip("[]().")
        if label in choices:
            return label
        # A labelled option copied in full is unambiguous. Normalize whitespace
        # only: preserve case, numbers, punctuation and negation in the option text.
        match = re.fullmatch(r"(?:\[([A-Fa-f])\]|\(([A-Fa-f])\)|([A-Fa-f])(?:[).:]|(?=\s)))\s*(\S[\s\S]*)", text)
        if match:
            label = (match[1] or match[2] or match[3]).upper()
            normalized = lambda value: " ".join(value.replace("\\n", "\n").split())
            if label in choices and normalized(match[4]) == normalized(choices[label]):
                return label
        return None
    if kind == "number":
        return text if numeric(text) is not None else None
    if kind == "short" and len(text) <= 300 and len(text.split()) <= 40 and "\n" not in text:
        if re.search(r"^(assistant:|answer:|based on|according to the (?:context|document))", text, re.I):
            return None
        return text
    return None


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def answer_score(expected, actual, kind):
    if kind in ("yesno", "choice"):
        return float(expected.lower() == actual.lower())
    if kind == "number":
        a, b = numeric(expected), numeric(actual)
        return float(1 / (1 + abs(a - b))) if a is not None and b is not None else 0.0
    if not expected:
        return float(actual == expected)
    return 1 / (1 + 2 * levenshtein(expected, actual) / len(expected))


def evaluate(questions, gold, predictions):
    qmap = {r["question_id"]: r["question_text"] for r in questions}
    gmap = {r["question_id"]: r for r in gold}
    pmap = {r["question_id"]: r for r in predictions}
    if set(qmap) != set(gmap) or set(qmap) != set(pmap):
        raise ValueError("Questions, gold and predictions must have exactly the same IDs")
    details = []
    for qid, question in qmap.items():
        g, p = gmap[qid], pmap[qid]
        kind = gold_type(question, g["answer_text"])
        ds = float(g["document_id"] == p["document_id"])
        score = answer_score(g["answer_text"], p["answer_text"], kind)
        details.append(dict(question_id=qid, kind=kind, document_score=ds, answer_score=score,
                            expected=g["answer_text"], actual=p["answer_text"],
                            format_valid=validate_answer(p["answer_text"], kind, question) is not None))
    n = len(details)
    return {"count": n, "document_accuracy": sum(x["document_score"] for x in details) / n,
            "score_out_of_200": 100 * sum(x["document_score"] + x["answer_score"] for x in details) / n,
            "format_valid_fraction": sum(x["format_valid"] for x in details) / n,
            "answer_score_by_type": {k: sum(x["answer_score"] for x in details if x["kind"] == k) /
                                     sum(x["kind"] == k for x in details) for k in sorted({x["kind"] for x in details})},
            "scoring_assumptions": "Inferred type from gold; raw case-sensitive short-text LD; no whitespace normalization; invalid numeric=0. Not an official grader.",
            "details": details}


def audit_and_split(questions, gold, output, seed=20261009):
    qmap = {r["question_id"]: r for r in questions}
    gmap = {r["question_id"]: r for r in gold}
    if set(qmap) != set(gmap):
        raise ValueError("Question and answer IDs differ")
    issues, groups = [], defaultdict(list)
    for row in questions:
        qid, text = row["question_id"], row["question_text"]
        g = gmap[qid]
        groups[" ".join(text.casefold().split())].append(qid)
        opts = options(text)
        if opts and len(set(x.casefold() for x in opts.values())) != len(opts):
            issues.append(dict(question_id=qid, issue="duplicate_option_text"))
        if re.search(r"correct answer", text, re.I):
            issues.append(dict(question_id=qid, issue="answer_leaked_in_question"))
        kind = gold_type(text, g["answer_text"])
        if validate_answer(g["answer_text"], kind, text) is None:
            issues.append(dict(question_id=qid, issue="invalid_or_long_gold_answer"))
        if classify(text) == "number" and kind != "number":
            issues.append(dict(question_id=qid, issue="numeric_question_non_scalar_gold"))
        if re.search(r"[\u0e00-\u0e7f]", g["answer_text"]):
            issues.append(dict(question_id=qid, issue="non_english_gold"))
    for ids in groups.values():
        if len(ids) > 1:
            conflict = len({(gmap[i]["answer_text"], gmap[i]["document_id"]) for i in ids}) > 1
            issues.extend(dict(question_id=i, issue="conflicting_duplicate" if conflict else "duplicate_question") for i in ids)
    # Connected components prevent identical questions from leaking across documents.
    docs = sorted({g["document_id"] for g in gold}, key=int)
    parent = {d: d for d in docs}
    def root(d):
        while parent[d] != d:
            d = parent[d]
        return d
    for ids in groups.values():
        for qid in ids[1:]:
            parent[root(gmap[qid]["document_id"])] = root(gmap[ids[0]]["document_id"])
    components = defaultdict(list)
    for d in docs:
        components[root(d)].append(d)
    components = list(components.values())
    random.Random(seed).shuffle(components)
    held = set()
    for component in components:
        if len(held) >= math.ceil(len(docs) * .2):
            break
        held.update(component)
    output = Path(output)
    for name, select in [("dev", lambda d: d not in held), ("holdout", lambda d: d in held)]:
        selected = [r for r in questions if select(gmap[r["question_id"]]["document_id"])]
        write_csv(output / f"{name}_questions.csv", selected, ["question_id", "question_text"])
        write_csv(output / f"{name}_answers.csv", [gmap[r["question_id"]] for r in selected],
                  ["question_id", "answer_text", "document_id"])
    write_json(output / "audit.json", {"count": len(questions), "issues": issues, "seed": seed,
               "holdout_document_ids": sorted(held, key=int), "source_digest": digest([questions, gold]),
               "policy": "No source edits; all rows retained, flagged rows require human review."})
    return {"count": len(questions), "issues": len(issues), "dev_count": sum(g["document_id"] not in held for g in gold),
            "holdout_count": sum(g["document_id"] in held for g in gold)}


def extract_documents(folder):
    from pypdf import PdfReader
    documents, seen = [], set()
    for path in sorted(Path(folder).glob("*.pdf")):
        match = re.match(r"^(\d{3})_", path.name)
        if not match:
            raise ValueError(f"PDF name must start with NNN_: {path.name}")
        docid = integer_id(match.group(1), "document_id", 150)
        if docid in seen:
            raise ValueError(f"Duplicate document_id: {docid}")
        seen.add(docid)
        raw = [p.extract_text() or "" for p in PdfReader(path).pages]
        if not raw or any(not p.strip() for p in raw):
            raise ValueError(f"Empty/scanned page in {path.name}; text extraction needs review")
        # Conservative: remove repeated, recognized footer lines only.
        pages = []
        for number, text in enumerate(raw, 1):
            lines = unicodedata.normalize("NFKC", text).splitlines()
            keep = [line for line in lines if not re.search(r"Page\s+\d+\s+of\s+\d+\s*$", line)
                    and not line.startswith("Fictional document created for testing a question-answering system.")]
            pages.append({"page": number, "text": "\n".join(keep)})
        documents.append({"document_id": docid, "filename": path.name, "sha256": file_hash(path), "pages": pages})
    if not documents:
        raise ValueError(f"No PDFs in {folder}")
    return sorted(documents, key=lambda d: int(d["document_id"]))
