"""Rank complete source sections/FAQ answers; never truncate a selected unit."""
import re
from .retrieval import BM25


BOUNDARY = re.compile(r'^(?:\d{1,2}(?:\.\d{1,2})*[.)]?[ \t]+[A-Z][^\n]{0,69}[^.!\n]$|Table\s+\d|(?:What|Which|Who|When|Where|How|Can|Does|Do|Is|Are)\b[^\n]*\?[ \t]*$)', re.M)
STOP = set('a an the is are was were do does did there it of to in on at for from with and or what which who how many much this that according document'.split())


def content_query(question):
    return ' '.join(w for w in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", question.lower()) if w not in STOP)


def source_units(document):
    # Concatenation lets a unit continue across a page boundary. Offsets map back
    # to original page text; page headers are retained rather than guessed away.
    joined, pages = '', []
    for page in document['pages']:
        start = len(joined)
        joined += page['text'] + '\n'
        pages.append((page['page'], start, start + len(page['text'])))
    starts = sorted({0, len(joined)} | {m.start() for m in BOUNDARY.finditer(joined)})
    units = []
    for a, b in zip(starts, starts[1:]):
        pieces = [(p, joined[max(a, s):min(b, e)].strip()) for p, s, e in pages if max(a,s) < min(b,e)]
        pieces = [(p, t) for p, t in pieces if t]
        if pieces:
            units.append(dict(order=a, pieces=pieces, text='\n'.join(t for _, t in pieces)))
    return units


def complete_evidence(document, codec, budget, question, focused=False):
    units = source_units(document)
    # Light plural normalization for matching only, never alter source text.
    def normalized(text):
        return ' '.join(w[:-1] if len(w)>4 and w.endswith('s') and not w.endswith(('ss','us','is')) else w
                        for w in content_query(text).split())
    query = normalized(question)
    if focused and re.search(r'\b(smoke|burning|fire|injury|injured|emergency)\b', question, re.I):
        query += ' emergency urgent immediate safety'
    texts = [normalized(u['text']) for u in units]
    scores = BM25(texts).scores(query)
    headers = [normalized(u['text'].splitlines()[0]) for u in units]
    header_scores = BM25(headers).scores(query)
    scores = [s + 2*h for s,h in zip(scores,header_scores)]
    requested = set(re.findall(r'\bsection\s+(\d+(?:\.\d+)*)', question, re.I))
    codes = re.findall(r'\b[A-Za-z]{1,8}-\d+[A-Za-z]?\b', question) if focused else []
    phase = re.search(r'\bphase\s+(\d+)\b', question, re.I) if focused else None
    safety_terms = set(re.findall(r'\b(?:smoke|fire|gas|injury|violence|emergency)\b', question.lower())) if focused else set()
    count_list = focused and bool(re.search(r'\b(?:how many|\d+)\b', question, re.I))
    list_nouns = set(normalized(question).split()) & {'code','category','plan','level','type','phase'} if count_list else set()
    def exact(i):
        text = units[i]['text']
        header = normalized(text.splitlines()[0])
        row = any(re.search(r'^\s*'+re.escape(c)+r'\s+[+-]?\d',text,re.M|re.I) for c in codes)
        phase_match = phase and re.search(r'\bphase\s+'+phase[1]+r'\b',text,re.I)
        emergency = safety_terms and (bool(safety_terms & set(normalized(text).split())) or
                                     bool(re.search(r'\b(?:urgent|emergency)\b',header)))
        return bool(row or phase_match or emergency or (list_nouns & set(header.split())))
    def priority(i):
        direct = any(re.match(re.escape(s) + r'[.)]?\s', units[i]['text']) for s in requested)
        return (-int(direct), -int(exact(i)), -scores[i], units[i]['order'])
    def render(selected):
        return '\n\n'.join(f"[Document {document['document_id']}, page {p}]\n{t}"
                           for i in sorted(selected) for p,t in units[i]['pieces'])
    selected = []
    for i in sorted(range(len(units)), key=priority):
        if scores[i] <= 0:
            continue
        # Keep complete units, but do not fill all spare space with weak matches.
        # A second unit can supply an exception; exact targets always compete.
        if focused and selected and not exact(i):
            if len(selected) >= 2 or scores[i] < .45 * max(scores[j] for j in selected):
                continue
        if len(codec.encode(render(selected + [i]))) <= budget:
            selected.append(i)
    # Empty evidence deliberately triggers NOT_FOUND/retry, never a clipped list.
    return render(selected)
