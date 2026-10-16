"""Question-focused excerpts, without answer labels or dataset identifiers."""
import re
from .blocks import source_units, content_query
from .retrieval import BM25


def precise_evidence(document, codec, budget, question):
    candidates=[]
    for unit in source_units(document):
        for page,text in unit['pieces']:
            # Preserve tables/lists together. Split long prose into overlapping
            # sentence windows so an oversized opening section is not discarded.
            table=bool(re.search(r'^Table\s+\d',text,re.I))
            if table or len(codec.encode(text))<=220:
                candidates.append((page,text))
            else:
                sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z"\d])',text)
                for start in range(0,len(sentences),2):
                    block=' '.join(sentences[start:start+3]).strip()
                    if block: candidates.append((page,block))
    if not candidates:return ''
    def norm(t):
        return ' '.join(w[:-1] if len(w)>4 and w.endswith('s') and not w.endswith(('ss','us','is')) else w
                        for w in content_query(t).split())
    query=norm(question)
    expansions=[(r'\b(cost|price|fee|charge)\b','cost price fee charge budget million'),
                (r'\b(votes?|highest|second-highest)\b','votes result tally candidate'),
                (r'\b(smoke|burning|fire|emergency)\b','urgent emergency safety immediately'),
                (r'\b(phone|telephone)\b','telephone phone call contact'),
                (r'\b(job title|position)\b','engineer manager director officer'),
                (r'\b(approved|signed|who)\b','approved signed name')]
    for pattern,extra in expansions:
        if re.search(pattern,question,re.I):query+=' '+extra
    scores=BM25([norm(t) for _,t in candidates]).scores(query)
    codes=re.findall(r'\b[A-Za-z]{1,8}-\d+[A-Za-z]?\b',question)
    exact=[]
    for i,(_,text) in enumerate(candidates):
        exact.append(any(re.search(r'^\s*'+re.escape(code)+r'\s+\d',text,re.M|re.I) for code in codes))
    ranking=sorted(range(len(candidates)),key=lambda i:(-int(exact[i]),-scores[i],i))
    selected=[]
    def render(ids):
        return '\n\n'.join(f'[Document {document["document_id"]}, page {candidates[i][0]}]\n{candidates[i][1]}' for i in sorted(ids))
    for i in ranking:
        if scores[i]<=0:continue
        if selected and len(selected)>=3:break
        if selected and not exact[i] and scores[i]<scores[ranking[0]]*.35:continue
        if len(codec.encode(render(selected+[i])))<=budget:selected.append(i)
    return render(selected)
