"""General query attributes; no document IDs, labels or expected answers."""
import re

def attributes(question):
    groups=[]
    for pattern,terms in [
        (r'\bsalary\b',('salary','pay')),
        (r'\bjob title|\bposition\b',('engineer','manager','officer','director','technician')),
        (r'\bphone|\btelephone',('phone','telephone','call')),
        (r'\bvotes?\b',('votes','vote','tally')),
        (r'\bdynamic factor\b',('dynamic factor',)),
        (r'\bexpired\b.*\bcertificate\b',('expired','certificate')),
    ]:
        if re.search(pattern,question,re.I):groups.append(terms)
    material=re.search(r'\bmade (?:of|from)\s+([\w-]+(?:\s+[\w-]+)?)(?:[?.]|$)',question,re.I)
    if material:groups.append((material[1].lower(),))
    return groups

def attribute_score(text,groups):
    normalized=' '.join(text.lower().split())
    return sum(any(re.search(r'\b'+re.escape(term)+r'\b',normalized) for term in group) for group in groups)

def short_instruction(question):
    q=question.lower()
    for pattern,attribute in [
        (r'job title|\bposition\b','job title, not the person name'),
        (r'\bstreet|\broad\b','street name only, without city or address'),
        (r'\broom\b','room designation only'),
        (r'\bphone|\btelephone','telephone number only'),
        (r'\bclass\b','class designation only'),
        (r'\bcategory\b','category designation only'),
        (r'\bwho pays\b','responsible party only'),
        (r'\bwhich company\b','company name only')]:
        if re.search(pattern,q):return 'Return the '+attribute+'. Copy the relevant phrase; no label or sentence.'
    return 'Copy only the requested name or phrase. No labels or full sentences.'
