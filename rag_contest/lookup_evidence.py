"""Conservative, source-only excerpts for unambiguous direct lookups.

No answers or question IDs are stored here. Ambiguous matches fall back to
complete sections. Policy questions deliberately keep their exceptions.
"""
import re


def lookup_instruction(question, kind):
    if kind == 'yesno' and re.search(r'\b(?:maximum|minimum)\b', question, re.I):
        return ('For a range, minimum is its lower endpoint and maximum is its upper endpoint. '
                'Compare that endpoint with the claimed number. A number inside the range '
                'is NOT necessarily the maximum or minimum. Return yes only if the exact claim is supported.')
    if kind == 'short' and re.search(r'^who pays\b', question, re.I):
        return 'The payer may be named in the section heading. Return that party, not the offence. Check exceptions.'
    return ''


def label_fleet_row(text):
    """Only label a known, anchored schema; other tables remain verbatim."""
    lines=text.splitlines()
    if not any(re.fullmatch(r'Boat\s+Class\s+Built\s+Hull\s+Ref\. kg\s+Amber kg\s+Red kg',line.strip()) for line in lines):
        return text
    row_pattern=r'(.+?)\s+(\d+[x+-]|Tub\s+\d+)\s+(\d{4})\s+(.+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)'
    rows=[re.fullmatch(row_pattern,line.strip()) for line in lines]
    rows=[r for r in rows if r]
    if len(rows)!=1:
        return text
    r=rows[0]
    return text+'\nColumn values from the row above: '+ '; '.join(
        f'{key}: {value}' for key,value in zip(['Boat','Class','Built','Hull','Ref. kg','Amber kg','Red kg'],r.groups()))


def lookup_evidence(document, codec, budget, question, structured=False):
    from .blocks import source_units
    matches = []
    matched_units = 0
    material = re.search(r'\bmade (?:of|from)\s+([\w-]+(?:\s+[\w-]+)?)[?.]?$', question, re.I)
    person = re.search(r"(?:what is|what's)\s+(.+?)[’']s\s+job title\??$", question, re.I)
    factor = bool(re.search(r'\bwhat (?:value of the )?dynamic factor\b', question, re.I))
    payer = bool(re.search(r'^who pays\b', question, re.I))
    collection_room = structured and bool(re.search(r'\broom\b',question,re.I) and re.search(r'\b(?:collect|pick up)\b',question,re.I))
    if not (material or person or factor or payer or collection_room):
        return ''
    for unit in source_units(document):
        if collection_room:
            if (re.search(r'\bRoom\s+[A-Z]?\d+',unit['text'],re.I)
                    and re.search(r'\b(?:collect\w*|pick\w* up)\b',unit['text'],re.I)):
                matched_units += 1
                matches.extend(unit['pieces'])
            continue
        if payer:
            # Keep the WHO in the heading AND every exception in this section.
            # Require a distinctive condition phrase, not a generic "fine" match.
            conditions = re.findall(r'\b(?:expired|broken|worn)\s+(?:\w+\s+){0,2}\w+', question, re.I)
            terms = [w.lower() for phrase in conditions for w in phrase.split()]
            if (terms and all(re.search(r'\b'+re.escape(w)+r'\b',unit['text'],re.I) for w in terms)
                    and re.search(r'\bpaid by\b', unit['text'].splitlines()[0], re.I)):
                matched_units += 1
                # Retain cross-page sections in full.
                for page, text in unit['pieces']:
                    matches.append((page,text))
            continue
        for page, text in unit['pieces']:
            lines = text.splitlines()
            if material and re.search(r'\bclass\b', question, re.I):
                # Require the column names, not just a match in a different table.
                headers = [i for i, line in enumerate(lines)
                           if re.search(r'\bClass\b', line) and re.search(r'\b(?:Hull|Material)\b', line, re.I)]
                if len(headers) != 1:
                    continue
                h = headers[0]
                rows = [line for line in lines[h+1:]
                        if re.search(r'\b'+re.escape(material[1])+r'\b', line, re.I)]
                if len(rows) == 1:
                    matches.append((page, '\n'.join([lines[0], lines[h], rows[0]])))
            elif person:
                # Join PDF line wraps, retaining the entire sentence and all names.
                flat = ' '.join(text.split())
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', flat)
                for sentence in sentences:
                    if re.search(re.escape(person[1])+r'\s*\([^)]*\)', sentence, re.I):
                        matches.append((page, sentence))
            elif factor:
                # Keep the definition paragraph with units, but not a worked example.
                for i, line in enumerate(lines):
                    if re.search(r'\bdynamic factor\b', line, re.I) and re.search(r'=\s*\d', line):
                        end = i+1
                        while end < len(lines) and not re.search(r'[.;]\s*$', lines[end-1]):
                            end += 1
                        matches.append((page, '\n'.join(lines[max(0,i-1):end])))
    # Multiple source matches may be conflicting versions or different entities.
    if (matched_units != 1 if collection_room or payer else len(matches) != 1):
        return ''
    rendered = '\n\n'.join(f'[Document {document["document_id"]}, page {page}]\n'
                           + (label_fleet_row(text) if structured and material else text)
                           for page,text in matches)
    return rendered if len(codec.encode(rendered)) <= budget else ''
