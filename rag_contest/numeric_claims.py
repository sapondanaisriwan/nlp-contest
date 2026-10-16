"""Bounded numeric equality checks from quoted model extractions, not gold."""
import json
import re
from decimal import Decimal
from .calculation import canonical

NUMBER = r'\d+(?:,\d{3})*(?:\.\d+)?'


def claim_spec(question):
    # Reject negations, inequalities and temporal versions rather than guessing
    # their meaning. The ordinary answer path handles these questions.
    if re.search(r'\b(not|no|never|before|after|until|since|at least|at most|over|under|more|less)\b', question, re.I):
        return None
    numbers = re.findall(NUMBER, question)
    if len(numbers) != 1:
        return None
    endpoint = re.search(r'\b(maximum|minimum)\s+base salary\b', question, re.I)
    warranty = re.search(r'\bwarranty\b.*\bvalid for\s+('+NUMBER+r')\s+years?\b', question, re.I)
    if not (endpoint or warranty) or not re.match(r'Is\b',question,re.I):
        return None
    return {'claimed': numbers[0].replace(',',''),
            'attribute': endpoint[1].lower()+' base salary' if endpoint else 'warranty duration for the exact item asked about',
            'unit': 'salary' if endpoint else 'years',
            'endpoint': endpoint[1].lower() if endpoint else None}


def instruction(spec):
    return ('Extract the actual '+spec['attribute']+' from the evidence. Do not answer yes/no. '
            'Return JSON with exactly value and quote (both strings). value is one number without commas. '
            'quote is an exact continuous excerpt containing that value, its entity and units. '
            'For a salary range copy the whole range; use its requested endpoint. '
            'For warranty choose the requested item, not a different item with another duration. '
            'Do not convert units. If not supported return NOT_FOUND.')


def inspect_claim(raw, evidence, spec):
    try:
        item = json.loads(raw)
        if not isinstance(item,dict) or set(item) != {'value','quote'}:
            raise ValueError('expected_value_quote_json')
        value,quote=item['value'],item['quote']
        if not isinstance(value,str) or not isinstance(quote,str) or not quote.strip():
            raise ValueError('invalid_fields')
        if not re.fullmatch(r'\d+(?:\.\d+)?',value):
            raise ValueError('invalid_scalar')
        if canonical(quote) not in canonical(evidence):
            raise ValueError('quote_not_in_evidence')
        number=Decimal(value)
        if number not in {Decimal(n.replace(',','')) for n in re.findall(NUMBER,quote)}:
            raise ValueError('value_not_in_quote')
        if spec['endpoint']:
            ranges=re.findall('('+NUMBER+r')\s*(?:to|–|-)\s*('+NUMBER+')',quote)
            if len(ranges)!=1 or not re.search(r'\b(?:base salary|salary)\b',quote,re.I):
                raise ValueError('need_one_salary_range')
            low,high=(Decimal(n.replace(',','')) for n in ranges[0])
            if low>high or number != (high if spec['endpoint']=='maximum' else low):
                raise ValueError('wrong_range_endpoint')
        elif not any(Decimal(n.replace(',',''))==number for n in re.findall('('+NUMBER+r')\s+years?\b',quote,re.I)):
            raise ValueError('value_must_be_in_years')
        return {'answer':'yes' if number==Decimal(spec['claimed']) else 'no',**item,'claimed':spec['claimed']},None
    except (ValueError,TypeError,ArithmeticError) as error:
        return None,str(error)
