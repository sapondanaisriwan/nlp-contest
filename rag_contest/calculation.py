"""Bounded arithmetic, never eval. Source selection/units still require evaluation."""
import ast
import json
import re
from decimal import Decimal, InvalidOperation, localcontext

INSTRUCTION = ('For a calculation, return a JSON object with expression, quote and units. '
               'quote must be an exact continuous evidence excerpt containing the input values. '
               'units must state input and requested output units and conversion assumptions. '
               'expression may use only numbers, parentheses, + - * / and ceil(number expression) for started periods. '
               'Use applicable rates and discounts. Do not invent missing quantities. '
               'Use quantities from THIS question, not a worked example. Use the active rate, not a withdrawn rate. '
               'Operands may also come from the question. State any conversion assumptions in units. '
               'Return the JSON, not a bare input value. Keep quote and units brief. If unsupported return NOT_FOUND.')


def needs_calculation(question):
    # "total" and "per day" alone can be direct lookups, not computations.
    return bool(re.search(r'\b(calculate|difference|daily cost|one year|annual cost|'
                          r'brining time|total (?:cost|credit|detention charge)|total number of calendar days|'
                          r'combined|minimum.*time.*weight)\b', question, re.I))


def canonical(text):
    """Normalize typography, not words, quantities, case or negation."""
    return re.sub(r'\s+', '', text.translate(str.maketrans({'×':'*','·':'*','÷':'/','−':'-','–':'-'})))


def inspect_calculation(raw, evidence, question=''):
    """Return (verified arithmetic, error code); never execute generated code."""
    try:
        return _calculate(raw, evidence, question), None
    except (ValueError, TypeError, SyntaxError, InvalidOperation, ArithmeticError) as error:
        return None, str(error) or type(error).__name__


def calculate(raw, evidence, question=''):
    return inspect_calculation(raw, evidence, question)[0]


def _calculate(raw, evidence, question):
    item = json.loads(raw)
    if not isinstance(item, dict) or set(item) != {'expression', 'quote', 'units'}:
        raise ValueError('expected_expression_quote_units_json')
    expression, quote, units = (item[k] for k in ('expression', 'quote', 'units'))
    if not all(isinstance(v, str) for v in (expression, quote, units)):
        raise ValueError('json_fields_must_be_strings')
    if not quote.strip() or canonical(quote) not in canonical(evidence):
        raise ValueError('quote_not_in_evidence')
    if not units.strip() or len(expression) > 100:
        raise ValueError('missing_units_or_expression_too_long')
    expression = expression.translate(str.maketrans({'×':'*','·':'*','÷':'/','−':'-','–':'-'})).strip()
    if not re.fullmatch(r'[0-9.()+*/\sceil-]+', expression):
        raise ValueError('unsupported_expression_characters')
    tree = ast.parse(expression.strip(), mode='eval')
    if not any(isinstance(node, ast.BinOp) for node in ast.walk(tree)):
        raise ValueError('expression_must_calculate_not_copy_input')
    if len(list(ast.walk(tree))) > 32:
        raise ValueError('expression_too_complex')
    source_numbers = {Decimal(n.replace(',', '')) for n in re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', quote)}
    source_numbers |= {Decimal(n.replace(',', '')) for n in re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', question)}
    # Admit conversion constants only when the units text explicitly names
    # the corresponding units/assumption. This is not semantic validation.
    allowed = source_numbers | {Decimal(1)}
    conversions = {7:('week','day'),12:('year','month'),24:('day','hour'),
                   60:('hour','minute'),100:('percent',),365:('year','day')}
    for number, labels in conversions.items():
        if all(label in units.lower() for label in labels): allowed.add(Decimal(number))
    if re.search(r'\b30\s*days?\b', units, re.I) and 'month' in units.lower(): allowed.add(Decimal(30))
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(ast.get_source_segment(expression.strip(), node))
            if value not in allowed:
                raise ValueError('Unquoted operand')
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ceil' and len(node.args) == 1 and not node.keywords:
            from decimal import ROUND_CEILING
            return walk(node.args[0]).to_integral_value(rounding=ROUND_CEILING)
        raise ValueError('Unsupported arithmetic')
    with localcontext() as ctx:
        ctx.prec = 28
        value = walk(tree.body)
    if not value.is_finite() or abs(value) > Decimal('1e15'):
        raise ValueError('nonfinite_or_excessive_result')
    answer = format(value, 'f')
    if '.' in answer: answer = answer.rstrip('0').rstrip('.')
    return {'answer': answer, **item}
