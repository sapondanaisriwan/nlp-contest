"""General answer guidance, without dataset examples or expected answers."""
import re


def verification_guidance(question, kind):
    rules={
        'yesno': 'Judge the exact claim, not whether its topic is mentioned. Compare the stated value or action with the evidence. A contradicted claim is no. Exceptions override general rules.',
        'choice': 'Compare only the listed options. For NOT/EXCEPT choose the exception. Oldest means earliest construction date; maximum/minimum means compare the relevant values.',
        'number': 'Copy the value for the exact entity, row and column asked for. Preserve decimals. Ignore example numbers in the question. Do not substitute a nearby value.',
        'short': 'Answer the requested attribute: who = person or responsible party, where = place, which company = company name. Copy its exact name, not a document title, address or code unless asked.'}
    text=rules[kind]
    if re.search(r'\b(price|cost|fee|rate|salary|current|maximum|minimum)\b',question,re.I):
        text+=' Use the applicable rate and date; a withdrawn or previous rate is not the current rate.'
    return text+' If the evidence does not determine the answer, return NOT_FOUND. Output no explanation.'


def guidance(question, kind):
    rules = ["Use the evidence to check the exact entity, condition and units asked about."]
    if kind == "number":
        rules.append("For a direct lookup, copy the value at the matching row AND column; do not calculate or round it unnecessarily. "
                     "If calculation is required, identify the applicable inputs and units, calculate the requested quantity, and check the result. "
                     "Do not substitute a monthly rate for a yearly total.")
    elif kind == "yesno":
        rules.append("Evaluate the claim made by the question, including negation. Check applicable exceptions before the general rule. "
                     "Answer no if the evidence contradicts the claim; lack of evidence alone does not establish no.")
    elif kind == "choice":
        rules.append("Compare the options against the evidence, including words such as NOT or EXCEPT. Select the option that answers the question.")
    else:
        rules.append("Match the requested level of detail. Do not replace a general category with only one example or species. "
                     "For a month, return the month; for a date or time, retain the requested date or time precision.")
    if re.search(r"\b(?:20\d\d|price|fee|cost|rate|before|after|since|until|current|previous|old|new|year|month)\b", question, re.I):
        rules.append("Use the version effective at the time asked about. Distinguish withdrawn rates from active rates, and apply discounts only when their conditions are supported.")
    rules.append("Do these checks before answering. Do not output working or explanations. If required evidence is missing, return NOT_FOUND.")
    return " ".join(rules)
