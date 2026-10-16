"""General answer guidance, without dataset examples or expected answers."""
import re


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
