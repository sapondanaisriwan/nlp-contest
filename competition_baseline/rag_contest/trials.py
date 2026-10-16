"""Development-only selection. Never imported by the inference pipeline."""


def quick_subset(questions, score):
    details = {str(d['question_id']): d for d in score['details']}
    if set(details) != {q['question_id'] for q in questions} or score.get('evaluation_set') != 'dev':
        raise ValueError('Quick trial requires the matching dev score and questions')
    selected = set()
    for kind in ('choice', 'number', 'short', 'yesno'):
        rows = [q for q in questions if details[q['question_id']]['kind'] == kind]
        good = [q for q in rows if details[q['question_id']]['answer_score'] == 1 and details[q['question_id']]['document_score'] == 1]
        bad = [q for q in rows if q not in good]
        # Two failures and one successful control, then fill if a bucket is sparse.
        chosen = (bad[:1] + bad[-1:] if len(bad) > 1 else bad) + good[:1]
        chosen += [q for q in rows if q not in chosen][:3-len(chosen)]
        selected.update(q['question_id'] for q in chosen)
    return [q for q in questions if q['question_id'] in selected]
