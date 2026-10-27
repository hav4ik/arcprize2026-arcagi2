"""Classify every attempt of a per-pair V5 run from its trace."""
import collections, gzip, json, re, sys
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *

SHADOW = re.compile(r'(?m)^\s*(def\s+(eval|compile|exec|globals|hasattr|print|json|isinstance|len|list)\s*\(|(eval|compile|exec|globals|hasattr|print)\s*=[^=])')

def trace_info(path):
    t = json.load(gzip.open(path))
    msgs, events = t['messages'], t['events']
    calls = {}  # id -> (name, code)
    submits = []  # (turn idx, code, output)
    shadow = []
    run_errors = 0
    for m in msgs:
        if m['role'] == 'assistant':
            for tc in m.get('tool_calls') or []:
                fn = tc['function']; name = fn['name']
                try:
                    a = json.loads(fn['arguments'])
                except Exception:
                    a = {'_raw': fn['arguments']}
                code = a.get('code') or a.get('expr') or a.get('expression') or a.get('answer') or json.dumps(a)
                calls[tc['id']] = (name, code)
                if name == 'run_python' and SHADOW.search(code or ''):
                    shadow.append(SHADOW.search(code).group(0).strip())
        elif m['role'] == 'tool':
            name, code = calls.get(m.get('tool_call_id'), ('?', ''))
            content = m.get('content') or ''
            if name == 'submit_answer':
                submits.append((code, content))
            elif name == 'run_python' and ('Traceback' in content or 'Error' in content[:400]):
                run_errors += 1
    fallbacks = [e for e in events if 'fallback' in e]
    other = [e for e in events if not ('usage' in e or 'tool_call_id' in e or 'fallback' in e)]
    last_asst = next((m for m in reversed(msgs) if m['role'] == 'assistant'), None)
    return {'submits': submits, 'shadow': shadow, 'fallbacks': fallbacks, 'other_events': other,
            'n_msgs': len(msgs), 'run_errors': run_errors,
            'last_finish': next((e.get('finish_reason') for e in reversed(events) if 'finish_reason' in e), None)}

def classify(rec, ok, info, sol_grid):
    if ok:
        return 'correct'
    rej = [(c, o) for c, o in info['submits'] if 'not accepted' in o]
    acc = [(c, o) for c, o in info['submits'] if 'answer accepted' in o]
    if rec.get('pred') is None:
        if rej:
            return 'no_answer:submit_rejected'
        if info['shadow']:
            return 'no_answer:shadow'
        return f"no_answer:{rec.get('stop_reason')}"
    # wrong answer
    if rej and not acc:
        return 'wrong:after_rejects'
    return 'wrong:submitted' if rec.get('answer_source') == 'submit' else f"wrong:{rec.get('answer_source')}"
