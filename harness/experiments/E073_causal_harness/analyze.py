#!/usr/bin/env python3
"""Two-answer scoring, with explicit subsets; never report an oracle over three answers as pass@2."""
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'E072_harness_loop/bin'))
import arcscore as A


def records_for(records, policy):
    first = {(r['task_id'], r.get('pair', 0), r['attempt']): r for r in records}
    chosen = []
    for r in records:
        t, p, a = r['task_id'], r.get('pair', 0), r['attempt']
        use = a in ((0, 1) if policy == 'TT' else (0, 2))
        if policy == 'retry':
            parents = [first.get((t, p, k)) for k in (0, 1)]
            passed = lambda x: x.get('train_pass') is True and A.valid_grid(dict(A.split(x)).get(p))
            use = a in (0, 1) or (a == 2 and all(parents) and not any(passed(x) for x in parents))
        if use:
            chosen.append(dict(r, attempt=1 if policy == 'TI' and a == 2 else a))
    return chosen


def values(records, sol):
    result = A.task_results(records, sol)
    return {t: sum(p['p2'] for p in ps) / len(ps) for t, ps in result.items()}


def paired(arm, base):
    ts = sorted(arm.keys() & base.keys()); ds = [arm[t] - base[t] for t in ts]
    rng = random.Random(2026); n = len(ds)
    boot = sorted(sum(rng.choices(ds, k=n))/n for _ in range(10000))
    return {'n': n, 'difference_pp': 100*sum(ds)/n,
            'ci95_pp': [100*boot[250], 100*boot[9749]]}


def screen(gold, draws):
    # The holdout key is never opened during selection.
    sol = json.load(open(HERE/'workloads/e073_dev_solutions.json'))
    gold = [r for r in gold if r['task_id'] in sol]
    draws = [r for r in draws if r['task_id'] in sol]
    b = values(gold, sol); cost = sum(r.get('usage', {}).get('completion', 0) for r in gold)
    out = {}
    for policy in ['TT', 'TI', 'retry']:
        rs = records_for(draws, policy); v = values(rs, sol)
        out[policy] = {'score_pass2': 100*sum(v.values())/len(v), **paired(v, b),
                       'completion_ratio': sum(r.get('usage', {}).get('completion', 0) for r in rs)/max(1, cost)}
    eligible = [k for k, v in out.items() if v['completion_ratio'] <= 1.15]
    winner = max(eligible, key=lambda k: (out[k]['score_pass2'], -out[k]['completion_ratio'])) if eligible else None
    # This is a screening choice, not proof of a win. Failed screens preserve the holdout.
    if winner and out[winner]['difference_pp'] <= 0:
        winner = None
    return {'split': 'dev', 'tasks': 60, 'version': 'Kaggle', 'metric': 'task-weighted pass@2',
            'policies': out, 'selected': winner, 'gold_pass2': 100*sum(b.values())/len(b)}


def validate(gold, candidate, policy):
    sol = json.load(open(HERE/'workloads/e073_holdout_solutions.json'))
    # Holdout TI is run directly as slots 0=text, 1=image. Retry is already conditional in the solver.
    v, b = values(candidate, sol), values(gold, sol)
    diff = paired(v, b)
    return {'split': 'holdout', 'tasks': 60, 'version': 'Kaggle', 'metric': 'task-weighted pass@2',
            'policy': policy, 'score_pass2': 100*sum(v.values())/len(v),
            'gold_pass2': 100*sum(b.values())/len(b), **diff,
            'verdict': 'accuracy win; full deadline replica still required' if diff['ci95_pp'][0] > 0 else 'inconclusive or loss'}
