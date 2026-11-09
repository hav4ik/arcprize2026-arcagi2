import collections, hashlib, json, sys
from pathlib import Path
ROOT = Path('/home/vu/Workspace/ARC-AGI-2')
sys.path.insert(0, str(ROOT / 'experiments/E042_dsv41_l4_synthetic/scripts'))
import bench_accuracy as ba
PREFIX = ROOT / 'evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_'
SP = Path('/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad')

def kaggle_keys():
    sol = json.loads((ROOT/'data/arc-agi_evaluation_solutions.json').read_text())
    return ba.load_set(str(ROOT/'data/arc-agi_evaluation_challenges.json'), sol), sol

def x2_keys():
    sol = json.loads((ROOT/'experiments/E055_kaggle_replica/eval_x2_solutions.json').read_text())
    return ba.load_set(str(ROOT/'experiments/E055_kaggle_replica/eval_x2_challenges.json'), sol), sol

_sets = {}
def ref_records(name):
    """list of (pair_key, record) from a harness result file"""
    out = []
    for r in map(json.loads, open(f'{PREFIX}{name}.jsonl')):
        if 'task_id' not in r or r.get('error'):
            continue
        ch = r['config']['challenges']
        if ch not in _sets:
            _sets[ch] = ba.load_set(ch, r['config'].get('solutions'))
        k = _sets[ch].get((r['task_id'], r['pair']))
        if k:
            out.append((k, r))
    return out

def bench_records(path, keys, sol):
    """list of (pair_key, record, correct) from a solver attempts.jsonl (latest per attempt; multi split)"""
    latest = {}
    for r in map(json.loads, open(path)):
        if 'task_id' not in r or r.get('error'):
            continue
        t, n = r['task_id'], r.get('multi_test')
        pred = r.get('pred')
        preds = (pred if isinstance(pred, list) and len(pred) == n else [None]*n) if n else [pred]
        for q, g in (enumerate(preds) if n else [(r['pair'], pred)]):
            k = keys.get((t, q))
            if k is None:
                continue
            latest[(t, q, r['attempt'])] = (k, dict(r, pair=q, pred=g, _multi=bool(n)), g is not None and g == sol[t][q])
    return list(latest.values())
