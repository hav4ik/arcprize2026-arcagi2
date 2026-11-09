"""Measured loss curves, exact-budget allocation, and joint-schedule ranking."""
import json
import math
import statistics


def key(r):
    return (r['batch'].replace('e065t_','e065_'),r['task_id'],r['pair'],r['attempt'])


def nll(exp, tag, pool):
    answer = {}
    manifest = json.loads((exp/'data/manifest.json').read_text())
    expected = {tuple(k) for k in manifest['paired_keys'][pool]}
    for mode in ('img','txt'):
        rows = [json.loads(s) for s in (exp/'results'/tag/f'{pool}_{mode}.jsonl').read_text().splitlines()]
        assert len(rows) == len(expected) and {key(r) for r in rows} == expected
        assert all(r['n_gen'] > 0 and r['n_blk'] > 0 for r in rows)
        answer[mode] = {'gen':sum(r['nll_gen'] for r in rows)/sum(r['n_gen'] for r in rows),
                       'tool':sum(r['nll_blk'] for r in rows)/sum(r['n_blk'] for r in rows),
                       'rows':rows}
        assert all(math.isfinite(answer[mode][k]) for k in ('gen','tool'))
    return answer


def compare(current, baseline):
    deltas = {m:{k:current[m][k]/baseline[m][k]-1 for k in ('gen','tool')}
              for m in ('img','txt')}
    gen = statistics.mean(deltas[m]['gen'] for m in deltas)
    tool = statistics.mean(deltas[m]['tool'] for m in deltas)
    costs = {}
    for mode in deltas:
        b = {key(r):r for r in baseline[mode]['rows']}
        costs[mode] = {key(r):((r['nll_gen']/r['n_gen']-b[key(r)]['nll_gen']/b[key(r)]['n_gen'])/baseline[mode]['gen'] +
                              .25*(r['nll_blk']/r['n_blk']-b[key(r)]['nll_blk']/b[key(r)]['n_blk'])/baseline[mode]['tool'])
                       for r in current[mode]['rows']}
    paired = [(costs['img'][k]+costs['txt'][k])/2 for k in costs['img']]
    stderr = statistics.stdev(paired)/math.sqrt(len(paired)) if len(paired)>1 else 0.
    return {'modalities':deltas,'gen':gen,'tool':tool,'stderr':stderr,
            'balanced':gen+.25*tool,
            'robust':gen+.25*tool+.25*abs(deltas['img']['gen']-deltas['txt']['gen'])+stderr,
            'joint_score':gen+.25*tool+.25*max(deltas[m]['gen'] for m in deltas)}


def allocate_exact(curves, slots, metric):
    states = {0:(0.,[])}
    for choices in curves:
        nxt = {}
        for used,(loss,ks) in states.items():
            for k,row in sorted(choices.items()):
                n = used+k
                score = loss+row[metric]
                if n<=slots and (n not in nxt or score<nxt[n][0]):
                    nxt[n] = (score,ks+[k])
        states=nxt
    if slots not in states:
        raise ValueError(f'Cannot meet {slots} slots')
    return states[slots]


def swap_neighbors(ks, curves, limit=3):
    """Transfer one slot, preserving the exact budget; return proxy-ranked alternatives."""
    base = sum(curves[i][k]['balanced'] for i,k in enumerate(ks))
    options=[]
    for i,ki in enumerate(ks):
        if ki<=3:
            continue
        for j,kj in enumerate(ks):
            if i==j or kj>=6:
                continue
            candidate=ks.copy();candidate[i]-=1;candidate[j]+=1
            score=base-curves[i][ki]['balanced']-curves[j][kj]['balanced']+curves[i][ki-1]['balanced']+curves[j][kj+1]['balanced']
            options.append((score,i,j,candidate))
    options.sort()
    return [x[-1] for x in options[:limit]]
