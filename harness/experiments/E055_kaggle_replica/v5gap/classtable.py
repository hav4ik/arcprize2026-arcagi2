import collections, json, sys
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts/refcompare.py').read())
w10 = R['w10_u2 (E043 8x, chat path)']
# reference no-answer rates
def noans(name):
    d = collections.defaultdict(list)
    for k, r in ref_records(name):
        d[k].append(r.get('pred') is None)
    return d
na_w10 = noans('e128t_w10_u2_gptq_eng64_sub80'); na_pp = noans('e128t_pp_sub80_tx'); na_f7 = noans('e128t_final_all')
cls = {}
for f, tag in [('classified_v5_w36.json', 'v5_w36'), ('classified_v5rep.json', 'v5rep')]:
    for o in json.load(open(SP/'v5gap'/f)):
        cls[(tag, o['task'], o['pair'])] = o
def sub(o):
    c = o['cls']
    if c == 'wrong:submitted':
        return 'wrong, submitted, train_pass=True' if o['train_pass'] else 'wrong, submitted, train_pass=False'
    if c.startswith('no_answer'):
        return 'no answer: budget/turn cap, no answer/solve in session'
    return c
rows = collections.defaultdict(lambda: collections.Counter())
details = []
for tag, bench, kk in [('v5_w36', v5, keys), ('v5rep', rep, xkeys)]:
    for k, r, ok in bench:
        o = cls[(tag, r['task_id'], r['pair'])]
        c = 'correct' if ok else sub(o)
        for refname, ref in [('w10', w10), ('kref', kref)]:
            if not ref.get(k):
                continue
            p = mean(ref[k])
            rows[(tag, refname, c)]['n'] += 1
            rows[(tag, refname, c)]['loss'] += p if not ok else 0
            rows[(tag, refname, c)]['gain'] += (1 - p) if ok else 0
            rows[(tag, refname, c)]['refright'] += (p >= 0.5) and not ok
        if not ok:
            details.append({'run': tag, 'task': r['task_id'], 'pair': r['pair'], 'n_test': ntest.get(k), 'class': c,
                            'w10': w10.get(k), 'kref': kref.get(k), 'v6': R['V6-code L4/4090 runs'].get(k),
                            'new': R['new-code E052 runs'].get(k), 'out': o['out'], 'turns': o['turns'], 'stop': o['stop'],
                            'fallbacks': [f.get('output', '')[-160:] for f in (o['fallbacks'] or [])]})
for tag in ['v5_w36', 'v5rep']:
    for refname in ['w10', 'kref']:
        tot_n = sum(v['n'] for (t, rn, c), v in rows.items() if t == tag and rn == refname)
        print(f'--- {tag} vs {refname}: {tot_n} attempts')
        net = 0
        for (t, rn, c), v in sorted(rows.items()):
            if t == tag and rn == refname:
                net += v['gain'] - v['loss']
                print(f"   {c:58s} n={v['n']:3d}  ref-weighted loss {v['loss']:5.2f} ({100*v['loss']/tot_n:4.1f} pp)  gain {v['gain']:5.2f}  wrong-where-ref>=0.5: {v['refright']}")
        print(f'   net {net:+.2f} attempts = {100*net/tot_n:+.1f} pp')
json.dump(details, open(SP/'v5gap/failures.json', 'w'), indent=1)
# no-answer rates
for tag, bench in [('v5_w36', v5), ('v5rep', rep)]:
    s = [(k, r) for k, r, ok in bench if na_w10.get(k)]
    print(tag, 'no-answer rate', sum(r.get('pred') is None for k, r in s) / len(s), 'w10 on same pairs', mean([mean(na_w10[k]) for k, _ in s]),
          'kaggle-flag runs: pp', mean([mean(na_pp[k]) for k, _ in s if na_pp.get(k)]), 'run7', mean([mean(na_f7[k]) for k, _ in s if na_f7.get(k)]))
