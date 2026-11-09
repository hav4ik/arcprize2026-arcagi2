import collections, json, sys
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *
keys, sol = kaggle_keys()
xkeys, xsol = x2_keys()
chal = json.loads((ROOT/'data/arc-agi_evaluation_challenges.json').read_text())
ntest = {}
for (t, q), k in keys.items():
    ntest[k] = len(chal[t]['test'])
xchal = json.loads((ROOT/'experiments/E055_kaggle_replica/eval_x2_challenges.json').read_text())
for (t, q), k in xkeys.items():
    ntest[k] = len(xchal[t]['test'])

def rates(recs):
    d = collections.defaultdict(list)
    for k, r in recs:
        d[k].append(bool(r.get('correct')))
    return d
refs = {n: rates(ref_records(n)) for n in ['e128t_w10_u2_gptq_eng64_sub80', 'e128t_pp_sub80_tx', 'e128t_final_all']}
P = ROOT/'experiments/E046_multi_test_rollout/pods/runs'
v6runs = {'eff50': ROOT/'experiments/E050_effort_ab/runs/eff50/dsv41_logs/attempts.jsonl',
          'eff25': ROOT/'experiments/E050_effort_ab/runs/eff25/dsv41_logs/attempts.jsonl',
          'mt36g60': P/'mt36_b05_gate60/out/dsv41_logs/attempts.jsonl',
          'mt36g60kv90': P/'mt36_b05_gate60_kv90/out/dsv41_logs/attempts.jsonl',
          'c36_full': P/'c36_full/out/dsv41_logs/attempts.jsonl', 'c36_heavy': P/'c36_heavy/out/dsv41_logs/attempts.jsonl',
          'c36_heavy_kv90': P/'c36_heavy_kv90/out/dsv41_logs/attempts.jsonl'}
newruns = {'new_ar': ROOT/'experiments/E052_next_code_gate/runs/new_ar/out/dsv41_logs/attempts.jsonl',
           'new_ar_bg': ROOT/'experiments/E052_next_code_gate/runs/new_ar_bg/out/dsv41_logs/attempts.jsonl',
           'new_ar_pp': ROOT/'experiments/E052_next_code_gate/runs/new_ar_pp/out/dsv41_logs/attempts.jsonl'}
def pool(runs):
    d = collections.defaultdict(list)
    for n, p in runs.items():
        for k, r, ok in bench_records(p, keys, sol):
            d[k].append(ok)
    return d
v6 = pool(v6runs); new = pool(newruns)
v5 = bench_records(ROOT/'experiments/E046_multi_test_rollout/pods/runs/v5_w36/out/dsv41_logs/attempts.jsonl', keys, sol)
rep = bench_records(SP/'v5gap/v5rep_attempts.jsonl', xkeys, xsol)
def mean(x): return sum(x)/len(x) if x else float('nan')
def summary(name, bench):
    for grp in ['single', 'multi', 'all']:
        sel = [(k, r, ok) for k, r, ok in bench if grp == 'all' or (ntest[k] == 1) == (grp == 'single')]
        line = f'{name:8s} {grp:6s} n={len(sel):3d} acc={100*mean([ok for _,_,ok in sel]):5.1f}'
        for rn, rd in list(refs.items()) + [('v6pool', v6), ('newpool', new)]:
            s = [(k, ok) for k, _, ok in sel if rd.get(k)]
            if s:
                line += f' | {rn[:14]} n={len(s)} bench={100*mean([ok for _,ok in s]):.1f} ref={100*mean([mean(rd[k]) for k,_ in s]):.1f}'
        print(line)
summary('v5_w36', v5)
summary('v5rep', rep)
json.dump({'note': 'cache'}, open('/dev/null', 'w'))
