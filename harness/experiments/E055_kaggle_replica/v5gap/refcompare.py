"""How do the references compare with each other on the same pairs? And V5 vs a Kaggle-flag per-pair reference."""
import collections, json, math, random, sys
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *
keys, sol = kaggle_keys(); xkeys, xsol = x2_keys()
chal = json.loads((ROOT/'data/arc-agi_evaluation_challenges.json').read_text())
ntest = {k: len(chal[t]['test']) for (t, q), k in keys.items()}
def rates_ref(name, single_only=False):
    d = collections.defaultdict(list)
    for k, r in ref_records(name):
        d[k].append(bool(r.get('correct')))
    return d
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
R = {'w10_u2 (E043 8x, chat path)': rates_ref('e128t_w10_u2_gptq_eng64_sub80'),
     'pp_sub80_tx (per pair, Kaggle flags)': rates_ref('e128t_pp_sub80_tx'),
     'final_all (E046 run 7)': rates_ref('e128t_final_all'),
     'V6-code L4/4090 runs': pool(v6runs), 'new-code E052 runs': pool(newruns)}
def mean(x): return sum(x) / len(x) if x else float('nan')
print('== references against each other (per-attempt accuracy on shared pairs, pair-weighted)')
names = list(R)
for grp in ['single', 'multi']:
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ks = [k for k in R[a] if k in R[b] and ((ntest.get(k) == 1) == (grp == 'single'))]
            if len(ks) < 10: continue
            print(f'{grp:6s} {a:38s} {100*mean([mean(R[a][k]) for k in ks]):5.1f} vs {b:38s} {100*mean([mean(R[b][k]) for k in ks]):5.1f}  pairs {len(ks)} att {sum(len(R[a][k]) for k in ks)}/{sum(len(R[b][k]) for k in ks)}')
# Kaggle-flag per-pair reference: single-test pairs -> final_all + V6-code runs (single path); multi pairs -> pp_sub80_tx
kref = {}
for k in set(R['final_all (E046 run 7)']) | set(R['V6-code L4/4090 runs']) | set(R['pp_sub80_tx (per pair, Kaggle flags)']):
    if ntest.get(k) == 1:
        v = R['final_all (E046 run 7)'].get(k, []) + R['V6-code L4/4090 runs'].get(k, [])
    else:
        v = R['pp_sub80_tx (per pair, Kaggle flags)'].get(k, [])
    if v: kref[k] = v
v5 = bench_records(ROOT/'experiments/E046_multi_test_rollout/pods/runs/v5_w36/out/dsv41_logs/attempts.jsonl', keys, sol)
rep = bench_records(SP/'v5gap/v5rep_attempts.jsonl', xkeys, xsol)
print()
print('== V5 runs vs references (paired by pair; null-model sd assumes V5 ~ reference rate)')
def comp(name, bench, ref, label):
    s = [(k, ok) for k, _, ok in bench if ref.get(k)]
    if not s: return
    acc = mean([ok for _, ok in s]); exp = mean([mean(ref[k]) for k, _ in s])
    var = sum(mean(ref[k]) * (1 - mean(ref[k])) * (1 + 1 / len(ref[k])) for k, _ in s) / len(s) ** 2
    print(f'{name:7s} vs {label:38s} n={len(s):3d}  {100*acc:5.1f} vs {100*exp:5.1f}  diff {100*(acc-exp):+5.1f}  (null sd {100*math.sqrt(var):.1f}, z {(acc-exp)/math.sqrt(var):+.2f})')
for name, bench in [('v5_w36', v5), ('v5rep', rep), ('both', v5 + rep)]:
    for label, ref in list(R.items()) + [('Kaggle-flag per-pair ref (combined)', kref)]:
        comp(name, bench, ref, label)
    for grp in ['single', 'multi']:
        sub = [(k, r, ok) for k, r, ok in bench if (ntest.get(k) == 1) == (grp == 'single')]
        comp(name + '/' + grp[0], sub, R['w10_u2 (E043 8x, chat path)'], 'w10_u2')
        comp(name + '/' + grp[0], sub, kref, 'Kaggle-flag per-pair ref')
    print()
