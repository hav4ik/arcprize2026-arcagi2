import collections, json, sys, contextlib, io
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts/refcompare.py').read())
def recs_by_key(name):
    d = collections.defaultdict(list)
    for k, r in ref_records(name):
        d[k].append(r)
    return d
W = recs_by_key('e128t_w10_u2_gptq_eng64_sub80'); PPr = recs_by_key('e128t_pp_sub80_tx'); F7 = recs_by_key('e128t_final_all')
def m(x): return sum(x)/len(x) if x else float('nan')
for tag, bench in [('v5_w36', v5), ('v5rep', rep)]:
    for grp in ['single', 'multi']:
        sel = [(k, r) for k, r, ok in bench if (ntest[k] == 1) == (grp == 'single') and W.get(k)]
        out = m([r['usage']['completion'] for k, r in sel]); turns = m([r['turns'] for k, r in sel])
        wout = m([m([x['usage']['completion'] for x in W[k]]) for k, r in sel]); wt = m([m([x['turns'] for x in W[k]]) for k, r in sel])
        tp = m([r.get('train_pass') is True for k, r in sel]); wtp = m([m([x.get('train_pass') is True for x in W[k]]) for k, r in sel])
        line = f'{tag} {grp:6s} n={len(sel)} V5 out {out/1e3:.1f}k turns {turns:.1f} train_pass {100*tp:.0f}% | w10 out {wout/1e3:.1f}k turns {wt:.1f} tp {100*wtp:.0f}%'
        ref2 = PPr if grp == 'multi' else F7
        s2 = [(k, r) for k, r in sel if ref2.get(k)]
        line += f' | {"pp_tx" if grp=="multi" else "run7"} n={len(s2)} out {m([m([x["usage"]["completion"] for x in ref2[k]]) for k, r in s2])/1e3:.1f}k turns {m([m([x["turns"] for x in ref2[k]]) for k, r in s2]):.1f} tp {100*m([m([x.get("train_pass") is True for x in ref2[k]]) for k, r in s2]):.0f}% acc {100*m([m([bool(x.get("correct")) for x in ref2[k]]) for k, r in s2]):.1f} (V5 on these {100*m([ok for k, r, ok in bench if ref2.get(k) and (ntest[k] == 1) == (grp == "single")]):.1f})'
        print(line)
# precision: P(correct | train_pass) and recall
for tag, bench in [('v5_w36', v5), ('v5rep', rep)]:
    sel = [(k, r, ok) for k, r, ok in bench if W.get(k)]
    tpc = [ok for k, r, ok in sel if r.get('train_pass') is True]
    wtpc = [bool(x.get('correct')) for k, r, ok in sel for x in W[k] if x.get('train_pass') is True]
    print(tag, f'P(correct|train_pass) V5 {100*m(tpc):.1f}% (n={len(tpc)}) vs w10 {100*m(wtpc):.1f}% (n={len(wtpc)})')
print('--- whole eval set, per source: train_pass rate, P(correct|tp), accuracy (single / multi-task pairs)')
def stats(recs):
    tp = [x.get('train_pass') is True for x in recs]
    c = [bool(x.get('correct')) for x in recs]
    ctp = [bool(x.get('correct')) for x in recs if x.get('train_pass') is True]
    return f'n={len(recs):4d} tp {100*m(tp):.1f}% P(c|tp) {100*m(ctp):.1f}% acc {100*m(c):.1f}% no-answer {100*m([x.get("pred") is None for x in recs]):.1f}%'
for name, D in [('w10_u2', W), ('pp_sub80_tx', PPr), ('final_all', F7)]:
    for grp in ['single', 'multi']:
        recs = [x for k, v in D.items() if (ntest.get(k) == 1) == (grp == 'single') and ntest.get(k) for x in v]
        if recs: print(f'{name:12s} {grp:6s}', stats(recs))
for name, runs in [('V6-code runs', v6runs), ('new-code runs', newruns)]:
    for grp in ['single']:
        recs = []
        for n_, p in runs.items():
            for k, r, ok in bench_records(p, keys, sol):
                if ntest.get(k) == 1:
                    recs.append(dict(r, correct=ok))
        print(f'{name:12s} {grp:6s}', stats(recs))
for tag, bench in [('v5_w36', v5), ('v5rep', rep)]:
    for grp in ['single', 'multi']:
        recs = [dict(r, correct=ok) for k, r, ok in bench if (ntest[k] == 1) == (grp == 'single')]
        print(f'{tag:12s} {grp:6s}', stats(recs))
print('--- same pairs as the V5 runs: tp rate / P(c|tp) / acc per source (pair-weighted by V5 attempts)')
V6R = collections.defaultdict(list)
for n_, p in v6runs.items():
    for k, r, ok in bench_records(p, keys, sol):
        V6R[k].append(dict(r, correct=ok))
for tag, bench in [('v5_w36', v5), ('v5rep', rep), ('both', v5 + rep)]:
    for grp in ['single', 'multi']:
        sel = [(k, r, ok) for k, r, ok in bench if (ntest[k] == 1) == (grp == 'single')]
        srcs = [('w10', W), ('pp_tx', PPr) if grp == 'multi' else ('run7', F7), ('V6runs', V6R) if grp == 'single' else ('run7(multi)', F7)]
        line = f'{tag:6s} {grp:6s} V5: n={len(sel)} tp {100*m([r.get("train_pass") is True for k,r,ok in sel]):.0f}% P(c|tp) {100*m([ok for k,r,ok in sel if r.get("train_pass") is True]):.1f}% acc {100*m([ok for k,r,ok in sel]):.1f}'
        for sn, D in srcs:
            ks = [k for k, r, ok in sel if D.get(k)]
            if not ks: continue
            tpr = m([m([x.get('train_pass') is True for x in D[k]]) for k in ks])
            num = sum(sum(bool(x.get('correct')) and x.get('train_pass') is True for x in D[k]) / len(D[k]) for k in ks)
            den = sum(sum(x.get('train_pass') is True for x in D[k]) / len(D[k]) for k in ks)
            acc = m([m([bool(x.get('correct')) for x in D[k]]) for k in ks])
            line += f' | {sn} n={len(ks)} tp {100*tpr:.0f}% P(c|tp) {100*num/den:.1f}% acc {100*acc:.1f}'
        print(line)
