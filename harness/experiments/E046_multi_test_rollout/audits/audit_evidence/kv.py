import json, sys, os, glob, statistics, re
base='/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/'
runs=sorted(glob.glob(base+'pods/runs/*/out/dsv41_logs'))+[base+'kaggle_bench/dsv41_logs']
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p*len(v)))] if v else float('nan')
print(f"{'run':24s} {'dur':>5s} {'n':>3s} {'maxKV':>6s} {'@h':>5s} {'KVend':>6s} {'slope_last1h':>7s} {'minHit':>6s} {'@h':>5s} {'maxUnc':>6s} {'tok p10/med/p90':>16s} {'pre':>4s} {'done':>4s} {'run_min':>7s} {'thrashwin':>9s}")
for d in runs:
    name=d.split('/')[-3] if 'pods' in d else 'KAGGLE_BENCH'
    ms=[json.loads(l) for l in open(d+'/metrics.jsonl')]
    ms=[m for m in ms if (m.get('window_s') or 0)>=200 and m.get('elapsed_h') is not None and m.get('kv') is not None and m.get('prefix_hit') is not None]
    if not ms: print(name,'no metrics'); continue
    ms2=[m for m in ms if m['elapsed_h']>0.3]
    mk=max(ms,key=lambda m:m['kv'])
    mh=min(ms2,key=lambda m:m['prefix_hit']) if ms2 else mk
    end=ms[-1]
    last=[m for m in ms if m['elapsed_h']>=end['elapsed_h']-1.0]
    slope=(last[-1]['kv']-last[0]['kv'])/max(1e-6,last[-1]['elapsed_h']-last[0]['elapsed_h']) if len(last)>1 else float('nan')
    tok=[m['gen_tok_s'] for m in ms2]
    pre=sum(m.get('preemptions') or 0 for m in ms)
    thr=[m for m in ms2 if m['prefix_hit']<0.85 and m['uncached_tok_s']>300]
    runmin=min((m.get('srv_running') or 0) for m in ms2) if ms2 else 0
    print(f"{name:24s} {end['elapsed_h']:5.2f} {len(ms):3d} {mk['kv']:6.3f} {mk['elapsed_h']:5.2f} {end['kv']:6.3f} {slope:+7.3f} {mh['prefix_hit']:6.3f} {mh['elapsed_h']:5.2f} {max(m['uncached_tok_s'] for m in ms2):6.0f} {q(tok,.1):5.0f}/{q(tok,.5):4.0f}/{q(tok,.9):4.0f} {pre:4.0f} {end.get('done',0):4d} {runmin:7.0f} {len(thr):3d} {','.join(f'{m[chr(101)+chr(108)+chr(97)+chr(112)+chr(115)+chr(101)+chr(100)+chr(95)+chr(104)]:.2f}' for m in thr[:6])}")
