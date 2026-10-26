import json,gzip,glob,os,collections
R='/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_final_all/'
first={}
for f in glob.glob(R+'*.messages.json.gz'):
    t=os.path.basename(f).split('_')[0]
    try:
        ev=json.load(gzip.open(f))['events']
        u=[e for e in ev if 'usage' in e]
        if u:
            us=u[0]['usage']; us=us if isinstance(us,dict) else eval(us)
            first.setdefault(t,[]).append(us['prompt_tokens'])
    except Exception as e: pass
fp={t:min(v) for t,v in first.items()}
print('first-prompt tokens known for',len(fp),'tasks; median',sorted(fp.values())[len(fp)//2],'max',max(fp.values()))
B='/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/pods/runs/'
for run,wins in (('mt36_b05_gate60',[(1.97,2.40)]),('mt36_b05_gate60_kv90',[(2.42,2.70)]),('c36_heavy',[(1.55,1.80),(2.90,3.16)])):
    att=[json.loads(l) for l in open(B+run+'/out/dsv41_logs/attempts.jsonl')]
    ms=[json.loads(l) for l in open(B+run+'/out/dsv41_logs/metrics.jsonl')]
    for a,b in wins:
        for m in ms:
            if m.get('elapsed_h') and a<=m['elapsed_h']<=b:
                lo=m['elapsed_h']-m['window_s']/3600
                st=[r for r in att if r.get('wall_s') and lo<=r['t_end_h']-r['wall_s']/3600<m['elapsed_h']]
                # also still-running at end (not in attempts.jsonl) unknown
                need=sum(fp.get(r['task_id'],0) for r in st)
                print(f"{run} {m['elapsed_h']:.2f}h uncached {m['uncached_tok_s']*m['window_s']/1e3:6.0f}k  hit {m['prefix_hit']:.3f} tok/s {m['gen_tok_s']:.0f} | started(finished later) {len(st)} first-prompts {need/1e3:.0f}k {[r['task_id'][:4] for r in st]}")
