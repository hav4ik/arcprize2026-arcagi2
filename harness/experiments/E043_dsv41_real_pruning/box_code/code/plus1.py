import json, sys, collections, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, '/workspace/dsv4/code/scripts')
from verify_tokenization import runs
d = Path('/workspace/dsv4/data/tok/shard0')
roles = np.fromfile(d/'roles.u8', dtype=np.uint8)
idx = [json.loads(l) for l in open(d/'index.jsonl')]
tabs = {}
for s in ['e031_arc2_train','e031_nvarc_b0','e036_nvarc_b1','e038_openrouter_deepinfra_b1r']:
    t = pq.read_table(f'/workspace/dsv4/data/rollouts_ds/rollouts/{s}.parquet', columns=['task_id','pair','attempt','messages','events']).to_pylist()
    tabs[s] = {(r['task_id'], r['pair'], r['attempt']): r for r in t}
stat = collections.Counter()
ex = []
for r in idx:
    row = tabs[r['split']][(r['task_id'], r['pair'], r['attempt'])]
    msgs = [m for m in json.loads(row['messages']) if m['role']=='assistant']
    ev = [e for e in json.loads(row['events']) if isinstance(e, dict) and e.get('usage')]
    rr = runs(roles[r['offset']: r['offset']+r['length']])
    for k, ((s, ln), (p, c)) in enumerate(zip(rr, r['api_usage'])):
        m = msgs[k]
        key = (c-(ln-1), bool(m.get('reasoning_content')), bool(m.get('content')), bool(m.get('tool_calls')), ev[k].get('finish_reason'))
        stat[key] += 1
        if c-(ln-1) == 1 and len(ex) < 3:
            ex.append((m.get('content','')[:200], (m.get('reasoning_content') or '')[-80:]))
for k, v in sorted(stat.items(), key=lambda x: -x[1])[:14]: print(v, k)
for e in ex: print(repr(e))
