import json, sys, numpy as np, requests, pyarrow.parquet as pq
sys.path.insert(0, '/workspace/dsv4/code/scripts')
from tokenize_rollouts import harness_tools
d = '/workspace/dsv4/data/tok/shard0'
toks = np.fromfile(d + '/tokens.u32', dtype=np.uint32)
idx = [json.loads(l) for l in open(d + '/index.jsonl')]
tabs = {}
ok = 0
for r in idx[:: max(1, len(idx) // 12)][:12]:
    s = r['split']
    if s not in tabs:
        tabs[s] = {(x['task_id'], x['pair'], x['attempt']): x for x in pq.read_table(f'/workspace/dsv4/data/rollouts_ds/rollouts/{s}.parquet', columns=['task_id','pair','attempt','messages']).to_pylist()}
    msgs = json.loads(tabs[s][(r['task_id'], r['pair'], r['attempt'])]['messages'])
    last = max(i for i, m in enumerate(msgs) if m['role'] == 'assistant')
    body = {"model": "dsv41", "messages": msgs[: last + 1], "tools": harness_tools(),
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "add_generation_prompt": False}
    resp = requests.post('http://127.0.0.1:18000/tokenize', json=body, timeout=120).json()
    srv = resp.get('tokens')
    mine = toks[r['offset']: r['offset'] + r['length']].tolist()
    if srv is None:
        print('ERR', resp); break
    same = srv == mine
    ok += same
    if not same:
        k = next((i for i, (a, b) in enumerate(zip(srv, mine)) if a != b), min(len(srv), len(mine)))
        print('DIFF', r['task_id'], len(srv), len(mine), 'first diff at', k, srv[k-3:k+3], mine[k-3:k+3])
print('identical', ok, 'of', min(12, len(idx)))
