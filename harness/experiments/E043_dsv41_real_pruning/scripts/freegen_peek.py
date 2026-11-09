# E043 diagnostic: free generation from dev no-tool prompts under several masks (server B), print the reasoning start.
import json, os, sys, time
import pyarrow.parquet as pq, requests
S, CTRL = "http://127.0.0.1:18001", "/dev/shm/e043/ctrlB.json"
dev = [json.loads(l) for l in open("/workspace/dsv4/data/tok_nt2_dev/index.jsonl")]
want = [(r["task_id"], int(r["pair"])) for r in dev if r["correct"]][:3]
rows = {}
for p in sorted(os.listdir("/workspace/dsv4/data/rollouts_ds2/rollouts")):
    for r in pq.read_table("/workspace/dsv4/data/rollouts_ds2/rollouts/" + p, columns=["task_id", "pair", "messages"]).to_pylist():
        k = (r["task_id"], int(r["pair"]))
        if k in want and k not in rows:
            rows[k] = json.loads(r["messages"])[:2]
M = "/workspace/dsv4/data/engram/masks_nt2_calib/engram_top48gib.npz"
K = "/workspace/dsv4/data/reap/keep_nt2/"
for tag, eng, exp in [("eng48", M, None), ("e72_w10", M, K + "keep_contrib_dec_sp0_w10_e72.npy"),
                      ("e72_w10_fullengram", None, K + "keep_contrib_dec_sp0_w10_e72.npy"), ("e96", M, K + "keep_contrib_dec_e96.npy")]:
    json.dump({"engram_keep": eng, "expert_keep": exp, "observe": None}, open(CTRL + ".tmp", "w")); os.replace(CTRL + ".tmp", CTRL)
    requests.post(S + "/v1/completions", json={"model": "dsv41", "prompt": [0, 5, 6, 7], "max_tokens": 1})
    salt = f"peek-{tag}-{time.time()}"
    for k in want:
        for temp in (0.0, 1.0):
            body = {"model": "dsv41", "messages": rows[k], "max_tokens": 1500, "temperature": temp,
                    "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "cache_salt": salt}
            c = requests.post(S + "/v1/chat/completions", json=body, timeout=900).json()["choices"][0]
            rc = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
            print(f"[{tag} T={temp} {k[0]}] finish={c['finish_reason']} reasoning={len(rc)} chars: {rc[:260]!r} || content: {(c['message'].get('content') or '')[:100]!r}", flush=True)
json.dump({"engram_keep": None, "expert_keep": None, "observe": None}, open(CTRL + ".tmp", "w")); os.replace(CTRL + ".tmp", CTRL)
