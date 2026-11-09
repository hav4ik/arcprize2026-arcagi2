"""Held-out paired NLL after E032: render N unseen E031 rollouts (incorrect but train_pass, i.e. never in the
correct-only set) with the E032 builder, then compute_logprobs for base and the given sampler checkpoints."""
import json, random, sys, time
from pathlib import Path
import numpy as np
E032 = Path("/home/vu/Workspace/ARC-AGI-2/experiments/E032_qwen38_sft_e031")
sys.path.insert(0, str(E032))
import build_dataset as B
import train as T
n, ckpts = int(sys.argv[1]), sys.argv[2:]
rows = []
for sub in B.SUBSETS:
    for l in open(B.RESULTS / f"{B.PREFIX}{sub}.jsonl"):
        r = json.loads(l); r["subset"] = sub
        if r.get("correct") is not True and r.get("train_pass") is True and r.get("answer_source") == "submit":
            rows.append(r)
random.Random(0).shuffle(rows)
B._init()
seqs = []
for r in rows:
    _, res, why = B.process((r, r["trace_path"], False, 65536, "drop"))
    if res is not None:
        seqs.append((r["subset"], res[0], res[1]))
    if len(seqs) == n:
        break
print("held-out seqs", len(seqs), "tokens", sum(len(s[1]) for s in seqs), {k: sum(s[0] == k for s in seqs) for k in B.SUBSETS})
T.tinker_key()
import tinker
from tinker import types
sc = tinker.ServiceClient()
clients = {"base": sc.create_sampling_client(base_model="Qwen/Qwen3.8-27B")}
clients.update({c.rsplit("/", 1)[-1]: sc.create_sampling_client(model_path=c) for c in ckpts})
res = [{"subset": s[0]} for s in seqs]
for name, cl in clients.items():
    t0 = time.time()
    futs = [cl.compute_logprobs(types.ModelInput.from_ints(s[1].tolist())) for s in seqs]
    for r, s, f in zip(res, seqs, futs):
        lp = f.result(); idx = np.nonzero(s[2])[0]
        r[name] = (float(-np.sum([lp[j] for j in idx])), len(idx))
    print(name, f"{time.time() - t0:.0f}s", flush=True)
for sub in (*B.SUBSETS, None):
    rs = [r for r in res if sub is None or r["subset"] == sub]
    if not rs:
        continue
    b = sum(r["base"][0] for r in rs) / sum(r["base"][1] for r in rs)
    line = f"{sub or 'all':10s} n={len(rs):3d} base {b:.4f}"
    for name in list(clients)[1:]:
        c = sum(r[name][0] for r in rs) / sum(r[name][1] for r in rs)
        d = np.array([r[name][0] / r[name][1] - r["base"][0] / r["base"][1] for r in rs])
        line += f" | {name} {c:.4f} ({c - b:+.4f}; better {int((d < 0).sum())}/{len(d)})"
    print(line)
