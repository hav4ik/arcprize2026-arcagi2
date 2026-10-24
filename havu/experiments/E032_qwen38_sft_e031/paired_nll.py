"""Paired NLL on not-yet-trained E032 sequences: base Qwen3.8-27B vs sampler checkpoints (compute_logprobs).

  paired_nll.py N SEEN_STEPS CKPT [CKPT ...]   # N sequences from batches >= SEEN_STEPS, seed 0
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
E032 = Path("/home/vu/Workspace/ARC-AGI-2/experiments/E032_qwen38_sft_e031")
sys.path.insert(0, str(E032))
import train as T
n, seen_steps, ckpts = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
data = T.Batches(T.DATA / "train.npz", 300_000, 0, 65536, 0, 1)
seen = {i for s in range(seen_steps) for i in data.rows(s)}
pool = [i for i in data.idx if i not in seen]
man = [json.loads(l) for l in open(T.DATA / "train_manifest.jsonl")]
pick = random.Random(0).sample(pool, n)
T.tinker_key()
import tinker
from tinker import types
sc = tinker.ServiceClient()
clients = {"base": sc.create_sampling_client(base_model="Qwen/Qwen3.8-27B")}
clients.update({c.rsplit("/", 1)[-1]: sc.create_sampling_client(model_path=c) for c in ckpts})
res = {i: {"subset": man[i]["subset"]} for i in pick}
for name, cl in clients.items():
    t0 = time.time()
    futs = {i: cl.compute_logprobs(types.ModelInput.from_ints(data.tokens[data.offsets[i]:data.offsets[i + 1]].tolist())) for i in pick}
    for i, f in futs.items():
        lp = f.result(); w = data.weights[data.offsets[i]:data.offsets[i + 1]]
        idx = np.nonzero(w)[0]
        res[i][name] = (float(-np.sum([lp[j] for j in idx])), len(idx))
    print(name, f"{time.time() - t0:.0f}s", flush=True)
for sub in ("arc2_train", "nvarc_b0", None):
    rs = [r for r in res.values() if sub is None or r["subset"] == sub]
    b = sum(r["base"][0] for r in rs) / sum(r["base"][1] for r in rs)
    line = f"{sub or 'all':10s} n={len(rs):3d} base {b:.4f}"
    for name in list(clients)[1:]:
        c = sum(r[name][0] for r in rs) / sum(r[name][1] for r in rs)
        d = np.array([r[name][0] / r[name][1] - r["base"][0] / r["base"][1] for r in rs])
        line += f" | {name} {c:.4f} ({c - b:+.4f}; per-seq {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, better {int((d < 0).sum())}/{len(d)})"
    print(line)
