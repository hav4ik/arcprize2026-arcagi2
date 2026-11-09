"""E028: reference outputs from Tinker (adapter applied server-side) for parity checks of merged / FP8 models.

For the final SFT sampler checkpoint and for the base model (no adapter), on a small subset:
1. compute_logprobs on fixed training sequences (exact, deterministic): per-position logprobs and NLL over the
   trained (assistant) positions;
2. greedy generation (top_k=1) of the first assistant turn for a few task prompts, rendered exactly like
   training (qwen3_8_xhigh_reasoning, E025 tool specs), max 1024 new tokens, stop at <|im_end|>.
Output: tinker_reference.json (tokens, weights, logprobs, greedy tokens) to compare against served models.

  python -I tinker_reference.py --parity parity_data.json --out tinker_reference.json
Cost: ~70k prefill tokens + ~8k sampled tokens per model (< $1 total at Qwen3.8-27B list prices).
"""
import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import DATA, tinker_key  # noqa: E402

SFT = "tinker://ccafea71-06e8-5045-b0f5-1b93f24e3c5e:train:0/sampler_weights/final"
BASE = "Qwen/Qwen3.8-27B"
RESULTS = Path("/home/vu/Workspace/ARC-AGI-2/evals/results")
PREFIX = "tools_persistent_grid_lean2roles_deepseek-flash_low_e025_"


def extra_sequences(n_have, rng):
    """3 more short training sequences: incorrect nvarc, incorrect arc2, correct nvarc (8k-15k tokens)."""
    z = np.load(DATA / "train.npz")
    t, w, o = z["tokens"], z["weights"], z["offsets"]
    man = [json.loads(l) for l in open(DATA / "train_manifest.jsonl")]
    out = []
    for subset, correct in [("nvarc_b0", False), ("arc2_train", False), ("nvarc_b0", True)]:
        cand = [i for i, m in enumerate(man) if m["subset"] == subset and m["correct"] == correct
                and 8000 <= m["tokens"] <= 15000]
        i = rng.choice(cand)
        out.append({"tag": f"extra_{subset}_{'correct' if correct else 'incorrect'}", "idx": i,
                    "len": int(o[i + 1] - o[i]), "trained": int(w[o[i]:o[i + 1]].sum()), "subset": subset,
                    "correct": correct, "tokens": t[o[i]:o[i + 1]].tolist(), "weights": w[o[i]:o[i + 1]].tolist()})
    return out


def greedy_prompts(rng, n=4):
    """First-turn generation prompts for n training rollouts (2 arc2, 2 nvarc), rendered as in training."""
    from build_dataset import RENDERER, TOKENIZER, to_tinker
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tools_spec import e025_tools
    rend = get_renderer(RENDERER, get_tokenizer(TOKENIZER))
    man = [json.loads(l) for l in open(DATA / "train_manifest.jsonl")]
    out = []
    for subset in ("arc2_train", "arc2_train", "nvarc_b0", "nvarc_b0"):
        m = rng.choice([m for m in man if m["subset"] == subset and m["correct"]])
        raw = json.load(gzip.open(RESULTS / f"{PREFIX}{subset}" / m["trace"]))["messages"]
        convo = rend.create_conversation_prefix_with_tools(e025_tools()) + to_tinker(raw[:1])
        ids = rend.build_generation_prompt(convo).to_ints()
        out.append({"tag": f"greedy_{subset}", "task_id": m["task_id"], "trace": m["trace"], "prompt": ids,
                    "stop": rend.get_stop_sequences()})
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", required=True, help="parity_data.json (tokens, weights, tinker_nll)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1024)
    args = ap.parse_args()
    tinker_key()
    import tinker
    from tinker import types

    rng = random.Random(0)
    seqs = json.load(open(args.parity)) + extra_sequences(5, rng)
    prompts = greedy_prompts(rng)
    sc = tinker.ServiceClient()
    clients = {"sft": sc.create_sampling_client(model_path=SFT), "base": sc.create_sampling_client(base_model=BASE)}

    t0 = time.time()
    for name, cl in clients.items():
        futs = [cl.compute_logprobs(types.ModelInput.from_ints(s["tokens"])) for s in seqs]
        for s, f in zip(seqs, futs):
            lp = f.result()
            assert len(lp) == len(s["tokens"]), (len(lp), len(s["tokens"]))
            s[f"lp_{name}"] = lp
            idx = [j for j, wj in enumerate(s["weights"]) if wj]
            s[f"nll_{name}"] = float(-np.mean([lp[j] for j in idx]))
        print(f"{name}: logprobs for {len(seqs)} sequences ({time.time() - t0:.0f}s)", flush=True)
        sp = types.SamplingParams(max_tokens=args.max_new, temperature=0.0, top_k=1, stop=prompts[0]["stop"])
        futs = [cl.sample(types.ModelInput.from_ints(p["prompt"]), num_samples=1, sampling_params=sp)
                for p in prompts]
        for p, f in zip(prompts, futs):
            seq = f.result().sequences[0]
            p[f"greedy_{name}"] = list(seq.tokens)
            p[f"greedy_lp_{name}"] = list(seq.logprobs) if seq.logprobs is not None else None
            p[f"stop_reason_{name}"] = str(getattr(seq, "stop_reason", ""))
        print(f"{name}: greedy for {len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)

    json.dump({"sft": SFT, "base": BASE, "sequences": seqs, "prompts": prompts}, open(args.out, "w"))
    print("\n%-34s %6s %7s %8s %8s %8s" % ("sequence", "len", "trained", "logged", "nll_sft", "nll_base"))
    for s in seqs:
        print("%-34s %6d %7d %8s %8.4f %8.4f" % (s["tag"], s["len"], s["trained"],
              "%.4f" % s["tinker_nll"] if "tinker_nll" in s else "-", s["nll_sft"], s["nll_base"]))
    for p in prompts:
        same = next((k for k, (a, b) in enumerate(zip(p["greedy_sft"], p["greedy_base"])) if a != b),
                    min(len(p["greedy_sft"]), len(p["greedy_base"])))
        print(f"{p['tag']} {p['task_id']}: prompt {len(p['prompt'])} tok; greedy sft {len(p['greedy_sft'])} tok "
              f"({p['stop_reason_sft']}), base {len(p['greedy_base'])} tok; sft/base identical for first {same}")


if __name__ == "__main__":
    main()
