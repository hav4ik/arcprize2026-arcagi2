"""Exact Qwen3.8 token counts of every E025 rollout, rendered as for Tinker SFT.

Renderer qwen3_8_xhigh_reasoning, tool specs prepended, one sequence per rollout,
TrainOnWhat.ALL_ASSISTANT_MESSAGES. Writes a per-rollout TSV and prints the distribution.

  python -I qwen_token_stats.py --tools e025_tools.json --out qwen_token_counts.tsv
"""
import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_sft_inspect import RESULTS, to_tinker  # noqa: E402

PREFIX = "tools_persistent_grid_lean2roles_deepseek-flash_low_e025_"
SUBSETS = ["arc2_train", "nvarc_b0"]
CUTS = [16384, 32768, 49152, 65536, 98304, 131072, 262144]
_state = {}


def init(tools_path):
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    _state["rend"] = get_renderer("qwen3_8_xhigh_reasoning", get_tokenizer("Qwen/Qwen3.8-27B"))
    _state["tools"] = [t["function"] for t in json.load(open(tools_path))]


def count(path):
    from tinker_cookbook.renderers import TrainOnWhat
    rend = _state["rend"]
    raw = json.load(gzip.open(path))["messages"]
    convo = rend.create_conversation_prefix_with_tools(_state["tools"]) + to_tinker(raw)
    try:
        model_input, weights = rend.build_supervised_example(convo, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    except Exception as exc:  # e.g. tool-call arguments cut off mid-JSON at the token cap
        return Path(path).name, 0, 0, 0, 0, f"{type(exc).__name__}: {str(exc)[:80]}"
    n_asst = sum(m["role"] == "assistant" for m in raw)
    n_empty = sum(m["role"] == "assistant" and not (m.get("reasoning_content") or "").strip() for m in raw)
    return Path(path).name, model_input.length, int(weights.sum()), n_asst, n_empty, ""


def describe(label, tot, trn):
    q = lambda a, p: int(np.percentile(a, p))
    print(f"  {label:<9} n={len(tot):5d}  total: mean {tot.mean():7.0f}  p50 {q(tot,50):6d}  p90 {q(tot,90):6d}  "
          f"p99 {q(tot,99):6d}  max {tot.max():6d}  sum {tot.sum()/1e6:6.1f}M | trained sum {trn.sum()/1e6:5.1f}M "
          f"({trn.sum()/tot.sum():.0%})")
    cov = "  ".join(f"≤{c//1024}k {np.mean(tot <= c):5.1%} ({tot[tot <= c].sum()/1e6:.1f}M)" for c in CUTS)
    print(f"  {'':<9} kept by max_length: {cov}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    rows = []
    for sub in SUBSETS:
        meta = {}
        for line in open(RESULTS / f"{PREFIX}{sub}.jsonl"):
            d = json.loads(line)
            meta[Path(d.get("trace_path") or "").name] = d
        files = sorted((RESULTS / f"{PREFIX}{sub}").glob("*.messages.json.gz"))
        with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.tools,)) as ex:
            for name, tot, trn, na, ne, err in ex.map(count, files, chunksize=16):
                d = meta.get(name, {})
                rows.append((sub, name, d.get("task_id"), bool(d.get("correct")), bool(d.get("train_pass")),
                             tot, trn, na, ne, err))
    with open(args.out, "w") as f:
        f.write("subset\ttrace\ttask_id\tcorrect\ttrain_pass\ttotal_tokens\ttrained_tokens\tassistant_turns\t"
                "empty_reasoning_turns\terror\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    bad = [r for r in rows if r[9]]
    print(f"render errors: {len(bad)} ({sum(r[3] for r in bad)} of them correct)")
    for r in bad[:5]:
        print("  ", r[0], r[1], r[9])
    rows = [r for r in rows if not r[9]]
    for sub in SUBSETS + ["both"]:
        rs = [r for r in rows if sub in ("both", r[0])]
        print(f"\n== {sub}")
        for label, keep in [("all", lambda r: True), ("correct", lambda r: r[3])]:
            sel = [r for r in rs if keep(r)]
            describe(label, np.array([r[5] for r in sel]), np.array([r[6] for r in sel]))
    tot = np.array([r[5] for r in rows if r[3]])
    edges = [0, 8192, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 131072, 10**9]
    hist, _ = np.histogram(tot, bins=edges)
    print("\nhistogram (correct, both subsets):")
    for lo, hi, h in zip(edges, edges[1:], hist):
        print(f"  {lo//1024:>4}k–{'∞' if hi == 10**9 else str(hi//1024)+'k':<5} {h:5d}  {'#' * int(60 * h / hist.max())}")


if __name__ == "__main__":
    main()
