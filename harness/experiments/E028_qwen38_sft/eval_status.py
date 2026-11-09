"""E028 eval status: per-arm accuracy and token consumption from the harness result files, plus live server TPS.

  python eval_status.py <tag>          # e.g. e028_dev56 → results/tools_persistent_grid_lean2roles_{sft,base}_low_<tag>_*.jsonl
"""
import glob
import gzip
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

RES = Path("/home/vu/Workspace/ARC-AGI-2/evals/results")
BOX = ["ssh", "-o", "ConnectTimeout=15", "-p", "22843", "root@104.6.29.19"]


def reasoning_share(trace_path):
    """Chars of reasoning vs visible content+tool args in a finished trace (token split isn't stored per kind)."""
    try:
        msgs = json.load(gzip.open(trace_path))["messages"]
    except Exception:
        return None
    r = c = 0
    for m in msgs:
        if m["role"] == "assistant":
            r += len(m.get("reasoning_content") or "")
            c += len(m.get("content") or "") + sum(len(t["function"]["arguments"]) for t in m.get("tool_calls") or [])
    return r / max(r + c, 1)


def arm_stats(rows):
    out = [r["usage"]["completion"] for r in rows if r.get("usage")]
    corr = sum(r.get("correct") is True for r in rows)
    tp = sum(r.get("train_pass") is True for r in rows)
    stops = {}
    for r in rows:
        stops[r.get("stop_reason")] = stops.get(r.get("stop_reason"), 0) + 1
    rs = [x for x in (reasoning_share(r["trace_path"]) for r in rows if r.get("trace_path")) if x is not None]
    q = lambda p: sorted(out)[min(len(out) - 1, int(p * len(out)))] if out else 0
    return (f"done {len(rows):3d} | correct {corr:3d} ({100 * corr / max(len(rows), 1):4.1f}%) | train_pass "
            f"{tp:3d} | out tok mean {st.mean(out) if out else 0:7.0f} p50 {q(.5):6d} p90 {q(.9):6d} | reasoning "
            f"{100 * st.mean(rs) if rs else 0:4.1f}% of chars | turns {st.mean([r.get('turns', 0) for r in rows]) if rows else 0:4.1f} | "
            f"stops {stops}")


def server_tps():
    cmd = ("cd /workspace/e028; for f in sg_e_base_0.log sg_e_base_1.log sg_e_sft_0.log sg_e_sft_1.log; do "
           "grep 'Decode batch' $f | tail -1; done")
    try:
        txt = subprocess.run(BOX + [cmd], capture_output=True, text=True, timeout=40).stdout
    except Exception as e:
        return f"tps n/a ({e})"
    tot = run = 0.0
    for line in txt.splitlines():
        if "gen throughput" in line:
            tot += float(line.split("gen throughput (token/s): ")[1].split(",")[0])
            run += float(line.split("#running-req: ")[1].split(",")[0])
    return f"global gen {tot:6.0f} tok/s over {run:.0f} running streams (last decode log line per GPU server)"


def main():
    tag = sys.argv[1]
    print(f"=== {time.strftime('%H:%M:%S')} E028 eval '{tag}' | {server_tps()}")
    for arm in ("sft", "base"):
        rows = []
        for f in sorted(glob.glob(str(RES / f"tools_persistent_grid_lean2roles_{arm}_low_{tag}*.jsonl"))):
            rows += [json.loads(l) for l in open(f) if l.strip()]
        print(f"  {arm:<4} {arm_stats(rows)}")


if __name__ == "__main__":
    main()
