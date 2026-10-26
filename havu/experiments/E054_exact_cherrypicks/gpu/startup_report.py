#!/usr/bin/env python3
"""E054 (c2): weights / KV / activation numbers from vLLM server logs, and the #58678 memory gate.

  python3 gpu/startup_report.py RUNDIR_OR_LOG [...]                 # one JSON line per log
  python3 gpu/startup_report.py --gate OLD_LOG NEW_LOG              # #58678 memory go/no-go (NEW = WKV_TP on)

Per log (all ranks; min/max when they differ): the E054 arm lines; "Model loading took" (GiB, 1 decimal);
"Actual usage is X GiB for consumed memory (weights + non-torch), Y GiB for peak activation, Z GiB for CUDAGraph
memory" and the byte-exact "--kv-cache-memory=<bytes> ... to fit into requested memory"; "Available KV cache
memory"; "Overriding num_gpu_blocks=N" (the profiled block count, logged before --num-gpu-blocks-override clamps
it) and "GPU KV cache size"; graph capture and init times; Traceback / ERROR counts.
Gate (E053 §5, per GPU): weights -0.18..-0.25 GiB; KV budget gain / Kaggle's 2.28 GiB >= +7%; peak activation
not up by more than 0.02 GiB; same KV page geometry (bytes per block unchanged: blocks gained x 113,920 B ~= KV
bytes gained).
"""

import argparse
import json
import re
import sys
from pathlib import Path

KAGGLE_KV_GIB = 2.28
KAGGLE_TOKENS = 1_554_183
KAGGLE_BLOCKS = 21_490
BLOCK_BYTES = 113_920
GIB = 1024 ** 3

PATTERNS = {
    "load_gib": r"Model loading took ([\d.]+) GiB",
    "consumed_gib": r"Actual usage is ([\d.]+) GiB for consumed memory",
    "peak_act_gib": r"([\d.]+) GiB for peak activation",
    "graph_gib": r"([\d.]+) GiB for CUDAGraph memory",
    "kv_fit_bytes": r"--kv-cache-memory=(\d+)` \([\d.]+ GiB\) to fit into requested memory",
    "kv_full_bytes": r"--kv-cache-memory=(\d+)` \([\d.]+ GiB\) to fully utilize",
    "avail_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "profiled_blocks": r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)",
    "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "graph_capture_s": r"Graph capturing finished in (\d+) secs",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
}


def parse(path: Path) -> dict:
    if path.is_dir():
        path = path / "server.log"
    text = path.read_text(errors="replace")
    out = {"log": str(path)}
    for key, pat in PATTERNS.items():
        vals = [m.group(1).replace(",", "") for m in re.finditer(pat, text)]
        if not vals:
            continue
        nums = sorted({float(v) for v in vals})
        out[key] = nums[0] if len(nums) == 1 else [nums[0], nums[-1]]
    m = re.search(r"E054 ARM=(\w+) DSV41_INDEXER_SKIP=(\d) DSV41_ENGRAM_WKV_TP=(\d)", text)
    if m:
        out["arm"] = {"arm": m.group(1), "skip": m.group(2), "wkv_tp": m.group(3)}
    out["engram_lines"] = sorted(set(re.findall(r"E054 Engram wkv: [^\n]*", text)))
    out["indexer_lines"] = sorted(set(re.findall(r"E054 DSv4\.1 indexer: [^\n]*", text)))
    out["tracebacks"] = text.count("Traceback (most recent call last)")
    out["error_lines"] = len(re.findall(r"^\S*\s*ERROR ", text, re.M)) + len(re.findall(r"\] ERROR|ERROR \d\d-\d\d", text))
    return out


def lo(v):
    return v[0] if isinstance(v, list) else v


def hi(v):
    return v[1] if isinstance(v, list) else v


def gate(old: dict, new: dict) -> dict:
    r = {"old": old["log"], "new": new["log"]}
    if "consumed_gib" in old and "consumed_gib" in new:
        r["weights_delta_gib"] = round(hi(new["consumed_gib"]) - hi(old["consumed_gib"]), 3)
    if "kv_fit_bytes" in old and "kv_fit_bytes" in new:
        d = lo(new["kv_fit_bytes"]) - lo(old["kv_fit_bytes"])
        r["kv_budget_delta_gib"] = round(d / GIB, 3)
        r["kaggle_kv_gain_pct"] = round(100 * d / GIB / KAGGLE_KV_GIB, 1)
        r["kaggle_kv_tokens_projected"] = int(KAGGLE_TOKENS * (1 + d / GIB / KAGGLE_KV_GIB))
        r["kaggle_blocks_projected"] = int(KAGGLE_BLOCKS + d // BLOCK_BYTES)
    if "profiled_blocks" in old and "profiled_blocks" in new:
        db = lo(new["profiled_blocks"]) - lo(old["profiled_blocks"])
        r["profiled_blocks_delta"] = int(db)
        if "kv_budget_delta_gib" in r:
            # same page geometry <=> the extra bytes buy blocks of the same 113,920 B
            r["bytes_per_extra_block"] = round(r["kv_budget_delta_gib"] * GIB / db) if db else None
    if "peak_act_gib" in old and "peak_act_gib" in new:
        r["peak_activation_delta_gib"] = round(hi(new["peak_act_gib"]) - hi(old["peak_act_gib"]), 3)
    checks = {
        "weights_in_band": "weights_delta_gib" in r and -0.25 <= r["weights_delta_gib"] <= -0.18,
        "kv_gain_ge_7pct": r.get("kaggle_kv_gain_pct", 0) >= 7.0,
        "peak_activation_ok": r.get("peak_activation_delta_gib", 1) <= 0.02,
        "page_geometry_same": r.get("bytes_per_extra_block") is None
        or abs(r["bytes_per_extra_block"] - BLOCK_BYTES) <= 0.05 * BLOCK_BYTES,
        "no_tracebacks": new["tracebacks"] == 0 and old["tracebacks"] == 0,
        "new_is_column_parallel": any("ColumnParallelLinear" in s for s in new["engram_lines"]),
        "old_is_replicated": not old["engram_lines"] or any("ReplicatedLinear" in s for s in old["engram_lines"]),
    }
    r["checks"] = checks
    r["go"] = all(checks.values())
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--gate", action="store_true", help="two logs: OLD then NEW (WKV_TP on)")
    a = p.parse_args()
    parsed = [parse(Path(x)) for x in a.logs]
    for d in parsed:
        print(json.dumps(d))
    if a.gate:
        if len(parsed) != 2:
            sys.exit("--gate needs OLD_LOG NEW_LOG")
        g = gate(*parsed)
        print(json.dumps(g, indent=1))
        sys.exit(0 if g["go"] else 1)


if __name__ == "__main__":
    main()
