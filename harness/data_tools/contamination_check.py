"""Check training data for leakage of ARC-AGI-2 *evaluation* grids.

Every grid is reduced to a canonical form that is invariant to the 8 rotations/reflections and to colour
relabelling (colours renumbered by first appearance in row-major order; the smallest of the 8 variants
wins). A training grid "matches" an eval task if its canonical form equals that of any grid in the task
(train inputs/outputs, test inputs and test outputs). Trivial grids (< MIN_CELLS cells or a single colour) are ignored,
because they collide by chance.

Sources checked (all intended as clean, i.e. not derived from eval tasks):
  - NVARC synthetic `nvarc_training` (JSON pairs per puzzle)
  - NVARC augmented subsets arc2_training, rearc, concept, mini, nvarc_training (HF datasets)
  - official ARC-AGI-2 training challenges

    .venv/bin/python data_tools/contamination_check.py [--workers 20] [--skip-augmented]
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = f"{ROOT}/data/arc-agi-2-github/data/evaluation"
MIN_CELLS = 16


def canon(grid):
    """Canonical key invariant to D8 symmetries and colour permutations; None for trivial grids."""
    a = np.asarray(grid, dtype=np.int8)
    if a.ndim != 2 or a.size < MIN_CELLS or len(np.unique(a)) < 2:
        return None
    best = None
    for k in range(4):
        for b in (np.rot90(a, k), np.rot90(a, k).T):
            _, first_idx, inv = np.unique(b.ravel(), return_index=True, return_inverse=True)
            order = np.argsort(np.argsort(first_idx))  # colour -> rank of first appearance
            key = (b.shape, order[inv].astype(np.int8).tobytes())
            if best is None or key < best:
                best = key
    return best


def eval_index():
    idx = defaultdict(set)
    for f in glob.glob(f"{EVAL_DIR}/*.json"):
        t = json.load(open(f))
        tid = os.path.basename(f)[:-5]
        grids = [g for p in t["train"] for g in (p["input"], p["output"])]
        grids += [g for p in t["test"] for g in (p["input"], p["output"])]  # test outputs = the answers
        for g in grids:
            c = canon(g)
            if c is not None:
                idx[c].add(tid)
    return idx


def parse_text_grid(s):
    return [[int(ch) for ch in row] for row in s.strip().split("\n") if row]


def scan_synthetic_file(path):
    pairs = json.load(open(path))
    return [canon(g) for p in pairs for g in (p["input"], p["output"])]


def scan_augmented_rows(args):
    subset, start, stop = args
    from datasets import load_from_disk
    ds = load_from_disk(f"{ROOT}/data/nvarc/nvarc-augmented-puzzles/{subset}").select(range(start, stop))
    keys = []
    for msgs in ds["messages"]:
        for m in msgs:
            try:
                keys.append(canon(parse_text_grid(m["content"])))
            except ValueError:
                pass
    return keys


def report(name, keys, idx):
    keys = [k for k in keys if k is not None]
    hits = Counter()
    for k in keys:
        for tid in idx.get(k, ()):
            hits[tid] += 1
    print(f"{name:32s} grids checked {len(keys):>10,}  matching eval grids {sum(hits.values()):>7,}  "
          f"eval tasks hit {len(hits):>3}/120  top: {hits.most_common(5)}", flush=True)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--skip-augmented", action="store_true")
    args = ap.parse_args()
    idx = eval_index()
    print(f"eval index: {len(idx):,} distinct non-trivial canonical grids from 120 tasks", flush=True)
    results = {}

    tr = json.load(open(f"{ROOT}/data/arc-agi_training_challenges.json"))
    keys = [canon(g) for t in tr.values() for p in t["train"] for g in (p["input"], p["output"])]
    keys += [canon(p["input"]) for t in tr.values() for p in t["test"]]
    results["arc2_training_challenges"] = report("ARC-AGI-2 training (official)", keys, idx)

    files = glob.glob(f"{ROOT}/data/nvarc/nvarc-synthetic-puzzles/nvarc_training/*/*.json")
    with Pool(args.workers) as pool:
        keys = [k for ks in pool.imap_unordered(scan_synthetic_file, files, chunksize=200) for k in ks]
    results["nvarc_synthetic_training"] = report("NVARC synthetic nvarc_training", keys, idx)

    if not args.skip_augmented:
        from datasets import load_from_disk
        for subset in ["arc2_training", "rearc", "concept", "mini", "nvarc_training"]:
            n = len(load_from_disk(f"{ROOT}/data/nvarc/nvarc-augmented-puzzles/{subset}"))
            step = max(1, n // (args.workers * 4))
            chunks = [(subset, s, min(n, s + step)) for s in range(0, n, step)]
            with Pool(args.workers) as pool:
                keys = [k for ks in pool.imap_unordered(scan_augmented_rows, chunks) for k in ks]
            results[f"augmented_{subset}"] = report(f"NVARC augmented {subset}", keys, idx)

    out = f"{ROOT}/data_tools/contamination_report.json"
    json.dump({k: dict(v) for k, v in results.items()}, open(out, "w"), indent=1)
    print(f"\nper-source hits by eval task written to {out}")


if __name__ == "__main__":
    main()
