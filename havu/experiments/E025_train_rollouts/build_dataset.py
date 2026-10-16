"""Build the E025 task pool: the 1000 Kaggle ARC-AGI-2 training tasks plus NVARC synthetic tasks.

Everything is deterministic and resumable, so the pool can be extended later:
- `nvarc_order.txt`: all nvarc_training puzzles in one fixed seeded order (rank, stem). Batches take the next
  eligible puzzles in this order, so a later batch never repeats an earlier one.
- `manifest.jsonl`: one row per task with its source, file, sha1 and the exact pair indices used as train / test,
  plus the unused pair indices (available for extra test pairs later).
- `<batch>_challenges.json` / `<batch>_solutions.json`: Kaggle-format files for `run_deepseek_tools.py`.

NVARC tasks: n_train is drawn from the ARC-AGI-2 training distribution of train-pair counts, 1 test pair.
Puzzles are skipped when any grid matches an ARC-AGI-2 eval grid (D8 + colour invariant, as
data_tools/contamination_check.py), a grid is larger than 30x30, or the chosen pairs repeat an input.

Usage: .venv/bin/python experiments/E025_train_rollouts/build_dataset.py --batch nvarc_b0 --n 4000
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data_tools"))
from contamination_check import canon, eval_index  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
NV_DIR = ROOT / "data/nvarc/nvarc-synthetic-puzzles/nvarc_training"
SEED = 2026


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def train_count_dist():
    tr = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
    c = Counter(len(t["train"]) for t in tr.values())
    ks = sorted(c)
    return ks, [c[k] for k in ks]


def write_order():
    path = OUT / "nvarc_order.txt"
    if path.exists():
        return [line.split("\t")[1] for line in path.read_text().splitlines()]
    stems = sorted(f.stem for f in NV_DIR.glob("*/*.json"))
    random.Random(SEED).shuffle(stems)
    path.write_text("".join(f"{i}\t{s}\n" for i, s in enumerate(stems)))
    return stems


def manifest_rows():
    path = OUT / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def build_arc2(rows):
    if any(r["batch"] == "arc2_train" for r in rows):
        print("arc2_train already in manifest")
        return []
    ch = ROOT / "data/arc-agi_training_challenges.json"
    tasks = json.loads(ch.read_text())
    out = []
    for tid in sorted(tasks):
        t = tasks[tid]
        out.append(dict(task_id=tid, batch="arc2_train", source="kaggle_arc2_training",
                        source_file=str(ch.relative_to(ROOT)), source_sha1=sha1(ch),
                        n_train=len(t["train"]), test_pairs=list(range(len(t["test"])))))
    return out


def build_nvarc(rows, batch, n):
    if any(r["batch"] == batch for r in rows):
        sys.exit(f"batch {batch} already in manifest")
    used = {r["task_id"] for r in rows if r["source"] == "nvarc_training"}
    last_rank = max((r["order_rank"] for r in rows if r["source"] == "nvarc_training"), default=-1)
    order = write_order()
    idx = eval_index()
    ks, ws = train_count_dist()
    challenges, solutions, out, skipped = {}, {}, [], Counter()
    for rank in range(last_rank + 1, len(order)):
        if len(out) >= n:
            break
        stem = order[rank]
        if stem in used:
            continue
        parent = stem.split("_")[0]
        path = NV_DIR / parent / f"{stem}.json"
        pairs = json.loads(path.read_text())
        rng = random.Random(f"{SEED}:{stem}")
        n_train = rng.choices(ks, ws)[0]
        perm = list(range(len(pairs)))
        rng.shuffle(perm)
        if len(pairs) < n_train + 1:
            skipped["too_few_pairs"] += 1
            continue
        train_idx, test_idx, unused = perm[:n_train], perm[n_train:n_train + 1], perm[n_train + 1:]
        chosen = [pairs[i] for i in train_idx + test_idx]
        grids = [g for p in pairs for g in (p["input"], p["output"])]
        if any(len(g) > 30 or len(g[0]) > 30 or len(g) == 0 for g in grids):
            skipped["too_large"] += 1
            continue
        if len({json.dumps(p["input"]) for p in chosen}) < len(chosen):
            skipped["repeated_input"] += 1
            continue
        if any(canon(g) in idx for g in grids if canon(g) is not None):
            skipped["eval_grid_match"] += 1
            continue
        challenges[stem] = {"train": [pairs[i] for i in train_idx],
                            "test": [{"input": pairs[i]["input"]} for i in test_idx]}
        solutions[stem] = [pairs[i]["output"] for i in test_idx]
        out.append(dict(task_id=stem, batch=batch, source="nvarc_training", parent=parent,
                        source_file=str(path.relative_to(ROOT)), source_sha1=sha1(path), order_rank=rank,
                        n_pairs=len(pairs), n_train=n_train, train_idx=train_idx, test_idx=test_idx,
                        unused_idx=unused, test_pairs=[0]))
    print(f"{batch}: {len(out)} tasks, ranks {last_rank + 1}..{rank}, skipped {dict(skipped)}")
    (OUT / f"{batch}_challenges.json").write_text(json.dumps(challenges))
    (OUT / f"{batch}_solutions.json").write_text(json.dumps(solutions))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="arc2_train, or a new NVARC batch name such as nvarc_b0")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = manifest_rows()
    new = build_arc2(rows) if args.batch == "arc2_train" else build_nvarc(rows, args.batch, args.n)
    with open(OUT / "manifest.jsonl", "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
