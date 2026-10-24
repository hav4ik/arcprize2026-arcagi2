"""E036 analysis: run stats (reuses E031 analyze.stats) for nvarc_b1 + nvarc_b1x, unrun tasks, per-parent NVARC
coverage for E036 alone and E031-nvarc + E036, and per-attempt speed vs worker count.
Usage: .venv/bin/python experiments/E036_nvarc_balanced_rollouts/scripts/analyze.py"""
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments/E036_nvarc_balanced_rollouts"
sys.path.insert(0, str(ROOT / "experiments/E031_flash1017_hdr_rollouts/scripts"))
from analyze import R, load, stats  # noqa: E402

RES = "tools_persistent_grid_lean2sub_deepseek-flash_low_{}.jsonl"
BATCHES = ["nvarc_b1", "nvarc_b1x"]
E031N = R / RES.format("e031_nvarc_b0")
E031A = R / RES.format("e031_arc2_train")


def parent(tid):
    return tid.split("_")[0]


def planned(b):
    return [t.strip() for t in (HERE / f"pairs_{b}.txt").read_text().replace("\n", ",").split(",") if t.strip()]


def spread(label, recs, universe):
    n = collections.Counter(parent(r["task_id"]) for r in recs)
    c = collections.Counter(parent(r["task_id"]) for r in recs if r.get("correct"))
    counts = [n.get(p, 0) for p in universe]
    print(f"\n## coverage {label}: {len(recs)} rollouts, parents covered {sum(x > 0 for x in counts)}/{len(universe)}; "
          f"per-parent rollouts min {min(counts)} median {statistics.median(counts):g} max {max(counts)}")
    print("  rollouts/parent histogram", dict(sorted(collections.Counter(counts).items())))
    cov = [p for p in universe if n.get(p)]
    acc = [c.get(p, 0) / n[p] for p in cov]
    print(f"  per-parent accuracy: mean {statistics.mean(acc):.3f} median {statistics.median(acc):.3f}; "
          f"0 correct: {sum(not c.get(p) for p in cov)} parents; all correct: {sum(c.get(p, 0) == n[p] for p in cov)}; "
          f"<50%: {sum(a < .5 for a in acc)}")
    print("  correct/parent histogram", dict(sorted(collections.Counter(c.get(p, 0) for p in cov).items())))


def speed(label, recs):
    """Per-attempt wall time and output tok/s; seconds include tool and train-check time."""
    ok = [r for r in recs if "error" not in r and r.get("seconds")]
    if ok:
        tps = [r["usage"]["completion"] / r["seconds"] for r in ok]
        print(f"  {label}: n={len(ok)} sec/attempt median {statistics.median(r['seconds'] for r in ok):.0f} | "
              f"out tok/s per attempt median {statistics.median(tps):.1f} | pooled "
              f"{sum(r['usage']['completion'] for r in ok) / sum(r['seconds'] for r in ok):.1f}")


if __name__ == "__main__":
    runs = {b: list(load(R / RES.format(f"e036_{b}")).values()) for b in BATCHES}
    e36 = [r for b in BATCHES for r in runs[b]]
    e31 = [r for r in load(E031N).values() if "error" not in r]
    for b in BATCHES:
        stats(runs[b], f"E036 {b}")
    stats(e36, "E036 total")
    plan = {b: planned(b) for b in BATCHES}
    universe = sorted({parent(t) for b in BATCHES for t in plan[b]} | {parent(r["task_id"]) for r in e31})
    for b in BATCHES:
        done = {f"{r['task_id']}:{r['pair']}" for r in runs[b]}
        unrun = sorted(t for t in plan[b] if t not in done)  # the harness runs pairs in task-id order
        print(f"\n{b}: planned {len(plan[b])}, done {len(runs[b])}, unrun {len(unrun)}")
        if unrun:
            sp = collections.Counter(parent(t) for t in unrun)
            print(f"  unrun span {unrun[0]} .. {unrun[-1]}; {len(sp)} parents, per-parent max {max(sp.values())}")
    ok36 = [r for r in e36 if "error" not in r]
    spread("E036 alone", ok36, universe)
    spread("E031-nvarc + E036", ok36 + e31, universe)
    na, nn = len(load(E031A)), len(load(E031N))
    print(f"\ncombined dataset: E031 {na + nn} ({na} arc2_train + {nn} nvarc_b0) + E036 {len(e36)} = {na + nn + len(e36)}; "
          f"correct+train_pass E036 {sum(bool(r.get('correct') and r.get('train_pass')) for r in ok36)}")
    raw = [json.loads(line) for line in (R / RES.format("e036_nvarc_b1")).open()]
    print("\n## speed (per attempt)")
    speed("E036 b1, 64 workers (first 94 records, before the 19:33 restart)", raw[:94])
    speed("E036 b1, 128 workers", raw[94:])
    speed("E036 b1x (first 21 at 32 workers alongside b1, rest 64 workers alone)", runs["nvarc_b1x"])
    speed("E031 nvarc_b0, 60 workers", e31)
    errs = collections.Counter(r["error"][:60] for r in e36 if "error" in r)
    print("  errors", dict(errs.most_common(5)))
