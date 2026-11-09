#!/usr/bin/env python3
"""E055 toolkit: derive the Kaggle-vs-pod throughput constants (k, d) of replica_report.py from a twin pair (the same
settings once on Kaggle, once on a pod) and check the Kaggle-clock stretch against the tokens Kaggle really generated.

  calibrate_twin.py <kaggle_run_dir> <pod_run_dir> [--band 34-36]

k = pod / Kaggle effective prefill rate (prompt tokens per second of decode stall, server.log 10-s stats);
d = Kaggle / pod pure-decode per-stream rate (median of prompt-0 windows in the running band).
Check: generated tokens 1, 1.5, 2 h after each server came up, Kaggle / pod, measured vs predicted by the stretch.
Default pair: experiments/E046_multi_test_rollout/kaggle_bench (Kaggle bench V1, V6 settings) and
experiments/E046_multi_test_rollout/pods/runs/mt36_b05_gate60 (pod, same settings).
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica_report as rr  # noqa: E402

ROOT = rr.ROOT


def load(d):
    a = argparse.Namespace(solutions="", challenges="", idmap="", baseline="", no_traces=True, out="")
    run = rr.Run(d, a)
    return run, rr.analyse(run, a)


def cum_tokens(R, t_h):
    """Generated tokens up to t_h (hours), interpolating inside the 5-min window that contains t_h."""
    tot = 0.0
    for w in R["W"]:
        if w["t1"] <= t_h:
            tot += w["gen_tok_s"] * w["window_s"]
        elif w["t0"] < t_h:
            tot += w["gen_tok_s"] * (t_h - w["t0"]) * 3600
    return tot


def tokens_after_up(run, R, dh):
    return cum_tokens(R, run.events["server_up"][0][0] + dh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaggle", nargs="?", default=str(ROOT / "experiments/E046_multi_test_rollout/kaggle_bench"))
    ap.add_argument("pod", nargs="?", default=str(ROOT / "experiments/E046_multi_test_rollout/pods/runs/mt36_b05_gate60"))
    ap.add_argument("--band", default="34-36", help="running-request band for the pure-decode rate")
    a = ap.parse_args()
    lo, hi = map(int, a.band.split("-"))
    (rk, Rk), (rp, Rp) = load(a.kaggle), load(a.pod)
    dec = {}
    for name, R in (("kaggle", Rk), ("pod", Rp)):
        pure = [r["gen"] / r["running"] for r in R["srv_rows"][30:] if r["prompt"] < 1 and lo <= r["running"] <= hi and r["gen"] > 0]
        dec[name] = statistics.median(pure) if pure else None
        print(f"{name:6s} {Path(a.kaggle if name == 'kaggle' else a.pod).name}: prefill share {100 * R['prefill_share']:.1f}%, "
              f"effective prefill {R['implied_prefill_rate']:.0f} tok/s, pure decode {dec[name]:.2f} tok/s per stream "
              f"({len(pure)} windows at {a.band} running), server up at {rk.events['server_up'][0][0] if name == 'kaggle' else rp.events['server_up'][0][0]:.2f} h")
    k = Rp["implied_prefill_rate"] / Rk["implied_prefill_rate"]
    d = dec["kaggle"] / dec["pod"]
    print(f"k = {k:.2f}, d = {d:.3f} (replica_report.py central: k {rr.KCLOCK['central'][0]}, d {rr.KCLOCK['central'][1]})")
    # stretch check: Kaggle tokens dh after its server-up vs pod tokens at the pod time with the same stretched age
    f = rr.kaggle_clock(rp, Rp, k, d)
    up = rp.events["server_up"][0][0] * 3600
    print("| hours after server up | Kaggle / pod tokens, measured | predicted by the stretch |")
    print("|---|---|---|")
    for dh in (1.0, 1.5, 2.0):
        target = f(up) + dh * 3600
        lo_, hi_ = up, up + dh * 3600
        for _ in range(60):
            mid = (lo_ + hi_) / 2
            lo_, hi_ = (mid, hi_) if f(mid) < target else (lo_, mid)
        tp = lo_ / 3600
        pod_at_dh = tokens_after_up(rp, Rp, dh)
        pod_at_tp = cum_tokens(Rp, tp)
        meas = tokens_after_up(rk, Rk, dh) / pod_at_dh
        print(f"| {dh} | {meas:.3f} | {pod_at_tp / pod_at_dh:.3f} |")


if __name__ == "__main__":
    main()
