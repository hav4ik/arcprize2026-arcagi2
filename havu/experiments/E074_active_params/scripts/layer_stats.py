#!/usr/bin/env python3
"""Summarize prior causal routing and the live E128 active-k ablations.

Run with system python3 (numpy/matplotlib installed) for figures; the project
venv can generate data with --no-plot. --watch-hours refreshes only local files
as monitor.py downloads completed arms. No inference process is changed.
"""
import argparse
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/E074_active_params"
OLD = ROOT / "experiments/E066_dsv41_image_pruning"
OUT = EXP / "analysis/layer_stats"
CLASSES = ("prompt", "reasoning", "tool_block", "image")


def atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def effective_count(x):
    p = x[x > 0] / x.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def concentration(a, n, w, positions, keep):
    assert positions > 0 and a.sum() > 0 and n.sum() > 0 and w.sum() > 0
    assert np.isclose(n.sum(), 6 * positions)
    order = np.argsort(-a, kind="stable")
    share = np.cumsum(a[order]) / a.sum()
    expert = int(order[0])
    return {
        "dominant_expert_original_id": expert,
        "dominant_amplitude_share": float(share[0]),
        "top8_amplitude_share": float(share[7]),
        "experts_for_50pct_amplitude": int(np.searchsorted(share, .5) + 1),
        "experts_for_90pct_amplitude": int(np.searchsorted(share, .9) + 1),
        "experts_for_95pct_amplitude": int(np.searchsorted(share, .95) + 1),
        "amplitude_effective_experts": effective_count(a),
        "selection_effective_experts": effective_count(n),
        "dominant_selection_rate": float(n[expert] / positions),
        "dominant_router_mass_share": float(w[expert] / w.sum()),
        "dominant_mean_selected_router_weight": float(w[expert] / n[expert]),
        "dominant_mean_selected_weighted_output_norm": float(a[expert] / n[expert]),
        "dominant_kept_in_causal_e128": bool(keep[expert]),
        "causal_keep_original_amplitude_coverage": float(a[keep].sum() / a.sum()),
        "causal_keep_original_selection_coverage": float(n[keep].sum() / n.sum()),
    }


def historical():
    d = OLD / "results/node5_fix/data"
    keep = np.load(d / "c1f_keep.npy")
    assert keep.shape == (40, 384) and keep.dtype == bool
    assert np.all(keep.sum(1) == 128)
    paths = {"combined": d / "stats_both_causal.npz",
             "img": d / "stats_img_causal.npz", "txt": d / "stats_txt.npz"}
    loaded = {mode: dict(np.load(p)) for mode, p in paths.items()}
    for key in ("N", "A", "W", "pos"):
        assert np.allclose(loaded["combined"][key], loaded["img"][key] + loaded["txt"][key])
    rows, sources = [], {}
    for mode, z in loaded.items():
        assert np.all(z["N"].sum(2) == 6 * z["pos"][None])
        sources[mode] = {"path": str(paths[mode].relative_to(ROOT)),
                         "sha256": hashlib.sha256(paths[mode].read_bytes()).hexdigest(),
                         "positions": dict(zip(CLASSES, z["pos"].tolist()))}
        for layer in range(40):
            views = [(name, np.eye(4)[c]) for c, name in enumerate(CLASSES)]
            views.append(("pruning_recipe", np.array([1, 1, 10, 1] if layer < 20 else [0, 1, 10, 1])))
            for name, weights in views:
                positions = float(z["pos"] @ weights)
                if not positions:
                    continue
                values = [weights @ z[key][layer] for key in ("A", "N", "W")]
                rows.append({"layer": layer, "modality": mode, "token_class": name,
                             "weighted_positions": positions,
                             **concentration(*values, positions, keep[layer])})
    return rows, sources


def key(row):
    return (row["batch"].replace("e065t_", "e065_"), row["task_id"], row["pair"], row["attempt"])


def records(tag, modality):
    path = EXP / "results" / tag / f"{modality}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), f"Duplicate rollout in {path}"
    return result


def compare(tag, baseline_tag, families):
    # Resample families, keeping their paired image/text observations together.
    b = [records(baseline_tag, m) for m in ("img", "txt")]
    c = [records(tag, m) for m in ("img", "txt")]
    keys = sorted(b[0])
    assert all(set(x) == set(keys) for x in b + c), f"Unpaired data: {tag}"
    names = [families[k] for k in keys]
    assert len(set(names)) == len(names), "Expected one paired rollout per family"
    n = len(keys)
    draws = (np.array(list(itertools.product(range(n), repeat=n))) if n <= 5 else
             np.random.default_rng(74).integers(n, size=(5000, n)))
    result, family_rows = {"families": n}, []
    for metric, suffix in (("generated", "gen"), ("tool_block", "blk")):
        counts = np.array([[x[k][f"n_{suffix}"] for k in keys] for x in b], dtype=float)
        assert np.all(counts > 0)
        assert all(x[k][f"n_{suffix}"] == y[k][f"n_{suffix}"] for x, y in zip(b, c) for k in keys)
        losses = np.array([[x[k][f"nll_{suffix}"] for k in keys] for x in b])
        delta = np.array([[y[k][f"nll_{suffix}"] - x[k][f"nll_{suffix}"] for k in keys]
                          for x, y in zip(b, c)])
        base_modality = losses.sum(1) / counts.sum(1)
        delta_modality = delta.sum(1) / counts.sum(1)
        samples = (delta[:, draws].sum(2) / counts[:, draws].sum(2)).mean(0)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        family_deltas = (delta / counts).mean(0)
        result.update({f"{metric}_baseline_nll": float(base_modality.mean()),
                       f"{metric}_delta_nll": float(delta_modality.mean()),
                       f"{metric}_delta_relative_pct": float(100 * delta_modality.mean() / base_modality.mean()),
                       f"{metric}_delta_img": float(delta_modality[0]),
                       f"{metric}_delta_txt": float(delta_modality[1]),
                       f"{metric}_family_mean_delta": float(family_deltas.mean()),
                       f"{metric}_bootstrap95_lo": float(lo), f"{metric}_bootstrap95_hi": float(hi),
                       f"{metric}_worst_family_delta": float(family_deltas.max()),
                       f"{metric}_positive_families": int((family_deltas > 0).sum())})
        for i, family in enumerate(names):
            family_rows.append({"arm": tag, "baseline": baseline_tag, "family": family,
                                "metric": metric, "img_tokens": int(counts[0, i]),
                                "txt_tokens": int(counts[1, i]), "img_delta_nll": float(delta[0, i] / counts[0, i]),
                                "txt_delta_nll": float(delta[1, i] / counts[1, i]),
                                "family_mean_delta_nll": float(family_deltas[i])})
    return result, family_rows


def current():
    families = {}
    for mode in ("img", "txt"):
        for line in (EXP / "data" / mode / "index.jsonl").read_text().splitlines():
            row = json.loads(line)
            k = key(row)
            assert k not in families or families[k] == row["parent"]
            families[k] = row["parent"]
    training = set(json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text()))
    evaluation = set(json.loads((ROOT / "data/arc-agi_evaluation_challenges.json").read_text()))
    assert set(families.values()) <= training and not set(families.values()) & evaluation
    rows, family_rows, controls = [], [], {}
    for layer in range(40):
        tag = f"layer{layer:02d}_k4"
        if not (EXP / "results" / tag / "complete.json").exists():
            continue
        score, per_family = compare(tag, "baseline_screen", families)
        rows.append({"layer": layer, **score})
        family_rows.extend(per_family)
    for tag, baseline in (("uniform5_screen", "baseline_screen"),
                          ("uniform5_verify", "uniform6_verify"),
                          ("mean5_profiled_verify", "uniform6_verify")):
        if (EXP / "results" / tag / "complete.json").exists():
            controls[tag], per_family = compare(tag, baseline, families)
            family_rows.extend(per_family)
    return rows, family_rows, controls


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    x = np.arange(40)
    old = report["historical_recipe_combined"]
    new = report["active_k_sensitivity"]
    ax = axes[0]
    width = .36
    ax.bar(x - width / 2, [r["amplitude_effective_experts"] for r in old], width,
           color="#177b8c", label="Output amplitude distribution")
    ax.bar(x + width / 2, [r["selection_effective_experts"] for r in old], width,
           color="#d58a3a", label="Selection frequency distribution")
    ax.set_title("Historical causal full model: expert diversity differs across layers", loc="left", fontweight="bold")
    ax.set_ylabel("Effective experts: exp(entropy)")
    ax.legend(loc="upper right", frameon=False)
    ax = axes[1]
    ax.bar(x, [r["dominant_amplitude_share"] for r in old], color="#177b8c", label="Share of total weighted output norm")
    ax.plot(x, [r["dominant_selection_rate"] for r in old], "o-", color="#9d452d", ms=4,
            label="Fraction of positions selecting that expert")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1)
    ax.set_title("Largest-amplitude expert: late-layer totals are dominated by rare activations", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax = axes[2]
    ax.set_title(f"Current quantized causal E128: one layer at k=4, all others at k=6 ({len(new)}/40 complete)",
                 loc="left", fontweight="bold")
    ax.axhline(0, color="#555555", lw=.8)
    completed = {r["layer"] for r in new}
    for layer in set(range(40)) - completed:
        ax.axvspan(layer - .44, layer + .44, color="#eeeeee", zorder=0)
    if new:
        xx = np.array([r["layer"] for r in new])
        yy = np.array([r["generated_delta_nll"] for r in new]) * 1000
        lo = np.array([r["generated_bootstrap95_lo"] for r in new]) * 1000
        hi = np.array([r["generated_bootstrap95_hi"] for r in new]) * 1000
        ax.vlines(xx, lo, hi, color="#c2c7cb", lw=3, label="95% family bootstrap (4 families only)")
        ax.plot(xx, yy, "o", color="#194d78", label="Image/text average")
        ax.scatter(xx - .13, [r["generated_delta_img"] * 1000 for r in new], marker="_", color="#177b8c", label="Image")
        ax.scatter(xx + .13, [r["generated_delta_txt"] * 1000 for r in new], marker="_", color="#d58a3a", label="Text")
    ax.set_ylabel("Generated NLL change × 1,000\nPositive = worse")
    ax.set_xlabel("Layer index (0–39)")
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9)
    for ax in axes:
        ax.set_xlim(-.7, 39.7)
        ax.set_xticks(x[::2])
        ax.grid(axis="y", alpha=.15)
        ax.axvline(19.5, color="#aaaaaa", ls=":", lw=1)
    positions = sum(report["sources"]["combined"]["positions"].values())
    fig.suptitle("E074 · Layer statistics", fontsize=18, fontweight="bold")
    fig.supxlabel(f"Historical: {positions:,} scored positions; pruning weights enc=[1,1,10,1], dec=[0,1,10,1].\n"
                  "Aggregate diversity is not per-token top-k. Current NLL is a small training-only screen, not ARC accuracy.\n"
                  f"Updated {report['updated_utc']}; gray columns are pending.", fontsize=9)
    for ext in ("png", "pdf"):
        tmp = OUT / f"layer_stats.tmp.{ext}"
        fig.savefig(tmp, dpi=160)
        tmp.replace(OUT / f"layer_stats.{ext}")
    plt.close(fig)


def refresh(make_plot=True):
    OUT.mkdir(parents=True, exist_ok=True)
    historical_rows, sources = historical()
    active, family_rows, controls = current()
    hist = [r for r in historical_rows if r["modality"] == "combined" and r["token_class"] == "pruning_recipe"]
    old_sensitivity = json.loads((OLD / "results/sens_merged.json").read_text())
    by_layer = {r["layer"]: r for r in active}
    active_fields = list(active[0]) if active else []
    combined = [{**h, "old_stored128_to64_delta_nll_pre_image_fix": old_sensitivity["delta"][h["layer"]],
                 **{f"current_k6_to4_{k}": by_layer.get(h["layer"], {}).get(k) for k in active_fields if k != "layer"}}
                for h in hist]
    report = {"updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
              "sources": sources,
              "provenance": {
                  "historical": "E066 causal full384 calibration before E128 pruning/GPTQ, native top6, original expert IDs",
                  "weights": {"classes": CLASSES, "encoder_layers_0_19": [1, 1, 10, 1], "decoder_layers_20_39": [0, 1, 10, 1]},
                  "current": "Causal E128 2bit; 4 paired training families; one layer k6->4; native re-normalization",
                  "old_sensitivity": "Stored128->64, top6 unchanged; earlier E066 image-bug path; context only",
              },
              "limitations": [
                  "Amplitude is the sum of individual weighted expert-output L2 norms, not a share of the final summed vector.",
                  "Aggregate effective counts and top-expert mass do not measure the number of useful experts per token.",
                  "Rare large activations dominate late-layer amplitude; selection frequency provides a different view.",
                  "E128 keep coverage is conditional on original full384 routing, not measured routing after pruning.",
                  "Small-screen bootstrap does not include server numerical variation or unobserved task families; no multiple-testing correction.",
                  "Layer costs need not add. Verify the combined schedule and actual ARC task-weighted pass@1 before acceptance.",
              ],
              "historical_recipe_combined": hist, "active_k_sensitivity": active,
              "controls": controls, "completed_layers": len(active)}
    atomic_text(OUT / "layer_stats.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    csv_write(OUT / "layer_stats.csv", combined)
    csv_write(OUT / "routing_by_class.csv", historical_rows)
    csv_write(OUT / "paired_family_deltas.csv", family_rows)
    if make_plot:
        plot(report)
    print(json.dumps({"updated_utc": report["updated_utc"], "completed_layers": len(active),
                      "historical_positions": sum(sources["combined"]["positions"].values()),
                      "out": str(OUT)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--watch-hours", type=float, default=0)
    args = ap.parse_args()
    deadline = time.monotonic() + args.watch_hours * 3600
    previous = None
    consecutive_failures = 0
    while True:
        complete = sorted((EXP / "results").glob("*/complete.json"))
        signature = [(str(p), p.stat().st_mtime_ns) for p in complete]
        if signature != previous:
            try:
                refresh(make_plot=not args.no_plot)
            except (FileNotFoundError, json.JSONDecodeError, AssertionError) as error:
                # rsync replaces files individually; complete.json can arrive first.
                consecutive_failures += 1
                if args.watch_hours <= 0 or consecutive_failures >= 3:
                    raise
                print(f"Result sync may be in progress: {type(error).__name__}: {error}", flush=True)
                time.sleep(60)
                continue
            consecutive_failures = 0
            previous = signature
        if time.monotonic() >= deadline:
            return
        status_path = EXP / "results/batch_status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("state") in ("complete", "stopped"):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
