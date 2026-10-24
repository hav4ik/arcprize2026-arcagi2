#!/usr/bin/env python3
"""E043 no-tool round: expert-routing statistics for the report / tech report (small JSON), from the REAP sufficient
statistics of the no-tool replay, its answer-block statistics, and (for comparison) the tool round's statistics.

  notool_report_stats.py --stats stats_nt2.npz --answer answerblock_nt2.npz --tool-stats stats_all.npz \
      --tool-block toolblock_all.npz --keep-dirs keep_nt2 [--tool-keep-dirs keep_all protect] --out expert_stats_notool.json
Per layer: selections, effective experts exp(H), activity (experts never selected / below 0.1% of traffic, share of
traffic carried by the least-used half and by the top-16/64), coverage curves (top-E by frequency / contribution /
contribution on decoder generation targets / plain REAP / answer-weighted xW), per-layer kept routing and
answer-block share for every keep set, answer-block vs general top-E overlap, and no-tool vs tool comparisons
(effective experts, top-E overlap, cross coverage, Jaccard between keep sets of the two rounds).
"""
import argparse
import json
from pathlib import Path

import numpy as np

L, E = 40, 384
ES = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 192, 256, 384]


def load(stats):
    z = np.load(stats)
    N = z["N"].sum(axis=(1, 2)).astype(np.float64)
    A = z["A"].sum(axis=(1, 2)).astype(np.float64)
    Adec = np.where(np.arange(L)[:, None] >= 20, z["A"][:, :, 1].sum(1), A)
    return N, A, Adec


def topk(score, e):
    keep = np.zeros((L, E), bool)
    for li in range(L):
        keep[li, np.lexsort((np.arange(E), -score[li]))[:e]] = True
    return keep


def eff(N):
    p = N / N.sum(1, keepdims=True)
    return np.exp(-(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1))


def share(N, keep):
    return (N * keep).sum(1) / N.sum(1)


def activity(N):
    p = N / N.sum(1, keepdims=True)
    srt = np.sort(p, axis=1)
    return {"never_selected": (N == 0).sum(1).tolist(), "below_0.1pct": (p < 0.001).sum(1).tolist(),
            "least_used_half_share": srt[:, : E // 2].sum(1).round(4).tolist(),
            "top16_share": srt[:, -16:].sum(1).round(4).tolist(), "top64_share": srt[:, -64:].sum(1).round(4).tolist(),
            "max_share": srt[:, -1].round(4).tolist(), "never_selected_total": int((N == 0).sum())}


def load_keeps(dirs):
    keeps = {}
    for d in dirs:
        for f in sorted(Path(d).glob("keep_*.npy")):
            keeps[f"{Path(d).name}/{f.stem[5:]}"] = np.load(f).astype(bool)
    return keeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--tool-stats", required=True)
    ap.add_argument("--tool-block", required=True)
    ap.add_argument("--keep-dirs", nargs="+", required=True)
    ap.add_argument("--tool-keep-dirs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    N, A, Adec = load(a.stats)
    ab = np.load(a.answer)
    Nab, Aab = ab["N"].astype(np.float64), ab["A"].astype(np.float64)
    NT, AT, AdecT = load(a.tool_stats)
    tb = np.load(a.tool_block)
    Ntb, Atb = tb["N"].astype(np.float64), tb["A"].astype(np.float64)

    out = {"layers": L, "experts": E, "selections_per_layer": N.sum(1).tolist(),
           "answer_block_selections_per_layer": Nab.sum(1).tolist(),
           "effective_experts": eff(N).round(2).tolist(), "effective_experts_answer_block": eff(Nab).round(2).tolist(),
           "effective_experts_tool": eff(NT).round(2).tolist(), "effective_experts_tool_block": eff(Ntb).round(2).tolist(),
           "mean_amp": (A / np.maximum(N, 1)).mean(1).round(4).tolist(),
           "activity": activity(N), "activity_tool": activity(NT)}
    ee = np.array(out["effective_experts"])
    et = np.array(out["effective_experts_tool"])
    out["effective_summary"] = {"notool_mean": float(ee.mean()), "tool_mean": float(et.mean()),
                                "notool_enc_dec": [float(ee[:20].mean()), float(ee[20:].mean())],
                                "tool_enc_dec": [float(et[:20].mean()), float(et[20:].mean())],
                                "layers_notool_lower": int((ee < et).sum())}
    rules = {"frequency (oracle for routing share)": N, "contribution": A, "contribution, decoder on gen": Adec,
             "plain REAP": np.where(N > 0, A / np.maximum(N, 1), 0), "answer-block contribution": Aab,
             "answer x10": Adec + 9 * Aab, "answer x20": Adec + 19 * Aab, "answer x30": Adec + 29 * Aab}
    out["coverage"] = {name: {e: {"mean": float(share(N, topk(s, e)).mean()), "min": float(share(N, topk(s, e)).min()),
                                  "answer_mean": float(share(Nab, topk(s, e)).mean())} for e in ES}
                       for name, s in rules.items()}
    out["coverage_tool_contribution_dec"] = {e: {"mean": float(share(NT, topk(AdecT, e)).mean())} for e in ES}

    keeps = load_keeps(a.keep_dirs)
    out["per_layer_kept_share"] = {k: share(N, v).round(4).tolist() for k, v in keeps.items()}
    out["per_layer_kept_answer_share"] = {k: share(Nab, v).round(4).tolist() for k, v in keeps.items()}
    out["kept_summary"] = {k: {"routing_mean": float(share(N, v).mean()), "routing_min": float(share(N, v).min()),
                               "answer_mean": float(share(Nab, v).mean()), "answer_min": float(share(Nab, v).min()),
                               "answer_contrib_kept": float((Aab * v).sum() / Aab.sum())} for k, v in keeps.items()}
    tkeeps = load_keeps(a.tool_keep_dirs)
    out["tool_sets_on_notool_traffic"] = {k: {"routing_mean": float(share(N, v).mean()),
                                              "answer_mean": float(share(Nab, v).mean())}
                                          for k, v in tkeeps.items() if v.sum(1).max() <= 128}
    allk = {**keeps, **{f"tool:{k}": v for k, v in tkeeps.items()}}
    names = [k for k in allk if allk[k].sum(1)[0] in (48, 64, 72)]
    out["jaccard"] = {"names": names, "matrix": [[round(float(((allk[x] & allk[y]).sum(1) / (allk[x] | allk[y]).sum(1)).mean()), 4)
                                                  for y in names] for x in names]}
    for e in (48, 64, 72):
        ta, tg = topk(Aab, e), topk(Adec, e)
        out[f"answer_vs_general_top{e}_overlap"] = ((ta & tg).sum(1) / e).round(3).tolist()
        tn, tt = topk(Adec, e), topk(AdecT, e)
        out[f"notool_vs_tool_top{e}_overlap"] = ((tn & tt).sum(1) / e).round(3).tolist()
        out[f"notool_traffic_kept_by_tool_top{e}"] = share(N, tt).round(4).tolist()
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["effective_summary"]), "never selected (no-tool / tool):",
          out["activity"]["never_selected_total"], out["activity_tool"]["never_selected_total"])
    for k, v in out["kept_summary"].items():
        print(f"{k:42s} routing {v['routing_mean']:.3f} (min {v['routing_min']:.3f}) answer {v['answer_mean']:.3f}")


if __name__ == "__main__":
    main()
