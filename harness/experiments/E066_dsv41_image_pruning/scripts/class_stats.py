#!/usr/bin/env python3
"""E066: expert statistics per token class, then keep sets from per-class weights (uniform or per-layer counts).

aggregate --obs DIR [DIR ...] --out class_stats.npz
    Re-reads E043 observer step files (same stream check as reap_aggregate.py) and accumulates per (layer, class,
    expert): selections N, native amplitude sum A, router weight sum W. Each scored input position gets one class:
      0 prompt  text input whose target is not model-generated (task prompt, tool output, template)
      1 gen     target is model-generated, outside tool-call blocks (reasoning)
      2 block   target is model-generated inside a tool-call block (</think> through EOS: content + DSML calls)
      3 image   image-span input position (id 129264; the router adds bias_vl there)
    Requests replayed by replay_mm.py (manifest "sent": "full") carry the whole sequence: its last position has no
    target and is not scored. E043 replays (n - 1 inputs) are read as before.

select --stats class_stats.npz --w-enc 1,1,10,1 --w-dec 0,1,10,1 --avg 128
       [--alloc uniform|norm|freq|blend|calibrated] [--sens sens.json --shape blend] [--floor 96] [--cap 160]
       [--step 8] --out-dir DIR
    Score per layer = sum over classes of weight x A (contribution; --score freq uses N). Layers 0-19 use --w-enc,
    layers 20-39 --w-dec. E043's tool E128 rule (contribution, decoder-gen-only, tool-block weight 10) is
    --w-enc 1,1,10,1 --w-dec 0,1,10,0 on text traffic; images then count in layers 0-19 only.
    --alloc uniform keeps the top avg experts in every layer. norm / freq / blend give per-layer counts that sum to
    40 x avg: greedy on the per-layer kept share of the score (norm), of the weighted selection count (freq), or of
    their mean (blend), each layer normalised to its own total, within [floor, cap], in steps of --step experts.
    The proxies disagree on layers 36-39 (one expert holds most of the norm there, E043 analysis/layer_alloc), so
    --alloc calibrated scales each layer's --shape curve to its measured dev-NLL cost (sweep below): predicted loss
    of layer l at count n = s_l x (share_l(ref) - share_l(n)), s_l = delta_l / (share_l(ref) - share_l(cut)), and the
    greedy allocation minimises the summed prediction at the budget (additive per-layer losses, as in mixed-precision
    bit allocation).

sweep-masks --stats class_stats.npz --w-enc ... --w-dec ... --ref 128 --cut 64 --out-dir DIR
    The uniform ref keep set plus 40 masks that cut one layer to --cut experts (its top-cut by the same score).
sens --nll dev_sweep.jsonl --ref-tag base --cut 64 --ref 128 --out sens.json
    Per-layer cost from nll_eval_mm.py runs tagged base / base2 (the ref set twice: noise) and l00..l39 (one layer
    cut): delta_l = NLL on generated targets (cut) - NLL (ref, mean of the base runs), paired on the same rollouts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

L, E, K, C = 40, 384, 6, 4
IMAGE_TOKEN_ID, THINK_END, EOS = 129264, 128822, 1
CLASS_NAMES = ["prompt", "gen", "block", "image"]


def load_manifest(d):
    warm, reqs = None, []
    for line in open(Path(d) / "manifest.jsonl"):
        r = json.loads(line)
        if "warmup" in r:
            warm = r["warmup"]
        else:
            reqs.append(r)
    return warm, reqs


def classes(ids, roles, full):
    """Class per input position of one request; -1 = not scored. ids/roles: the rollout's whole token stream."""
    n_in = len(ids) if full else len(ids) - 1
    tgt_role = np.zeros(n_in, np.int64)
    tgt_role[: len(ids) - 1] = roles[1:n_in + 1] if full else roles[1:]
    tgt = np.full(n_in, -1, np.int64)
    tgt[: len(ids) - 1] = ids[1:n_in + 1] if full else ids[1:]
    gen = tgt_role == 1
    # tool-call blocks: inside each generated run (one assistant turn), from the target </think> to the run's end
    start = gen & ~np.concatenate([[False], gen[:-1]])
    marks = np.cumsum(gen & (tgt == THINK_END))
    base = np.maximum.accumulate(np.where(start, np.concatenate([[0], marks[:-1]]), 0))
    block = gen & (marks - base > 0)
    cls = np.where(block, 2, np.where(gen, 1, 0))
    cls = np.where(ids[:n_in] == IMAGE_TOKEN_ID, 3, cls)
    if full:
        cls[-1] = -1
    return ids[:n_in], cls


def cmd_aggregate(a):
    N = np.zeros((L, C, E), np.int64)
    A = np.zeros((L, C, E), np.float64)
    W = np.zeros((L, C, E), np.float64)
    pos = np.zeros(C, np.int64)
    n_req = 0
    for d in a.obs:
        d = Path(d)
        warm, reqs = load_manifest(d)
        toks, exp_ids, exp_cls = {}, [np.asarray(warm, np.int64)], [np.full(len(warm), -1, np.int64)]
        for r in reqs:
            if r["dir"] not in toks:
                toks[r["dir"]] = (np.memmap(Path(r["dir"]) / "tokens.u32", dtype=np.uint32, mode="r"),
                                  np.memmap(Path(r["dir"]) / "roles.u8", dtype=np.uint8, mode="r"))
            t, ro = toks[r["dir"]]
            o, n = r["offset"], r["length"]
            i, c = classes(np.asarray(t[o:o + n], np.int64), np.asarray(ro[o:o + n], np.int64),
                           r.get("sent") == "full")
            exp_ids.append(i)
            exp_cls.append(c)
        exp_ids, exp_cls = np.concatenate(exp_ids), np.concatenate(exp_cls)
        cur = 0
        for sp in sorted(d.glob("step_*.npz")):
            z = np.load(sp)
            ids_in = z["input_ids"].astype(np.int64)
            T = len(ids_in)
            exp = exp_ids[cur:cur + T]
            if not np.array_equal(ids_in, exp):
                if len(exp) < T:
                    raise SystemExit(f"{sp}: more recorded positions than expected ({cur + T} > {len(exp_ids)})")
                bad = cur + int(np.flatnonzero(ids_in != exp)[0])
                raise SystemExit(f"{sp}: input_ids differ from the expected stream at offset {bad}")
            cl = exp_cls[cur:cur + T]
            cur += T
            ok = cl >= 0
            if not ok.any():
                continue
            ids = z["ids"][ok].astype(np.int64)
            if (ids < 0).any():
                raise SystemExit(f"{sp}: unobserved layer slots")
            amp, w, cl = z["amp"][ok].astype(np.float64), z["w"][ok].astype(np.float64), cl[ok]
            pos += np.bincount(cl, minlength=C)
            strat = cl[:, None, None] * E + ids
            for li in range(L):
                s = strat[:, li].ravel()
                N[li] += np.bincount(s, minlength=C * E).reshape(C, E)
                A[li] += np.bincount(s, weights=amp[:, li].ravel(), minlength=C * E).reshape(C, E)
                W[li] += np.bincount(s, weights=w[:, li].ravel(), minlength=C * E).reshape(C, E)
        if cur != len(exp_ids):
            raise SystemExit(f"{d}: recorded {cur} positions, expected {len(exp_ids)}")
        n_req += len(reqs)
    assert (N.sum(axis=2) == K * pos[None]).all(), "selections != 6 x scored positions"
    np.savez(a.out, N=N, A=A, W=W, pos=pos)
    print(json.dumps({"requests": n_req, "positions": dict(zip(CLASS_NAMES, pos.tolist())),
                      "observed_experts_per_layer_min": int((N.sum(1) > 0).sum(1).min())}))


def weighted(z, score, w_enc, w_dec):
    X = z["A"] if score == "contrib" else z["N"].astype(np.float64)
    we, wd = np.asarray(w_enc, np.float64), np.asarray(w_dec, np.float64)
    return np.stack([np.tensordot(we if li < 20 else wd, X[li], axes=(0, 0)) for li in range(L)])


def share_curves(S):
    """kept share of each layer's total by its top-k experts under the order of S: [L, E] (index k-1)."""
    srt = -np.sort(-S, axis=1)
    return np.cumsum(srt, axis=1) / np.maximum(srt.sum(axis=1, keepdims=True), 1e-30)


def greedy(curves, total, floor, cap, step=1):
    """Per-layer counts in [floor, cap] summing to total, adding `step` experts at a time where the curve gains most."""
    cnt = np.full(L, floor)
    assert (total - cnt.sum()) % step == 0, (total, floor, step)
    for _ in range((total - cnt.sum()) // step):
        nxt = np.minimum(cnt + step, E)
        gain = np.where(cnt + step <= cap, curves[np.arange(L), nxt - 1] - curves[np.arange(L), cnt - 1], -np.inf)
        cnt[int(np.argmax(gain))] += step
    return cnt


def calibration_scale(shape, sens):
    """s_l: the measured NLL cost of cutting layer l from ref to cut, per unit of the shape curve's lost share."""
    ref, cut = sens["ref"], sens["cut"]
    delta = np.maximum(np.asarray(sens["delta"], np.float64), sens.get("floor_delta", 1e-6))
    span = np.maximum(shape[:, ref - 1] - shape[:, cut - 1], 1e-12)
    return delta / span


def cmd_select(a):
    z = np.load(a.stats)
    w_enc = [float(x) for x in a.w_enc.split(",")]
    w_dec = [float(x) for x in a.w_dec.split(",")]
    S = weighted(z, a.score, w_enc, w_dec)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((np.tile(np.arange(E), (L, 1)), -S), axis=1)  # score desc, lowest id first on ties
    # selection-count curve under the same order (weighted N, the "freq" proxy)
    Nw = weighted(z, "freq", w_enc, w_dec)
    Nsorted = np.take_along_axis(Nw, order, axis=1)
    curve_freq = np.cumsum(Nsorted, axis=1) / np.maximum(Nsorted.sum(axis=1, keepdims=True), 1e-30)
    curve_norm = share_curves(S)
    curves = {"norm": curve_norm, "freq": curve_freq, "blend": (curve_norm + curve_freq) / 2}
    pred = None
    if a.alloc == "calibrated":
        sens = json.load(open(a.sens))
        scale = calibration_scale(curves[a.shape], sens)
        curves["calibrated"] = scale[:, None] * curves[a.shape]  # higher = lower predicted loss
        ref = sens["ref"]
        pred = lambda cnt: float(sum(scale[li] * (curves[a.shape][li, ref - 1] - curves[a.shape][li, cnt[li] - 1])
                                     for li in range(L)))  # noqa: E731
    N_all = z["N"].sum(axis=1)
    tag_w = f"enc{a.w_enc.replace(',', '-')}_dec{a.w_dec.replace(',', '-')}"
    for avg in a.avg:
        total = int(round(40 * avg))
        if a.alloc == "uniform":
            assert total % L == 0
            cnt = np.full(L, total // L)
        else:
            assert L * a.floor <= total <= L * a.cap, (avg, a.floor, a.cap)
            cnt = greedy(curves[a.alloc], total, a.floor, a.cap, a.step)
        keep = np.zeros((L, E), bool)
        for li in range(L):
            keep[li, order[li, :cnt[li]]] = True
        alloc = a.alloc if a.alloc != "calibrated" else f"cal{a.shape}"
        tag = f"{a.score}_{tag_w}_{alloc}{'' if a.alloc == 'uniform' else f'_f{a.floor}c{a.cap}s{a.step}'}_avg{avg:g}"
        np.save(out / f"keep_{tag}.npy", keep)
        res = {"mask": tag, "experts_total": int(keep.sum()), "per_layer": cnt.tolist(),
               "kept_selection_share": round(float((N_all * keep).sum() / N_all.sum()), 4)}
        for name, cv in curves.items():
            if name != "calibrated":
                res[f"lost_share_{name}"] = round(float(sum(1 - cv[li, cnt[li] - 1] for li in range(L))), 4)
        if pred is not None:
            res["predicted_nll_increase_vs_ref"] = round(pred(cnt), 6)
            res["predicted_nll_increase_uniform"] = round(pred(np.full(L, total // L)), 6) if total % L == 0 else None
        for c, name in enumerate(CLASS_NAMES):
            Nc = z["N"][:, c]
            if Nc.sum():
                res[f"kept_sel_{name}"] = round(float((Nc * keep).sum() / Nc.sum()), 4)
        print(json.dumps(res))


def cmd_sweep_masks(a):
    z = np.load(a.stats)
    S = weighted(z, a.score, [float(x) for x in a.w_enc.split(",")], [float(x) for x in a.w_dec.split(",")])
    order = np.lexsort((np.tile(np.arange(E), (L, 1)), -S), axis=1)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = np.zeros((L, E), bool)
    for li in range(L):
        base[li, order[li, :a.ref]] = True
    np.save(out / "keep_base.npy", base)
    for li in range(L):
        k = base.copy()
        k[li] = False
        k[li, order[li, :a.cut]] = True
        np.save(out / f"keep_l{li:02d}.npy", k)
    print(json.dumps({"out_dir": str(out), "ref": a.ref, "cut": a.cut, "masks": L + 1}))


def cmd_sens(a):
    runs = {}
    for line in open(a.nll):
        r = json.loads(line)
        runs.setdefault(r["tag"], {})[(r["task_id"], r["pair"], r["attempt"], r.get("batch"))] = r
    keys = set(runs[a.ref_tag])
    for t in runs:
        keys &= set(runs[t])
    assert keys, "no rollouts common to every run"

    def nll(tag, field="gen"):
        rs = [runs[tag][k] for k in keys]
        return sum(r[f"nll_{field}"] for r in rs) / sum(r[f"n_{field}"] for r in rs)

    bases = [t for t in (a.ref_tag, a.ref_tag + "2") if t in runs]
    base = float(np.mean([nll(t) for t in bases]))
    noise = abs(nll(bases[0]) - nll(bases[1])) if len(bases) == 2 else None
    delta = [nll(f"l{li:02d}") - base for li in range(L)]
    base_blk = float(np.mean([nll(t, "blk") for t in bases]))
    delta_blk = [nll(f"l{li:02d}", "blk") - base_blk for li in range(L)]
    res = {"ref": a.ref, "cut": a.cut, "rollouts": len(keys), "base_nll_gen": base, "noise_nll_gen": noise,
           "delta": delta, "delta_blk": delta_blk, "floor_delta": max(1e-6, (noise or 0) / 4)}
    json.dump(res, open(a.out, "w"), indent=1)
    order = np.argsort(delta)[::-1]
    print(json.dumps({"rollouts": len(keys), "base_nll_gen": round(base, 5), "noise": noise,
                      "most_sensitive": [(int(i), round(delta[i], 5)) for i in order[:6]],
                      "least_sensitive": [(int(i), round(delta[i], 5)) for i in order[-6:]]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["aggregate", "select", "sweep-masks", "sens"])
    ap.add_argument("--obs", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--stats", default="")
    ap.add_argument("--score", default="contrib", choices=["contrib", "freq"])
    ap.add_argument("--w-enc", default="1,1,10,1", help="class weights prompt,gen,block,image for layers 0-19")
    ap.add_argument("--w-dec", default="0,1,10,1", help="class weights prompt,gen,block,image for layers 20-39")
    ap.add_argument("--avg", type=float, nargs="+", default=[128])
    ap.add_argument("--alloc", default="uniform", choices=["uniform", "norm", "freq", "blend", "calibrated"])
    ap.add_argument("--shape", default="blend", choices=["norm", "freq", "blend"], help="curve shape for calibrated")
    ap.add_argument("--sens", default="", help="sens.json from the layer sweep (--alloc calibrated)")
    ap.add_argument("--floor", type=int, default=96)
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--step", type=int, default=1, help="experts added per greedy step (8 for the candidates)")
    ap.add_argument("--ref", type=int, default=128)
    ap.add_argument("--cut", type=int, default=64)
    ap.add_argument("--nll", default="", help="nll_eval_mm.py output with the sweep runs (sens)")
    ap.add_argument("--ref-tag", default="base")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    {"aggregate": cmd_aggregate, "select": cmd_select, "sweep-masks": cmd_sweep_masks, "sens": cmd_sens}[a.cmd](a)


if __name__ == "__main__":
    main()
