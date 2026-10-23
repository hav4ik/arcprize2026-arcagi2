#!/usr/bin/env python3
"""E043: turn observer step files into REAP sufficient statistics, then expert keep masks.

aggregate --obs DIR [DIR ...] --out stats.npz
    Concatenates step_*.npz in order, checks input_ids against the manifest's request streams (warmup first),
    and accumulates per (layer, input role, target role, expert): selections N, sum amp A, sum amp^2 Q,
    sum router weight W, max amp. Roles: 0 = prompt/tool/template, 1 = model-generated; the target of
    input t is token t+1 (the last input of a request predicts the final, unconsumed token).
select --stats stats.npz --experts 64 72 96 128 --out-dir DIR [--score reap|freq|contrib|weight]
       [--decoder-gen-only]
    Per-layer top-E keep masks (uniform budget per layer). reap = conditional mean amplitude A/N
    (reap_native_amp: ||native weighted expert output||, gate inside), freq = N, contrib = A,
    weight = mean router weight, wsum = total router weight mass. Unobserved experts score 0. --decoder-gen-only scores layers 20-39 on
    positions whose target is a generated token (CED decoder-generation mask); default: all consumed inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

L, E, K = 40, 384, 6


def load_manifest(d):
    warm, reqs = None, []
    for line in open(Path(d) / "manifest.jsonl"):
        r = json.loads(line)
        if "warmup" in r:
            warm = r["warmup"]
        else:
            reqs.append(r)
    return warm, reqs


def cmd_aggregate(a):
    shape = (L, 2, 2, E)
    N = np.zeros(shape, np.int64)
    A = np.zeros(shape, np.float64)
    Q = np.zeros(shape, np.float64)
    W = np.zeros(shape, np.float64)
    MX = np.zeros(shape, np.float32)
    pos_count = np.zeros((2, 2), np.int64)
    n_req = n_tok = 0
    for d in a.obs:
        d = Path(d)
        warm, reqs = load_manifest(d)
        toks = {}
        steps = sorted(d.glob("step_*.npz"))
        # Expected stream: warmup, then each request's consumed inputs.
        exp_ids, exp_in_role, exp_tgt_role = [np.asarray(warm, np.int64)], [None], [None]
        for r in reqs:
            if r["dir"] not in toks:
                toks[r["dir"]] = (np.memmap(Path(r["dir"]) / "tokens.u32", dtype=np.uint32, mode="r"),
                                  np.memmap(Path(r["dir"]) / "roles.u8", dtype=np.uint8, mode="r"))
            t, ro = toks[r["dir"]]
            o, n = r["offset"], r["length"]
            exp_ids.append(np.asarray(t[o:o + n - 1], np.int64))
            exp_in_role.append(np.asarray(ro[o:o + n - 1], np.int64))
            exp_tgt_role.append(np.asarray(ro[o + 1:o + n], np.int64))
        exp_all = np.concatenate(exp_ids)
        in_role = np.concatenate([np.full(len(warm), -1)] + exp_in_role[1:])
        tgt_role = np.concatenate([np.full(len(warm), -1)] + exp_tgt_role[1:])
        cur = 0
        for sp in steps:
            z = np.load(sp)
            ids_in = z["input_ids"].astype(np.int64)
            T = len(ids_in)
            if not np.array_equal(ids_in, exp_all[cur:cur + T]):
                raise SystemExit(f"{sp}: input_ids do not match the expected stream at offset {cur}")
            ir, tr = in_role[cur:cur + T], tgt_role[cur:cur + T]
            cur += T
            ok = ir >= 0
            if not ok.any():
                continue
            ids = z["ids"][ok].astype(np.int64)            # [t, L, K]
            amp = z["amp"][ok].astype(np.float64)
            w = z["w"][ok].astype(np.float64)
            ir, tr = ir[ok], tr[ok]
            if (ids < 0).any():
                raise SystemExit(f"{sp}: unobserved layer slots")
            for i in range(2):
                for j in range(2):
                    pos_count[i, j] += int(((ir == i) & (tr == j)).sum())
            strat = (ir * 2 + tr)[:, None, None] * E + ids   # [t, L, K] in [0, 4E)
            for li in range(L):
                s = strat[:, li].ravel()
                N[li] += np.bincount(s, minlength=4 * E).reshape(2, 2, E)
                A[li] += np.bincount(s, weights=amp[:, li].ravel(), minlength=4 * E).reshape(2, 2, E)
                Q[li] += np.bincount(s, weights=(amp[:, li] ** 2).ravel(), minlength=4 * E).reshape(2, 2, E)
                W[li] += np.bincount(s, weights=w[:, li].ravel(), minlength=4 * E).reshape(2, 2, E)
                mx = np.zeros(4 * E, np.float32)
                np.maximum.at(mx, s, amp[:, li].ravel().astype(np.float32))
                MX[li] = np.maximum(MX[li], mx.reshape(2, 2, E))
            n_tok += int(ok.sum())
        if cur != len(exp_all):
            raise SystemExit(f"{d}: recorded {cur} tokens, expected {len(exp_all)}")
        n_req += len(reqs)
    assert (N.sum(axis=3) == K * pos_count[None]).all(), "selection counts != 6 x scored positions"
    np.savez(a.out, N=N, A=A, Q=Q, W=W, MX=MX, pos_count=pos_count)
    print(json.dumps({"requests": n_req, "tokens": n_tok, "pos_count": pos_count.tolist(),
                      "observed_experts_per_layer_min": int((N.sum(axis=(1, 2)) > 0).sum(axis=1).min())}))


def scores(z, score, decoder_gen_only):
    N, A, W = z["N"].astype(np.float64), z["A"], z["W"]
    out = np.zeros((L, E))
    for li in range(L):
        if decoder_gen_only and li >= 20:
            n, s, w = N[li][:, 1].sum(0), A[li][:, 1].sum(0), W[li][:, 1].sum(0)
        else:
            n, s, w = N[li].sum((0, 1)), A[li].sum((0, 1)), W[li].sum((0, 1))
        if score == "reap":
            out[li] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        elif score == "freq":
            out[li] = n
        elif score == "contrib":
            out[li] = s
        elif score == "weight":
            out[li] = np.where(n > 0, w / np.maximum(n, 1), 0.0)
        elif score == "wsum":
            out[li] = w
        elif score == "contrib_sqrt":  # between REAP (A/N) and contribution total (A): A / sqrt(N)
            out[li] = np.where(n > 0, s / np.sqrt(np.maximum(n, 1)), 0.0)
    return out


def cmd_select(a):
    z = np.load(a.stats)
    sc = scores(z, a.score, a.decoder_gen_only)
    if a.within:  # iterative pruning: only experts of the current keep set are eligible
        sc = np.where(np.load(a.within).astype(bool), sc, -np.inf)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    N = z["N"].sum(axis=(1, 2))
    for e in a.experts:
        keep = np.zeros((L, E), bool)
        for li in range(L):
            order = np.lexsort((np.arange(E), -sc[li]))  # score desc, lowest id first on ties
            keep[li, order[:e]] = True
        tag = f"{a.score}{'_dec' if a.decoder_gen_only else ''}_e{e}"
        np.save(out / f"keep_{tag}.npy", keep)
        kept_sel = float((N * keep).sum() / N.sum())
        print(json.dumps({"mask": tag, "experts": e, "kept_selection_share": kept_sel,
                          "kept_selection_share_min_layer": float(((N * keep).sum(1) / N.sum(1)).min())}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["aggregate", "select"])
    ap.add_argument("--obs", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--stats", default="")
    ap.add_argument("--experts", type=int, nargs="*", default=[64, 72])
    ap.add_argument("--score", default="reap", choices=["reap", "freq", "contrib", "weight", "wsum", "contrib_sqrt"])
    ap.add_argument("--decoder-gen-only", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--within", default=None, help="bool [40, 384] keep mask the selection must stay inside")
    a = ap.parse_args()
    {"aggregate": cmd_aggregate, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
