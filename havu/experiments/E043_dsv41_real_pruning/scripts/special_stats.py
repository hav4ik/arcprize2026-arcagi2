#!/usr/bin/env python3
"""E043: expert statistics restricted to positions whose target token is a special/format token.

Pruned models (E72 and below) lose the rare structural tokens that make tool calls parse (`</think>`, DSML
markers, EOS). This re-reads the observer step files (same alignment as reap_aggregate.py) and accumulates, per layer
and expert, selections N and native amplitude sums A over positions whose *target* (next) token is one of the
tokenizer's added/special tokens and is model-generated. Then `select` builds keep sets that protect the top-k
experts per layer by this structural contribution and fill the rest by an existing score.

  special_stats.py aggregate --obs DIR [...] --tokenizer tokenizer.json --out special.npz
  special_stats.py select --stats stats_all.npz --special special.npz --protect 8 --experts 48 64 72 --out-dir DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

L, E = 40, 384


def special_ids(tokenizer_json):
    t = json.load(open(tokenizer_json))
    return np.array(sorted(a["id"] for a in t["added_tokens"]), dtype=np.int64)


def cmd_aggregate(a):
    from reap_aggregate import load_manifest
    sp = special_ids(a.tokenizer)
    N = np.zeros((L, E), np.int64)
    A = np.zeros((L, E), np.float64)
    tgt_counts = {}
    for d in a.obs:
        d = Path(d)
        warm, reqs = load_manifest(d)
        toks = {}
        tgt, role_t = [np.full(len(warm), -1, np.int64)], [np.zeros(len(warm), np.int64)]
        for r in reqs:
            if r["dir"] not in toks:
                toks[r["dir"]] = (np.memmap(Path(r["dir"]) / "tokens.u32", dtype=np.uint32, mode="r"),
                                  np.memmap(Path(r["dir"]) / "roles.u8", dtype=np.uint8, mode="r"))
            t, ro = toks[r["dir"]]
            o, n = r["offset"], r["length"]
            tt = np.asarray(t[o + 1:o + n], np.int64)
            rr = np.asarray(ro[o + 1:o + n], np.int64)
            if a.region:  # targets inside generated tool-call blocks: after </think> (128822) up to EOS (1)
                inblk = np.zeros(len(tt), bool)
                on = False
                for i in range(len(tt)):
                    if rr[i] != 1:
                        on = False
                        continue
                    if on:
                        inblk[i] = True
                        if tt[i] == 1:
                            on = False
                    elif tt[i] == 128822:
                        inblk[i] = True
                        on = True
                rr = np.where(inblk, 1, 0)
            tgt.append(tt)
            role_t.append(rr)
        tgt, role_t = np.concatenate(tgt), np.concatenate(role_t)
        cur = 0
        for sp_file in sorted(d.glob("step_*.npz")):
            z = np.load(sp_file)
            T = len(z["input_ids"])
            tg, rt = tgt[cur:cur + T], role_t[cur:cur + T]
            cur += T
            sel = (rt == 1) if a.region else (np.isin(tg, sp) & (rt == 1))
            if not sel.any():
                continue
            for k, c in zip(*np.unique(tg[sel], return_counts=True)):
                tgt_counts[int(k)] = tgt_counts.get(int(k), 0) + int(c)
            ids = z["ids"][sel].astype(np.int64)
            amp = z["amp"][sel].astype(np.float64)
            for li in range(L):
                N[li] += np.bincount(ids[:, li].ravel(), minlength=E)
                A[li] += np.bincount(ids[:, li].ravel(), weights=amp[:, li].ravel(), minlength=E)
    np.savez(a.out, N=N, A=A)
    print(json.dumps({"special_target_positions": int(N[0].sum() // 6), "by_token": tgt_counts}))


def cmd_select(a):
    from reap_aggregate import scores
    z = np.load(a.stats)
    base = scores(z, a.score, a.decoder_gen_only)
    sp = np.load(a.special)["A"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.weight:  # smooth alternative: contribution total + (w - 1) x tool-block contribution
        assert a.score == "contrib"
        base = base + (a.weight - 1.0) * sp
    for e in a.experts:
        keep = np.zeros((L, E), bool)
        for li in range(L):
            prot = np.lexsort((np.arange(E), -sp[li]))[: a.protect]
            keep[li, prot] = True
            rest = np.lexsort((np.arange(E), -np.where(keep[li], -np.inf, base[li])))
            keep[li, rest[: e - a.protect]] = True
        assert (keep.sum(1) == e).all()
        tag = f"{a.score}{'_dec' if a.decoder_gen_only else ''}_sp{a.protect}{f'_w{a.weight:g}' if a.weight else ''}_e{e}"
        np.save(out / f"keep_{tag}.npy", keep)
        cov = float((sp * keep).sum() / sp.sum())
        print(json.dumps({"mask": tag, "special_contrib_kept": round(cov, 3)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["aggregate", "select"])
    ap.add_argument("--obs", nargs="*", default=[])
    ap.add_argument("--tokenizer", default="/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/tokenizer.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--region", action="store_true", help="all targets in generated tool-call blocks")
    ap.add_argument("--stats", default="")
    ap.add_argument("--special", default="")
    ap.add_argument("--score", default="contrib")
    ap.add_argument("--decoder-gen-only", action="store_true")
    ap.add_argument("--protect", type=int, default=8)
    ap.add_argument("--weight", type=float, default=0.0, help="tool-block weight w (0 = off)")
    ap.add_argument("--experts", type=int, nargs="*", default=[48, 64, 72])
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    {"aggregate": cmd_aggregate, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
