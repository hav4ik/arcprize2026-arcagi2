#!/usr/bin/env python3
"""E043 Engram row census, frequency selection and coverage (CPU only).

Addresses come from the official inference/engram.py of the pinned 1017 revision (compressed token map,
primes, multipliers, offsets), re-implemented in numpy: for each consumed input position t the 2/3/4-gram
hashes of the canonical ids ending at x[t] (pad before the sequence start), 3 orders x 8 heads per Engram
layer, row id = hash % prime + head offset (original geometry, never resized).

Consumed inputs of a rollout are x[0:N-1] (the final EOS is never fed back). Every eligible input is counted
once per layer and head (48 row lookups per token); counts are split by the role of the input token
(0 = prompt/tool output, 1 = model-generated).

  count    --tok DIR [DIR ...] --out counts.npz [--workers N]
  select   --counts counts.npz --budget-gib 32 64 96 125 --out-dir masks/   (global top-frequency rows)
  coverage --masks masks/*.npz --tok DIR [...]   (lookup hit rates on held-out token dirs)
  hashes   --tok DIR --n 4096                    (first sequence's ids, for parity with the GPU hash)
"""
import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROW_BYTES = 264  # 256 fp8 values + 8 ue8m0 scales


def load_geometry(model_dir, cache):
    """token_map (int64 [vocab]), mult [2,4], primes [2,3,8], offsets [2,24], num_embeddings [2], pad."""
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    spec = importlib.util.spec_from_file_location("dsv41_engram", Path(model_dir) / "inference/engram.py")
    eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eng)
    from transformers import PreTrainedTokenizerFast
    cfg = json.load(open(Path(model_dir) / "config.json"))["text_config"]
    tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(model_dir) / "tokenizer.json"))
    token_map, vocab = eng.build_compressed_token_map(tok)
    assert vocab == cfg["engram_compressed_vocab_size"], (vocab, cfg["engram_compressed_vocab_size"])

    class A:  # EngramLayout.from_args needs these attributes
        engram_layer_ids = cfg["engram_layer_ids"]
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = cfg["engram_num_embeddings"]
        engram_head_dim = cfg["engram_head_dim"]
    layout = eng.EngramLayout.from_args(A)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)
    mult = eng.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab).numpy().astype(np.int64)
    geo = {"token_map": np.asarray(token_map, dtype=np.int64), "mult": mult,
           "primes": np.asarray(layout.primes, dtype=np.int64), "offsets": offsets,
           "num_embeddings": np.asarray(layout.num_embeddings, dtype=np.int64),
           "pad": np.int64(token_map[cfg["engram_pad_token_id"]])}
    assert all(int(offsets[l, -1] + flat[l][-1]) == int(layout.num_embeddings[l]) for l in range(2))
    np.savez(cache, **geo)
    return geo


def hash_ids(geo, ids, pos):
    """ids: int64 [T] token ids of consumed inputs; pos: int64 [T] position within its sequence.
    Returns int64 [T, 2, 24] original row ids per Engram layer (order-major, 8 heads per order)."""
    comp = geo["token_map"][ids]
    T = len(ids)
    toks = np.empty((T, 4), dtype=np.int64)
    toks[:, 0] = comp
    for s in range(1, 4):
        sh = np.empty(T, dtype=np.int64)
        sh[s:] = comp[:-s]
        sh[:s] = geo["pad"]
        toks[:, s] = np.where(pos >= s, sh, geo["pad"])
    out = np.empty((T, 2, 24), dtype=np.int64)
    for l in range(2):
        prod = toks * geo["mult"][l][None, :]
        rolling = prod[:, 0]
        for i in range(1, 4):
            rolling = np.bitwise_xor(rolling, prod[:, i])
            out[:, l, (i - 1) * 8: i * 8] = rolling[:, None] % geo["primes"][l, i - 1][None, :] \
                + geo["offsets"][l, (i - 1) * 8: i * 8][None, :]
    return out


def iter_consumed(tokdir, chunk=2_000_000):
    """Yield (ids, pos, role) chunks of consumed inputs (each sequence minus its last token)."""
    tokdir = Path(tokdir)
    toks = np.fromfile(tokdir / "tokens.u32", dtype=np.uint32)
    roles = np.fromfile(tokdir / "roles.u8", dtype=np.uint8)
    buf_ids, buf_pos, buf_role, n = [], [], [], 0
    for line in open(tokdir / "index.jsonl"):
        r = json.loads(line)
        o, L = r["offset"], r["length"]
        buf_ids.append(toks[o:o + L - 1].astype(np.int64))
        buf_pos.append(np.arange(L - 1, dtype=np.int64))
        buf_role.append(roles[o:o + L - 1])
        n += L - 1
        if n >= chunk:
            yield np.concatenate(buf_ids), np.concatenate(buf_pos), np.concatenate(buf_role)
            buf_ids, buf_pos, buf_role, n = [], [], [], 0
    if buf_ids:
        yield np.concatenate(buf_ids), np.concatenate(buf_pos), np.concatenate(buf_role)


def _count_worker(job):
    geo_path, tokdirs, out = job
    geo = dict(np.load(geo_path))
    ne = geo["num_embeddings"]
    counts = {(l, r): np.zeros(int(ne[l]), dtype=np.uint32) for l in range(2) for r in range(2)}
    ntok = np.zeros(2, dtype=np.int64)
    for d in tokdirs:
        for ids, pos, role in iter_consumed(d):
            h = hash_ids(geo, ids, pos)
            for r in range(2):
                sel = role == r
                ntok[r] += int(sel.sum())
                for l in range(2):
                    counts[(l, r)] += np.bincount(h[sel, l].ravel(), minlength=int(ne[l])).astype(np.uint32)
    np.savez(out, ntok=ntok, **{f"l{l}_r{r}": c for (l, r), c in counts.items()})
    return out


def cmd_count(a):
    geo_path = Path(a.geo)
    load_geometry(a.model_dir, geo_path)
    dirs = sorted(a.tok)
    groups = [dirs[i::a.workers] for i in range(a.workers)]
    tmp = Path(a.out).with_suffix(".parts")
    tmp.mkdir(exist_ok=True)
    jobs = [(str(geo_path), g, str(tmp / f"part{i}.npz")) for i, g in enumerate(groups) if g]
    with ProcessPoolExecutor(len(jobs)) as ex:
        parts = list(ex.map(_count_worker, jobs))
    total, ntok = None, np.zeros(2, dtype=np.int64)
    for p in parts:
        z = np.load(p)
        ntok += z["ntok"]
        if total is None:
            total = {k: z[k].astype(np.uint64) for k in z.files if k != "ntok"}
        else:
            for k in total:
                total[k] += z[k]
    assert max(int(v.max()) for v in total.values()) < 2**32
    np.savez(a.out, ntok=ntok, **{k: v.astype(np.uint32) for k, v in total.items()})
    seen = [int(((total[f"l{l}_r0"] + total[f"l{l}_r1"]) > 0).sum()) for l in range(2)]
    print(json.dumps({"tokens_prompt": int(ntok[0]), "tokens_gen": int(ntok[1]),
                      "distinct_rows_per_layer": seen,
                      "distinct_gib": sum(seen) * ROW_BYTES / 2**30}))


def cmd_select(a):
    z = np.load(a.counts)
    cnt = [z[f"l{l}_r0"].astype(np.int64) + z[f"l{l}_r1"].astype(np.int64) for l in range(2)]
    if a.gen_weight != 1.0:
        cnt = [z[f"l{l}_r0"].astype(np.float64) + a.gen_weight * z[f"l{l}_r1"] for l in range(2)]
    # Global ranking over both layers: count desc, then (layer, row id) asc.
    nz = [np.flatnonzero(c) for c in cnt]
    lay = np.concatenate([np.full(len(nz[l]), l, dtype=np.int8) for l in range(2)])
    rid = np.concatenate(nz)
    val = np.concatenate([cnt[l][nz[l]] for l in range(2)])
    order = np.lexsort((rid, lay, -val))
    total_rows = sum(len(c) for c in cnt)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_lookups = float(val.sum())
    cum = np.cumsum(val[order])
    for gib in a.budget_gib:
        max_rows = int(gib * 2**30 // ROW_BYTES)
        k = min(max_rows, len(order))
        keep_sel = order[:k]
        bits = {}
        n_keep = []
        for l in range(2):
            m = np.zeros(len(cnt[l]), dtype=bool)
            m[rid[keep_sel][lay[keep_sel] == l]] = True
            n_keep.append(int(m.sum()))
            bits[f"layer{l}"] = np.packbits(m, bitorder="little")
        name = f"engram_top{gib:g}gib"
        np.savez(out_dir / f"{name}.npz", **bits)
        man = {"name": name, "budget_gib": gib, "rows_kept": n_keep, "rows_total": [len(c) for c in cnt],
               "kept_fraction": sum(n_keep) / total_rows, "kept_gib": sum(n_keep) * ROW_BYTES / 2**30,
               "observed_rows": len(order), "all_observed_kept": k == len(order),
               "calib_lookup_coverage": float(cum[k - 1] / total_lookups) if k else 0.0,
               "min_count_kept": float(val[order[k - 1]]) if k else None, "counts": str(a.counts),
               "rule": "global top-frequency rows over both layers, ties by (layer,row); miss -> zero row before Wkv"}
        json.dump(man, open(out_dir / f"{name}.json", "w"), indent=1)
        print(json.dumps(man))


def cmd_coverage(a):
    geo = load_geometry(a.model_dir, a.geo)
    masks = {Path(p).stem: [np.unpackbits(np.load(p)[f"layer{l}"], bitorder="little").astype(bool)
                            for l in range(2)] for p in a.masks}
    stats = {k: {"lookups": 0, "hits": 0, "tokens": 0, "all48": 0, "gen_lookups": 0, "gen_hits": 0} for k in masks}
    for d in a.tok:
        for ids, pos, role in iter_consumed(d):
            h = hash_ids(geo, ids, pos)
            for k, m in masks.items():
                hit = np.stack([m[l][h[:, l]] for l in range(2)], axis=1)  # [T, 2, 24]
                s = stats[k]
                s["lookups"] += hit.size
                s["hits"] += int(hit.sum())
                s["tokens"] += len(ids)
                s["all48"] += int(hit.all(axis=(1, 2)).sum())
                g = role == 1
                s["gen_lookups"] += int(hit[g].size)
                s["gen_hits"] += int(hit[g].sum())
    for k, s in stats.items():
        print(json.dumps({"mask": k, "row_hit_rate": s["hits"] / max(1, s["lookups"]),
                          "gen_row_hit_rate": s["gen_hits"] / max(1, s["gen_lookups"]),
                          "all48_token_rate": s["all48"] / max(1, s["tokens"]), "tokens": s["tokens"]}))


def cmd_hashes(a):
    geo = load_geometry(a.model_dir, a.geo)
    ids, pos, _ = next(iter_consumed(a.tok[0], chunk=1))
    n = min(a.n, len(ids))
    h = hash_ids(geo, ids[:n], pos[:n])
    np.savez(a.out, ids=ids[:n], hashes=h)
    print(a.out, h.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "select", "coverage", "hashes"])
    ap.add_argument("--model-dir", default="/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--geo", default="/workspace/dsv4/data/engram_geometry.npz")
    ap.add_argument("--tok", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--counts", default="")
    ap.add_argument("--budget-gib", type=float, nargs="*", default=[])
    ap.add_argument("--gen-weight", type=float, default=1.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--masks", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    {"count": cmd_count, "select": cmd_select, "coverage": cmd_coverage, "hashes": cmd_hashes}[a.cmd](a)


if __name__ == "__main__":
    main()
