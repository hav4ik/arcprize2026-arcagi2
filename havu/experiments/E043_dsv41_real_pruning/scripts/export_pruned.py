#!/usr/bin/env python3
"""E043: write a physically pruned DeepSeek-V4.1-Flash-1017 checkpoint (HF layout) from the original shards.

Experts: for each backbone layer keep the experts of --expert-keep (bool [40, 384]), renumbered 0..E-1 in ascending
original id; ffn.gate.weight / gate.bias / gate.bias_vl are sliced to the same rows; native MXFP4 bytes and UE8M0
scales are copied unchanged. config.json: text_config.n_routed_experts = E (top-6 unchanged).
Engram (--engram-keep, packbits per layer): `layers.{1,14}.engram.embed.weight` / `.scale` become the compact
kept rows in original-id order, plus `layers.{1,14}.engram.embed_tokens.row_map` int32 [num_embeddings] mapping each
original hashed row id to its compact row (-1 = dropped: read as a zero row before Wkv). Hash geometry unchanged.
DSpark (mtp.*) and vision tensors are dropped with --text-only (served with --language-model-only, no DSpark).
--no-mtp (E066) drops only DSpark and keeps the vision tower (vision.*, aligner.*, image_*) for image prompts.
--per-layer (E066) allows a different expert count per layer: experts renumbered 0..E_l-1 per layer, and
text_config.n_routed_experts_per_layer = the 40 counts (the dynamic-count fork patch, E062 port/04_per_layer_experts),
n_routed_experts = their maximum.
--no-routed-experts writes everything but the routed-expert tensors: the base of a GPTQ-only make_lowbit_export.py
checkpoint, without the ~96 GB of MXFP4 E128 experts on disk.

  export_pruned.py --src /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017 --expert-keep keep.npy \
      [--engram-keep engram.npz] --out /dev/shm/dsv4/export/e64 [--text-only] [--shard-gb 4.9]
"""
import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")
GATE_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")
ENGRAM_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
TEXT_ONLY_DROP = re.compile(r"^(mtp\.|vision\.|aligner\.|image_)")
MTP_DROP = re.compile(r"^mtp\.")


class Writer:
    def __init__(self, out, shard_bytes, prefix="model"):
        self.out, self.shard_bytes, self.prefix = out, shard_bytes, prefix
        self.buf, self.size, self.n, self.weight_map, self.total = {}, 0, 0, {}, 0

    def add(self, name, t):
        # A cap: flush before a tensor would push the shard past shard_bytes (only a single larger tensor, i.e. an
        # Engram table not taken from --engram-from, gets its own bigger file; reshard_safetensors.py splits those).
        n = t.numel() * t.element_size()
        if self.buf and self.size + n > self.shard_bytes:
            self.flush()
        self.buf[name] = t.contiguous()
        self.size += n

    def flush(self):
        if not self.buf:
            return
        self.n += 1
        fn = f"{self.prefix}-{self.n:05d}.safetensors"
        save_file(self.buf, str(self.out / fn), metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fn
        self.total += self.size
        self.buf, self.size = {}, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--expert-keep", required=True)
    ap.add_argument("--engram-keep", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--no-mtp", action="store_true", help="drop DSpark (mtp.*) only; keep the vision tower")
    ap.add_argument("--no-routed-experts", action="store_true")
    ap.add_argument("--per-layer", action="store_true", help="(E066) per-layer expert counts, n_routed_experts_per_layer")
    ap.add_argument("--shard-gb", type=float, default=4.9, help="tensor bytes per shard (cap; files stay < 5 GB)")
    ap.add_argument("--engram-chunk", type=int, default=8_000_000)
    ap.add_argument("--engram-from", default=None,
                    help="existing export whose model-engram-* files to hardlink instead of rebuilding the table")
    ap.add_argument("--geo", default="/workspace/dsv4/data/engram_geometry.npz",
                    help="engram_census.py geometry (per-column offsets/primes) for engram_compact_kept")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = np.load(a.expert_keep).astype(bool)
    n_layers, n_exp = keep.shape
    counts = keep.sum(1).astype(int)
    E = int(counts.max())
    assert a.per_layer or (counts == E).all(), "uniform per-layer budget expected (--per-layer for per-layer counts)"
    newid = np.full(keep.shape, -1, np.int64)
    for li in range(n_layers):
        newid[li, keep[li]] = np.arange(counts[li])
    eng_bits = None
    if a.engram_keep:
        z = np.load(a.engram_keep)
        eng_bits = [z["layer0"], z["layer1"]]
    cfg = json.load(open(src / "config.json"))
    eng_layers = cfg["text_config"]["engram_layer_ids"]

    wm = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    by_file = defaultdict(list)
    for name, fn in wm.items():
        by_file[fn].append(name)
    w = Writer(out, int(a.shard_gb * 1e9))
    # Engram tables/scales/maps in their own files so one Engram artifact can be shared by E48/E64/E72.
    we = Writer(out, int(a.shard_gb * 1e9), prefix="model-engram")
    per_layer = counts.tolist() if a.per_layer else None
    manifest = {"source": str(src), "experts_per_layer": per_layer or E, "engram": {}, "dropped": 0}
    compact_kept = [None, None]
    for fn in sorted(by_file):
        with safe_open(str(src / fn), framework="pt") as f:
            for name in sorted(by_file[fn]):
                if (a.text_only and TEXT_ONLY_DROP.match(name)) or (a.no_mtp and MTP_DROP.match(name)):
                    manifest["dropped"] += 1
                    continue
                m = EXPERT_RE.match(name)
                if m:
                    li, ei = int(m.group(1)), int(m.group(2))
                    if li < n_layers:
                        if keep[li, ei] and not a.no_routed_experts:
                            w.add(f"layers.{li}.ffn.experts.{newid[li, ei]}.{m.group(3)}", f.get_tensor(name))
                        continue
                m = GATE_RE.match(name)
                if m and int(m.group(1)) < n_layers:
                    li = int(m.group(1))
                    t = f.get_tensor(name)
                    idx = torch.from_numpy(np.flatnonzero(keep[li]))
                    w.add(name, t.index_select(0, idx))
                    continue
                m = ENGRAM_RE.match(name)
                if m and a.engram_from:
                    continue  # taken from the existing export below
                if m and eng_bits is not None:
                    li = int(m.group(1))
                    hi = eng_layers.index(li)
                    sl = f.get_slice(name)
                    rows = sl.get_shape()[0]
                    bits = np.unpackbits(eng_bits[hi], bitorder="little")[:rows].astype(bool)
                    parts = []
                    for s in range(0, rows, a.engram_chunk):
                        e = min(rows, s + a.engram_chunk)
                        sel = torch.from_numpy(np.flatnonzero(bits[s:e]))
                        if len(sel):
                            chunk = sl[s:e]  # fp8 / e8m0: select on the raw bytes, then view back
                            parts.append(chunk.view(torch.uint8).index_select(0, sel).view(chunk.dtype))
                    we.add(name, torch.cat([p.view(torch.uint8) for p in parts]).view(parts[0].dtype))
                    if m.group(2) == "weight":
                        rm = np.full(rows, -1, np.int32)
                        kept = np.flatnonzero(bits)
                        rm[kept] = np.arange(len(kept), dtype=np.int32)
                        we.add(f"layers.{li}.engram.embed_tokens.row_map", torch.from_numpy(rm))
                        geo = np.load(a.geo)
                        cols = [(int(geo["offsets"][hi, j]), int(geo["primes"][hi].reshape(-1)[j])) for j in range(24)]
                        compact_kept[hi] = [int(bits[o:o + n].sum()) for o, n in cols]
                        assert sum(compact_kept[hi]) == len(kept)
                        manifest["engram"][str(li)] = {"rows": int(rows), "kept": int(len(kept)),
                                                       "mask_sha256": hashlib.sha256(eng_bits[hi].tobytes()).hexdigest()}
                    continue
                w.add(name, f.get_tensor(name))
    w.flush()
    we.flush()
    if a.engram_from:  # share the Engram files of an earlier export (hardlinks on the same filesystem)
        src_idx = json.load(open(Path(a.engram_from) / "model.safetensors.index.json"))["weight_map"]
        for k, fn in src_idx.items():
            if fn.startswith("model-engram-"):
                if not (out / fn).exists():
                    try:
                        (out / fn).hardlink_to(Path(a.engram_from) / fn)
                    except OSError:  # another filesystem: copy
                        shutil.copy2(Path(a.engram_from) / fn, out / fn)
                we.weight_map[k] = fn
        src_cfg = json.load(open(Path(a.engram_from) / "config.json"))["text_config"]
        compact_kept = src_cfg["engram_compact_kept"]
        eng_bits = True
        manifest["engram"] = json.load(open(Path(a.engram_from) / "e043_manifest.json"))["engram"]
        manifest["engram_from"] = str(a.engram_from)
    w.weight_map.update(we.weight_map)
    w.total += we.total
    json.dump({"metadata": {"total_size": w.total}, "weight_map": w.weight_map},
              open(out / "model.safetensors.index.json", "w"), indent=1)
    cfg["text_config"]["n_routed_experts"] = E
    if per_layer:
        cfg["text_config"]["n_routed_experts_per_layer"] = per_layer
    if eng_bits is not None:
        cfg["text_config"]["engram_compact_kept"] = compact_kept  # per Engram layer, kept rows per hash column
    cfg["e043_pruning"] = {"experts_per_layer": per_layer or E, "expert_keep_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
                           "engram_compact": bool(eng_bits is not None), "text_only": a.text_only, "no_mtp": a.no_mtp,
                           **({"routed_experts": "none (GPTQ-only base)"} if a.no_routed_experts else {})}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "README.md", "LICENSE"):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)
    for d in ("encoding", "inference"):
        if (src / d).exists():
            shutil.copytree(src / d, out / d, dirs_exist_ok=True)
    manifest["keep_original_ids"] = {str(li): np.flatnonzero(keep[li]).tolist() for li in range(n_layers)}
    manifest["total_bytes"] = w.total
    json.dump(manifest, open(out / "e043_manifest.json", "w"), indent=1)
    print(json.dumps({"out": str(out), "bytes": w.total, "shards": w.n, "engram": manifest["engram"]}))


if __name__ == "__main__":
    main()
