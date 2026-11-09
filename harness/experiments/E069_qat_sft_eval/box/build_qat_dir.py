#!/usr/bin/env python3
"""E068: zero-copy serving dir for a weights-only E128 checkpoint (QAT + SFT) with B32's compact Engram table.

Usage: build_qat_dir.py --qat /root/e068/model-qat025 --b32 /root/e068/model-e128t --out /root/e068/model-qat025-eng

- every top-level file/dir of the QAT repo except its index (config, tokenizer, encoding, gptq/, model shards, ...)
  is symlinked into --out;
- B32's model-engram-*.safetensors (the whole-tensor HF layout) are symlinked at the top level (--engram-subdir ""),
  because this wheel's DefaultModelLoader globs only <model>/*.safetensors and then requires every index file to be in
  that list (a file in a subdir fails with "Weight files referenced in index but missing");
- model.safetensors.index.json = the QAT weight_map minus its Engram *table* entries
  (layers.{L}.engram.embed.weight[.rows_*], embed.scale[.rows_*], embed_tokens.row_map), plus B32's entries for those
  tensors (rewritten to <subdir>/<file> when --engram-subdir is set). The QAT's own Engram projections (q_weight, k_weight, wkv.*) are kept.
Checks: config text_config Engram keys equal between the two repos; per layer, B32's embed.weight / embed.scale rows ==
sum(engram_compact_kept[layer]); the QAT rows_<start> chunks tile [0, kept) (chunk sizes from consecutive starts);
row_map length == engram_num_embeddings; every index file exists.
"""
import argparse
import json
import os
import re
import struct
from pathlib import Path

TABLE = re.compile(r"^layers\.(\d+)\.engram\.(embed\.weight|embed\.scale|embed_tokens\.row_map)(\.rows_(\d+))?$")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat", type=Path, required=True)
    ap.add_argument("--b32", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engram-subdir", default="", help="subdir for the Engram symlinks ('' = top level)")
    a = ap.parse_args()
    qcfg = json.load(open(a.qat / "config.json"))["text_config"]
    bcfg = json.load(open(a.b32 / "config.json"))["text_config"]
    ekeys = sorted(k for k in set(qcfg) | set(bcfg) if "engram" in k)
    diff = [k for k in ekeys if qcfg.get(k) != bcfg.get(k)]
    assert not diff, f"Engram config keys differ: {diff}"
    print("engram config keys equal:", ekeys)
    kept = {lid: sum(r) for lid, r in zip(bcfg["engram_layer_ids"], bcfg["engram_compact_kept"])}
    nemb = dict(zip(bcfg["engram_layer_ids"], bcfg["engram_num_embeddings"]))

    qidx = json.load(open(a.qat / "model.safetensors.index.json"))
    bidx = json.load(open(a.b32 / "model.safetensors.index.json"))
    # QAT table entries (dropped) and their row-chunk starts
    starts, dropped = {}, []
    for k in qidx["weight_map"]:
        m = TABLE.match(k)
        if m:
            dropped.append(k)
            if m.group(4) is not None:
                starts.setdefault((int(m.group(1)), m.group(2)), []).append(int(m.group(4)))
    for (lid, t), st in sorted(starts.items()):
        st = sorted(st)
        sizes = [b - s for s, b in zip(st, st[1:] + [kept[lid]])]
        assert st[0] == 0 and all(s > 0 for s in sizes), (lid, t, st)
        print(f"QAT layer {lid} {t}: {len(st)} row chunks, starts {st}, sizes {sizes}, sum {sum(sizes)} "
              f"== kept {kept[lid]}")
    # B32 table entries and their shapes
    added = {}
    for k, f in bidx["weight_map"].items():
        m = TABLE.match(k)
        if not m:
            continue
        assert m.group(4) is None, f"B32 index unexpectedly row-chunked: {k}"
        h = header(a.b32 / f)[k]
        lid, t = int(m.group(1)), m.group(2)
        want = nemb[lid] if t == "embed_tokens.row_map" else kept[lid]
        assert h["shape"][0] == want, (k, h["shape"], want)
        print(f"B32 {k}: {h['dtype']} {h['shape']} in {f} (rows == {want})")
        added[k] = f"{a.engram_subdir}/{f}" if a.engram_subdir else f
    assert len(added) == 3 * len(kept), added

    a.out.mkdir(parents=True, exist_ok=True)
    if a.engram_subdir:
        (a.out / a.engram_subdir).mkdir(exist_ok=True)
    for p in sorted(a.qat.iterdir()):
        if p.name in ("model.safetensors.index.json", ".cache", "engram"):
            continue
        dst = a.out / p.name
        if not dst.is_symlink():
            dst.symlink_to(p.resolve())
    for v in sorted(set(added.values())):
        dst = a.out / v
        assert not dst.exists() or dst.is_symlink(), f"{dst} exists and is not a symlink"
        if not dst.is_symlink():
            dst.symlink_to((a.b32 / Path(v).name).resolve())
    wm = {k: v for k, v in qidx["weight_map"].items() if k not in set(dropped)}
    overlap = set(wm) & set(added)
    assert not overlap, overlap
    wm.update(added)
    for f in set(wm.values()):
        assert (a.out / f).exists(), f"missing shard {f}"
    out = {"metadata": dict(qidx.get("metadata", {}), e068_note="QAT weight_map + B32 HF Engram tables"),
           "weight_map": dict(sorted(wm.items()))}
    tmp = a.out / "model.safetensors.index.json.tmp"
    json.dump(out, open(tmp, "w"), indent=2)
    os.replace(tmp, a.out / "model.safetensors.index.json")
    proj = sorted(k for k in wm if ".engram." in k and k not in added)
    print(f"dropped {len(dropped)} QAT table entries, added {len(added)} B32 entries; "
          f"{len(wm)} tensors in the merged index; QAT Engram projections kept: "
          + ", ".join(f"{k} -> {wm[k]}" for k in proj))
    print("BUILD_OK", a.out)


if __name__ == "__main__":
    main()
