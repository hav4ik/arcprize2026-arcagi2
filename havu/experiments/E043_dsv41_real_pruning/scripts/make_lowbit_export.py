#!/usr/bin/env python3
"""E043 low-bit round: package a GPTQ-only low-bit checkpoint (no routed MXFP4 experts) for deployment.

  make_lowbit_export.py --base export/notool_e128_x10_eng56 --precomputed /dev/shm/e043/gptq_e128_u2 \
      --precomputed-ids original --keep keep.npy --schema '<json>' [--schema-w2 '<json>'] --name E128 \
      --out export/notool_e128_x10_eng56_gptq

--base is a compact export from export_pruned.py (router, dense, shared expert, compact Engram; routed experts as
MXFP4). The output keeps every file of --base except the routed-expert tensors: shards without experts and the
model-engram-* files are hardlinked, shards with experts are rewritten without them, the index is updated. The
low-bit experts go to gptq/layer_LL.safetensors (e{E}.{w1,w3,w2}.{q,s,z} in Humming's packed layout, compact expert
ids = rank among the kept original ids). Serve with the E043 patches (apply_e043_patches.py, apply_compact_engram.py,
apply_e043_humming.py) and DSV41_EXPERT_HUMMING / DSV41_EXPERT_HUMMING_W2 / E043_GPTQ_DIR=<out>/gptq (recorded in
config.json text_config.e043_lowbit and README.md).
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$")
PKEY = re.compile(r"^e(\d+)\.(w[123])\.([qsz])$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--precomputed-ids", choices=["original", "compact"], required=True)
    ap.add_argument("--keep", required=True, help="bool [40, 384] keep set of the base export")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--schema-w2", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--quantizer", default="E043 GPTQ (gptq_experts.py)")
    ap.add_argument("--keep-desc", default="no-tool keep set", help="model-card wording for the keep set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    keep = np.load(a.keep).astype(bool)
    idx = json.load(open(base / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    dropped = 0
    for f in sorted(base.iterdir()):
        if f.is_dir():  # encoding/, inference/ (DeepSeek's chat encoder and reference code)
            shutil.copytree(f, out / f.name)
            continue
        if f.name == "model.safetensors.index.json":
            continue
        if f.suffix == ".safetensors" and not f.name.startswith("model-engram"):
            with safe_open(str(f), framework="pt") as sf:
                keys = list(sf.keys())
                exp = [k for k in keys if EXPERT.match(k)]
                if exp:
                    rest = {k: sf.get_tensor(k) for k in keys if not EXPERT.match(k)}
                    dropped += len(exp)
                    for k in exp:
                        wm.pop(k, None)
                    if rest:
                        save_file(rest, str(out / f.name), metadata={"format": "pt"})
                    continue
        if f.name == "config.json":
            continue
        os.link(f, out / f.name)
    idx["weight_map"] = wm
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=1)
    pd = out / "gptq"
    pd.mkdir()
    n_exp = None
    for L in range(40):
        src = Path(a.precomputed) / f"layer_{L:02d}.safetensors"
        kept = np.nonzero(keep[L])[0]
        n_exp = len(kept)
        comp = {int(o): i for i, o in enumerate(kept)}
        with safe_open(str(src), framework="pt") as sf:
            t = {}
            for k in sf.keys():
                m = PKEY.match(k)
                e = int(m.group(1))
                if a.precomputed_ids == "original":
                    if e not in comp:
                        continue
                    e = comp[e]
                t[f"e{e}.{m.group(2)}.{m.group(3)}"] = sf.get_tensor(k)
        assert len(t) == 9 * n_exp, (L, len(t), n_exp)
        save_file(t, str(pd / f"layer_{L:02d}.safetensors"), metadata={"format": "pt", "experts": str(n_exp)})
    if (Path(a.precomputed) / "manifest.json").exists():
        shutil.copy(Path(a.precomputed) / "manifest.json", pd / "source_manifest.json")
    lowbit = {"name": a.name, "quantizer": a.quantizer, "experts_per_layer": n_exp, "humming_schema": json.loads(a.schema),
              "humming_schema_w2": json.loads(a.schema_w2) if a.schema_w2 else None, "precomputed_dir": "gptq",
              "env": {"DSV41_EXPERT_HUMMING": a.schema, **({"DSV41_EXPERT_HUMMING_W2": a.schema_w2} if a.schema_w2 else {}),
                      "E043_GPTQ_DIR": "<model_dir>/gptq"}}
    cfg = json.load(open(base / "config.json"))
    (cfg.get("text_config", cfg))["e043_lowbit"] = lowbit
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    readme = f"""# DeepSeek-V4.1-Flash-1017, {a.name} (E043 low-bit, GPTQ-only layout)

Routed experts: {n_exp} per backbone layer ({a.keep_desc}, original ids in e043_manifest.json), stored only as
low-bit Humming tensors in `gptq/` (no MXFP4 copies): {a.quantizer}. Schema (gate/up): `{a.schema}`;
down: `{a.schema_w2 or a.schema}`. Attention, router (sliced to the kept experts), shared expert, norms and mHC as
released; Engram compact as in the base export (`text_config.engram_compact_kept`).

Serve with the vLLM DeepSeek-V4.1 SM89/SM120 fork plus E043's patches (apply_e043_patches.py, apply_compact_engram.py,
apply_e043_humming.py, which add the router shape ({n_exp}, 6)) and:

    export DSV41_EXPERT_HUMMING='{a.schema}'
""" + (f"    export DSV41_EXPERT_HUMMING_W2='{a.schema_w2}'\n" if a.schema_w2 else "") + """    export E043_GPTQ_DIR=<model_dir>/gptq

The loader gives each routed expert a zero-allocation placeholder and Humming takes the packed q/s/z directly.
Details and quality numbers: havu/experiments/E043_dsv41_real_pruning in hav4ik/arcprize2026-arcagi2.
"""
    (out / "README.md").write_text(readme)
    tot = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"out": str(out), "dropped_expert_tensors": dropped, "experts_per_layer": n_exp, "bytes": tot}))


if __name__ == "__main__":
    main()
