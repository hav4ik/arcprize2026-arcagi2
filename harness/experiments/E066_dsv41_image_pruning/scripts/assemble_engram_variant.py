#!/usr/bin/env python3
"""E066: an export variant with a replaced Engram table, everything else hardlinked from the source export.

--engram-dir is either a fp4_engram_chunked.py output (same file names and keys as the source's Engram files: only the
files are swapped) or an engram_subset.py output (new files and keys, listed in engram_subset.json with the new
text_config.engram_compact_kept: the index and config are rewritten).

  assemble_engram_variant.py --src export/b32_e128 --engram-dir export/engram_b32_fp4 --out export/b32_fp4
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--engram-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, ed, out = Path(a.src), Path(a.engram_dir), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    old_eng = {f for f in wm.values() if f.startswith("model-engram-")}
    sub = json.load(open(ed / "engram_subset.json")) if (ed / "engram_subset.json").exists() else None
    for p in src.iterdir():
        if p.name in old_eng or p.name in ("model.safetensors.index.json", "config.json") or p.name.startswith("."):
            continue
        if p.is_dir():
            shutil.copytree(p, out / p.name, copy_function=os.link)
        else:
            os.link(p, out / p.name)
    cfg = json.load(open(src / "config.json"))
    if sub:
        wm = {k: v for k, v in wm.items() if not v.startswith("model-engram-")}
        wm.update(sub["weight_map"])
        (cfg.get("text_config", cfg))["engram_compact_kept"] = sub["engram_compact_kept"]
        cfg.setdefault("e066_engram", {})["subset"] = sub["stats"]
        new_eng = set(sub["weight_map"].values())
    else:
        new_eng = old_eng
        missing = [f for f in new_eng if not (ed / f).exists()]
        assert not missing, f"Engram files missing in {ed}: {missing}"
    if (ed / "fp4_engram_stats.json").exists():
        st = json.load(open(ed / "fp4_engram_stats.json"))
        cfg.setdefault("e066_engram", {})["fp4"] = {"format": "MXFP4 values in the MXFP8 container (E2M1 + UE8M0 / 32)",
                                                    "rel_mse_vs_mxfp8": st.get("rel_mse_vs_mxfp8_row_weighted")}
    for f in new_eng:
        os.link(ed / f, out / f)
    idx["weight_map"] = wm
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=1)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    missing = sorted({f for f in wm.values() if not (out / f).exists()})
    assert not missing, missing
    print(json.dumps({"out": str(out), "engram_files": len(new_eng), "subset": bool(sub)}))


if __name__ == "__main__":
    main()
