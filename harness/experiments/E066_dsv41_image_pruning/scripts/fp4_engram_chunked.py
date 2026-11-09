#!/usr/bin/env python3
"""E066: MXFP4 Engram rows (E043 make_fp4_engram.py rounding) for a row-chunked export, on CPU.

The row-chunked layout (E043 reshard_safetensors.py) keeps `layers.L.engram.embed.weight.rows_<start>` chunks in
separate files and the E8M0 scale [rows, 8] whole in another file, which make_fp4_engram.py does not handle. This
re-rounds every weight chunk with its scale rows (the same requant(): per group of 32 the lower-MSE of the two E8M0
scale choices, E2M1 values stored exactly in FP8 E4M3), writes the same file names and keys to --out, and copies the
other Engram tensors (row maps) unchanged. Serving is unchanged: the engine runs true MXFP4 Engram values from the
MXFP8 container, so dev NLL measures the quality of a 4-bit Engram table (a deployment kernel stores 4.25 bits).

  fp4_engram_chunked.py --src export/c1_gptq --out export/engram64_fp4 [--chunk 2000000] [--threads 96]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "E043_dsv41_real_pruning" / "scripts"))
from make_fp4_engram import requant  # noqa: E402

WKEY = re.compile(r"^(layers\.\d+\.engram\.embed)\.weight(?:\.rows_(\d+))?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=96)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wm = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    eng = {k: f for k, f in wm.items() if f.startswith("model-engram-")}
    by_file = defaultdict(list)
    for k, f in eng.items():
        by_file[f].append(k)
    scales, new_scales, stats = {}, {}, {}
    for k, f in eng.items():  # whole scale tensors (rows x 8 bytes), rewritten per weight chunk below
        if k.endswith(".embed.scale"):
            with safe_open(str(src / f), framework="pt") as sf:
                scales[k[: -len(".scale")]] = sf.get_tensor(k)
            new_scales[k[: -len(".scale")]] = scales[k[: -len(".scale")]].clone()
    for f in sorted(by_file):
        tensors = {}
        with safe_open(str(src / f), framework="pt") as sf:
            for k in by_file[f]:
                m = WKEY.match(k)
                if not m:
                    if not k.endswith(".embed.scale"):
                        tensors[k] = sf.get_tensor(k)  # row maps etc.
                    continue
                base, start = m.group(1), int(m.group(2) or 0)
                ws = sf.get_slice(k)
                n = ws.get_shape()[0]
                new_w = torch.empty((n, 256), dtype=torch.float8_e4m3fn)
                err = tot = 0.0
                for r0 in range(0, n, a.chunk):
                    r1 = min(n, r0 + a.chunk)
                    w, s, e_, t_ = requant(ws[r0:r1], scales[base][start + r0:start + r1])
                    new_w[r0:r1] = w
                    new_scales[base][start + r0:start + r1] = s
                    err, tot = err + float(e_), tot + float(t_)
                tensors[k] = new_w
                stats[k] = {"rows": n, "rel_mse_vs_mxfp8": err / tot}
                print(json.dumps({k: stats[k]}), flush=True)
        if tensors:
            save_file(tensors, str(out / f), metadata={"format": "pt"})
    for base, s in new_scales.items():  # scale files: rewrite with the new scales (and whatever else they hold)
        k = base + ".scale"
        f = eng[k]
        tensors = {}
        if (out / f).exists():
            with safe_open(str(out / f), framework="pt") as sf:
                tensors = {x: sf.get_tensor(x) for x in sf.keys()}
        else:
            with safe_open(str(src / f), framework="pt") as sf:
                tensors = {x: sf.get_tensor(x) for x in by_file[f] if not x.endswith(".embed.scale") and not WKEY.match(x)}
        tensors[k] = s
        others = [x for x in by_file[f] if x.endswith(".embed.scale") and x != k]
        if others:  # several scale tensors in one file: add the others (rewritten or not) too
            for x in others:
                tensors[x] = new_scales[x[: -len(".scale")]]
        save_file(tensors, str(out / f), metadata={"format": "pt"})
    missing = sorted(set(eng.values()) - {p.name for p in out.glob("model-engram-*.safetensors")})
    assert not missing, missing
    tot_err = sum(v["rel_mse_vs_mxfp8"] * v["rows"] for v in stats.values()) / sum(v["rows"] for v in stats.values())
    json.dump({"chunks": stats, "rel_mse_vs_mxfp8_row_weighted": tot_err}, open(out / "fp4_engram_stats.json", "w"),
              indent=1)
    print(json.dumps({"files": len(by_file), "weight_chunks": len(stats), "rel_mse_vs_mxfp8": round(tot_err, 5)}))


if __name__ == "__main__":
    main()
