#!/usr/bin/env python3
"""E043 low-bit round: reorganize the calibration dump (observer step files with dump_x) into per-layer arrays.

Each step file holds x [Tsub, 40, 5120] (BF16 bits as int16) for the sampled rows x_rows, plus the routing of every
row: ids [T, 40, 6] (int16) and w [T, 40, 6] (fp16, the top-k weights handed to the experts). Writes, per layer L,
x_LL.npy int16 [N, 5120], ids_LL.npy int16 [N, 6], w_LL.npy float16 [N, 6] (memory-mappable) and manifest.json.
The warmup step (manifest line {"warmup": ...}) is skipped.

  calx_transpose.py --obs /dev/shm/e043/obs/calx --out /dev/shm/e043/calx_by_layer
"""
import argparse
import json
from pathlib import Path

import numpy as np

L = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke tests: only the first N step files")
    a = ap.parse_args()
    steps = sorted(Path(a.obs).glob("step_*.npz"))[1:]  # step 0 = warmup prompt
    if a.max_steps:
        steps = steps[: a.max_steps]
    sizes = []
    for p in steps:
        with np.load(p) as z:
            sizes.append(len(z["x_rows"]) if "x_rows" in z.files else z["x"].shape[0])
    n = int(sum(sizes))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    xs = [np.lib.format.open_memmap(out / f"x_{li:02d}.npy", mode="w+", dtype=np.int16, shape=(n, 5120)) for li in range(L)]
    ids = np.zeros((L, n, 6), np.int16)
    ws = np.zeros((L, n, 6), np.float16)
    o = 0
    for p, k in zip(steps, sizes):
        with np.load(p) as z:
            rows = z["x_rows"] if "x_rows" in z.files else np.arange(z["x"].shape[0])
            x = z["x"]
            for li in range(L):
                xs[li][o:o + k] = x[:, li]
            ids[:, o:o + k] = z["ids"][rows].transpose(1, 0, 2)
            ws[:, o:o + k] = z["w"][rows].transpose(1, 0, 2)
        o += k
    for li in range(L):
        xs[li].flush()
        np.save(out / f"ids_{li:02d}.npy", ids[li])
        np.save(out / f"w_{li:02d}.npy", ws[li])
    bad = int((ids < 0).sum())
    json.dump({"rows": n, "steps": len(steps), "obs": a.obs, "negative_ids": bad}, open(out / "manifest.json", "w"))
    print(json.dumps({"rows": n, "steps": len(steps), "negative_ids": bad}))


if __name__ == "__main__":
    main()
