"""Averages ("soups") Full-SFT FSDP2 distributed checkpoints into a new checkpoint directory.

All members must come from the same run (identical world_size / num_parts / parameter layout). Shards are
averaged elementwise in fp32 and written back in the members' dtype, so the output is a drop-in
`full_sft_ckpt_dir` for `export_and_quantize.ensure_sharded_fp8_export` and for
`launch_pilot2_xm.py --full_sft_ckpt_dir=...` (which exports FP8 inside the eval workers).

Example:
  soup_checkpoints --member_dirs=<run>/checkpoints/step-00366,<run>/checkpoints/step-00427,<run>/checkpoints/final \
    --output_dir=<run>/checkpoints/soup_366_427_final
"""

from __future__ import annotations

from concurrent import futures
import io
import json
import os
import time

from absl import app
from absl import flags
import torch

from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_list("member_dirs", [], "Comma-separated Full-SFT checkpoint dirs (each with COMPLETED.json).")
flags.DEFINE_list("weights", [], "Optional comma-separated member weights (default uniform).")
flags.DEFINE_string("output_dir", "", "Output checkpoint dir on CNS.")
flags.DEFINE_integer("parallel_parts", 4, "Number of (rank, part) shard files processed concurrently.")


def _read(path: str) -> bytes:
    chunks: list[bytes] = []
    with gfile.Open(path, "rb") as f:
        while True:
            b = f.read(64 * 1024 * 1024)
            if not b:
                break
            chunks.append(b)
    return b"".join(chunks)


def _write(path: str, payload: bytes) -> None:
    with gfile.Open(path, "wb") as f:
        mv = memoryview(payload)
        for off in range(0, len(payload), 64 * 1024 * 1024):
            f.write(mv[off : off + 64 * 1024 * 1024])


def _soup_one(fname: str, member_dirs: list[str], weights: list[float], out_dir: str) -> tuple[str, int]:
    acc: dict[str, torch.Tensor] | None = None
    dtype: dict[str, torch.dtype] = {}
    for m_dir, w in zip(member_dirs, weights):
        payload = torch.load(io.BytesIO(_read(os.path.join(m_dir, fname))), map_location="cpu", weights_only=False)
        model = payload["model"]
        if acc is None:
            acc = {k: v.to(torch.float32) * w for k, v in model.items()}
            dtype = {k: v.dtype for k, v in model.items()}
        else:
            if set(model.keys()) != set(acc.keys()):
                raise ValueError(f"{fname}: parameter set differs between members")
            for k, v in model.items():
                if v.shape != acc[k].shape:
                    raise ValueError(f"{fname}: shape mismatch for {k}: {v.shape} vs {acc[k].shape}")
                acc[k].add_(v.to(torch.float32), alpha=w)
        del payload, model
    assert acc is not None
    out = {"model": {k: v.to(dtype[k]).contiguous() for k, v in acc.items()}, "optimizer": {}}
    buf = io.BytesIO()
    torch.save(out, buf, _use_new_zipfile_serialization=False)
    data = buf.getvalue()
    _write(os.path.join(out_dir, fname), data)
    return fname, len(data)


def main(argv: list[str]) -> None:
    del argv
    members = [m.rstrip("/") for m in FLAGS.member_dirs if m.strip()]
    if len(members) < 2 or not FLAGS.output_dir:
        raise app.UsageError("Need >= 2 --member_dirs and --output_dir")
    weights = [float(w) for w in FLAGS.weights] if FLAGS.weights else [1.0] * len(members)
    if len(weights) != len(members):
        raise app.UsageError("--weights must match --member_dirs")
    s = sum(weights)
    weights = [w / s for w in weights]

    metas = []
    for m in members:
        with gfile.Open(os.path.join(m, "COMPLETED.json"), "r") as f:
            metas.append(json.load(f))
    ws = {int(mt["world_size"]) for mt in metas}
    nps = {int(mt["num_parts"]) for mt in metas}
    if len(ws) != 1 or len(nps) != 1:
        raise ValueError(f"Members disagree on layout: world_size={ws}, num_parts={nps}")
    world_size, num_parts = ws.pop(), nps.pop()
    part_files = [f"rank_{r:02d}_part_{p:02d}.pt" for r in range(world_size) for p in range(num_parts)]

    export_and_quantize.ensure_cns_quota_dir(FLAGS.output_dir)
    t0 = time.time()
    print(f"Souping {len(members)} members (weights={[round(w, 4) for w in weights]}) -> {FLAGS.output_dir}", flush=True)
    total = 0
    with futures.ThreadPoolExecutor(max_workers=FLAGS.parallel_parts) as pool:
        futs = [pool.submit(_soup_one, fn, members, weights, FLAGS.output_dir) for fn in part_files]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            fname, nbytes = fut.result()
            total += nbytes
            print(f"  [{i}/{len(part_files)}] {fname} {nbytes/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)

    # Rank state files: copy from the last member (RNG/order are irrelevant for export; keeps shard_files complete).
    last = members[-1]
    for r in range(world_size):
        fn = f"rank_{r:02d}_state.pt"
        _write(os.path.join(FLAGS.output_dir, fn), _read(os.path.join(last, fn)))

    meta = dict(metas[-1])
    meta.update(
        {
            "ckpt_name": os.path.basename(FLAGS.output_dir),
            "soup_members": members,
            "soup_weights": weights,
            "soup_member_steps": [int(mt.get("step", -1)) for mt in metas],
            "rank0_bytes": total,
            "elapsed_sec": round(time.time() - t0, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    with gfile.Open(os.path.join(FLAGS.output_dir, "COMPLETED.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done: {total/1e9:.1f} GB written in {time.time()-t0:.0f}s -> {FLAGS.output_dir}", flush=True)


if __name__ == "__main__":
    app.run(main)
