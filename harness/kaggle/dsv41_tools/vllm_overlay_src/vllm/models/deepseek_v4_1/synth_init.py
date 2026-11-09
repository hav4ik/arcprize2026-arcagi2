# SPDX-License-Identifier: Apache-2.0
"""E042: bounded, format-valid synthetic weights for DeepSeek-V4.1 on 4xL4.

Enabled with ``DSV41_SYNTH_INIT=1`` and ``--load-format dummy``. Runs after the
stock dummy fill and before ``process_weights_after_loading``, so the normal
Marlin / MXFP8 / wo_a / mHC finalizers still run on the filled tensors.

Why: the stock dummy fill skips integer tensors (packed MXFP4 experts stay
zero, ``torch.empty`` MXFP8 scales stay garbage) and sets norms and the router
to +-1e-3, so the correction bias alone picks the same 6 experts for every
token. That makes MoE decode look much cheaper than it is.

Rules (by dtype, then name):
- uint8 with "scale" in the name: constant UE8M0 exponent giving fan-in scaled
  weights (experts: 2^-8 / 2^-7, dense: 1/sqrt(fan_in), engram: 2^-4).
- other uint8 (packed E2M1 pairs): uniform random bytes (E2M1 has no NaN/inf).
- float8_e4m3fn: N(0, 1) clamped to +-6.
- floating ``*norm*`` vectors: 1; ``e_score_correction_bias*``: 0;
  ``attn_sink``: 0 on real heads, -inf on padded heads.
- other floating matrices (>= 2-D, not mHC): N(0, 1/fan_in).
- everything else keeps the dummy +-1e-3 values (mHC fn/base/scale, biases).

Values are keyed by (seed, parameter name, rank-local position): replicated
parameters match across ranks, but TP shards are not slices of one global
tensor (unlike the plan's global-coordinate initializer). Temporaries are
bounded by ``_TILE`` elements.
"""

import json
import math
import os
import re
import time
import zlib
from collections import defaultdict

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_TILE = 1 << 24  # elements per generated tile (64 MiB fp32 temporaries)
_E2M1_RMS = math.sqrt(68.5 / 8)  # RMS of a uniformly random E2M1 code
_SEED = int(os.environ.get("DSV41_SYNTH_SEED", "20261020"))


def _generator(name: str, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed((_SEED * 1_000_003 + zlib.crc32(name.encode())) % (1 << 62))
    return g


def _tiles(t: torch.Tensor):
    flat = t.view(-1)
    for start in range(0, flat.numel(), _TILE):
        yield flat[start : start + _TILE]


def _scale_byte(std: float) -> int:
    return int(round(127 + math.log2(std)))


def _fill_bytes(t, g, dev):
    for tile in _tiles(t):
        r = torch.randint(0, 256, (tile.numel(),), generator=g, device=dev,
                          dtype=torch.int16)
        tile.copy_(r.to(torch.uint8))


def _fill_normal(t, g, dev, std, clamp=None):
    for tile in _tiles(t):
        r = torch.randn(tile.numel(), generator=g, device=dev, dtype=torch.float32)
        if clamp is not None:
            r.clamp_(-clamp, clamp)
        if std != 1.0:
            r.mul_(std)
        tile.copy_(r.to(tile.dtype))


def _pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", name)


@torch.no_grad()
def apply_synthetic_init(model: nn.Module) -> None:
    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device())
    summary: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])

    sink_heads = {}
    for mod_name, module in model.named_modules():
        if isinstance(getattr(module, "attn_sink", None), torch.Tensor):
            sink_heads[f"{mod_name}.attn_sink"] = getattr(module, "n_local_heads", None)

    for name, param in model.named_parameters():
        data = param.data
        if data.numel() == 0:
            continue
        if not data.is_contiguous():
            summary[(_pattern(name), str(data.dtype), "skip_noncontig", "")][0] += 1
            continue
        g = _generator(name, dev)
        dtype = data.dtype
        if dtype == torch.uint8:
            if "scale" in name:
                if "experts" in name:
                    fan_in = 5120 if "w13" in name else 2304
                    byte = _scale_byte(1 / math.sqrt(fan_in) / _E2M1_RMS)
                elif "engram.embed_tokens" in name:
                    byte = 123
                else:
                    fan_in = 32 * data.shape[-1]
                    byte = _scale_byte(1 / math.sqrt(fan_in))
                data.fill_(byte)
                rule = f"scale_byte={byte}"
            else:
                _fill_bytes(data, g, dev)
                rule = "random_e2m1_bytes"
        elif dtype == torch.float8_e4m3fn:
            _fill_normal(data, g, dev, 1.0, clamp=6.0)
            rule = "fp8_normal"
        elif torch.is_floating_point(data):
            leaf = name.rsplit(".", 1)[-1]
            if name in sink_heads:
                n = sink_heads[name] if sink_heads[name] is not None else data.numel()
                data.fill_(-float("inf"))
                data[:n] = 0.0
                rule = f"sink_0_first_{n}"
            elif "e_score_correction_bias" in leaf:
                data.zero_()
                rule = "zero"
            elif data.ndim == 1 and "norm" in name:
                data.fill_(1.0)
                rule = "ones"
            elif data.ndim >= 2 and not leaf.startswith("hc_"):
                std = 1 / math.sqrt(data.shape[-1])
                _fill_normal(data, g, dev, std, clamp=6 * std)
                rule = "normal_fan_in"
            else:
                rule = "keep_dummy"
        else:
            rule = "keep"
        entry = summary[(_pattern(name), str(dtype), rule, str(data.device.type))]
        entry[0] += 1
        entry[1] += data.numel() * data.element_size()

    torch.cuda.synchronize()
    rows = [
        {"pattern": p, "dtype": d, "rule": r, "device": dv, "count": c, "bytes": b}
        for (p, d, r, dv), (c, b) in sorted(summary.items())
    ]
    by_dev = defaultdict(int)
    for row in rows:
        by_dev[row["device"]] += row["bytes"]
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    logger.info(
        "DSV41 synthetic init done in %.1fs: %s parameter bytes, %d patterns",
        time.time() - t0,
        {k: f"{v / 1024**3:.2f} GiB" for k, v in by_dev.items()},
        len(rows),
    )
    out = os.environ.get("DSV41_SYNTH_MANIFEST")
    if out:
        path = f"{out}.rank{torch.cuda.current_device()}.json"
        with open(path, "w") as f:
            json.dump({"seed": _SEED, "rank_env": rank, "rows": rows}, f, indent=1)
