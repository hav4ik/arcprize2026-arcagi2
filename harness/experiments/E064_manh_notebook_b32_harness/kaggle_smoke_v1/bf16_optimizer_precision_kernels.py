"""Fused FP32 candidate arithmetic and BF16 writeback; no model-wide scratch."""

import triton
import triton.language as tl


@triton.jit
def _writeback(P, U, MASTER, N: tl.constexpr, SCALE, DECAY, SEED,
               HAS_MASTER: tl.constexpr, STOCHASTIC: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    if HAS_MASTER:
        weight = tl.load(MASTER + i, valid, 0).to(tl.float32)
    else:
        weight = tl.load(P + i, valid, 0).to(tl.float32)
    update = tl.load(U + i, valid, 0).to(tl.float32)
    candidate = weight * DECAY - update * SCALE
    if HAS_MASTER:
        tl.store(MASTER + i, candidate, valid)
    if STOCHASTIC and P.dtype.element_ty == tl.bfloat16:
        bits = candidate.to(tl.uint32, bitcast=True)
        random = tl.randint(SEED, i).to(tl.uint32) & 65535
        rounded = ((bits + random) & 0xFFFF0000).to(tl.float32, bitcast=True)
        candidate = tl.where(tl.abs(candidate) < float("inf"), rounded, candidate)
    tl.store(P + i, candidate, valid)


def writeback_cuda(parameter, update, master, scale, decay, seed, *, stochastic):
    if parameter.numel() == 0:
        return
    _writeback[(triton.cdiv(parameter.numel(), 1024),)](
        parameter, update, master if master is not None else parameter,
        parameter.numel(), scale, decay, seed,
        HAS_MASTER=master is not None, STOCHASTIC=stochastic, BLOCK=1024,
        enable_fp_fusion=False,
    )
