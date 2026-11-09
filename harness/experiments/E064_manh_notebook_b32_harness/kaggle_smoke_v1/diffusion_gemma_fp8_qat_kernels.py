"""E4M3 quantization kernels for fake-QAT and real FP8 GEMM inputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_fp8(X, S, Q, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    matrix = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + matrix * SIZE + offsets, offsets < SIZE, other=0).to(tl.float32)
    scale = tl.load(S + matrix)
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + matrix * SIZE + offsets, q, offsets < SIZE)


def quantize_e4m3_tensor(x, scales):
    size = x.shape[-2] * x.shape[-1]
    output = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    _quantize_fp8[(triton.cdiv(size, 4096), x.numel() // size)](
        x, scales, output, size, 4096
    )
    return output


@triton.jit
def _tensor_qdq(X, Amax, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    matrix = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + matrix * SIZE + offsets, offsets < SIZE, other=0).to(tl.float32)
    amax = tl.load(Amax + matrix).to(tl.float32)
    scale = tl.where(amax > 0, tl.div_rn(448.0, amax), 1.0)
    inverse = tl.div_rn(1.0, scale)
    quantized = tl.minimum(tl.maximum(x * scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Y + matrix * SIZE + offsets, quantized.to(tl.float32) * inverse, offsets < SIZE)


@triton.jit
def _block_qdq(X, Y, ROWS: tl.constexpr, COLUMNS: tl.constexpr):
    rows = tl.program_id(0) * 128 + tl.arange(0, 128)
    columns = tl.program_id(1) * 128 + tl.arange(0, 128)
    offsets = tl.program_id(2) * ROWS * COLUMNS + rows[:, None] * COLUMNS + columns[None, :]
    valid = (rows[:, None] < ROWS) & (columns[None, :] < COLUMNS)
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(x), 0), 0)
    scale = tl.where(amax > 0, tl.div_rn(448.0, amax), 1.0)
    scale = tl.exp2(tl.floor(tl.log2(scale)))
    inverse = tl.div_rn(1.0, scale)
    quantized = tl.minimum(tl.maximum(x * scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Y + offsets, quantized.to(tl.float32) * inverse, valid)


def quantize_dequantize_weight(weight, scaling):
    weight = weight.detach().contiguous()
    rows, columns = weight.shape[-2:]
    matrices = weight.numel() // (rows * columns)
    output = torch.empty_like(weight)
    if scaling == "tensor":
        # BF16 absolute/max is exact for BF16 inputs; only the tiny scale vector
        # is promoted to FP32. Avoid an FP32 copy of a complete expert tensor.
        amax = weight.view(matrices, -1).abs().amax(dim=1).float()
        _tensor_qdq[(triton.cdiv(rows * columns, 4096), matrices)](
            weight, amax, output, rows * columns, 4096
        )
    elif scaling == "block128":
        _block_qdq[(triton.cdiv(rows, 128), triton.cdiv(columns, 128), matrices)](
            weight, output, rows, columns, num_warps=8
        )
    else:
        raise ValueError(f"unknown FP8 QAT scaling: {scaling}")
    return output
