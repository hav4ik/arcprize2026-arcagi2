#!/usr/bin/env python3
"""E042: numerical parity of the two MXFP8 dense GEMM kernels on L4: HummingMxfp8LinearKernel (submit Version 5,
`--linear-backend humming`) vs AdaMxfp8LinearKernel (v3 / Version 6). Same random MXFP8 weights (fp8 e4m3 values,
e8m0 scales per 32 along K) and the same bf16 inputs; each output is compared with an fp32 reference built from the
dequantized weights and (a) the bf16 inputs, (b) the inputs after MXFP8 quantization (Ada's contract). Run on a pod
GPU with the fork's venv: python mxfp8_dense_parity.py
"""
import copy
import itertools

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.ada import AdaMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.humming import HummingMxfp8LinearKernel
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize


def dequant(q, s):
    k = q.shape[1]
    return q.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)[:, :k]


def layer_for(wq, ws):
    layer = torch.nn.Module()
    n, k = wq.shape
    layer.weight = Parameter(wq.clone(), requires_grad=False)
    layer.weight_scale = Parameter(ws.clone(), requires_grad=False)
    layer.input_size_per_partition, layer.output_size_per_partition = k, n
    layer.input_size, layer.output_size, layer.output_partition_sizes = k, n, [n]
    layer.params_dtype, layer.bias, layer.has_bias = torch.bfloat16, None, False
    return layer


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    torch.manual_seed(0)
    cfg = Mxfp8LinearLayerConfig(model_profile="deepseek_v41")
    kernels = {"ada": AdaMxfp8LinearKernel(cfg), "humming": HummingMxfp8LinearKernel(cfg)}
    shapes = [(4096, 4096), (1536, 4096), (4096, 2048), (8192, 4096), (512, 4096)]  # (N out, K in)
    print(f"{'N x K':>11s} {'M':>5s} | rel err vs fp32(bf16 x): ada humming | vs fp32(mxfp8 x): ada humming | humming vs ada")
    worst = 0.0
    for (n, k), m in itertools.product(shapes, (1, 8, 36, 64, 2048)):
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False, min_amax=1e-4)
        wd = dequant(wq, ws)
        ref = x.float() @ wd.t()
        ref_q = dequant(xq, xs) @ wd.t()
        out = {}
        for name, kern in kernels.items():
            layer = layer_for(wq, ws)
            kern.process_weights_after_loading(layer)
            out[name] = kern.apply_weights(layer, x)
        e = [rel(out["ada"], ref), rel(out["humming"], ref), rel(out["ada"], ref_q), rel(out["humming"], ref_q),
             rel(out["humming"], out["ada"])]
        worst = max(worst, e[1])
        print(f"{n:5d}x{k:<5d} {m:5d} |            {e[0]:.2e} {e[1]:.2e} |            {e[2]:.2e} {e[3]:.2e} | {e[4]:.2e}")
    print(f"worst humming rel err vs fp32 reference: {worst:.2e}")


if __name__ == "__main__":
    main()
