# SPDX-License-Identifier: Apache-2.0
"""E054 CPU check for upstream #58678 (Engram wkv column-parallel over TP) in our engram.py.

A. Constructor switch (E128 shapes: 24 hash columns x 256 = 6144 in, 5120 x 5 = 25600 out):
   default -> ColumnParallelLinear(gather_output=True, disable_tp=use_sequence_parallel);
   DSV41_ENGRAM_WKV_TP=0 -> the old ReplicatedLinear with the old arguments.
B. Real vLLM linear classes + the production quant method on CPU: DeepseekV4FP8Config
   (V4.1, weight_block_size [32, 32], ue8m0) -> ModelOptLinearMethod MXFP8, with its real
   create_weights and weight loaders (32x32 checkpoint scales repeated per row, then the
   column-parallel narrow). Only the GEMM kernel is replaced (no MXFP8 kernel exists on
   CPU) by an exact float64 dequant reference. Checks, for TP 2 / 4 / 8:
   - rank r holds rows [r N/tp, (r+1) N/tp) of the replicated weight and per-row scale,
     bit for bit (so the shards partition the replicated parameters exactly);
   - the base communicator's dim=-1 all_gather (concat-style gather, movedim, reshape)
     of the rank outputs equals the replicated output bit for bit, i.e. the gather puts
     each rank's columns back in rank order;
   - per-rank parameter bytes are 1/tp of the replicated ones.
   What B cannot show on CPU: that the GPU GEMM (Humming MXFP8 on SM89) gives the same
   bits for an N/tp-column problem as for the N-column one. gpu/engram_wkv_ab.py checks
   that on the real E128 weights.
"""

import os
import unittest
from types import SimpleNamespace

import cpu_harness  # noqa: F401  (must precede torch/vllm imports)
import torch

import vllm.model_executor.layers.linear as lin
import vllm.model_executor.layers.quantization.modelopt as mo
import vllm.model_executor.parameter as vp
import vllm.models.deepseek_v4_1.common.engram as eg
from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config

assert eg.__file__.endswith("E054_exact_cherrypicks/overlay/vllm/models/deepseek_v4_1/common/engram.py"), eg.__file__

E2M1_DIM, HC, N_HASH, HEAD_DIM = 5120, 4, 24, 256


class Patch:
    def __init__(self, obj, **attrs):
        self.obj, self.attrs = obj, attrs

    def __enter__(self):
        self.saved = {k: getattr(self.obj, k) for k in self.attrs}
        for k, v in self.attrs.items():
            setattr(self.obj, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.obj, k, v)


class Env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ConstructorSwitch(unittest.TestCase):
    def build(self, sp):
        calls = []

        def rec(kind):
            def make(*args, **kwargs):
                calls.append((kind, args, kwargs))
                return torch.nn.Identity()
            return make

        class FakeEmbedding(torch.nn.Module):
            def __init__(self, *a, **k):
                super().__init__()
                self.part_n_hash_cols, self.n_hash_cols, self.tp_size = N_HASH // 4, N_HASH, 4

        vcfg = SimpleNamespace(engram_config=None,
                               scheduler_config=SimpleNamespace(max_num_batched_tokens=16))
        config = SimpleNamespace(hidden_size=E2M1_DIM, hc_mult=HC, rms_norm_eps=1e-6)
        layout = SimpleNamespace(num_embeddings=[1000], head_dim=HEAD_DIM,
                                 primes=[[[7] * 12, [11] * 12]], max_ngram_size=3, n_heads=12)
        qc = object()
        with Patch(eg, ColumnParallelLinear=rec("col"), ReplicatedLinear=rec("rep"),
                   ParallelEngramEmbedding=FakeEmbedding, get_current_vllm_config=lambda: vcfg):
            eg.Engram(config, qc, layout, 0, use_sequence_parallel=sp, prefix="model.layers.1.engram")
        self.assertEqual(len(calls), 1)
        kind, args, kwargs = calls[0]
        self.assertEqual(args, (N_HASH * HEAD_DIM, E2M1_DIM * (HC + 1)))  # 6144 -> 25600
        self.assertIs(kwargs["quant_config"], qc)
        self.assertEqual(kwargs["prefix"], "model.layers.1.engram.wkv")
        self.assertIs(kwargs["bias"], False)
        self.assertIs(kwargs["return_bias"], False)
        return kind, kwargs

    def test_default_is_column_parallel(self):
        with Env(DSV41_ENGRAM_WKV_TP=None):
            kind, kw = self.build(sp=False)
        self.assertEqual(kind, "col")
        self.assertIs(kw["gather_output"], True)
        self.assertIs(kw["disable_tp"], False)

    def test_sequence_parallel_keeps_full_projection(self):
        with Env(DSV41_ENGRAM_WKV_TP="1"):
            kind, kw = self.build(sp=True)
        self.assertEqual(kind, "col")
        self.assertIs(kw["disable_tp"], True)

    def test_kill_switch_restores_replicated(self):
        with Env(DSV41_ENGRAM_WKV_TP="0"):
            kind, kw = self.build(sp=False)
        self.assertEqual(kind, "rep")
        self.assertEqual(set(kw), {"bias", "quant_config", "return_bias", "prefix"})


class CpuMxfp8Kernel:
    """Exact CPU stand-in for the MXFP8 GEMM: dequantize (fp8 e4m3 x 2^(e8m0 - 127) per
    32 input columns) and multiply in float64. Every output column is an independent
    float64 dot product, so any difference below is a loader / ordering bug."""

    def process_weights_after_loading(self, layer):
        pass

    def apply_weights(self, layer, x, bias=None):
        w = layer.weight.data.view(torch.float8_e4m3fn).to(torch.float64)
        s = torch.pow(2.0, layer.weight_scale.data.view(torch.uint8).to(torch.float64) - 127.0)
        w = w * s.repeat_interleave(32, dim=1)
        return (x.to(torch.float64) @ w.T).to(torch.float32)


def gather_dim_last(outs):
    """vLLM's DeviceCommunicatorBase.all_gather(dim=-1) with all_gather_into_tensor(dim 0)."""
    world = len(outs)
    inp = outs[0]
    dim = -1 + inp.dim()  # all_gather converts a negative dim first
    out = torch.cat(outs, dim=0)  # what all_gather_into_tensor writes, rank-major
    out = out.reshape((world,) + inp.size()).movedim(0, dim)
    return out.reshape(inp.size()[:dim] + (world * inp.size(dim),) + inp.size()[dim + 1:])


class ColumnParallelMxfp8(unittest.TestCase):
    K, N, T = 256, 1280, 7  # N / 32 = 40 checkpoint scale blocks; shards stay block-aligned

    def setUp(self):
        self.qc = DeepseekV4FP8Config.from_config({
            "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0", "expert_dtype": "fp4",
        })
        g = torch.Generator().manual_seed(58678)
        raw = torch.randint(0, 256, (self.N, self.K), dtype=torch.uint8, generator=g)
        raw[(raw & 0x7F) == 0x7F] = 0x3C  # no e4m3 NaN codes
        self.w = raw.view(torch.float8_e4m3fn)
        self.s = torch.randint(118, 131, (self.N // 32, self.K // 32), dtype=torch.uint8, generator=g)
        self.x = torch.randn(self.T, self.K, generator=g).to(torch.bfloat16)

    def make(self, tp, rank):
        prefix = "model.layers.1.engram.wkv"
        with Patch(mo, select_linear_kernel=lambda *a, **k: CpuMxfp8Kernel()), \
             Patch(vp, get_tensor_model_parallel_rank=lambda: 0,
                   get_tensor_model_parallel_world_size=lambda: 1), \
             Patch(lin, get_tensor_model_parallel_rank=lambda: 0,
                   get_tensor_model_parallel_world_size=lambda: 1), \
             Patch(mo, expose_input_quant_key=lambda *a, **k: None):
            if tp is None:
                layer = lin.ReplicatedLinear(self.K, self.N, bias=False, quant_config=self.qc,
                                             return_bias=False, prefix=prefix)
            else:
                layer = lin.ColumnParallelLinear(self.K, self.N, bias=False, gather_output=True,
                                                 quant_config=self.qc, return_bias=False,
                                                 prefix=prefix, tp_rank=rank, tp_size=tp)
        self.assertIsInstance(layer.quant_method, mo.ModelOptLinearMethod)
        params = dict(layer.named_parameters())
        self.assertEqual(set(params), {"weight", "weight_scale"})
        params["weight"].weight_loader(params["weight"], self.w)
        params["weight_scale"].weight_loader(params["weight_scale"], self.s)
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_shards_partition_replicated_and_gather_restores_output(self):
        full = self.make(None, 0)
        self.assertEqual(tuple(full.weight.shape), (self.N, self.K))
        self.assertEqual(tuple(full.weight_scale.shape), (self.N, self.K // 32))
        ref = full.quant_method.apply(full, self.x)
        for tp in (2, 4, 8):
            shards = [self.make(tp, r) for r in range(tp)]
            n = self.N // tp
            for r, sh in enumerate(shards):
                self.assertEqual(tuple(sh.weight.shape), (n, self.K))
                self.assertTrue(torch.equal(sh.weight.data.view(torch.uint8),
                                            full.weight.data.view(torch.uint8)[r * n:(r + 1) * n]))
                self.assertTrue(torch.equal(sh.weight_scale.data.view(torch.uint8),
                                            full.weight_scale.data.view(torch.uint8)[r * n:(r + 1) * n]))
            got = gather_dim_last([sh.quant_method.apply(sh, self.x) for sh in shards])
            self.assertTrue(torch.equal(got.view(torch.int32), ref.view(torch.int32)), f"tp={tp}")
            per_rank = sum(p.numel() * p.element_size() for p in shards[0].parameters())
            total = sum(p.numel() * p.element_size() for p in full.parameters())
            self.assertEqual(per_rank * tp, total)

    def test_e128_memory_arithmetic(self):
        # E128: 2 Engram layers (1, 14), wkv 6144 -> 25600 MXFP8 + per-row-per-32 e8m0 scales.
        per_layer = 25600 * 6144 + 25600 * (6144 // 32)
        self.assertEqual(2 * 25600 * 6144, 314_572_800)  # the synth-init record per GPU
        self.assertEqual(2 * 25600 * (6144 // 32), 9_830_400)
        saved = 2 * per_layer * 3 // 4
        self.assertEqual(saved, 243_302_400)  # 0.2266 GiB per GPU at TP4


if __name__ == "__main__":
    unittest.main(verbosity=2)
