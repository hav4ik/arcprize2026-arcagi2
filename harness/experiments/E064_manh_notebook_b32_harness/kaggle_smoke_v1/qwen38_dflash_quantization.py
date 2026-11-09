# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded, in-memory backport for quantized DFlash context projections.

Projection logic adapted from vLLM (Apache-2.0), commit
73a78e6f1f38e280986b81e0f2a9aa5e1ee6fe47, PR #57934. Older DFlash code
uses F.linear on transposed/packed quantized QKV weights. Newer vLLM already
uses each layer's quantization method. Never modify installed package files.
License copy: third_party/VLLM_LICENSE.
"""

import ast
import hashlib
import inspect
import textwrap
from types import CodeType

import torch
import torch.nn.functional as F


METHODS = ("_build_context_kv_buffers", "_project_context_kv")
LEGACY_HASHES = (
    "78a8e175a949bc476fb3060034f4a8c871bf435c021c330ee256ce252c803952",
    "2a9025f42e2d2c30a95be60fcba4e4769daee1d1454bc3e257d2dc42a00083f8",
)


def method_hash(method):
    node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def code_names(code):
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(code_names(constant))
    return names


def projection_methods(ops, unquantized_linear_cls):
    """Keep fused BF16 projections; dispatch quantized weights through vLLM."""
    def build(self, layers_attn, has_bias):
        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._context_qkv_projs = [a.qkv_proj for a in layers_attn]
        self._context_q_sizes = [a.q_size for a in layers_attn]
        if all(isinstance(p.quant_method, unquantized_linear_cls)
               for p in self._context_qkv_projs):
            self._fused_kv_weight = torch.cat(
                [a.qkv_proj.weight[a.q_size:] for a in layers_attn], dim=0)
            self._fused_kv_bias = (torch.cat(
                [a.qkv_proj.bias[a.q_size:] for a in layers_attn], dim=0)
                if has_bias else None)
        else:
            self._fused_kv_weight = None
            self._fused_kv_bias = None
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0).contiguous()

    def project(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, self._hidden_norm_weight,
                     self._rms_norm_eps)
        if self._fused_kv_weight is not None:
            all_kv = F.linear(normed, self._fused_kv_weight, self._fused_kv_bias)
            all_kv = all_kv.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
        else:
            all_kv = []
            for proj, q_size in zip(self._context_qkv_projs,
                                    self._context_q_sizes, strict=True):
                qkv, _ = proj(normed)
                all_kv.append(qkv[:, q_size:].view(num_ctx, 2, num_kv_heads, head_dim))
            all_kv = torch.stack(all_kv, dim=1)
        all_kv = all_kv.permute(2, 1, 0, 3, 4).contiguous()
        return all_kv[0], all_kv[1]

    return build, project


def patch_model_class(model_class, ops, unquantized_linear_cls):
    if getattr(model_class, "_arc_quantized_context_fix", False):
        return "already_patched"
    methods = [getattr(model_class, name) for name in METHODS]
    # Upstream already has the dispatch; leave it intact, including later fixes.
    if ("UnquantizedLinearMethod" in code_names(methods[0].__code__)
            and all("_context_qkv_projs" in m.__code__.co_names for m in methods)):
        return "upstream_native"
    actual = tuple(method_hash(m) for m in methods)
    if actual != LEGACY_HASHES:
        raise RuntimeError(
            "Unrecognized vLLM DFlash context projection implementation; refusing "
            f"to override it ({actual}). Upgrade vLLM, or use an unquantized draft checkpoint "
            "with --draft-quantization auto.")
    for name, method in zip(METHODS, projection_methods(ops, unquantized_linear_cls), strict=True):
        setattr(model_class, name, method)
    model_class._arc_quantized_context_fix = True
    return "backported_57934"


def install():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    status = patch_model_class(DFlashQwen3Model, ops, UnquantizedLinearMethod)
    print(f"[ARC DFlash quantization] context projection compatibility: {status}", flush=True)
    return status
