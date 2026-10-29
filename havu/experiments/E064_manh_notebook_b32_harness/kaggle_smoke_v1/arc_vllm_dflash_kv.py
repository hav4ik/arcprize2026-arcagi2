# SPDX-License-Identifier: Apache-2.0
"""In-memory compatibility for independent DFlash2 KV dtype on vLLM 4def927fe.

V2 constructs target and draft metadata builders in one pass, but passes the
target cache dtype to both. Its draft-only pass also inherits the target dtype.
Patch only these known sources; never edit site-packages or change target KV.
The guarded source is vLLM (Apache-2.0); see third_party/VLLM_LICENSE.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap


INIT_HASH = "f2de63aa317eea777e18367f1483b8a0cdfc615827c6c9ca145145bc9fe4df2b"
PROPERTY_HASH = "1e8ea9b865023a1502209ac8e184e7b4721fe9db0d1e73d4725c22dadc6ea1b5"


def source_hash(fn):
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def draft_cache_config(config):
    spec = config.speculative_config
    dtype = getattr(spec, "kv_cache_dtype", None)
    if dtype is None or dtype == config.cache_config.cache_dtype:
        return config
    result = copy.copy(config)
    # dataclasses.replace resets init=False fields: in particular it loses the
    # physical KV layout already resolved before CUDA-graph profiling.
    result.cache_config = copy.copy(config.cache_config)
    result.cache_config.cache_dtype = dtype
    return result


def group_cache_config(config, layer_names, draft_layer_names):
    draft = set(draft_layer_names or ())
    names = set(layer_names)
    if not names.intersection(draft):
        return config
    if not names.issubset(draft):
        raise RuntimeError("DFlash2 target/draft attention metadata must be separate")
    return draft_cache_config(config)


def patched_init_source(source):
    """Minimal edits to upstream discovery; all allocation/workspace code stays."""
    replacements = (
        ("key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)",
         "key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q, "
         "layer_name in (draft_layer_names or ()))"),
        ("vllm_config=vllm_config,\n                device=device,",
         "vllm_config=_arc_group_cache_config(\n"
         "                    vllm_config, group.layer_names, draft_layer_names),\n"
         "                device=device,"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Unrecognized DFlash2 mixed-KV metadata source")
        source = source.replace(old, new)
    return source


def install():
    from vllm.v1.worker.gpu import attn_utils, model_runner
    from vllm.v1.worker.gpu.spec_decode import speculator

    cls = speculator.DraftModelSpeculator
    if getattr(cls, "_arc_independent_draft_kv", False):
        return "already_patched"
    fn = attn_utils.init_attn_backend
    getter = cls.attn_vllm_config.fget
    actual = source_hash(fn), source_hash(getter)
    if actual != (INIT_HASH, PROPERTY_HASH):
        raise RuntimeError(
            "Unrecognized vLLM DFlash2 mixed-KV implementation; refusing to "
            f"override it ({actual}). Use --vllm-draft-kv-cache-dtype auto "
            "until this runtime is validated.")
    source = patched_init_source(textwrap.dedent(inspect.getsource(fn)))
    namespace = dict(fn.__globals__, _arc_group_cache_config=group_cache_config)
    exec(compile(source, fn.__code__.co_filename + ":arc_draft_kv", "exec"), namespace)
    patched = namespace[fn.__name__]
    # Both callers imported the function by name before worker initialization.
    attn_utils.init_attn_backend = patched
    model_runner.init_attn_backend = patched
    speculator.init_attn_backend = patched

    def attn_config(self):
        return draft_cache_config(getter(self))

    cls.attn_vllm_config = property(attn_config)
    cls._arc_independent_draft_kv = True
    print("[ARC DFlash2] independent draft-KV metadata compatibility enabled; "
          "target cache unchanged", flush=True)
    return "patched_4def927fe"
