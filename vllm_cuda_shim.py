"""In-process Google3 vLLM CUDA H100 Shim for Qwen3.5 / GatedDeltaNet.

Bridges missing compiled C++/CUDA extensions and Copybara-excluded files in
//third_party/py/vllm so it runs natively on H100 GPUs in Google3:
  1. Restores vllm.platforms.cuda.CudaPlatform (excluded by copy.bara.sky)
  2. Bridges vllm.vllm_flash_attn to //third_party/py/flash_attn
  3. Bridges vllm.third_party.flash_linear_attention to vllm_fla_triton_ops
  4. Replaces mocked torch.ops._C / _C_cache_ops calls with PyTorch/Triton ops
  5. Enables in-process model registry inspection (_run_in_subprocess bypass)
"""

from __future__ import annotations

from datetime import timedelta
import importlib.metadata
import importlib.util
import os
import sys
import tempfile
import types
from typing import Any

# Redirect Triton, TorchInductor, and HuggingFace cache directories to /tmp
# BEFORE importing torch or triton (which captures default_cache_dir() at import
# time; on Borg containers, $HOME = /user/<ldap> is read-only).
_TMP_BASE = tempfile.gettempdir()
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_TMP_BASE, "triton_cache"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.join(_TMP_BASE, "torchinductor_cache")
)
os.environ.setdefault("HF_HOME", os.path.join(_TMP_BASE, "hf_home"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_TMP_BASE, "xdg_cache"))

import torch
import torch.nn.functional as F

try:
  from google3.experimental.users.arazdai.kaggle.arc_agi_2 import vllm_fla_triton_ops
except ImportError:
  from google3.experimental.users.arazdai.kaggle import vllm_fla_triton_ops

_SHIM_INSTALLED = False


def _patch_metadata_versions_early() -> None:
  """Mocks importlib.metadata.version before any vllm/transformers imports."""
  try:
    from google3.research.vision.huggingface import google3_version

    google3_version.MonkeyPatchVersion()
  except Exception:
    pass

  orig_version = importlib.metadata.version

  def _mock_version(name: str) -> str:
    if name == "transformers":
      return "5.5.3"
    if name == "vllm":
      return "0.24.0"
    if name in ("mistral_common", "mistral-common"):
      return "9.9.9"
    try:
      res = orig_version(name)
      return res if res is not None else "9.9.9"
    except Exception:
      return "9.9.9"

  importlib.metadata.version = _mock_version
  if "vllm._version" not in sys.modules:
    ver_mod = types.ModuleType("vllm._version")
    setattr(ver_mod, "version", "0.24.0")
    setattr(ver_mod, "__version__", "0.24.0")
    setattr(ver_mod, "__version_tuple__", (0, 24, 0))
    setattr(ver_mod, "commit_id", "google3")
    sys.modules["vllm._version"] = ver_mod


def _install_cuda_platform() -> None:
  """Restores vllm.platforms.cuda.CudaPlatform (excluded in //third_party/py/vllm/copy.bara.sky)."""
  import vllm
  vllm_platforms = importlib.import_module("vllm.platforms")
  setattr(vllm, "platforms", vllm_platforms)
  from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

  # Disable tpu_platform_plugin BEFORE importing AttentionBackendEnum (which
  # accesses vllm.platforms.current_platform and would otherwise trigger
  # tpu_platform_plugin() -> import tpu_inference -> import torch_xla2).
  vllm_platforms.tpu_platform_plugin = lambda: None
  if hasattr(vllm_platforms, "builtin_platform_plugins"):
    vllm_platforms.builtin_platform_plugins["tpu"] = lambda: None
    vllm_platforms.builtin_platform_plugins["cuda"] = (
        lambda: "vllm.platforms.cuda.CudaPlatform"
    )

  # Stub vllm.platforms.rocm (excluded by Copybara) so functions like
  # fp8_utils.w8a8_triton_block_scaled_mm (`from vllm.platforms.rocm import on_gfx1250`)
  # work cleanly on CUDA.
  class _RocmStubModule(types.ModuleType):

    def __getattr__(self, name: str) -> Any:
      if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
      if name.startswith("_ON_"):
        return False
      if name == "get_cdna_version":
        return lambda *a, **k: 0
      return lambda *a, **k: False

  rocm_stub = _RocmStubModule("vllm.platforms.rocm")
  rocm_stub.__file__ = "/dev/null"
  sys.modules["vllm.platforms.rocm"] = rocm_stub
  setattr(vllm_platforms, "rocm", rocm_stub)

  from vllm.v1.attention.backends.registry import AttentionBackendEnum

  torch.backends.cuda.enable_cudnn_sdp(False)

  class Google3CudaPlatform(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ]

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
      return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
      torch.cuda.set_device(device)
      _ = torch.zeros(1, device=device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
      torch.cuda.manual_seed_all(seed)

    @classmethod
    def get_device_capability(
        cls, device_id: int = 0
    ) -> DeviceCapability | None:
      if not torch.cuda.is_available():
        return DeviceCapability(major=9, minor=0)
      major, minor = torch.cuda.get_device_capability(device_id)
      return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
      if not torch.cuda.is_available():
        return "NVIDIA H100"
      return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
      if not torch.cuda.is_available():
        return f"cuda:{device_id}"
      props = torch.cuda.get_device_properties(device_id)
      return str(getattr(props, "uuid", f"cuda:{device_id}"))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
      return int(torch.cuda.get_device_properties(device_id).total_memory)

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
      del device_ids
      return True

    @classmethod
    def log_warnings(cls) -> None:
      pass

    @classmethod
    def check_and_update_config(cls, vllm_config: Any) -> None:
      parallel_config = vllm_config.parallel_config
      if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats(device)
      return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> str:
      del attn_selector_config, num_heads
      if selected_backend is not None:
        return selected_backend.get_path()
      return AttentionBackendEnum.TRITON_ATTN.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list[AttentionBackendEnum]:
      return [
          AttentionBackendEnum.TORCH_SDPA,
          AttentionBackendEnum.FLASH_ATTN,
          AttentionBackendEnum.TRITON_ATTN,
      ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: AttentionBackendEnum | None = None,
    ) -> AttentionBackendEnum:
      del head_size, dtype
      if backend is not None:
        return backend
      return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
      return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
      return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"

    @classmethod
    def supports_fp8(cls) -> bool:
      return cls.has_device_capability(89)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
      return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
      return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
      return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: Any,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> Any:
      del backend
      from torch.distributed import ProcessGroup
      from torch.distributed.distributed_c10d import ProcessGroupNCCL

      pg = ProcessGroup(prefix_store, group_rank, group_size)
      opts = ProcessGroupNCCL.Options()
      opts._timeout = timeout
      backend_cls = ProcessGroupNCCL(prefix_store, group_rank, group_size, opts)
      backend_type = ProcessGroup.BackendType.NCCL
      pg._set_default_backend(backend_type)
      backend_cls._set_sequence_number_for_group()
      pg._register_backend(torch.device("cuda"), backend_type, backend_cls)
      return pg

    @classmethod
    def device_count(cls) -> int:
      return torch.cuda.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
      del dtype

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
      _src_cache = src_cache[:, src_block_indices]
      dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
      _src_cache = src_cache[:, src_block_indices]
      dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
      return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
      return True

    @classmethod
    def support_deep_gemm(cls) -> bool:
      return False

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
      return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
      return False

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int) -> int:
      return device_id

    @classmethod
    def logical_device_id_to_visible_device_id(cls, device_id: int) -> int:
      return device_id

    @classmethod
    def visible_device_id_to_physical_device_id(cls, device_id: int) -> int:
      return device_id

  cuda_mod = types.ModuleType("vllm.platforms.cuda")
  cuda_mod.CudaPlatform = Google3CudaPlatform
  cuda_mod.NvmlCudaPlatform = Google3CudaPlatform
  cuda_mod.NonNvmlCudaPlatform = Google3CudaPlatform
  sys.modules["vllm.platforms.cuda"] = cuda_mod
  vllm.platforms.cuda = cuda_mod

  cuda_platform = Google3CudaPlatform()
  vllm.platforms.current_platform = cuda_platform

  # Update any already-imported vllm submodules holding UnspecifiedPlatform or TpuPlatform
  for mod_name, mod in list(sys.modules.items()):
    if (
        mod is not None
        and mod_name.startswith("vllm")
        and getattr(mod, "current_platform", None) is not None
    ):
      try:
        setattr(mod, "current_platform", cuda_platform)
      except Exception:
        pass


def _install_vllm_flash_attn_shim() -> None:
  """Bridges vllm.vllm_flash_attn to Google3's compiled //third_party/py/flash_attn."""
  from flash_attn import flash_attn_interface as fa_iface

  def _apply_rotary_emb(
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      interleaved: bool = False,
      inplace: bool = False,
      seqlen_offsets: int | torch.Tensor = 0,
      cu_seqlens: torch.Tensor | None = None,
      max_seqlen: int | None = None,
  ) -> torch.Tensor:
    del seqlen_offsets, cu_seqlens, max_seqlen
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    res = ApplyRotaryEmb.forward_static(
        x, cos, sin, is_neox_style=not interleaved
    )
    if inplace:
      x.copy_(res)
      return x
    return res

  def _unpage_kv_cache(
      k: torch.Tensor,
      v: torch.Tensor,
      block_table: torch.Tensor,
      seqused_k: torch.Tensor,
      max_seqlen_k: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpages [num_blocks, block_size, num_kv_heads, head_size] KV cache via block_table."""
    num_seqs = block_table.shape[0]
    block_size = k.shape[1]
    max_k_blocks = min(
        block_table.shape[1],
        max(1, (int(max_seqlen_k) + block_size - 1) // block_size),
    )
    b_ids = (
        block_table[:, :max_k_blocks]
        .to(torch.int64)
        .unsqueeze(-1)
        .expand(-1, -1, block_size)
        .reshape(num_seqs, -1)
    )
    o_ids = (
        torch.arange(block_size, device=k.device, dtype=torch.int64)
        .view(1, 1, block_size)
        .expand(num_seqs, max_k_blocks, -1)
        .reshape(num_seqs, -1)
    )
    if num_seqs == 1:
      seqlen_k = int(max_seqlen_k)
      valid_b = b_ids[0, :seqlen_k]
      valid_o = o_ids[0, :seqlen_k]
    else:
      t_idx = torch.arange(
          max_k_blocks * block_size, device=k.device, dtype=seqused_k.dtype
      ).unsqueeze(0)
      valid_mask = t_idx < seqused_k.unsqueeze(1)
      valid_b = b_ids[valid_mask]
      valid_o = o_ids[valid_mask]
    k_unpaged = k[valid_b, valid_o]
    v_unpaged = v[valid_b, valid_o]
    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=k.device)
    torch.cumsum(seqused_k, dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])
    return k_unpaged, v_unpaged, cu_seqlens_k

  def _wrapped_flash_attn_varlen_func(
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      max_seqlen_q: int,
      cu_seqlens_q: torch.Tensor,
      max_seqlen_k: int,
      cu_seqlens_k: torch.Tensor | None = None,
      seqused_q: torch.Tensor | None = None,
      seqused_k: torch.Tensor | None = None,
      dropout_p: float = 0.0,
      softmax_scale: float | None = None,
      causal: bool = False,
      window_size: tuple[int, int] | list[int] = (-1, -1),
      softcap: float = 0.0,
      alibi_slopes: torch.Tensor | None = None,
      deterministic: bool = False,
      return_attn_probs: bool = False,
      block_table: torch.Tensor | None = None,
      return_softmax_lse: bool = False,
      out: torch.Tensor | None = None,
      scheduler_metadata: Any = None,
      q_descale: Any = None,
      k_descale: Any = None,
      v_descale: Any = None,
      fa_version: int = 2,
      s_aux: Any = None,
      num_par_softmax_segments: int = 1,
      **kwargs: Any,
  ) -> Any:
    del (
        seqused_q,
        scheduler_metadata,
        q_descale,
        k_descale,
        v_descale,
        fa_version,
        s_aux,
        num_par_softmax_segments,
        kwargs,
    )
    if isinstance(window_size, list):
      window_size = (window_size[0], window_size[1])
    if (
        block_table is not None
        and seqused_k is not None
        and k.ndim == 4
        and k.shape[1] % 256 != 0
    ):
      k, v, cu_seqlens_k = _unpage_kv_cache(
          k, v, block_table, seqused_k, max_seqlen_k
      )
      block_table = None
    if cu_seqlens_k is None:
      cu_seqlens_k = cu_seqlens_q
    if q.shape[0] == 0 or k.shape[0] == 0:
      if out is not None:
        out.zero_()
        return (out, None) if return_softmax_lse else out
      zeros = torch.zeros_like(q)
      return (zeros, None) if return_softmax_lse else zeros
    res = fa_iface.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=bool(causal) if not isinstance(causal, torch.Tensor) else True,
        window_size=window_size,
        softcap=float(softcap) if softcap else 0.0,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs or return_softmax_lse,
        block_table=block_table,
    )
    if return_softmax_lse and isinstance(res, tuple):
      attn_out, lse = res[0], res[1]
      if out is not None:
        out.copy_(attn_out.view_as(out))
        return out, lse
      return attn_out, lse
    attn_out = res[0] if isinstance(res, tuple) else res
    if out is not None:
      out.copy_(attn_out.view_as(out))
      return out
    return attn_out

  def _wrapped_flash_attn_with_kvcache(
      q: torch.Tensor,
      k_cache: torch.Tensor,
      v_cache: torch.Tensor,
      k: torch.Tensor | None = None,
      v: torch.Tensor | None = None,
      rotary_cos: torch.Tensor | None = None,
      rotary_sin: torch.Tensor | None = None,
      cache_seqlens: int | torch.Tensor | None = None,
      cache_batch_idx: torch.Tensor | None = None,
      cache_leftpad: torch.Tensor | None = None,
      block_table: torch.Tensor | None = None,
      alibi_slopes: torch.Tensor | None = None,
      out: torch.Tensor | None = None,
      softmax_scale: float | None = None,
      causal: bool = False,
      window_size: tuple[int, int] | list[int] = (-1, -1),
      softcap: float = 0.0,
      rotary_interleaved: bool = True,
      limit_splits: int = 0,
      num_splits: int = 0,
      return_softmax_lse: bool = False,
      scheduler_metadata: Any = None,
      q_descale: Any = None,
      k_descale: Any = None,
      v_descale: Any = None,
      fa_version: int = 2,
      **kwargs: Any,
  ) -> Any:
    del (
        limit_splits,
        scheduler_metadata,
        q_descale,
        k_descale,
        v_descale,
        fa_version,
        kwargs,
    )
    if isinstance(window_size, list):
      window_size = (window_size[0], window_size[1])
    res = fa_iface.flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        block_table=block_table,
        alibi_slopes=alibi_slopes,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
        return_softmax_lse=return_softmax_lse,
    )
    if out is not None:
      if return_softmax_lse and isinstance(res, tuple):
        out.copy_(res[0])
        return out, res[1]
      out.copy_(res)
      return out
    return res

  fa_pkg = types.ModuleType("vllm.vllm_flash_attn")
  fa_pkg.__path__ = []
  fa_iface_mod = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
  fa_layers_pkg = types.ModuleType("vllm.vllm_flash_attn.layers")
  fa_layers_pkg.__path__ = []
  fa_rotary_mod = types.ModuleType("vllm.vllm_flash_attn.layers.rotary")

  attrs = {
      "FA2_AVAILABLE": True,
      "FA3_AVAILABLE": False,
      "FA4_AVAILABLE": False,
      "DEFAULT_FA_VERSION": 2,
      "flash_attn_varlen_func": _wrapped_flash_attn_varlen_func,
      "flash_attn_with_kvcache": _wrapped_flash_attn_with_kvcache,
      "get_scheduler_metadata": lambda *a, **kw: None,
      "is_fa_version_supported": lambda v, *a, **kw: v == 2,
      "fa_version_unsupported_reason": (
          lambda v, *a, **kw: None if v == 2 else "Only FA2 is built in Google3"
      ),
      "sparse_attn_func": getattr(fa_iface, "sparse_attn_func", None),
      "sparse_attn_varlen_func": getattr(
          fa_iface, "sparse_attn_varlen_func", None
      ),
      "compile_flash_attn_varlen_func_from_specs": None,
  }
  for k_name, val in attrs.items():
    setattr(fa_pkg, k_name, val)
    setattr(fa_iface_mod, k_name, val)

  fa_rotary_mod.apply_rotary_emb = _apply_rotary_emb
  fa_layers_pkg.rotary = fa_rotary_mod
  fa_pkg.flash_attn_interface = fa_iface_mod
  fa_pkg.layers = fa_layers_pkg

  sys.modules["vllm.vllm_flash_attn"] = fa_pkg
  sys.modules["vllm.vllm_flash_attn.flash_attn_interface"] = fa_iface_mod
  sys.modules["vllm.vllm_flash_attn.layers"] = fa_layers_pkg
  sys.modules["vllm.vllm_flash_attn.layers.rotary"] = fa_rotary_mod

  import vllm

  vllm.vllm_flash_attn = fa_pkg

  # Patch kernel_unified_attention so non-power-of-2 num_queries_per_kv (e.g. 6
  # in Qwen3.8-27B) does not emit v2i16 umul_lohi in LLVM NVPTX backend, while
  # keeping zero-copy in-place paged KV cache attention on H100!
  import vllm.v1.attention.ops.triton_unified_attention as triton_ua_mod

  ua_src_path = triton_ua_mod.__file__
  with open(ua_src_path, "r", encoding="utf-8") as f:
    ua_src = f.read()
  patched_ua_src = ua_src.replace(
      "    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv\n"
      "\n"
      "    query_offset_0 = cur_batch_in_all_start_index + query_pos\n"
      "    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv",
      "    _q_sub_pos = ((offs_m.to(tl.float32) + 0.5) / num_queries_per_kv).to(tl.int32)\n"
      "    _q_sub_head = offs_m - _q_sub_pos * num_queries_per_kv\n"
      "    query_pos = q_block_local_idx * BLOCK_Q + _q_sub_pos\n"
      "\n"
      "    query_offset_0 = cur_batch_in_all_start_index + query_pos\n"
      "    query_offset_1 = kv_head_idx * num_queries_per_kv + _q_sub_head",
  ).replace(
      "    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)",
      "    query_mask_0 = tl.where((query_pos < cur_batch_query_len) & (_q_sub_pos < BLOCK_Q), 1, 0).to(tl.int1)",
  )
  triton_cache_dir = os.environ.get("TRITON_CACHE_DIR", "/tmp/triton_cache")
  os.makedirs(triton_cache_dir, exist_ok=True)
  patched_ua_file = os.path.join(
      triton_cache_dir, f"patched_triton_ua_{os.getpid()}.py"
  )
  with open(patched_ua_file, "w", encoding="utf-8") as f:
    f.write(patched_ua_src)
  spec = importlib.util.spec_from_file_location(
      "vllm.v1.attention.ops._patched_triton_ua", patched_ua_file
  )
  if spec is not None and spec.loader is not None:
    patched_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = patched_mod
    spec.loader.exec_module(patched_mod)
    triton_ua_mod.kernel_unified_attention = (
        patched_mod.kernel_unified_attention
    )
    triton_ua_mod.unified_attention = patched_mod.unified_attention
    import vllm.v1.attention.backends.triton_attn as triton_attn_mod

    triton_attn_mod.unified_attention = patched_mod.unified_attention



def _install_fla_triton_modules() -> None:
  """Registers vllm.third_party.flash_linear_attention modules backed by vllm_fla_triton_ops."""
  tp_pkg = types.ModuleType("vllm.third_party")
  tp_pkg.__path__ = []
  fla_pkg = types.ModuleType("vllm.third_party.flash_linear_attention")
  fla_pkg.__path__ = []
  ops_pkg = types.ModuleType("vllm.third_party.flash_linear_attention.ops")
  ops_pkg.__path__ = []

  utils_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.utils"
  )
  utils_mod.FLA_CHUNK_SIZE = vllm_fla_triton_ops.FLA_CHUNK_SIZE
  utils_mod.tensor_cache = vllm_fla_triton_ops.tensor_cache
  utils_mod.input_guard = vllm_fla_triton_ops.input_guard

  index_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.index"
  )
  index_mod.prepare_lens = vllm_fla_triton_ops.prepare_lens
  index_mod.prepare_chunk_indices = vllm_fla_triton_ops.prepare_chunk_indices
  index_mod.prepare_chunk_offsets = vllm_fla_triton_ops.prepare_chunk_offsets

  chunk_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.chunk"
  )
  chunk_mod.chunk_gated_delta_rule = vllm_fla_triton_ops.chunk_gated_delta_rule
  chunk_mod.l2norm_fwd = vllm_fla_triton_ops.l2norm_fwd

  post_conv_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv"
  )
  post_conv_mod.fused_post_conv_prep = vllm_fla_triton_ops.fused_post_conv_prep

  sig_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating"
  )
  sig_mod.fused_sigmoid_gating_delta_rule_update = (
      vllm_fla_triton_ops.fused_sigmoid_gating_delta_rule_update
  )

  rec_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
  )
  rec_mod.fused_recurrent_gated_delta_rule_packed_decode = (
      vllm_fla_triton_ops.fused_recurrent_gated_delta_rule_packed_decode
  )

  ln_mod = types.ModuleType(
      "vllm.third_party.flash_linear_attention.ops.layernorm_guard"
  )
  ln_mod.rmsnorm_fn = vllm_fla_triton_ops.rmsnorm_fn
  ln_mod.layernorm_fn = vllm_fla_triton_ops.layernorm_fn

  ops_pkg.chunk_gated_delta_rule = vllm_fla_triton_ops.chunk_gated_delta_rule
  ops_pkg.fused_post_conv_prep = vllm_fla_triton_ops.fused_post_conv_prep
  ops_pkg.fused_recurrent_gated_delta_rule_packed_decode = (
      vllm_fla_triton_ops.fused_recurrent_gated_delta_rule_packed_decode
  )
  ops_pkg.fused_sigmoid_gating_delta_rule_update = (
      vllm_fla_triton_ops.fused_sigmoid_gating_delta_rule_update
  )
  ops_pkg.l2norm_fwd = vllm_fla_triton_ops.l2norm_fwd
  ops_pkg.FLA_CHUNK_SIZE = vllm_fla_triton_ops.FLA_CHUNK_SIZE
  ops_pkg.utils = utils_mod
  ops_pkg.index = index_mod
  ops_pkg.chunk = chunk_mod
  ops_pkg.fused_gdn_prefill_post_conv = post_conv_mod
  ops_pkg.fused_sigmoid_gating = sig_mod
  ops_pkg.fused_recurrent = rec_mod
  ops_pkg.layernorm_guard = ln_mod

  fla_pkg.ops = ops_pkg
  tp_pkg.flash_linear_attention = fla_pkg

  modules_to_register = {
      "vllm.third_party": tp_pkg,
      "vllm.third_party.flash_linear_attention": fla_pkg,
      "vllm.third_party.flash_linear_attention.ops": ops_pkg,
      "vllm.third_party.flash_linear_attention.ops.utils": utils_mod,
      "vllm.third_party.flash_linear_attention.ops.index": index_mod,
      "vllm.third_party.flash_linear_attention.ops.chunk": chunk_mod,
      "vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv": (
          post_conv_mod
      ),
      "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating": (
          sig_mod
      ),
      "vllm.third_party.flash_linear_attention.ops.fused_recurrent": rec_mod,
      "vllm.third_party.flash_linear_attention.ops.layernorm_guard": ln_mod,
  }
  for mod_name, mod_obj in modules_to_register.items():
    sys.modules[mod_name] = mod_obj

  import vllm

  vllm.third_party = tp_pkg


def _install_custom_ops_and_c_fallbacks() -> None:
  """Replaces mocked torch.ops._C / _C_cache_ops calls with real PyTorch/Triton kernels."""
  from vllm import _custom_ops
  from vllm.model_executor.custom_op import CustomOp
  from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
      triton_reshape_and_cache_flash,
  )

  # 1. Force all CustomOp subclasses (RMSNorm, GemmaRMSNorm, RMSNormGated,
  #    SiluAndMul, RotaryEmbedding, MRotaryEmbedding) to default to forward_native.
  CustomOp.default_on = staticmethod(lambda: False)

  # 2. Wire real PyTorch/Triton implementations onto torch.ops._C and _custom_ops
  def _c_weak_ref_tensor(tensor: Any) -> Any:
    return tensor.detach() if isinstance(tensor, torch.Tensor) else tensor

  def _c_rms_norm(
      out: torch.Tensor,
      x: torch.Tensor,
      weight: torch.Tensor | None,
      epsilon: float,
  ) -> None:
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    res = x_f * torch.rsqrt(var + epsilon)
    if weight is not None:
      res = res * weight.float()
    out.copy_(res.to(orig_dtype))

  def _c_fused_add_rms_norm(
      x: torch.Tensor,
      residual: torch.Tensor,
      weight: torch.Tensor | None,
      epsilon: float,
  ) -> None:
    orig_dtype = x.dtype
    res_f = x.float() + residual.float()
    residual.copy_(res_f.to(orig_dtype))
    var = res_f.pow(2).mean(dim=-1, keepdim=True)
    normed = res_f * torch.rsqrt(var + epsilon)
    if weight is not None:
      normed = normed * weight.float()
    x.copy_(normed.to(orig_dtype))

  def _c_silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])

  def _c_gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(F.gelu(x[..., :d]) * x[..., d:])

  def _c_gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(F.gelu(x[..., :d], approximate="tanh") * x[..., d:])

  def _c_rotary_embedding(
      positions: torch.Tensor,
      query: torch.Tensor,
      key: torch.Tensor | None,
      head_size: int,
      cos_sin_cache: torch.Tensor,
      is_neox: bool,
  ) -> None:
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rotary_dim = cos_sin_cache.shape[-1]
    q_out, k_out = RotaryEmbedding.forward_static(
        positions,
        query,
        key,
        head_size,
        rotary_dim,
        cos_sin_cache,
        is_neox,
    )
    query.copy_(q_out)
    if key is not None and k_out is not None:
      key.copy_(k_out)

  def _c_per_token_group_fp8_quant(
      x: torch.Tensor,
      x_q: torch.Tensor,
      x_s: torch.Tensor,
      group_size: int,
      eps: float,
      fp8_min: float,
      fp8_max: float,
      use_ue8m0: bool = False,
      column_major_scales: bool = False,
      tma_aligned_scales: bool = False,
  ) -> None:
    del tma_aligned_scales
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.triton_utils import triton

    m = x.numel() // group_size
    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    num_stages = 1
    num_cols = x.shape[-1]
    row_stride = x.stride(-2) if x.ndim >= 2 else num_cols
    if column_major_scales:
      fp8_utils._per_token_group_quant_fp8_colmajor[(m,)](
          x,
          x_q,
          x_s,
          group_size,
          num_cols,
          row_stride,
          x_s.stride(-1),
          eps,
          fp8_min=fp8_min,
          fp8_max=fp8_max,
          use_ue8m0=use_ue8m0,
          BLOCK=block,
          num_warps=num_warps,
          num_stages=num_stages,
      )
    else:
      fp8_utils._per_token_group_quant_fp8[(m,)](
          x,
          x_q,
          x_s,
          group_size,
          num_cols,
          row_stride,
          eps,
          fp8_min=fp8_min,
          fp8_max=fp8_max,
          use_ue8m0=use_ue8m0,
          BLOCK=block,
          num_warps=num_warps,
          num_stages=num_stages,
      )

  torch.ops._C.weak_ref_tensor = _c_weak_ref_tensor
  torch.ops._C.rms_norm = _c_rms_norm
  torch.ops._C.fused_add_rms_norm = _c_fused_add_rms_norm
  torch.ops._C.silu_and_mul = _c_silu_and_mul
  torch.ops._C.gelu_and_mul = _c_gelu_and_mul
  torch.ops._C.gelu_tanh_and_mul = _c_gelu_tanh_and_mul
  torch.ops._C.rotary_embedding = _c_rotary_embedding
  torch.ops._C.per_token_group_fp8_quant = _c_per_token_group_fp8_quant
  torch.ops._C.apply_repetition_penalties_ = (
      _custom_ops.apply_repetition_penalties_torch
  )
  torch.ops._C.cutlass_scaled_mm_supports_fp4 = lambda *a, **kw: False
  torch.ops._C.cutlass_scaled_mm_supports_fp8 = lambda *a, **kw: False
  torch.ops._C.cutlass_scaled_mm_supports_block_fp8 = lambda *a, **kw: False
  torch.ops._C.cutlass_group_gemm_supported = lambda *a, **kw: False
  torch.ops._C_cache_ops.reshape_and_cache_flash = (
      triton_reshape_and_cache_flash
  )

  _custom_ops.reshape_and_cache_flash = triton_reshape_and_cache_flash
  _custom_ops.apply_repetition_penalties = (
      _custom_ops.apply_repetition_penalties_torch
  )

  # 3. Ensure vllm.ir.ops.rms_norm and fused_add_rms_norm prefer native impl
  try:
    from vllm import ir
    import vllm.kernels.vllm_c  # noqa: F401

    if "vllm_c" in ir.ops.rms_norm.impls:
      ir.ops.rms_norm.impls["vllm_c"].supported = False
    if "vllm_c" in ir.ops.fused_add_rms_norm.impls:
      ir.ops.fused_add_rms_norm.impls["vllm_c"].supported = False
  except Exception:
    pass


def _install_fp8_triton_linear_shim() -> None:
  """Wires vllm's Fp8LinearMethod to native Triton 128x128 block-FP8 GEMM on H100."""
  from vllm.model_executor.layers.quantization.utils import fp8_utils

  class _CutlassFP8ScaledMMLinearKernel:
    pass

  class _MarlinFP8ScaledMMLinearKernel:
    pass

  class Google3TritonFP8LinearKernel:
    """Executes 128x128 block-FP8 (and per-tensor FP8) linear layers via Triton on H100."""

    def __init__(
        self,
        activation_quant_key: Any = None,
        weight_quant_key: Any = None,
        weight_shape: Any = None,
        input_dtype: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
        module_name: str | None = None,
        **kwargs: Any,
    ) -> None:
      self.activation_quant_key = activation_quant_key
      self.weight_quant_key = weight_quant_key
      self.weight_shape = weight_shape
      self.input_dtype = input_dtype
      self.out_dtype = out_dtype
      self.module_name = module_name
      self.extra_kwargs = kwargs

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
      if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
        if not layer.weight.data.is_contiguous():
          layer.weight.data = layer.weight.data.contiguous()
      if hasattr(layer, "weight_scale_inv") and isinstance(
          layer.weight_scale_inv, torch.nn.Parameter
      ):
        if not layer.weight_scale_inv.data.is_contiguous():
          layer.weight_scale_inv.data = layer.weight_scale_inv.data.contiguous()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
      out_dtype = self.out_dtype if self.out_dtype is not None else x.dtype
      block_size = getattr(layer, "weight_block_size", None)
      if block_size is not None:
        block_n, block_k = int(block_size[0]), int(block_size[1])
        x_2d = x.contiguous().view(-1, x.shape[-1])
        x_q, x_s = fp8_utils.per_token_group_quant_fp8(
            x_2d,
            group_size=block_k,
            column_major_scales=False,
            tma_aligned_scales=False,
            use_ue8m0=False,
        )
        out_2d = fp8_utils.w8a8_triton_block_scaled_mm(
            A=x_q,
            B=layer.weight,
            As=x_s,
            Bs=layer.weight_scale_inv,
            block_size=[block_n, block_k],
            output_dtype=out_dtype,
        )
        out = out_2d.view(*x.shape[:-1], layer.weight.shape[0])
        if bias is not None:
          out = out + bias
        return out

      # Per-tensor / per-channel fallback (layer.weight was transposed to [K, N] in fp8.py:404)
      w_fp8 = layer.weight.to(out_dtype)
      w_scale = getattr(layer, "weight_scale", None)
      if w_scale is not None:
        w_s = w_scale.to(out_dtype)
        if w_s.numel() == 1:
          w_deq = w_fp8 * w_s
        elif w_s.ndim == 1 and w_s.shape[0] == w_fp8.shape[1]:
          w_deq = w_fp8 * w_s.unsqueeze(0)
        elif w_s.ndim == 1 and w_s.shape[0] == w_fp8.shape[0]:
          w_deq = w_fp8 * w_s.unsqueeze(1)
        else:
          w_deq = w_fp8 * w_s
      else:
        w_deq = w_fp8
      return torch.nn.functional.linear(x.to(out_dtype), w_deq.t(), bias)

  def _init_fp8_linear_kernel(*args: Any, **kwargs: Any) -> Google3TritonFP8LinearKernel:
    return Google3TritonFP8LinearKernel(*args, **kwargs)

  from vllm.g3_integration import patchers

  importlib.import_module("vllm.model_executor.kernels")
  for mod_name in (
      "vllm.model_executor.kernels.linear",
      "vllm.model_executor.kernels.linear.scaled_mm",
  ):
    try:
      mod = importlib.import_module(mod_name)
    except Exception:
      mod = patchers.MockModule()
      mod.__name__ = mod_name
      mod.__path__ = []
      sys.modules[mod_name] = mod
    setattr(mod, "init_fp8_linear_kernel", _init_fp8_linear_kernel)
    setattr(mod, "CutlassFP8ScaledMMLinearKernel", _CutlassFP8ScaledMMLinearKernel)
    setattr(mod, "MarlinFP8ScaledMMLinearKernel", _MarlinFP8ScaledMMLinearKernel)
    setattr(mod, "Google3TritonFP8LinearKernel", Google3TritonFP8LinearKernel)

  for quant_mod_name in (
      "vllm.model_executor.layers.quantization.fp8",
      "vllm.model_executor.layers.quantization.modelopt",
      "vllm.model_executor.layers.quantization.online.fp8",
      "vllm.model_executor.layers.quantization.fbgemm_fp8",
  ):
    try:
      qmod = importlib.import_module(quant_mod_name)
      setattr(qmod, "init_fp8_linear_kernel", _init_fp8_linear_kernel)
      if hasattr(qmod, "CutlassFP8ScaledMMLinearKernel"):
        setattr(qmod, "CutlassFP8ScaledMMLinearKernel", _CutlassFP8ScaledMMLinearKernel)
      if hasattr(qmod, "MarlinFP8ScaledMMLinearKernel"):
        setattr(qmod, "MarlinFP8ScaledMMLinearKernel", _MarlinFP8ScaledMMLinearKernel)
    except Exception:
      pass



def _patch_torch_accelerator_for_cuda() -> None:
  """Forces torch.accelerator and C++ deviceFromLong(int) to use CUDA on GPU runtimes."""
  import ctypes
  import struct
  import torch.accelerator
  import torch.accelerator._utils
  import torch.accelerator.memory
  import torch.accelerator.random
  import torch.utils._device

  if not torch.cuda.is_available():
    return

  # 0. Prevent future imports of torch_xla2 / tpu_inference from registering
  #    "jax" as the PrivateUse1 backend on a CUDA machine.
  try:
    torch.utils.rename_privateuse1_backend = lambda *a, **k: None
    torch._C._rename_privateuse1_backend = lambda *a, **k: None
    _orig_reg_dev_mod = getattr(
        torch, "_g3_orig_register_device_module", torch._register_device_module
    )
    setattr(torch, "_g3_orig_register_device_module", _orig_reg_dev_mod)

    def _safe_register_device_module(device_type: Any, module: Any) -> Any:
      if str(device_type) in ("jax", "tpu", "privateuseone"):
        return None
      return _orig_reg_dev_mod(device_type, module)

    torch._register_device_module = _safe_register_device_module
  except Exception:
    pass

  # 1. If pkg_colab_binary or torch_xla2 already autoloaded "jax" onto
  #    PrivateUse1, clear C++ privateuse1_backend_name_set atomic bool in .bss
  #    across the global symbol table and any loaded libc10.so so C++
  #    at::getAccelerator() and deviceFromLong(0) return kCUDA.
  candidate_libs = [None]
  try:
    with open("/proc/self/maps", "r", encoding="utf-8") as f:
      for line in f:
        if "libc10.so" in line or "libtorch_cpu.so" in line:
          parts = line.strip().split()
          if len(parts) >= 6 and parts[-1].startswith("/"):
            if parts[-1] not in candidate_libs:
              candidate_libs.append(parts[-1])
  except Exception:
    pass

  for lib_path in candidate_libs:
    try:
      lib = ctypes.CDLL(lib_path)
      fn_sym = getattr(lib, "_ZN3c1032is_privateuse1_backend_registeredEv", None)
      if fn_sym is None:
        continue
      fn_ptr = ctypes.cast(fn_sym, ctypes.c_void_p).value
      if fn_ptr:
        code = (ctypes.c_uint8 * 16).from_address(fn_ptr)
        if (
            code[0] == 0x0F
            and code[1] == 0xB6
            and code[2] == 0x05
            and code[7] == 0xC3
        ):
          disp = struct.unpack("<i", bytes(code[3:7]))[0]
          ctypes.c_uint8.from_address(fn_ptr + 7 + disp).value = 0
        elif code[0] == 0x8A and code[1] == 0x05:
          disp = struct.unpack("<i", bytes(code[2:6]))[0]
          ctypes.c_uint8.from_address(fn_ptr + 6 + disp).value = 0
    except Exception:
      pass

  # 1b. Also patch torch.utils._device.DeviceContext so that any integer device
  #     (e.g. device=0 from RMSNormGated) or 'jax'/'privateuseone' device inside
  #     a `with torch.device(...):` block is normalized to torch.device("cuda", idx).
  try:
    _dev_constructors = torch.utils._device._device_constructors

    def _normalize_cuda_device(dev: Any) -> Any:
      if isinstance(dev, int) and not isinstance(dev, bool):
        return torch.device("cuda", dev)
      if isinstance(dev, str) and dev.startswith(("jax", "privateuseone", "tpu")):
        return torch.device("cuda", torch.cuda.current_device())
      if isinstance(dev, torch.device) and dev.type in (
          "jax",
          "privateuseone",
          "tpu",
      ):
        return torch.device(
            "cuda",
            dev.index if dev.index is not None else torch.cuda.current_device(),
        )
      return dev

    def _patched_device_ctx_init(self, device: Any) -> None:
      self.device = torch.device(_normalize_cuda_device(device))

    def _patched_device_ctx_torch_function(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
      kwargs = kwargs or {}
      if func in _dev_constructors():
        if kwargs.get("device") is None:
          kwargs["device"] = _normalize_cuda_device(self.device)
        else:
          kwargs["device"] = _normalize_cuda_device(kwargs["device"])
      return func(*args, **kwargs)

    torch.utils._device.DeviceContext.__init__ = _patched_device_ctx_init
    torch.utils._device.DeviceContext.__torch_function__ = (
        _patched_device_ctx_torch_function
    )
  except Exception:
    pass

  # 2. Explicitly route torch.accelerator.*, torch.accelerator.memory.*,
  #    torch.accelerator._utils.*, and torch._C._accelerator_* to torch.cuda.
  def _cuda_get_device_index(device: Any = None, optional: bool = False) -> int:
    if device is None:
      if not optional:
        raise ValueError(
            "Expected a torch.device with a specified index or an integer, but"
            f" got:{device}"
        )
      return torch.cuda.current_device()
    if isinstance(device, int):
      return device
    if isinstance(device, str):
      device = torch.device(device)
    if isinstance(device, torch.device):
      if device.index is None:
        if not optional:
          raise ValueError(
              "Expected a torch.device with a specified index or an integer,"
              f" but got:{device}"
          )
        return torch.cuda.current_device()
      return device.index
    return torch.cuda._utils._get_device_index(device, optional=optional)

  def _cuda_current_accelerator(
      check_available: bool = False,
  ) -> torch.device | None:
    if check_available and not torch.cuda.is_available():
      return None
    return torch.device("cuda")

  def _cuda_set_device_index(device: Any, /) -> None:
    idx = _cuda_get_device_index(device, optional=False)
    if idx >= 0:
      torch.cuda.set_device(idx)

  def _cuda_current_stream(device: Any = None, /):
    return torch.cuda.current_stream(
        _cuda_get_device_index(device, optional=True)
    )

  def _cuda_synchronize(device: Any = None, /) -> None:
    torch.cuda.synchronize(_cuda_get_device_index(device, optional=True))

  class _CudaDeviceIndexCtx:

    def __init__(self, device: int | None, /) -> None:
      self.idx = device
      self.prev_idx = -1

    def __enter__(self) -> None:
      if self.idx is not None:
        self.prev_idx = torch._C._cuda_exchangeDevice(self.idx)

    def __exit__(self, *exc_info: object) -> None:
      if self.idx is not None:
        torch._C._cuda_maybeExchangeDevice(self.prev_idx)

  _empty_host_cache = getattr(torch.cuda, "empty_host_cache", lambda: None)
  _memory_stats = lambda d=None: torch.cuda.memory_stats(
      _cuda_get_device_index(d, optional=True)
  )
  _memory_allocated = lambda d=None: torch.cuda.memory_allocated(
      _cuda_get_device_index(d, optional=True)
  )
  _max_memory_allocated = lambda d=None: torch.cuda.max_memory_allocated(
      _cuda_get_device_index(d, optional=True)
  )
  _memory_reserved = lambda d=None: torch.cuda.memory_reserved(
      _cuda_get_device_index(d, optional=True)
  )
  _max_memory_reserved = lambda d=None: torch.cuda.max_memory_reserved(
      _cuda_get_device_index(d, optional=True)
  )
  _reset_accumulated = lambda d=None: torch.cuda.reset_accumulated_memory_stats(
      _cuda_get_device_index(d, optional=True)
  )
  _reset_peak = lambda d=None: torch.cuda.reset_peak_memory_stats(
      _cuda_get_device_index(d, optional=True)
  )
  _get_memory_info = lambda d=None: torch.cuda.mem_get_info(
      _cuda_get_device_index(d, optional=True)
  )

  torch.accelerator._utils._get_device_index = _cuda_get_device_index
  torch.accelerator._get_device_index = _cuda_get_device_index
  torch.accelerator.memory._get_device_index = _cuda_get_device_index
  torch.accelerator.random._get_device_index = _cuda_get_device_index

  torch.accelerator.current_accelerator = _cuda_current_accelerator
  torch.accelerator.is_available = torch.cuda.is_available
  torch.accelerator.device_count = torch.cuda.device_count
  torch.accelerator.current_device_index = torch.cuda.current_device
  torch.accelerator.current_device_idx = torch.cuda.current_device
  torch.accelerator.set_device_index = _cuda_set_device_index
  torch.accelerator.set_device_idx = _cuda_set_device_index
  torch.accelerator.current_stream = _cuda_current_stream
  torch.accelerator.set_stream = torch.cuda.set_stream
  torch.accelerator.synchronize = _cuda_synchronize
  torch.accelerator.device_index = _CudaDeviceIndexCtx

  for mod in (torch.accelerator, torch.accelerator.memory):
    mod.empty_cache = torch.cuda.empty_cache
    mod.empty_host_cache = _empty_host_cache
    mod.memory_stats = _memory_stats
    mod.memory_allocated = _memory_allocated
    mod.max_memory_allocated = _max_memory_allocated
    mod.memory_reserved = _memory_reserved
    mod.max_memory_reserved = _max_memory_reserved
    mod.reset_accumulated_memory_stats = _reset_accumulated
    mod.reset_peak_memory_stats = _reset_peak
    mod.get_memory_info = _get_memory_info

  try:
    torch._C._accelerator_getAccelerator = lambda: torch.device("cuda")
    torch._C._accelerator_getDeviceIndex = torch.cuda.current_device
    torch._C._accelerator_setDeviceIndex = (
        lambda idx: torch.cuda.set_device(idx) if idx >= 0 else None
    )
    torch._C._accelerator_getStream = torch.cuda.current_stream
    torch._C._accelerator_setStream = torch.cuda.set_stream
    torch._C._accelerator_synchronizeDevice = torch.cuda.synchronize
    torch._C._accelerator_exchangeDevice = torch._C._cuda_exchangeDevice
    torch._C._accelerator_maybeExchangeDevice = (
        torch._C._cuda_maybeExchangeDevice
    )
    torch._C._accelerator_isAllocatorInitialized = torch.cuda.is_initialized
    torch._C._accelerator_emptyCache = torch.cuda.empty_cache
    torch._C._accelerator_emptyHostCache = _empty_host_cache
    torch._C._accelerator_getMemoryInfo = torch.cuda.mem_get_info
    torch._C._accelerator_resetAccumulatedStats = (
        torch.cuda.reset_accumulated_memory_stats
    )
    torch._C._accelerator_resetPeakStats = torch.cuda.reset_peak_memory_stats
  except Exception:
    pass

  # 3. Ensure torch.Event() defaults to CUDA when device=None
  orig_event = getattr(torch, "_g3_orig_Event", torch.Event)
  setattr(torch, "_g3_orig_Event", orig_event)

  class _CudaDefaultEvent(orig_event):

    def __new__(cls, device: Any = None, **kwargs: Any):
      if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
      return orig_event.__new__(cls, device=device, **kwargs)

  torch.Event = _CudaDefaultEvent
  try:
    torch.get_device_module.cache_clear()
  except Exception:
    pass


def install_google3_vllm_cuda_shim() -> None:
  """Installs all shims needed to run //third_party/py/vllm on H100 CUDA in Google3."""
  global _SHIM_INSTALLED
  if _SHIM_INSTALLED:
    _patch_torch_accelerator_for_cuda()
    return

  os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
  os.environ["VLLM_NO_USAGE_STATS"] = "1"
  os.environ["TEST_TARGET"] = "1"
  os.environ.setdefault("VLLM_RPC_BASE_PATH", "/tmp")
  os.environ["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] = "0"
  os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
  os.environ["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] = "0"

  # 0. Mock importlib.metadata versions & force torch.accelerator to CUDA
  _patch_metadata_versions_early()
  _patch_torch_accelerator_for_cuda()

  # 1. Run Google3's base vllm patchers (CNS bridge, metadata versions, mocks)
  from vllm.g3_integration import cns_bridge
  from vllm.g3_integration import patchers

  cns_bridge.activate_cns_bridge()
  patchers.inject_wildcard_mocks()
  patchers.patch_metadata_versions()
  patchers.patch_online_serving_dependencies()
  patchers.redirect_caches_to_tmp()
  patchers.disable_usage_stats()
  patchers.apply_borg_distributed_bypass()
  patchers.shorten_ipc_sockets()

  # 2. Restore vllm.platforms.cuda.CudaPlatform and set current_platform
  _install_cuda_platform()
  # Re-run after _install_cuda_platform() in case importing vllm.platforms.tpu
  # imported tpu_inference -> torch_xla2 and registered 'jax' on PrivateUse1.
  _patch_torch_accelerator_for_cuda()

  # 3. Install real vllm.vllm_flash_attn and vllm.third_party.flash_linear_attention
  _install_vllm_flash_attn_shim()
  _install_fla_triton_modules()

  # 4. Install PyTorch/Triton fallbacks for torch.ops._C, _C_cache_ops, and FP8 Linear
  _install_custom_ops_and_c_fallbacks()
  _install_fp8_triton_linear_shim()

  # 5. Patch model registry subprocess inspection in-process & Colab suppress_stdout
  patchers.apply_single_process_patches()
  try:
    import contextlib
    from vllm.distributed import parallel_state
    from vllm.distributed import utils as vllm_dist_utils
    from vllm.utils import system_utils as vllm_sys_utils

    vllm_sys_utils.suppress_stdout = contextlib.nullcontext
    parallel_state.suppress_stdout = contextlib.nullcontext
    vllm_dist_utils.suppress_stdout = contextlib.nullcontext
  except Exception:
    pass

  # 6. Patch Qwen GatedDeltaNet prefill backend on Hopper (SM90) to use Triton/FLA
  #    instead of FlashInfer (which is not packaged in Google3).
  from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

  qwen_gdn_linear_attn._resolve_gdn_prefill_backend = (
      lambda vllm_config: ("triton", "triton")
  )
  qwen_gdn_linear_attn.fla_chunk_gated_delta_rule = (
      vllm_fla_triton_ops.chunk_gated_delta_rule
  )
  qwen_gdn_linear_attn.fused_post_conv_prep = (
      vllm_fla_triton_ops.fused_post_conv_prep
  )
  qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode = (
      vllm_fla_triton_ops.fused_recurrent_gated_delta_rule_packed_decode
  )
  qwen_gdn_linear_attn.fused_sigmoid_gating_delta_rule_update = (
      vllm_fla_triton_ops.fused_sigmoid_gating_delta_rule_update
  )
  qwen_gdn_linear_attn.l2norm_fwd = vllm_fla_triton_ops.l2norm_fwd
  qwen_gdn_linear_attn.FLA_CHUNK_SIZE = vllm_fla_triton_ops.FLA_CHUNK_SIZE

  _orig_rmsnorm_gated = getattr(
      qwen_gdn_linear_attn,
      "_g3_orig_RMSNormGated",
      qwen_gdn_linear_attn.RMSNormGated,
  )
  setattr(qwen_gdn_linear_attn, "_g3_orig_RMSNormGated", _orig_rmsnorm_gated)

  def _cuda_rmsnorm_gated(*args: Any, **kwargs: Any) -> Any:
    dev = kwargs.get("device")
    if isinstance(dev, int) and not isinstance(dev, bool):
      kwargs["device"] = torch.device("cuda", dev)
    return _orig_rmsnorm_gated(*args, **kwargs)

  qwen_gdn_linear_attn.RMSNormGated = _cuda_rmsnorm_gated

  _patch_torch_accelerator_for_cuda()
  _SHIM_INSTALLED = True
