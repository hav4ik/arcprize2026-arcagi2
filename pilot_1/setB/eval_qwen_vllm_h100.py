"""ARC-AGI-2 Evaluation on H100 with Qwen3.8-27B ("Qwen 28B") + Python Tools.

Replicates the collaborator's Qwen 27B/28B tool-assisted setup from:
  https://claude.ai/artifact/9y7Fu5GbK43bpmjQzSq8Sq

Key features:
1. Multi-turn stateful Python REPL (`PythonToolSession`) pre-loaded with:
   - `np`, `train`, `train_inputs`, `train_outputs`, `test_input`, `test_input_list`
   - DSL helpers: `find_objects`, `crop_bbox`, `most_common_color`
   - `submit_answer(grid)` (E012 / K0 direct grid submission tool)
   - `submit_program(solve)` (E020 K2/K3 live train-pair verification + rich diff)
2. Prompt & Token Budget Control (`E012 Qwen tools full` & `E020 K3 xhigh-48k v4`):
   - `reasoning_effort="xhigh"` via Qwen3.8's native chat template
   - `48,000` hard token budget per attempt, `24,000` soft token target
   - `[Token Budget: used / 48,000 used (soft target: 24,000)]` injected after every
     Python tool execution, plus forced final submission turn when hitting budget
   - `temperature=1.0, top_p=0.95, top_k=20`, 2 attempts per test pair, ranked by
     `train_passed` (`live train check`)
3. Incrementally saves predictions and recomputes live `pass@1` and `pass@2`
   (both Kaggle pair-weighted across 167 test pairs and ARC Prize task-weighted
   across 120 tasks) to CNS after every single test pair:
   - `<output_dir>/predictions.jsonl` (or `predictions_shard_XXX.jsonl`)
   - `<output_dir>/summary.json`
   - `<output_dir>/kaggle_submission.json`
"""

from collections import Counter, deque
import builtins
import concurrent.futures
import copy
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

# Configure single-process vLLM and writable /tmp cache dirs BEFORE importing
# torch or triton (which reads TRITON_CACHE_DIR at import time; on Borg,
# $HOME=/user/<ldap> is read-only).
_TMP_BASE = tempfile.gettempdir()
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_TMP_BASE, "triton_cache"))
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.join(_TMP_BASE, "torchinductor_cache")
)
os.environ.setdefault("HF_HOME", os.path.join(_TMP_BASE, "hf_home"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_TMP_BASE, "xdg_cache"))

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

from google3.experimental.users.arazdai.kaggle import vllm_cuda_shim
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    "Primary CNS path to Qwen3.8-27B model weights (native BF16 54GB for 80GB H100).",
)
flags.DEFINE_string(
    "fp8_model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "Optional CNS path to FP8 Qwen3.8-27B-FP8 weights.",
)
flags.DEFINE_string(
    "data_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "CNS or local directory containing the 120 ARC-AGI-2 evaluation JSON files.",
)
flags.DEFINE_string(
    "output_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_qwen_h100_runs/qwen28b_tools_eval",
    "CNS directory where predictions.jsonl, summary.json, and kaggle_submission.json are saved.",
)
flags.DEFINE_string(
    "recipe",
    "e020_k3_v4",
    "Collaborator recipe: 'e012_tools_full' (first Qwen 28B tools setup with "
    "submit_answer + submit_program) or 'e020_k3_v4' (winning v4 recipe with "
    "lean2progroles + host perception summary + xhigh-48k).",
)
flags.DEFINE_integer("num_attempts", 2, "Number of attempts per test pair (2 for pass@2).")
flags.DEFINE_integer("max_turns", 12, "Max Python REPL turns per attempt.")
flags.DEFINE_integer("total_token_budget", 48000, "Hard token budget per attempt.")
flags.DEFINE_integer("soft_token_target", 24000, "Soft token pacing target per attempt.")
flags.DEFINE_integer("max_new_tokens_per_turn", 8192, "Max generated tokens per single turn.")
flags.DEFINE_float("temperature", 1.0, "Sampling temperature.")
flags.DEFINE_float("top_p", 0.95, "Top-p nucleus sampling.")
flags.DEFINE_integer("top_k", 20, "Top-k sampling.")
flags.DEFINE_string("reasoning_effort", "xhigh", "Qwen3.8 reasoning_effort: xhigh, medium, low.")
flags.DEFINE_float("gpu_memory_utilization", 0.90, "vLLM GPU memory utilization.")
flags.DEFINE_integer("max_model_len", 49152, "vLLM max_model_len.")
flags.DEFINE_integer("limit_tasks", 0, "If > 0, evaluate only the first N tasks.")
flags.DEFINE_string(
    "task_ids",
    "",
    "Optional comma-separated list of task_ids to evaluate.",
)
flags.DEFINE_string(
    "task_ids_file",
    "",
    "Optional JSON file (CNS or local) containing either a list of task_ids or a dict with key --task_ids_key.",
)
flags.DEFINE_string(
    "task_ids_key",
    "hard_25",
    "Key inside --task_ids_file when it is a dict (e.g. hard_25, hard_50, dev_hard_25, dev, holdout).",
)
flags.DEFINE_integer("num_shards", 1, "Total number of parallel worker shards.")
flags.DEFINE_integer("shard_idx", 0, "0-based shard index for this worker.")

COLOR_NAMES = {
    0: "0:black",
    1: "1:blue",
    2: "2:red",
    3: "3:green",
    4: "4:yellow",
    5: "5:grey",
    6: "6:magenta",
    7: "7:orange",
    8: "8:cyan",
    9: "9:maroon",
}

PYTHON_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a persistent stateful REPL session. "
                "Pre-loaded globals: `np`, `train`, `train_inputs`, `train_outputs`, "
                "`test_input`, `find_objects(grid)`, `crop_bbox(grid, r0, c0, r1, c1)`, "
                "`most_common_color(grid)`, `submit_program(solve)`, and "
                "`submit_answer(grid)`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute in the persistent REPL.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


# ==============================================================================
# 1. Zero-RAM Header-Only Streaming CNS Safetensors Loader & vLLM/H100 Engine
# ==============================================================================

_SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
    "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8),
}


class LazySafeSlice:
  """Zero-RAM header-backed slice object for safetensors on CNS."""

  def __init__(
      self, parent: "LazyGFileSafeOpen", key: str, info: dict[str, Any]
  ):
    self.parent = parent
    self.key = key
    self.info = info

  def get_shape(self) -> list[int]:
    return list(self.info["shape"])

  def get_dtype(self) -> str:
    return str(self.info["dtype"])

  def __getitem__(self, slice_args: Any) -> torch.Tensor:
    tensor = self.parent.get_tensor(self.key)
    return tensor[slice_args]


class LazyGFileSafeOpen:
  """Streams individual tensors on demand from CNS via GFile without reading whole shard files into RAM."""

  def __init__(
      self,
      filename: Any,
      framework: str = "pt",
      device: str | int = "cpu",
  ):
    del framework
    self.filename = str(filename)
    self.device = device
    self._lock = threading.Lock()
    self._f = gfile.Open(self.filename, "rb")
    raw_len = self._f.read(8)
    self.header_len = struct.unpack("<Q", raw_len)[0]
    header_bytes = self._f.read(self.header_len)
    self.header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
    self.data_base_offset = 8 + self.header_len

  def keys(self) -> list[str]:
    return [k for k in self.header.keys() if k != "__metadata__"]

  def metadata(self) -> dict[str, Any]:
    return self.header.get("__metadata__", {})

  def get_slice(self, key: str) -> LazySafeSlice:
    return LazySafeSlice(self, key, self.header[key])

  def get_tensor(self, key: str) -> torch.Tensor:
    info = self.header[key]
    dtype_str = info["dtype"]
    shape = info["shape"]
    start_off, end_off = info["data_offsets"]
    nbytes = end_off - start_off
    torch_dtype = _SAFETENSORS_DTYPE_MAP[dtype_str]
    if nbytes == 0:
      t = torch.empty(shape, dtype=torch_dtype)
    else:
      with self._lock:
        self._f.seek(self.data_base_offset + start_off)
        raw = self._f.read(nbytes)
      t = torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape)
    if self.device and str(self.device) != "cpu":
      target_dev = (
          f"cuda:{self.device}"
          if isinstance(self.device, int)
          else str(self.device)
      )
      t = t.to(target_dev)
    return t

  def close(self) -> None:
    if getattr(self, "_f", None) is not None:
      try:
        self._f.close()
      except Exception:
        pass
      self._f = None

  def __enter__(self) -> "LazyGFileSafeOpen":
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()


def install_cns_safetensors_hooks() -> None:
  """Installs LazyGFileSafeOpen so CNS safetensors stream tensor-by-tensor with <1GB CPU RAM."""
  import safetensors
  import safetensors.torch

  orig_safe_open = getattr(
      safetensors, "_g3_orig_safe_open", safetensors.safe_open
  )
  safetensors._g3_orig_safe_open = orig_safe_open

  def lazy_safe_open(
      filename: Any,
      framework: str = "pt",
      device: str | int = "cpu",
      *args: Any,
      **kwargs: Any,
  ):
    if str(filename).startswith(("/cns/", "/placer/")):
      return LazyGFileSafeOpen(filename, framework=framework, device=device)
    return orig_safe_open(filename, framework, device=device, *args, **kwargs)

  safetensors.safe_open = lazy_safe_open
  safetensors.torch.safe_open = lazy_safe_open
  for mod_name in list(sys.modules):
    if mod_name.startswith(("transformers.", "accelerate.", "vllm.")):
      mod = sys.modules.get(mod_name)
      mod_dict = getattr(mod, "__dict__", None)
      if isinstance(mod_dict, dict) and "safe_open" in mod_dict:
        mod_dict["safe_open"] = lazy_safe_open
  logging.info("Installed streaming header-only LazyGFileSafeOpen for CNS.")


class QwenVllmH100Engine:
  """Loads Qwen3.8-27B on H100 via vLLM (with automatic PyTorch SDPA fallback)."""

  def __init__(
      self,
      model_path: str,
      fp8_model_path: str,
      max_model_len: int = 49152,
      gpu_memory_utilization: float = 0.90,
  ):
    self.model_path = model_path
    self.fp8_model_path = fp8_model_path
    self.max_model_len = max_model_len
    self.gpu_memory_utilization = gpu_memory_utilization
    self.vllm_llm = None
    self.SamplingParams = None
    self.pytorch_model = None
    self.tokenizer = None
    self.backend_name = "uninitialized"

  def load(self) -> None:
    install_cns_safetensors_hooks()
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
      gpu_name = torch.cuda.get_device_name(0)
      vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
      logging.info("CUDA GPU detected: %s (%.2f GB VRAM)", gpu_name, vram_gb)
    else:
      logging.warning("WARNING: torch.cuda.is_available() is False!")

    # 1. Try vLLM LLM engine first (with Google3 H100 CUDA + Triton/FLA shim)
    vllm_cuda_shim.install_google3_vllm_cuda_shim()
    for candidate_path in [self.model_path, self.fp8_model_path]:
      if not candidate_path:
        continue
      try:
        from vllm import LLM, SamplingParams

        install_cns_safetensors_hooks()
        logging.info(
            "Attempting to initialize vLLM engine from %s ...", candidate_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            candidate_path, trust_remote_code=True
        )
        self.vllm_llm = LLM(
            model=candidate_path,
            trust_remote_code=True,
            language_model_only=True,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            attention_backend="TRITON_ATTN",
            compilation_config={"custom_ops": ["none"]},
            additional_config={"gdn_prefill_backend": "triton"},
        )
        self.SamplingParams = SamplingParams
        self.backend_name = f"vllm:{os.path.basename(candidate_path)}"
        logging.info("vLLM engine successfully loaded (%s)!", self.backend_name)
        return
      except Exception as vllm_err:
        logging.warning(
            "vLLM initialization for %s failed (%s); trying next/fallback...\n%s",
            candidate_path,
            vllm_err,
            traceback.format_exc(),
        )
        import gc

        self.vllm_llm = None
        try:
          torch.compiler.reset()
        except Exception:
          pass
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Automatic H100 PyTorch 2.5 cuDNN SDPA fallback with streaming LazyGFileSafeOpen
    logging.info("Initializing H100 PyTorch SDPA engine with streaming loader...")
    install_cns_safetensors_hooks()
    try:
      from transformers.quantizers import quantizer_finegrained_fp8

      orig_validate_env = (
          quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer.validate_environment
      )

      def dequant_validate_environment(q_self: Any, *args: Any, **kwargs: Any):
        if getattr(q_self, "pre_quantized", False):
          q_self.quantization_config.dequantize = True
        return orig_validate_env(q_self, *args, **kwargs)

      quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer.validate_environment = (
          dequant_validate_environment
      )
      quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer.update_tp_plan = (
          lambda q_self, config: config
      )
    except Exception as patch_err:
      logging.warning("FineGrainedFP8HfQuantizer patch note: %s", patch_err)

    classes_to_try = []
    try:
      from transformers.models.qwen3_5.modeling_qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
      )

      classes_to_try.append(Qwen3_5ForConditionalGeneration)
    except ImportError:
      pass
    try:
      from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

      classes_to_try.append(Qwen3_5ForCausalLM)
    except ImportError:
      pass
    from transformers import AutoModelForCausalLM

    classes_to_try.append(AutoModelForCausalLM)

    # Prefer native BF16 model path on 80GB H100 (54GB streams straight to cuda:0)
    paths_to_try = [p for p in [self.model_path, self.fp8_model_path] if p]
    for path in paths_to_try:
      try:
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True
        )
      except Exception as tok_err:
        logging.warning("Tokenizer load from %s failed: %s", path, tok_err)
        continue

      for model_cls in classes_to_try:
        try:
          install_cns_safetensors_hooks()
          logging.info(
              "Streaming %s from %s directly onto H100 GPU (device_map={'': 0}, low_cpu_mem_usage=True)...",
              model_cls.__name__,
              path,
          )
          self.pytorch_model = model_cls.from_pretrained(
              path,
              dtype=torch.bfloat16,
              device_map={"": 0} if torch.cuda.is_available() else None,
              low_cpu_mem_usage=True,
              attn_implementation="sdpa",
              trust_remote_code=True,
          )
          self.pytorch_model.eval()
          self.backend_name = (
              f"pytorch_sdpa:{model_cls.__name__}:{os.path.basename(path)}"
          )
          logging.info("Loaded model successfully via %s!", self.backend_name)
          return
        except Exception as e:
          logging.warning(
              "%s load from %s failed: %s",
              model_cls.__name__,
              path,
              e,
          )
          torch.cuda.empty_cache()

    raise RuntimeError(
        f"Failed to load Qwen3.8-27B from {paths_to_try} in both vLLM and PyTorch."
    )

  def generate_turn(
      self,
      messages: list[dict[str, Any]],
      temperature: float,
      top_p: float,
      top_k: int,
      max_new_tokens: int,
      reasoning_effort: str = "xhigh",
      use_native_tools_schema: bool = True,
  ) -> dict[str, Any]:
    """Generates one assistant turn from the multi-turn `messages` history."""
    t0 = time.perf_counter()
    tools_arg = PYTHON_TOOLS_SCHEMA if use_native_tools_schema else None

    try:
      prompt_text = self.tokenizer.apply_chat_template(
          messages,
          tools=tools_arg,
          tokenize=False,
          add_generation_prompt=True,
          enable_thinking=True,
          reasoning_effort=reasoning_effort,
      )
    except Exception:
      prompt_text = self.tokenizer.apply_chat_template(
          messages,
          tools=tools_arg,
          tokenize=False,
          add_generation_prompt=True,
      )

    if self.vllm_llm is not None:
      sampling_params = self.SamplingParams(
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          max_tokens=max_new_tokens,
      )
      outputs = self.vllm_llm.generate(
          [prompt_text], sampling_params=sampling_params, use_tqdm=False
      )
      out = outputs[0]
      raw_text = out.outputs[0].text if out.outputs else ""
      out_tokens = len(out.outputs[0].token_ids) if out.outputs else 0
      prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0
      finish_reason = str(out.outputs[0].finish_reason) if out.outputs else "stop"
    else:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      inputs = self.tokenizer(
          prompt_text, return_tensors="pt", add_special_tokens=False
      ).to(device)
      prompt_tokens = int(inputs["input_ids"].shape[-1])
      with torch.inference_mode():
        output_ids = self.pytorch_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=(temperature > 0),
            pad_token_id=self.tokenizer.eos_token_id,
        )
      gen_ids = output_ids[0][prompt_tokens:]
      raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
      out_tokens = int(len(gen_ids))
      finish_reason = "length" if out_tokens >= max_new_tokens else "stop"

    latency_s = time.perf_counter() - t0

    # Qwen3.8's chat_template ends with `<|im_start|>assistant\n<think>\n`,
    # so raw_text starts inside `<think>` and closes with `</think>`.
    if "</think>" in raw_text:
      thought_part, visible_part = raw_text.split("</think>", 1)
      thought_text = re.sub(r"^<think>\s*", "", thought_part).strip()
      visible_text = visible_part.strip()
    else:
      # Model hit max_new_tokens while still inside `<think>`
      thought_text = re.sub(r"^<think>\s*", "", raw_text).strip()
      visible_text = ""

    think_tokens_est = (
        len(self.tokenizer.encode(thought_text, add_special_tokens=False))
        if thought_text
        else 0
    )
    ans_tokens_est = max(0, out_tokens - think_tokens_est)

    return {
        "raw_text": raw_text,
        "thought_text": thought_text,
        "visible_text": visible_text,
        "prompt_tokens": prompt_tokens,
        "thinking_tokens": think_tokens_est,
        "answer_tokens": ans_tokens_est,
        "total_output_tokens": out_tokens,
        "finish_reason": finish_reason,
        "latency_s": round(latency_s, 2),
        "backend": self.backend_name,
    }


# ==============================================================================
# 2. ARC-AGI-2 Grid Helpers, Host Perception Summary, and Prompt Builder
# ==============================================================================


def fmt_grid(g: list[list[int]]) -> str:
  return "[\n" + ",\n".join("  " + json.dumps(row) for row in g) + "\n]"


def grid_meta(g: list[list[int]]) -> str:
  h = len(g)
  w = len(g[0]) if h > 0 else 0
  counts: dict[int, int] = {}
  for r in g:
    for v in r:
      counts[v] = counts.get(v, 0) + 1
  hist = ", ".join(
      f"{COLOR_NAMES.get(k, str(k))}={v}"
      for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
  )
  return f"shape={h}x{w} ({h} rows, {w} cols), colors=[{hist}]"


def dsl_most_common_color(grid: Any) -> int:
  arr = np.asarray(grid, dtype=int)
  vals, counts = np.unique(arr, return_counts=True)
  return int(vals[int(np.argmax(counts))])


def dsl_crop_bbox(
    grid: Any, r0: int, c0: int, r1: int, c1: int
) -> np.ndarray:
  arr = np.asarray(grid, dtype=int)
  return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def dsl_find_objects(
    grid: Any, bg: int | None = None, diag: bool = False
) -> list[dict[str, Any]]:
  arr = np.asarray(grid, dtype=int)
  if bg is None:
    bg = dsl_most_common_color(arr)
  h, w = arr.shape
  visited = np.zeros((h, w), dtype=bool)
  dirs = (
      [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
      if diag
      else [(-1, 0), (1, 0), (0, -1), (0, 1)]
  )
  objs = []
  for r in range(h):
    for c in range(w):
      color = int(arr[r, c])
      if color == bg or visited[r, c]:
        continue
      q = deque([(r, c)])
      visited[r, c] = True
      cells = []
      while q:
        cr, cc = q.popleft()
        cells.append((cr, cc))
        for dr, dc in dirs:
          nr, nc = cr + dr, cc + dc
          if (
              0 <= nr < h
              and 0 <= nc < w
              and not visited[nr, nc]
              and int(arr[nr, nc]) == color
          ):
            visited[nr, nc] = True
            q.append((nr, nc))
      rs = [p[0] for p in cells]
      cs = [p[1] for p in cells]
      r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
      sub = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
      for pr, pc in cells:
        sub[pr - r0, pc - c0] = color
      objs.append({
          "color": color,
          "size": len(cells),
          "bbox": (r0, c0, r1, c1),
          "shape": (r1 - r0 + 1, c1 - c0 + 1),
          "cells": cells,
          "subgrid": sub,
      })
  objs.sort(key=lambda x: (-x["size"], x["bbox"][0], x["bbox"][1]))
  return objs


def build_host_task_summary(
    task: dict[str, Any], test_input: list[list[int]]
) -> str:
  """Pre-computes a perception summary and test-vs-train novelty audit (E017/E020)."""
  lines = ["### Host-Computed Perception & Structural Summary"]
  train_in_shapes = []
  train_in_colors: list[set[int]] = []
  train_comp_counts = []

  for i, pair in enumerate(task["train"]):
    inp, out = pair["input"], pair["output"]
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    train_in_shapes.append((ih, iw))
    in_c = {v for r in inp for v in r}
    out_c = {v for r in out for v in r}
    train_in_colors.append(in_c)
    added_c = sorted(out_c - in_c)
    removed_c = sorted(in_c - out_c)
    flat_in = [v for r in inp for v in r]
    bg = Counter(flat_in).most_common(1)[0][0] if flat_in else 0
    in_comps = dsl_find_objects(inp, bg=bg)
    out_comps = dsl_find_objects(out, bg=bg)
    train_comp_counts.append(len(in_comps))
    comp_preview = ", ".join(
        f"c{c['color']}(n={c['size']},box={c['bbox'][0]}..{c['bbox'][2]}x{c['bbox'][1]}..{c['bbox'][3]})"
        for c in in_comps[:8]
    )
    if len(in_comps) > 8:
      comp_preview += f", ... (+{len(in_comps) - 8} more)"
    lines.append(
        f"- Train {i+1}: in={ih}x{iw} -> out={oh}x{ow} | bg={bg} |"
        f" in_comps={len(in_comps)} -> out_comps={len(out_comps)} |"
        f" colors_added={added_c} colors_removed={removed_c}\n"
        f"  In objects: [{comp_preview}]"
    )

  th, tw = len(test_input), len(test_input[0])
  flat_test = [v for r in test_input for v in r]
  t_bg = Counter(flat_test).most_common(1)[0][0] if flat_test else 0
  t_comps = dsl_find_objects(test_input, bg=t_bg)
  t_colors = set(flat_test)
  t_preview = ", ".join(
      f"c{c['color']}(n={c['size']},box={c['bbox'][0]}..{c['bbox'][2]}x{c['bbox'][1]}..{c['bbox'][3]})"
      for c in t_comps[:10]
  )
  if len(t_comps) > 10:
    t_preview += f", ... (+{len(t_comps) - 10} more)"
  lines.append(
      f"- Test Input: shape={th}x{tw} | bg={t_bg} | comps={len(t_comps)} |"
      f" colors={sorted(t_colors)}\n"
      f"  Test objects: [{t_preview}]"
  )

  warnings = []
  min_h = min(s[0] for s in train_in_shapes)
  max_h = max(s[0] for s in train_in_shapes)
  min_w = min(s[1] for s in train_in_shapes)
  max_w = max(s[1] for s in train_in_shapes)
  if not (min_h <= th <= max_h and min_w <= tw <= max_w):
    warnings.append(
        f"Test input dimensions ({th}x{tw}) lie OUTSIDE training input range"
        f" (rows {min_h}..{max_h}, cols {min_w}..{max_w}). Do NOT hardcode"
        " grid dimensions!"
    )
  all_train_colors = set().union(*train_in_colors) if train_in_colors else set()
  new_test_colors = sorted(t_colors - all_train_colors)
  if new_test_colors:
    warnings.append(
        f"Test input contains NEW color(s) {new_test_colors} never seen in any"
        " training input! Ensure your rule handles arbitrary object/marker"
        " colors."
    )
  if warnings:
    lines.append("### [!] TEST-VS-TRAIN NOVELTY AUDIT WARNINGS:")
    for w_msg in warnings:
      lines.append(f"  * WARNING: {w_msg}")
  return "\n".join(lines)


def build_qwen_tool_prompt(
    task: dict[str, Any],
    test_input: list[list[int]],
    recipe: str,
    soft_target: int,
    hard_budget: int,
) -> str:
  """Builds the collaborator's Qwen 27B/28B tool prompt (`lean2sub` or `lean2progroles`)."""
  ex_blocks = []
  for i, pair in enumerate(task["train"]):
    inp, out = pair["input"], pair["output"]
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    same_shape = (
        f"same shape {ih}x{iw}"
        if (ih, iw) == (oh, ow)
        else f"shape changes {ih}x{iw} -> {oh}x{ow}"
    )
    ex_blocks.append(
        f"### Training Example {i + 1} ({same_shape})\n"
        f"INPUT ({grid_meta(inp)}):\n{fmt_grid(inp)}\n\n"
        f"OUTPUT ({grid_meta(out)}):\n{fmt_grid(out)}"
    )
  train_section = "\n\n".join(ex_blocks)
  host_summary = build_host_task_summary(task, test_input)

  role_rules = ""
  if recipe == "e020_k3_v4":
    role_rules = (
        "\nCritical Reasoning Rules (lean2progroles):\n"
        "- Explicitly assign a functional ROLE to every color/component before"
        " hypothesizing a rule (e.g., background, frame/border, divider,"
        " template/stamp, position marker, color indicator, legend/key).\n"
        "- NEVER dismiss any pixel or cell as 'random noise'. Every single"
        " non-background cell in an ARC puzzle has a deterministic purpose.\n"
        "- Carefully check the TEST-VS-TRAIN NOVELTY AUDIT below: if the test"
        " input has a different grid size, new colors, or more components,"
        " make sure your rule is parameterized rather than hardcoded to"
        " training constants.\n"
    )

  tool_instructions = (
      "\nInteractive Python Environment & Verification Harness:\n"
      "- You have access to a persistent Python REPL across turns via the"
      " `python` tool (or by writing a ```python ... ``` code block).\n"
      "- Pre-loaded variables & helpers in your Python session:\n"
      "  * `np` (numpy)\n"
      "  * `train`: list of `(input_grid_list, output_grid_list)` pairs\n"
      "  * `train_inputs`: list of 2D numpy arrays\n"
      "  * `train_outputs`: list of 2D numpy arrays\n"
      "  * `test_input`: 2D numpy array (and `test_input_list` as 2D list)\n"
      "  * `find_objects(grid, bg=None, diag=False)`: returns list of"
      " connected-component dicts `{'color': c, 'size': n, 'bbox':"
      " (r0,c0,r1,c1), 'cells': [(r,c),...], 'subgrid': np.ndarray}`\n"
      "  * `crop_bbox(grid, r0, c0, r1, c1)` and `most_common_color(grid)`\n"
      "  * `submit_program(solve)`: tests your `def solve(grid):` function"
      " (which takes a 2D list or numpy array and returns a 2D list/array) on"
      " EVERY training example. If all training examples match 100%, it"
      " automatically computes `solve(test_input)` and records your final"
      " answer! If any training example fails, it shows the exact diff so you"
      " can fix your `solve` function.\n"
      "  * `submit_answer(grid)`: directly submits a 2D integer grid `[[...]]`"
      " for `test_input`.\n"
      f"- Token Budget: Aim to solve and verify the puzzle within the soft"
      f" target of {soft_target:,} tokens (hard ceiling: {hard_budget:,}"
      " tokens). Keep each thinking turn focused before calling Python to test"
      " your hypothesis.\n"
  )

  return (
      "You are an expert solver for the ARC-AGI-2 (Abstraction and Reasoning"
      " Corpus) competition.\n"
      "Each puzzle consists of 2D grids of integers from 0 to 9 (representing"
      " colors, where 0 is typically background/black).\n"
      "All Training Examples below obey a single, exact, deterministic"
      " transformation rule that maps each INPUT grid to its OUTPUT grid.\n"
      f"{role_rules}{tool_instructions}\n"
      f"{host_summary}\n\n"
      f"{train_section}\n\n"
      f"### Test Puzzle\n"
      f"TEST INPUT ({grid_meta(test_input)}):\n{fmt_grid(test_input)}\n\n"
      "Use the `python` tool (or a ```python ... ``` block) to inspect the"
      " grids, test your transformation rule on all training examples via"
      " `submit_program(solve)` (or `submit_answer(grid)`), and verify 100%"
      " training accuracy."
  )


# ==============================================================================
# 3. Python Tool Session & Code Extraction
# ==============================================================================


def normalize_grid(val: Any) -> list[list[int]] | None:
  """Converts a numpy array or nested list into a validated 2D list[list[int]]."""
  if isinstance(val, np.ndarray):
    val = val.tolist()
  if not isinstance(val, (list, tuple)) or len(val) == 0:
    return None
  out: list[list[int]] = []
  w = None
  for r in val:
    if isinstance(r, np.ndarray):
      r = r.tolist()
    if not isinstance(r, (list, tuple)) or len(r) == 0:
      return None
    if w is None:
      w = len(r)
    elif len(r) != w:
      return None
    row_ints: list[int] = []
    for x in r:
      if isinstance(x, bool):
        return None
      if isinstance(x, (int, np.integer)):
        iv = int(x)
      elif isinstance(x, (float, np.floating)) and float(x).is_integer():
        iv = int(x)
      else:
        return None
      if not (0 <= iv <= 9):
        return None
      row_ints.append(iv)
    out.append(row_ints)
  return out


def parse_grid(text: str) -> list[list[int]] | None:
  """Scans backward from the end of `text` for the last valid 2D JSON grid [[...]]."""
  if not text:
    return None
  cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
  for candidate_text in (cleaned, text):
    for end in range(len(candidate_text) - 1, -1, -1):
      if candidate_text[end] == "]":
        depth = 0
        for start in range(end, -1, -1):
          if candidate_text[start] == "]":
            depth += 1
          elif candidate_text[start] == "[":
            depth -= 1
            if depth == 0:
              try:
                val = json.loads(candidate_text[start : end + 1])
                norm = normalize_grid(val)
                if norm is not None:
                  return norm
              except Exception:
                pass
              break
  return None


def extract_python_from_response(visible_text: str) -> list[str]:
  """Extracts Python code from Qwen3.8 `<tool_call>` tags OR ```python blocks."""
  if not visible_text:
    return []
  blocks: list[str] = []

  # 1. Qwen3.8 XML <tool_call><function=python><parameter=code>...</parameter></function></tool_call>
  for tc_match in re.finditer(
      r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", visible_text, re.IGNORECASE
  ):
    inner = tc_match.group(1).strip()
    param_m = re.search(
        r"<parameter=code>\s*([\s\S]*?)\s*(?:</parameter>|$)",
        inner,
        re.IGNORECASE,
    )
    if param_m and param_m.group(1).strip():
      blocks.append(param_m.group(1).strip())
      continue
    # Check JSON format inside <tool_call>
    if inner.startswith("{"):
      try:
        obj = json.loads(inner)
        args = obj.get("arguments", obj.get("parameters", obj))
        if isinstance(args, str):
          args = json.loads(args)
        if isinstance(args, dict) and isinstance(args.get("code"), str):
          blocks.append(args["code"].strip())
          continue
      except Exception:
        pass

  if blocks:
    return blocks

  # 2. Markdown ```python ... ``` blocks
  md_blocks = re.findall(r"```(?:python|py)\s*\n([\s\S]*?)```", visible_text)
  return [b.strip() for b in md_blocks if b.strip()]


def format_grid_diff(expected: list[list[int]], got: list[list[int]]) -> str:
  eh, ew = len(expected), len(expected[0])
  gh, gw = len(got), len(got[0])
  if (eh, ew) != (gh, gw):
    return f"Shape mismatch: expected {eh}x{ew}, but solve() returned {gh}x{gw}."
  mismatches = []
  for r in range(eh):
    for c in range(ew):
      if expected[r][c] != got[r][c]:
        mismatches.append((r, c, expected[r][c], got[r][c]))
  if not mismatches:
    return "Exact match!"
  msg = [
      f"Cell mismatch on {len(mismatches)}/{eh*ew} cells (shape {eh}x{ew} matches)."
  ]
  sample_m = ", ".join(
      f"({r},{c}): exp={e} got={g}" for r, c, e, g in mismatches[:12]
  )
  if len(mismatches) > 12:
    sample_m += f", ... (+{len(mismatches) - 12} more)"
  msg.append(f"  First mismatches: {sample_m}")
  if eh <= 20 and ew <= 20:
    diff_rows = []
    for r in range(eh):
      row_cells = []
      for c in range(ew):
        if expected[r][c] == got[r][c]:
          row_cells.append(f" {got[r][c]} ")
        else:
          row_cells.append(f"{got[r][c]}!{expected[r][c]}")
      diff_rows.append(" ".join(row_cells))
    msg.append("  Grid diff (got!expected):\n    " + "\n    ".join(diff_rows))
  return "\n".join(msg)


class PythonToolSession:
  """Stateful per-attempt Python REPL + train-pair verification harness."""

  def __init__(self, task: dict[str, Any], test_input: list[list[int]]):
    self.task = task
    self.test_input = copy.deepcopy(test_input)
    self.train_passed = False
    self.best_train_pairs_ratio = 0.0
    self.best_train_cell_ratio = 0.0
    self.submitted_grid: list[list[int]] | None = None
    self.candidate_test_grid: list[list[int]] | None = None
    self.submitted_code: str = ""
    self._out_buf = io.StringIO()

    train_pairs = [
        (copy.deepcopy(p["input"]), copy.deepcopy(p["output"]))
        for p in task["train"]
    ]
    self.globals: dict[str, Any] = {
        "__builtins__": builtins.__dict__,
        "np": np,
        "math": math,
        "copy": copy,
        "Counter": Counter,
        "deque": deque,
        "train": train_pairs,
        "train_inputs": [np.array(p["input"], dtype=int) for p in task["train"]],
        "train_outputs": [
            np.array(p["output"], dtype=int) for p in task["train"]
        ],
        "test_input": np.array(test_input, dtype=int),
        "test_input_list": copy.deepcopy(test_input),
        "find_objects": dsl_find_objects,
        "crop_bbox": dsl_crop_bbox,
        "most_common_color": dsl_most_common_color,
        "print": self._custom_print,
        "submit_program": self._submit_program,
        "submit_answer": self._submit_answer,
    }

  def _custom_print(
      self, *args: Any, sep: str = " ", end: str = "\n", **kw: Any
  ) -> None:
    del kw
    s = sep.join(str(a) for a in args) + end
    if self._out_buf.tell() < 12000:
      self._out_buf.write(s[:4000])

  def _submit_answer(self, grid: Any) -> None:
    norm = normalize_grid(grid)
    if norm is None:
      self._custom_print(
          "ERROR in submit_answer(grid): grid must be a valid 2D rectangular"
          " grid of integers 0..9."
      )
      return
    self.submitted_grid = norm
    self.candidate_test_grid = norm
    self._custom_print(
        f"SUBMIT_ANSWER RECORDED: valid {len(norm)}x{len(norm[0])} grid."
    )

  def _verify_solve_fn(self, solve_fn: Any) -> bool:
    if not callable(solve_fn):
      self._custom_print("ERROR: argument to submit_program must be callable.")
      return False
    all_ok = True
    passed_pairs = 0
    cell_ratios: list[float] = []
    results_msg = []
    num_train = max(1, len(self.task["train"]))
    for idx, pair in enumerate(self.task["train"]):
      inp = copy.deepcopy(pair["input"])
      exp = pair["output"]
      try:
        try:
          raw_out = solve_fn(np.array(inp, dtype=int))
        except Exception:
          raw_out = solve_fn(copy.deepcopy(inp))
        norm_out = normalize_grid(raw_out)
        if norm_out is None:
          all_ok = False
          cell_ratios.append(0.0)
          results_msg.append(
              f"  - Train {idx+1}: FAILED (returned invalid grid: {type(raw_out)})"
          )
        elif norm_out != exp:
          all_ok = False
          eh, ew = len(exp), len(exp[0])
          if (len(norm_out), len(norm_out[0])) == (eh, ew):
            match_c = sum(
                1
                for r in range(eh)
                for c in range(ew)
                if exp[r][c] == norm_out[r][c]
            )
            cell_ratios.append(match_c / max(1, eh * ew))
          else:
            cell_ratios.append(0.0)
          diff_s = format_grid_diff(exp, norm_out)
          results_msg.append(f"  - Train {idx+1}: FAILED -> {diff_s}")
        else:
          passed_pairs += 1
          cell_ratios.append(1.0)
          results_msg.append(
              f"  - Train {idx+1}: PASSED ({len(exp)}x{len(exp[0])})"
          )
      except Exception as e:
        all_ok = False
        cell_ratios.append(0.0)
        results_msg.append(f"  - Train {idx+1}: EXCEPTION -> {e}")

    pairs_ratio = passed_pairs / num_train
    mean_cell_ratio = (
        sum(cell_ratios) / len(cell_ratios) if cell_ratios else 0.0
    )

    test_norm = None
    try:
      try:
        raw_t = solve_fn(np.array(self.test_input, dtype=int))
      except Exception:
        raw_t = solve_fn(copy.deepcopy(self.test_input))
      test_norm = normalize_grid(raw_t)
    except Exception as e:
      results_msg.append(f"  - Test Input execution raised exception: {e}")

    if test_norm is not None:
      if (pairs_ratio, mean_cell_ratio) >= (
          self.best_train_pairs_ratio,
          self.best_train_cell_ratio,
      ) or self.candidate_test_grid is None:
        self.best_train_pairs_ratio = pairs_ratio
        self.best_train_cell_ratio = mean_cell_ratio
        self.candidate_test_grid = test_norm

    if all_ok and test_norm is not None:
      self.train_passed = True
      self.best_train_pairs_ratio = 1.0
      self.best_train_cell_ratio = 1.0
      self.submitted_grid = test_norm
      self._custom_print(
          "=== TRAIN CHECK PASSED (ALL TRAINING EXAMPLES MATCH 100%) ===\n"
          + "\n".join(results_msg)
          + f"\nTest Output successfully generated: shape"
          f" {len(test_norm)}x{len(test_norm[0])}."
      )
      return True
    else:
      self._custom_print(
          f"=== TRAIN CHECK FAILED ({passed_pairs}/{num_train} pairs passed,"
          f" cell accuracy={mean_cell_ratio*100:.1f}%) ===\n"
          + "\n".join(results_msg)
          + "\nPlease inspect the failure diff above, fix `solve(grid)`, and"
          " call `submit_program(solve)` again."
      )
      return False

  def _submit_program(self, solve_fn: Any) -> bool:
    return self._verify_solve_fn(solve_fn)

  def run_code(self, code: str, timeout_s: float = 6.0) -> str:
    self._out_buf = io.StringIO()
    prev_submitted = self.submitted_grid
    err_holder: list[str] = []

    def _target() -> None:
      try:
        exec(code, self.globals)  # pylint: disable=exec-used
        if (
            self.submitted_grid is prev_submitted
            and "def solve" in code
            and callable(self.globals.get("solve"))
        ):
          self._verify_solve_fn(self.globals["solve"])
        for var_name in ("answer", "test_output", "pred", "output_grid"):
          if var_name in self.globals and self.candidate_test_grid is None:
            norm_ans = normalize_grid(self.globals[var_name])
            if norm_ans is not None:
              self.candidate_test_grid = norm_ans
      except Exception:
        err_holder.append(traceback.format_exc(limit=4))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
      return f"TIMEOUT: Code execution exceeded {timeout_s}s limit."
    out_str = self._out_buf.getvalue()
    if err_holder:
      out_str += "\n" + "\n".join(err_holder)
    if self.train_passed and "def solve" in code:
      self.submitted_code = code
    return out_str.strip() or "(Executed cleanly with no stdout)"


# ==============================================================================
# 4. Multi-Turn Attempt Execution & Pass@1 / Pass@2 Aggregation
# ==============================================================================


def run_single_attempt(
    engine: QwenVllmH100Engine,
    task: dict[str, Any],
    test_input: list[list[int]],
    target: list[list[int]],
    attempt_idx: int,
) -> dict[str, Any]:
  """Runs one multi-turn tool-assisted attempt with Qwen3.8-27B on H100."""
  t_start = time.time()
  session = PythonToolSession(task=task, test_input=test_input)
  prompt = build_qwen_tool_prompt(
      task=task,
      test_input=test_input,
      recipe=FLAGS.recipe,
      soft_target=FLAGS.soft_token_target,
      hard_budget=FLAGS.total_token_budget,
  )
  messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

  cum_output_tokens = 0
  cum_thinking_tokens = 0
  cum_answer_tokens = 0
  last_prompt_tokens = 0
  turns_taken = 0
  tool_calls_count = 0
  pred_grid: list[list[int]] | None = None
  forced_final_used = False

  for turn_idx in range(FLAGS.max_turns):
    remaining_budget = FLAGS.total_token_budget - cum_output_tokens
    if remaining_budget <= 512:
      break

    req_max_tokens = min(FLAGS.max_new_tokens_per_turn, remaining_budget)
    t_turn = time.time()
    res = engine.generate_turn(
        messages=messages,
        temperature=FLAGS.temperature,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
        max_new_tokens=req_max_tokens,
        reasoning_effort=FLAGS.reasoning_effort,
        use_native_tools_schema=True,
    )
    turn_dur = time.time() - t_turn
    turns_taken += 1
    cum_output_tokens += res["total_output_tokens"]
    cum_thinking_tokens += res["thinking_tokens"]
    cum_answer_tokens += res["answer_tokens"]
    last_prompt_tokens = res["prompt_tokens"]
    logging.info(
        "[Shard %d] Attempt %d | Turn %d/%d | gen_tokens=%d (cum=%d/%d) |"
        " turn_time=%.1fs (%.1f tok/s)",
        FLAGS.shard_idx,
        attempt_idx,
        turns_taken,
        FLAGS.max_turns,
        res["total_output_tokens"],
        cum_output_tokens,
        FLAGS.total_token_budget,
        turn_dur,
        res["total_output_tokens"] / max(turn_dur, 1e-3),
    )

    messages.append({
        "role": "assistant",
        "reasoning_content": res["thought_text"],
        "content": res["visible_text"],
    })

    code_blocks = extract_python_from_response(res["visible_text"])
    if code_blocks:
      tool_calls_count += len(code_blocks)
      outputs = [session.run_code(code) for code in code_blocks]
      combined_out = "\n---\n".join(outputs)

      if session.train_passed and session.submitted_grid is not None:
        pred_grid = session.submitted_grid
        break

      budget_note = (
          f"\n\n[Token Budget: {cum_output_tokens:,} /"
          f" {FLAGS.total_token_budget:,} used (soft target:"
          f" {FLAGS.soft_token_target:,})]"
      )
      if cum_output_tokens >= FLAGS.soft_token_target:
        budget_note += (
            " WARNING: Soft token target reached! Finalize your `solve(grid)`"
            " function or call `submit_answer(grid)` now."
        )
      messages.append({"role": "tool", "content": combined_out + budget_note})
      continue

    # No code blocks in this turn: check if session or text already has a grid
    if session.submitted_grid is not None:
      pred_grid = session.submitted_grid
      break

    parsed = parse_grid(res["visible_text"]) or parse_grid(res["thought_text"])
    if parsed is not None:
      pred_grid = parsed
      break

    # If the model got cut off inside `<think>` (finish_reason == length),
    # prompt it to finalize its tool call or grid.
    if cum_output_tokens < FLAGS.total_token_budget - 1024:
      messages.append({
          "role": "user",
          "content": (
              f"[Token Budget: {cum_output_tokens:,} /"
              f" {FLAGS.total_token_budget:,} used] Your previous turn ended"
              " without calling the `python` tool or submitting a grid. Please"
              " call `submit_program(solve)` or `submit_answer(grid)` now."
          ),
      })
    else:
      break

  # Fallback 1: session candidate grid from best train check or `submit_answer`
  if pred_grid is None and session.submitted_grid is not None:
    pred_grid = session.submitted_grid
  if pred_grid is None and session.candidate_test_grid is not None:
    pred_grid = session.candidate_test_grid

  # Fallback 2: Forced final turn if budget exhausted without a grid
  if pred_grid is None:
    forced_final_used = True
    messages.append({
        "role": "user",
        "content": (
            "FINAL SUBMISSION TURN: Token budget reached. Output ONLY the"
            " final 2D JSON integer grid `[[...]]` for TEST INPUT immediately"
            " (or call `submit_answer(grid)`):"
        ),
    })
    res_f = engine.generate_turn(
        messages=messages,
        temperature=0.2,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
        max_new_tokens=2048,
        reasoning_effort="low",
        use_native_tools_schema=True,
    )
    turns_taken += 1
    cum_output_tokens += res_f["total_output_tokens"]
    for cb in extract_python_from_response(res_f["visible_text"]):
      session.run_code(cb)
    pred_grid = (
        session.submitted_grid
        or session.candidate_test_grid
        or parse_grid(res_f["visible_text"])
        or parse_grid(res_f["thought_text"])
    )

  exact_match = pred_grid is not None and pred_grid == target
  shape_match = (
      pred_grid is not None
      and len(pred_grid) == len(target)
      and len(pred_grid[0]) == len(target[0])
  )
  cell_acc = 0.0
  if shape_match and pred_grid is not None:
    h, w = len(target), len(target[0])
    cell_acc = sum(
        1 for r in range(h) for c in range(w) if pred_grid[r][c] == target[r][c]
    ) / max(1, h * w)

  return {
      "attempt_idx": attempt_idx,
      "exact_match": exact_match,
      "shape_match": shape_match,
      "cell_accuracy": round(cell_acc, 4),
      "parsed_ok": pred_grid is not None,
      "train_passed": session.train_passed,
      "best_train_pairs_ratio": round(session.best_train_pairs_ratio, 4),
      "best_train_cell_ratio": round(session.best_train_cell_ratio, 4),
      "turns": turns_taken,
      "tool_calls": tool_calls_count,
      "forced_final_used": forced_final_used,
      "prompt_tokens": last_prompt_tokens,
      "thinking_tokens": cum_thinking_tokens,
      "answer_tokens": cum_answer_tokens,
      "total_output_tokens": cum_output_tokens,
      "latency_s": round(time.time() - t_start, 2),
      "pred_grid": pred_grid,
      "backend": engine.backend_name,
  }


def rank_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Ranks attempts prioritizing `train_passed` (live train check), then train cell ratio."""

  def _score_key(a: dict[str, Any]) -> tuple[Any, ...]:
    g_str = (
        json.dumps(a["pred_grid"], separators=(",", ":"))
        if a.get("pred_grid") is not None
        else ""
    )
    freq = (
        sum(
            1
            for x in attempts
            if x.get("pred_grid") is not None
            and json.dumps(x["pred_grid"], separators=(",", ":")) == g_str
        )
        if g_str
        else 0
    )
    return (
        1 if a.get("train_passed") else 0,
        float(a.get("best_train_pairs_ratio", 0.0)),
        float(a.get("best_train_cell_ratio", 0.0)),
        freq,
        1 if a.get("parsed_ok") else 0,
        -int(a.get("attempt_idx", 0)),
    )

  sorted_atts = sorted(attempts, key=_score_key, reverse=True)
  distinct: list[dict[str, Any]] = []
  seen_grids: set[str] = set()
  for a in sorted_atts:
    if a.get("pred_grid") is not None:
      g_str = json.dumps(a["pred_grid"], separators=(",", ":"))
      if g_str not in seen_grids:
        seen_grids.add(g_str)
        distinct.append(a)
  for a in sorted_atts:
    if a not in distinct:
      distinct.append(a)
  return distinct[:2]


def compute_metrics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  """Computes live pass@1 and pass@2 (pair-weighted and official task-weighted)."""
  n = len(rows)
  if n == 0:
    return {"pairs_completed": 0}

  p1 = sum(1 for r in rows if r["pass_at_1"])
  p2 = sum(1 for r in rows if r["pass_at_2"])

  by_task: dict[str, list[dict[str, Any]]] = {}
  for r in rows:
    by_task.setdefault(r["task_id"], []).append(r)
  num_tasks = len(by_task)
  task_p1 = sum(
      sum(1.0 for r in t_rows if r["pass_at_1"]) / len(t_rows)
      for t_rows in by_task.values()
  )
  task_p2 = sum(
      sum(1.0 for r in t_rows if r["pass_at_2"]) / len(t_rows)
      for t_rows in by_task.values()
  )

  all_atts = [a for r in rows for a in r.get("attempts", [])]
  n_att = max(1, len(all_atts))

  return {
      "recipe": FLAGS.recipe,
      "model_path": FLAGS.model_path,
      "backend": rows[-1].get("backend", ""),
      "pairs_completed": n,
      "tasks_completed": num_tasks,
      "pass_at_1_pairs": p1,
      "pass_at_1_pair_pct": round(100.0 * p1 / n, 2),
      "pass_at_2_pairs": p2,
      "pass_at_2_pair_pct": round(100.0 * p2 / n, 2),
      "official_task_pass_at_1_score": round(task_p1, 2),
      "official_task_pass_at_1_pct": round(
          100.0 * task_p1 / max(1, num_tasks), 2
      ),
      "official_task_pass_at_2_score": round(task_p2, 2),
      "official_task_pass_at_2_pct": round(
          100.0 * task_p2 / max(1, num_tasks), 2
      ),
      "train_passed_attempt_rate_pct": round(
          100.0 * sum(1 for a in all_atts if a.get("train_passed")) / n_att, 2
      ),
      "mean_turns_per_attempt": round(
          sum(a.get("turns", 0) for a in all_atts) / n_att, 2
      ),
      "mean_output_tokens_per_attempt": round(
          sum(a.get("total_output_tokens", 0) for a in all_atts) / n_att, 1
      ),
      "mean_latency_s_per_attempt": round(
          sum(a.get("latency_s", 0.0) for a in all_atts) / n_att, 2
      ),
      "solved_task_ids_pass_at_1": sorted(
          tid
          for tid, trs in by_task.items()
          if all(r["pass_at_1"] for r in trs)
      ),
      "solved_task_ids_pass_at_2": sorted(
          tid
          for tid, trs in by_task.items()
          if all(r["pass_at_2"] for r in trs)
      ),
      "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
  }


def load_all_completed_rows(
    output_dir: str,
) -> dict[tuple[str, int], dict[str, Any]]:
  """Loads completed rows across all shard JSONL files in `output_dir`."""
  completed: dict[tuple[str, int], dict[str, Any]] = {}
  if not gfile.Exists(output_dir):
    return completed
  try:
    filenames = sorted(gfile.ListDir(output_dir))
  except Exception:
    return completed
  for fn in filenames:
    if fn.startswith("predictions") and fn.endswith(".jsonl"):
      fpath = os.path.join(output_dir, fn)
      try:
        with gfile.Open(fpath, "r") as f:
          for line in f:
            line = line.strip()
            if not line:
              continue
            row = json.loads(line)
            completed[(row["task_id"], int(row["pair_idx"]))] = row
      except Exception as e:
        logging.warning("Error reading %s: %s", fpath, e)
  return completed


def save_kaggle_submission(
    output_dir: str, rows: list[dict[str, Any]]
) -> None:
  """Writes official Kaggle submission JSON (`attempt_1`, `attempt_2` per task)."""
  submission: dict[str, list[dict[str, Any]]] = {}
  by_task: dict[str, list[dict[str, Any]]] = {}
  for r in rows:
    by_task.setdefault(r["task_id"], []).append(r)
  for tid, t_rows in sorted(by_task.items()):
    t_rows.sort(key=lambda x: int(x["pair_idx"]))
    task_preds = []
    for r in t_rows:
      att1 = r.get("attempt_1_grid") or [[0]]
      att2 = r.get("attempt_2_grid") or att1
      task_preds.append({"attempt_1": att1, "attempt_2": att2})
    submission[tid] = task_preds

  sub_path = os.path.join(output_dir, "kaggle_submission.json")
  with gfile.Open(sub_path, "w") as f:
    json.dump(submission, f)


def load_arc_tasks(data_dir: str) -> list[tuple[str, dict[str, Any]]]:
  files = sorted(f for f in gfile.ListDir(data_dir) if f.endswith(".json"))

  def _load_one(fname: str) -> tuple[str, dict[str, Any]]:
    tid = fname[:-5]
    with gfile.Open(os.path.join(data_dir, fname), "r") as f:
      return tid, json.load(f)

  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    return list(ex.map(_load_one, files))


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  gfile.MakeDirs(FLAGS.output_dir)
  shard_suffix = (
      f"_shard_{FLAGS.shard_idx:03d}of{FLAGS.num_shards:03d}"
      if FLAGS.num_shards > 1
      else ""
  )
  jsonl_path = os.path.join(
      FLAGS.output_dir, f"predictions{shard_suffix}.jsonl"
  )
  summary_path = os.path.join(FLAGS.output_dir, "summary.json")

  tasks = load_arc_tasks(FLAGS.data_dir)
  wanted_ids: list[str] = []
  if FLAGS.task_ids_file:
    with gfile.Open(FLAGS.task_ids_file, "r") as tf:
      obj = json.load(tf)
    wanted_ids = list(obj[FLAGS.task_ids_key] if isinstance(obj, dict) else obj)
  if FLAGS.task_ids:
    wanted_ids += [t.strip() for t in FLAGS.task_ids.split(",") if t.strip()]
  if wanted_ids:
    keep = set(wanted_ids)
    tasks = [(tid, t) for tid, t in tasks if tid in keep]
  if FLAGS.limit_tasks > 0:
    tasks = tasks[: FLAGS.limit_tasks]

  all_work: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
  for tid, task in tasks:
    for pair_idx, test_pair in enumerate(task["test"]):
      all_work.append((tid, pair_idx, task, test_pair))

  shard_work = [
      item
      for idx, item in enumerate(all_work)
      if idx % FLAGS.num_shards == FLAGS.shard_idx
  ]

  completed_map = load_all_completed_rows(FLAGS.output_dir)
  todo = [
      (tid, pidx, task, tpair)
      for tid, pidx, task, tpair in shard_work
      if (tid, pidx) not in completed_map
  ]

  logging.info(
      "ARC-AGI-2 Qwen 28B Tool Evaluation | Shard %d/%d | Total pairs: %d | "
      "Shard pairs: %d | Remaining todo: %d | Output dir: %s",
      FLAGS.shard_idx,
      FLAGS.num_shards,
      len(all_work),
      len(shard_work),
      len(todo),
      FLAGS.output_dir,
  )

  engine = QwenVllmH100Engine(
      model_path=FLAGS.model_path,
      fp8_model_path=FLAGS.fp8_model_path,
      max_model_len=FLAGS.max_model_len,
      gpu_memory_utilization=FLAGS.gpu_memory_utilization,
  )
  engine.load()

  for idx, (tid, pair_idx, task, test_pair) in enumerate(todo, start=1):
    test_in = test_pair["input"]
    target = test_pair["output"]
    attempts = []
    for att_i in range(1, FLAGS.num_attempts + 1):
      att_res = run_single_attempt(
          engine=engine,
          task=task,
          test_input=test_in,
          target=target,
          attempt_idx=att_i,
      )
      attempts.append(att_res)
      logging.info(
          "[Shard %d/%d | %d/%d] %s pair %d att %d -> exact=%s train_passed=%s "
          "turns=%d tools=%d toks=%d (%.1fs)",
          FLAGS.shard_idx,
          FLAGS.num_shards,
          idx,
          len(todo),
          tid,
          pair_idx,
          att_i,
          att_res["exact_match"],
          att_res["train_passed"],
          att_res["turns"],
          att_res["tool_calls"],
          att_res["total_output_tokens"],
          att_res["latency_s"],
      )

    ranked = rank_attempts(attempts)
    att1 = ranked[0]
    att2 = ranked[1] if len(ranked) > 1 else ranked[0]
    pass_at_1 = bool(att1["exact_match"])
    pass_at_2 = bool(att1["exact_match"] or att2["exact_match"])

    row = {
        "task_id": tid,
        "pair_idx": pair_idx,
        "recipe": FLAGS.recipe,
        "backend": engine.backend_name,
        "pass_at_1": pass_at_1,
        "pass_at_2": pass_at_2,
        "attempt_1_grid": att1["pred_grid"],
        "attempt_2_grid": att2["pred_grid"],
        "attempts": attempts,
    }

    with gfile.Open(jsonl_path, "a") as f:
      f.write(json.dumps(row) + "\n")

    # Reload all shards' rows and update live pass@1 / pass@2 summary + Kaggle submission
    all_completed = load_all_completed_rows(FLAGS.output_dir)
    all_rows_sorted = [
        all_completed[k] for k in sorted(all_completed.keys())
    ]
    summary = compute_metrics_summary(all_rows_sorted)
    with gfile.Open(summary_path, "w") as f:
      json.dump(summary, f, indent=2)
    save_kaggle_submission(FLAGS.output_dir, all_rows_sorted)

    print(
        f"=== PROGRESS [{summary['pairs_completed']}/{len(all_work)} pairs |"
        f" {summary['tasks_completed']} tasks] "
        f"Pair pass@1={summary['pass_at_1_pair_pct']:.2f}%"
        f" ({summary['pass_at_1_pairs']}/{summary['pairs_completed']}) | "
        f"Pair pass@2={summary['pass_at_2_pair_pct']:.2f}%"
        f" ({summary['pass_at_2_pairs']}/{summary['pairs_completed']}) | "
        f"Official Task pass@1={summary['official_task_pass_at_1_pct']:.2f}% | "
        f"Official Task pass@2={summary['official_task_pass_at_2_pct']:.2f}% ===",
        flush=True,
    )


if __name__ == "__main__":
  app.run(main)
