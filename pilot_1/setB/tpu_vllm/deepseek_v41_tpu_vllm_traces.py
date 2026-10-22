"""Multi-TPU Full-Precision Reasoning Trace Generator for DeepSeek-V4.1-Flash-1017 on Top 10K ARC SFT Candidates.

Supports two full-precision TPU execution engines behind a unified vLLM interface:
1. `ds41_full_precision` (`DeepSeek-V4.1-Flash-1017` from `/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1`
   and pre-sliced TPU OCDBT weights `/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017-ocdbt-tp8/1`):
   - Dequantizes all FP8 (E4M3FN) and MXFP4 (E2M1) weights by their exact E8M0 block scales into `bfloat16`
     and executes all TPU MXU matmuls with full IEEE-754 `float32` accumulation (`BF16 x BF16 -> FP32`).
   - Uses Two-Phase Causal Encoder-Decoder (CED) + Bounded SWA Replay Prefill, 196.6B Engram conditional
     memory lookup (L1, L14), and 3-layer DSpark 1+5 speculative decoding (`speculative_step_fn`).
   - Encodes prompts with `ds4_encoding` (`Reasoning Effort: 100`, `<think>...</think>`) and supports both
     `<tool_call><function=run_python>...</tool_call>` and `<｜DSML｜ calls><｜DSML｜ invoke name="run_python">...`.
2. `vllm` (vLLM TPU engine with `dtype="bfloat16"`, `load_format="tpu_streaming_loader"`):
   - Runs vLLM's native TPU inference engine (`DeepseekV4ForCausalLM` / `Qwen3.8`) with continuous batching
     and prefix caching in `bfloat16` + `float32` MXU accumulation.

Reads the Top 10,000 highest-learning-signal candidates directly from CNS:
  `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/top_10k_deepseek_candidates.jsonl`
and performs 3-Level Verification on every trajectory:
  - Level 1: Demonstration pairs (`train`)
  - Level 2: Held-out test pair (`test`)
  - Level 3: Withheld extra generator check pairs (`extra_verification_pairs`, up to 25 NVARC / 10 ARC-GEN pairs)

Saves incremental outputs to CNS (`--output_dir`):
  - `verified_sft_trajectories_shard_XXXofYYY.jsonl`
  - `failed_or_overfit_trajectories_shard_XXXofYYY.jsonl`
  - `predictions_shard_XXXofYYY.jsonl`
  - `summary_shard_XXXofYYY.json` & `summary.json`
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent import futures
import copy
import dataclasses
import gc
import json
import math
import os
import re
import time
from typing import Any

from absl import flags
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import tensorstore as ts
import tokenizers
import tqdm
from vllm.g3_integration import app_runner

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import decode_megakernel as mk
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_encoding
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_engine_fns as engine_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_layout as layout_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_ref as ref
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import prompts_and_voting
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import qwen_tpu_vllm_eval as runner_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import sandbox
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "engine_backend",
    "auto",
    ["auto", "ds41_full_precision", "vllm"],
    "Execution backend: 'ds41_full_precision' (DeepSeek-V4.1-Flash-1017 Pallas/JAX full-precision BF16->FP32 megakernel), "
    "'vllm' (vLLM TPU engine in bfloat16), or 'auto' (try ds41_full_precision when 8 TensorCores are available, else vllm).",
)
flags.DEFINE_string(
    "ds41_raw_checkpoint_dir",
    "/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1",
    "TFHub/CNS path to DeepSeek-V4.1-Flash-1017 raw checkpoint & tokenizer.json.",
)
flags.DEFINE_string(
    "ds41_ocdbt_tp8_dir",
    "/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017-ocdbt-tp8/1",
    "TFHub/CNS path to pre-sliced 8-TensorCore OCDBT Zarr weights for DeepSeek-V4.1-Flash-1017.",
)
flags.DEFINE_string(
    "ds41_ocdbt_tp16_dir",
    "/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017-ocdbt-tp16/1",
    "TFHub/CNS path to pre-sliced 16-TensorCore OCDBT Zarr weights for DeepSeek-V4.1-Flash-1017.",
)
flags.DEFINE_string(
    "vllm_model_path",
    "/cns/yutulis-d/home/ml-gemini-prod-jobs/kedem/ttl=30d/models/DeepSeek-V4-Flash",
    "CNS path to vLLM-compatible model checkpoint when using --engine_backend=vllm.",
)
flags.DEFINE_string(
    "fallback_vllm_model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "Fallback vLLM model path if primary vLLM checkpoint fails to initialize.",
)
flags.DEFINE_string(
    "candidates_jsonl",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/top_10k_deepseek_candidates.jsonl",
    "CNS JSONL file containing the Top 10,000 highest-learning-signal SFT candidates.",
)
flags.DEFINE_integer(
    "reasoning_effort",
    100,
    "DeepSeek-V4.1 Reasoning Effort integer in [1, 100] (100 = maximum reasoning thoroughness).",
)


# ==============================================================================
# 1. Compatibility Wrappers for vLLM-Style `.generate()` on DeepSeek-V4.1-Flash-1017
# ==============================================================================


@dataclasses.dataclass
class CompatSamplingParams:
  temperature: float = 0.7
  top_p: float = 0.95
  top_k: int = 20
  repetition_penalty: float = 1.0
  max_tokens: int = 4096
  stop: list[str] | None = None


@dataclasses.dataclass
class CompatCompletionOutput:
  text: str
  token_ids: list[int]
  finish_reason: str


@dataclasses.dataclass
class CompatRequestOutput:
  prompt_token_ids: list[int]
  outputs: list[CompatCompletionOutput]


class HFTokenizerWrapper:
  """Adapts `tokenizers.Tokenizer` to the `.encode()` / `.decode()` interface expected by the runner."""

  def __init__(self, tok: tokenizers.Tokenizer):
    self._tok = tok

  def encode(self, text: str) -> list[int]:
    return list(self._tok.encode(text, add_special_tokens=False).ids)

  def decode(self, token_ids: Sequence[int]) -> str:
    return self._tok.decode(list(token_ids), skip_special_tokens=False)


class DeepSeekV41FullPrecisionLLM:
  """Full-precision (`bfloat16` dequantization + `float32` MXU accumulation) TPU engine for `DeepSeek-V4.1-Flash-1017`.

  Uses `ds4_engine_fns.make_ds4_engine_functions` with static shape `(max_seq_len,)`
  chunked Two-Phase CED prefill (`lax.fori_loop` over 512-token chunks up to `true_len`)
  and batched `batch_size=4`, `steps_per_call=4` speculative verification + DSpark drafting
  so JAX compiles once at startup and never recompiles across variable-length turns.
  """

  def __init__(
      self,
      raw_checkpoint_dir: str,
      ocdbt_tp8_dir: str,
      ocdbt_tp16_dir: str,
      max_seq_len: int = 16384,
      batch_size: int = 8,
      steps_per_call: int = 8,
  ):
    t0 = time.time()
    devs = sorted(jax.devices(), key=layout_lib.rank_sort_key)
    n_devs = len(devs)
    if n_devs not in (8, 16):
      raise ValueError(
          f"DeepSeekV41FullPrecisionLLM requires 8 (gf=2x2x1) or 16 (gf=2x2x2) TensorCores, got {n_devs}"
      )
    self.mesh = Mesh(np.asarray(devs[:n_devs]), (layout_lib.AXIS,))
    self.rep = NamedSharding(self.mesh, P())
    self.tp = NamedSharding(self.mesh, P(layout_lib.AXIS))

    cfg_path = os.path.join(raw_checkpoint_dir, "config.json")
    with gfile.Open(cfg_path, "r") as f:
      self.cfg = ref.Ds4Config.from_dict(json.load(f))
    self.lay = layout_lib.build_layout(self.cfg, num_devices=n_devs)

    tok_path = os.path.join(raw_checkpoint_dir, "tokenizer.json")
    with gfile.Open(tok_path, "r") as f:
      raw_tok = f.read()
    self._raw_tokenizer = tokenizers.Tokenizer.from_str(raw_tok)
    self._tokenizer = HFTokenizerWrapper(self._raw_tokenizer)

    export_dir = ocdbt_tp8_dir if n_devs == 8 else ocdbt_tp16_dir
    print(
        f"[DS4.1-FP] Loading 71 pre-sliced full-precision weight leaves from {export_dir}"
        f" onto {n_devs} TensorCores...",
        flush=True,
    )
    sharded_w = self._load_ocdbt_weights_via_tensorstore(export_dir)
    print(
        f"[DS4.1-FP] Loaded 71 weight leaves onto {n_devs} TensorCores in {time.time() - t0:.1f}s!",
        flush=True,
    )

    # Ensure max_seq_len is a multiple of 512 (4 * window_size) for static chunked CED prefill
    self.max_seq_len = max(4096, (min(max_seq_len, 16384) // 512) * 512)
    self.batch_size = batch_size
    self.steps_per_call = steps_per_call

    t_comp0 = time.time()
    aux_np, _ = engine_lib.aux_table_arrays(self.cfg, self.max_seq_len)

    def _stack(arr: np.ndarray) -> jax.Array:
      return jax.jit(
          lambda a: jnp.broadcast_to(a[None], (n_devs, *a.shape)),
          out_shardings=self.tp,
      )(jax.device_put(arr, self.rep))

    ds4_vars = dict(sharded_w)
    ds4_vars.update({k: _stack(v) for k, v in aux_np.items()})
    self.variables = {"params": {"ds4": ds4_vars}}

    prefill_fn, generate_fn, insert_fn, state_init_fn, _, _ = (
        engine_lib.make_ds4_engine_functions(
            self.mesh,
            self.lay,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            steps_per_call=self.steps_per_call,
        )
    )

    class _GenMod(engine_lib.nn.Module):

      @engine_lib.nn.compact
      def __call__(self, t, st, fn=generate_fn):
        return fn(t, st)

    class _PrefMod(engine_lib.nn.Module):

      @engine_lib.nn.compact
      def __call__(self, pr, fn=prefill_fn):
        return fn(pr)

    self._gen_jit = jax.jit(_GenMod().apply, donate_argnums=(2,))
    self._pref_jit = jax.jit(_PrefMod().apply)
    self._ins_jit = jax.jit(insert_fn, donate_argnums=(0,))
    self._state_init_jit = jax.jit(
        lambda: state_init_fn(self.batch_size), out_shardings=self.rep
    )
    print(
        f"[DS4.1-FP] Compiled static-shape Two-Phase CED Prefill + Batched DSpark Speculative Decode "
        f"(batch_size={self.batch_size}, steps_per_call={self.steps_per_call}, max_seq_len={self.max_seq_len}) "
        f"in {time.time() - t_comp0:.1f}s.",
        flush=True,
    )

  def _load_ocdbt_weights_via_tensorstore(
      self, export_dir: str
  ) -> dict[str, jax.Array]:
    """Streams all 71 pre-sliced OCDBT Zarr leaves in parallel onto the 8/16 TPU TensorCores."""
    specs = self.lay.abstract_device_arrays()
    dummy_tree = {name: 0 for name in specs}
    _, treedef = jax.tree.flatten(dummy_tree)
    ordered_names = jax.tree.unflatten(treedef, list(range(len(specs))))
    idx_to_name = {idx: name for name, idx in ordered_names.items()}

    n_devs = self.lay.num_devices
    ranks = list(self.mesh.devices.reshape(-1))
    ctx = ts.Context({
        "file_io_concurrency": {"limit": 256},
        "data_copy_concurrency": {"limit": 64},
    })

    num_leaves = len(specs)
    store_futs = [
        ts.open(
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "ocdbt",
                    "base": {"driver": "gfile", "path": export_dir},
                    "path": str(leaf_idx),
                },
            },
            context=ctx,
            read=True,
            open=True,
        )
        for leaf_idx in range(num_leaves)
    ]
    stores = [sf.result() for sf in store_futs]

    def _start_leaf_reads(leaf_idx: int) -> list[Any]:
      st = stores[leaf_idx]
      return [st[d : d + 1].read() for d in range(n_devs)]

    sharded_w: dict[str, jax.Array] = {}
    cur_read_futs = _start_leaf_reads(0) if num_leaves > 0 else []
    pbar = tqdm.tqdm(
        range(num_leaves),
        desc="[DS4.1-FP] Loading OCDBT leaves to TPU HBM",
        dynamic_ncols=True,
    )
    for leaf_idx in pbar:
      name = idx_to_name[leaf_idx]
      local_shape, want_dt = specs[name]
      next_read_futs = (
          _start_leaf_reads(leaf_idx + 1)
          if (leaf_idx + 1) < num_leaves
          else []
      )
      dev_arrays: list[jax.Array] = []
      for d, fut in enumerate(cur_read_futs):
        slice_np = np.asarray(fut.result(), dtype=want_dt)
        dev_arrays.append(
            jax.device_put(np.ascontiguousarray(slice_np), ranks[d])
        )
        del slice_np
      sharded_w[name] = jax.make_array_from_single_device_arrays(
          (n_devs, *local_shape), self.tp, dev_arrays
      )
      cur_read_futs = next_read_futs
      pbar.set_postfix(leaf=name)
    pbar.close()
    return sharded_w

  def get_tokenizer(self) -> HFTokenizerWrapper:
    return self._tokenizer

  def _generate_subbatch(
      self,
      sub_prompts: list[str],
      sub_params: list[CompatSamplingParams],
  ) -> list[CompatRequestOutput]:
    """Runs static-shape prefill + batched speculative decode for up to `self.batch_size` sequences."""
    b_active = len(sub_prompts)
    dec_state = self._state_init_jit()
    prompt_token_lists: list[list[int]] = []
    emitted_lists: list[list[int]] = [[] for _ in range(b_active)]
    finish_reasons: list[str] = ["length" for _ in range(b_active)]
    slot_done: list[bool] = [False for _ in range(b_active)]
    eos_ids = {self.cfg.eos_token_id, self.cfg.pad_token_id}

    # 1. Static-shape Two-Phase CED Prefill + Insert into batched decode state
    for slot_idx, (p_str, sp) in enumerate(zip(sub_prompts, sub_params)):
      p_ids = self._tokenizer.encode(p_str)
      if not p_ids or p_ids[0] != self.cfg.bos_token_id:
        p_ids = [self.cfg.bos_token_id] + p_ids
      max_prompt = self.max_seq_len - max(512, sp.max_tokens) - 128
      if len(p_ids) > max_prompt:
        p_ids = p_ids[:256] + p_ids[-(max_prompt - 256) :]
      prompt_token_lists.append(p_ids)

      padded_np = np.full(
          (self.max_seq_len,), engine_lib.DS4_PAD_ID, dtype=np.int32
      )
      padded_np[: len(p_ids)] = np.asarray(p_ids, dtype=np.int32)
      prompt_tuple = (
          jax.device_put(jnp.asarray(padded_np, dtype=jnp.int32), self.rep),
          jax.device_put(jnp.int32(len(p_ids)), self.rep),
      )
      _, _, pst = self._pref_jit(self.variables, prompt_tuple)
      dec_state = self._ins_jit(dec_state, pst, jnp.int32(slot_idx))
      del pst

    # Record the first token emitted by prefill for each active slot
    first_toks_np = np.asarray(dec_state.value["token"], dtype=np.int32)
    for slot_idx in range(b_active):
      ft = int(first_toks_np[slot_idx])
      if ft in eos_ids:
        finish_reasons[slot_idx] = "stop"
        slot_done[slot_idx] = True
      else:
        emitted_lists[slot_idx].append(ft)

    # 2. Batched speculative verify + DSpark draft loop (steps_per_call=4 -> up to 24 tokens/call)
    max_req_tokens = max(sp.max_tokens for sp in sub_params)
    while not all(slot_done):
      seed_np = np.full(
          (self.batch_size, 1), engine_lib.DS4_PAD_ID, dtype=np.int32
      )
      for slot_idx in range(b_active):
        if not slot_done[slot_idx]:
          seed_np[slot_idx, 0] = emitted_lists[slot_idx][-1]
      seed_dev = jax.device_put(jnp.asarray(seed_np, dtype=jnp.int32), self.rep)
      step_out, dec_state = self._gen_jit(self.variables, seed_dev, dec_state)
      emitted_bs_np = np.asarray(step_out.emitted, dtype=np.int32)
      emit_cnt_np = np.asarray(step_out.emit_count, dtype=np.int32)
      pos_np = np.asarray(dec_state.value["position"], dtype=np.int32)

      for slot_idx in range(b_active):
        if slot_done[slot_idx]:
          continue
        sp = sub_params[slot_idx]
        cnt = int(emit_cnt_np[slot_idx])
        # Note: emitted_bs_np[slot_idx, 0] is the anchor token already in emitted_lists[slot_idx][-1];
        # newly verified tokens are indices 1 .. cnt-1, plus step_out.last_token is the next anchor!
        row_toks = [int(x) for x in emitted_bs_np[slot_idx, 1:cnt]]
        row_toks.append(int( np.asarray(step_out.last_token, dtype=np.int32)[slot_idx] ))
        for tv in row_toks:
          if tv in eos_ids:
            finish_reasons[slot_idx] = "stop"
            slot_done[slot_idx] = True
            break
          if len(emitted_lists[slot_idx]) < sp.max_tokens:
            emitted_lists[slot_idx].append(tv)
          if len(emitted_lists[slot_idx]) >= sp.max_tokens:
            finish_reasons[slot_idx] = "length"
            slot_done[slot_idx] = True
            break

        if not slot_done[slot_idx] and int(pos_np[slot_idx]) + 32 >= self.max_seq_len:
          finish_reasons[slot_idx] = "length"
          slot_done[slot_idx] = True

        if not slot_done[slot_idx] and sp.stop:
          tail_txt = self._tokenizer.decode(emitted_lists[slot_idx][-64:])
          if any(st in tail_txt for st in sp.stop):
            finish_reasons[slot_idx] = "stop"
            slot_done[slot_idx] = True

      if all(len(emitted_lists[i]) >= max_req_tokens for i in range(b_active)):
        break

    results: list[CompatRequestOutput] = []
    for slot_idx in range(b_active):
      sp = sub_params[slot_idx]
      toks = emitted_lists[slot_idx]
      txt = self._tokenizer.decode(toks)
      f_reason = finish_reasons[slot_idx]
      if sp.stop:
        for st in sp.stop:
          if st in txt:
            txt = txt.split(st)[0]
            f_reason = "stop"
      results.append(
          CompatRequestOutput(
              prompt_token_ids=prompt_token_lists[slot_idx],
              outputs=[
                  CompatCompletionOutput(
                      text=txt,
                      token_ids=toks,
                      finish_reason=f_reason,
                  )
              ],
          )
      )
    return results

  def generate(
      self,
      prompts: Sequence[Any],
      sampling_params: Sequence[Any] | Any,
      use_tqdm: bool = False,
  ) -> list[CompatRequestOutput]:
    if not isinstance(sampling_params, Sequence):
      params_list = [sampling_params] * len(prompts)
    else:
      params_list = list(sampling_params)
    str_prompts = [
        p["prompt"] if isinstance(p, dict) else str(p) for p in prompts
    ]
    outs: list[CompatRequestOutput] = []
    rng = range(0, len(str_prompts), self.batch_size)
    if use_tqdm:
      rng = tqdm.tqdm(rng, desc="[DS4.1-FP] Batched Generate", leave=False)
    for start in rng:
      sub_p = str_prompts[start : start + self.batch_size]
      sub_sp = params_list[start : start + self.batch_size]
      outs.extend(self._generate_subbatch(sub_p, sub_sp))
    return outs


class CompatVllmNamespace:
  SamplingParams = CompatSamplingParams


# ==============================================================================
# 2. DSML + XML Tool Call Parser & 3-Level Verification Helper
# ==============================================================================

_DSML_INVOKE_RE = re.compile(
    r"<[｜|]DSML[｜|]\s*invoke\s+name=\"([^\"]+)\"\s*>(.*?)</[｜|]DSML[｜|]\s*invoke>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    r"<[｜|]DSML[｜|]\s*parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</[｜|]DSML[｜|]\s*parameter>",
    re.DOTALL,
)


def parse_any_tool_calls(text: str) -> list[tuple[str, str]]:
  """Parses both `<tool_call><function=...>` and DeepSeek-V4.1 `<｜DSML｜ invoke name=...>` tool calls."""
  calls = prompts_and_voting.parse_tool_calls(text)
  if calls:
    return calls
  dsml_calls: list[tuple[str, str]] = []
  for m in _DSML_INVOKE_RE.finditer(text):
    fn_name = m.group(1).strip()
    body = m.group(2)
    params = {pm.group(1).strip(): pm.group(2) for pm in _DSML_PARAM_RE.finditer(body)}
    code = params.get("code", "").strip()
    if fn_name in ("run_python", "submit_program") and code:
      dsml_calls.append((fn_name, code))
  return dsml_calls


def extract_last_submitted_code(transcript: str) -> str | None:
  """Extracts the latest `def solve(grid):` Python code from a multi-turn transcript."""
  matches = re.findall(
      r"<function=(?:submit_program|run_python)>.*?<parameter=code>\n?(.*?)</parameter>",
      transcript,
      flags=re.DOTALL,
  )
  for m in reversed(matches):
    if "def solve" in m:
      return m.strip()
  for m in reversed(list(_DSML_INVOKE_RE.finditer(transcript))):
    for pm in _DSML_PARAM_RE.finditer(m.group(2)):
      if pm.group(1).strip() == "code" and "def solve" in pm.group(2):
        return pm.group(2).strip()
  code_blocks = re.findall(r"```python\n(.*?)```", transcript, flags=re.DOTALL)
  for cb in reversed(code_blocks):
    if "def solve" in cb:
      return cb.strip()
  return None


def verify_level3_extra_pairs(
    code_str: str | None,
    extra_pairs: list[dict[str, Any]],
) -> tuple[bool, float, int]:
  """Executes `def solve(grid):` against withheld Level-3 extra check pairs (NVARC 25-pair / ARC-GEN 10-pair)."""
  if not extra_pairs:
    return True, 1.0, 0
  if not code_str or "def solve" not in code_str:
    return False, 0.0, len(extra_pairs)
  ns: dict[str, Any] = {
      "np": np,
      "Counter": sandbox.Counter,
      "deque": sandbox.deque,
      "math": sandbox.math,
      "objects": sandbox.find_objects,
      "find_objects": sandbox.find_objects,
      "bbox": sandbox.bbox,
      "crop": sandbox.crop,
      "bg_colour": sandbox.bg_colour,
      "d8": sandbox.d8,
  }
  try:
    import scipy.ndimage  # pylint: disable=g-import-not-at-top
    ns["scipy"] = scipy
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  try:
    exec(code_str, ns)  # pylint: disable=exec-used
    solve_fn = ns.get("solve")
    if not callable(solve_fn):
      return False, 0.0, len(extra_pairs)
    passed = 0
    for ep in extra_pairs:
      inp = copy.deepcopy(ep["input"])
      exp = ep["output"]
      got_raw = solve_fn(inp)
      got = sandbox.normalize_grid(got_raw)
      if got == exp:
        passed += 1
    rate = passed / len(extra_pairs)
    return (passed == len(extra_pairs)), round(rate, 4), len(extra_pairs)
  except Exception:  # pylint: disable=broad-exception-caught
    return False, 0.0, len(extra_pairs)


def build_ds41_initial_transcript(
    task: dict[str, Any],
    test_input: list[list[int]],
    soft_target: int,
    hard_budget: int,
    reasoning_effort: int,
    structured_summary: bool,
    diverse_attempt_idx: int | None,
    use_ds41_harmony: bool,
) -> str:
  """Builds the initial prompt for DeepSeek-V4.1-Flash-1017 (or Qwen/vLLM fallback)."""
  if not use_ds41_harmony:
    return prompts_and_voting.build_initial_transcript(
        task=task,
        test_input=test_input,
        use_tools=True,
        soft_target=soft_target,
        hard_budget=hard_budget,
        use_images=False,
        multi_hypothesis=True,
        coder_next_format=False,
        structured_summary=structured_summary,
        diverse_attempt_idx=diverse_attempt_idx,
    )
  user_body = prompts_and_voting.build_user_prompt(
      task=task,
      test_input=test_input,
      use_tools=True,
      soft_target=soft_target,
      hard_budget=hard_budget,
      use_images=False,
      multi_hypothesis=True,
      structured_summary=structured_summary,
      diverse_attempt_idx=diverse_attempt_idx,
  )
  sys_content = (
      f"Reasoning Effort: {reasoning_effort} (range 1-100, the higher the value,"
      " the more thorough the reasoning)\n\n"
      + prompts_and_voting.TOOLS_SYSTEM_PROGRAM
  )
  return (
      f"{ds4_encoding.bos_token}{sys_content}"
      f"{ds4_encoding.USER_SP_TOKEN}{user_body}"
      f"{ds4_encoding.ASSISTANT_SP_TOKEN}{ds4_encoding.thinking_start_token}\n"
  )


# ==============================================================================
# 3. Main Multi-TPU Runner with Live `tqdm` and 3-Level Verification
# ==============================================================================


def load_completed_task_ids(output_dir: str, shard_suffix: str) -> set[str]:
  done: set[str] = set()
  if not gfile.Exists(output_dir):
    return done
  fpath = os.path.join(output_dir, f"predictions{shard_suffix}.jsonl")
  if not gfile.Exists(fpath):
    return done
  try:
    with gfile.Open(fpath, "r") as f:
      for line in f:
        if line.strip():
          row = json.loads(line)
          done.add(row["task_id"])
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return done


def main(argv: Sequence[str]) -> None:
  del argv
  t0 = time.time()
  # 1. Load Top 10,000 candidates directly from CNS JSONL (fast single-stream read)
  candidates: list[dict[str, Any]] = []
  with gfile.Open(FLAGS.candidates_jsonl, "r") as f:
    for line in f:
      if not line.strip():
        continue
      candidates.append(json.loads(line))
      if FLAGS.limit_tasks > 0 and len(candidates) >= FLAGS.limit_tasks:
        break

  shard_candidates = [
      c
      for idx, c in enumerate(candidates)
      if idx % FLAGS.num_shards == FLAGS.shard_idx
  ]
  gfile.MakeDirs(FLAGS.output_dir)
  shard_suffix = (
      f"_shard_{FLAGS.shard_idx:03d}of{FLAGS.num_shards:03d}"
      if FLAGS.num_shards > 1
      else ""
  )
  preds_path = os.path.join(FLAGS.output_dir, f"predictions{shard_suffix}.jsonl")
  verified_sft_path = os.path.join(
      FLAGS.output_dir, f"verified_sft_trajectories{shard_suffix}.jsonl"
  )
  failed_sft_path = os.path.join(
      FLAGS.output_dir, f"failed_or_overfit_trajectories{shard_suffix}.jsonl"
  )
  shard_summary_path = os.path.join(
      FLAGS.output_dir, f"summary{shard_suffix}.json"
  )
  global_summary_path = os.path.join(FLAGS.output_dir, "summary.json")

  done_ids = load_completed_task_ids(FLAGS.output_dir, shard_suffix)
  todo_candidates = [
      c for c in shard_candidates if c["deepseek_task_id"] not in done_ids
  ]

  num_tpus = jax.device_count()
  device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
  print(
      f"[{FLAGS.exp_name}] Loaded {len(candidates)} top candidates in {time.time() - t0:.1f}s | "
      f"Shard {FLAGS.shard_idx}/{FLAGS.num_shards}: {len(shard_candidates)} tasks "
      f"({len(todo_candidates)} remaining) | Hardware: {num_tpus}x {device_kind}",
      flush=True,
  )
  if not todo_candidates:
    print("All tasks for this shard are already completed!", flush=True)
    return

  # 2. Initialize Full-Precision DeepSeek-V4.1-Flash-1017 TPU Engine (or vLLM TPU Engine)
  llm: Any = None
  vllm_mod: Any = CompatVllmNamespace
  active_backend = FLAGS.engine_backend
  active_model_name = "DeepSeek-V4.1-Flash-1017 (Full-Precision BF16->FP32)"
  use_ds41_harmony = True

  if active_backend in ("auto", "ds41_full_precision") and num_tpus in (8, 16):
    try:
      llm = DeepSeekV41FullPrecisionLLM(
          raw_checkpoint_dir=FLAGS.ds41_raw_checkpoint_dir,
          ocdbt_tp8_dir=FLAGS.ds41_ocdbt_tp8_dir,
          ocdbt_tp16_dir=FLAGS.ds41_ocdbt_tp16_dir,
          max_seq_len=FLAGS.max_model_len,
      )
      active_backend = "ds41_full_precision"
    except Exception as ds_exc:  # pylint: disable=broad-exception-caught
      print(
          f"[WARN] DeepSeekV41FullPrecisionLLM init raised ({ds_exc}); "
          "falling back to vLLM TPU engine in bfloat16.",
          flush=True,
      )
      llm = None

  if llm is None:
    import vllm  # pylint: disable=g-import-not-at-top

    vllm_mod = vllm
    active_backend = "vllm"
    tp_size = num_tpus if num_tpus > 0 else 4
    try:
      print(
          f"[vLLM-FP] Initializing vLLM TPU engine on {FLAGS.vllm_model_path} "
          f"(dtype=bfloat16, tp={tp_size})...",
          flush=True,
      )
      llm = vllm.LLM(
          model=FLAGS.vllm_model_path,
          dtype="bfloat16",
          tensor_parallel_size=tp_size,
          max_model_len=FLAGS.max_model_len,
          max_num_batched_tokens=8192,
          max_num_seqs=FLAGS.max_num_seqs,
          gpu_memory_utilization=0.90,
          enable_prefix_caching=True,
          load_format="tpu_streaming_loader",
      )
      active_model_name = f"DeepSeek-V4-Flash ({FLAGS.vllm_model_path}, bfloat16)"
      use_ds41_harmony = False
    except Exception as v_exc:  # pylint: disable=broad-exception-caught
      print(
          f"[WARN] Primary vLLM model {FLAGS.vllm_model_path} raised ({v_exc}); "
          f"initializing {FLAGS.fallback_vllm_model_path}...",
          flush=True,
      )
      llm = vllm.LLM(
          model=FLAGS.fallback_vllm_model_path,
          tensor_parallel_size=min(4, tp_size),
          max_model_len=FLAGS.max_model_len,
          max_num_batched_tokens=8192,
          max_num_seqs=FLAGS.max_num_seqs,
          gpu_memory_utilization=0.90,
          enable_prefix_caching=True,
          load_format="tpu_streaming_loader",
      )
      active_model_name = FLAGS.fallback_vllm_model_path
      use_ds41_harmony = False

  # 3. Run Batched Multi-Turn Generation + 3-Level Verification with Live `tqdm`
  from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import qwen_tpu_vllm_eval as runner_lib  # pylint: disable=g-import-not-at-top

  t_eval_start = time.time()
  cum_gen_tokens = 0
  cum_gen_seconds = 0.0
  puzzles_done = len(done_ids)
  l1_pass_puzzles = 0
  l12_pass_puzzles = 0
  l123_verified_puzzles = 0
  verified_trajectories_saved = 0
  failed_trajectories_saved = 0

  pbar = tqdm.tqdm(
      total=len(todo_candidates),
      desc=f"{FLAGS.exp_name} [shard {FLAGS.shard_idx}/{FLAGS.num_shards}]",
      dynamic_ncols=True,
  )

  for b_start in range(0, len(todo_candidates), FLAGS.pair_batch_size):
    batch_cands = todo_candidates[b_start : b_start + FLAGS.pair_batch_size]
    wave_states: list[runner_lib.AttemptState] = []
    cand_by_tid: dict[str, dict[str, Any]] = {}

    for c in batch_cands:
      tid = c["deepseek_task_id"]
      cand_by_tid[tid] = c
      raw_task = {"train": c["train"], "test": c["test"]}
      tpair = c["test"][0]
      for a_idx in range(FLAGS.num_attempts):
        st = runner_lib.AttemptState(
            task_id=tid,
            pair_idx=0,
            attempt_idx=a_idx,
            raw_task=raw_task,
            raw_test_input=tpair["input"],
            target_output=tpair["output"],
            use_tools=True,
            enable_tta=False,
            use_images=False,
            multi_hypothesis=True,
            soft_target=FLAGS.soft_token_target,
            hard_budget=FLAGS.total_token_budget,
            coder_next_format=False,
        )
        if use_ds41_harmony:
          st.transcript = build_ds41_initial_transcript(
              task=raw_task,
              test_input=tpair["input"],
              soft_target=FLAGS.soft_token_target,
              hard_budget=FLAGS.total_token_budget,
              reasoning_effort=FLAGS.reasoning_effort,
              structured_summary=FLAGS.structured_summary,
              diverse_attempt_idx=a_idx if FLAGS.diverse_hypotheses else None,
              use_ds41_harmony=True,
          )
        wave_states.append(st)

    w_desc = f"batch {b_start // FLAGS.pair_batch_size + 1}/{math.ceil(len(todo_candidates) / FLAGS.pair_batch_size)}"
    w_toks, w_secs = runner_lib.run_attempt_wave_batched(
        llm=llm,
        vllm_mod=vllm_mod,
        states=wave_states,
        wave_desc=w_desc,
    )
    cum_gen_tokens += w_toks
    cum_gen_seconds += w_secs

    by_tid_states: dict[str, list[runner_lib.AttemptState]] = {}
    for st in wave_states:
      by_tid_states.setdefault(st.task_id, []).append(st)

    pred_rows: list[dict[str, Any]] = []
    verified_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for c in batch_cands:
      tid = c["deepseek_task_id"]
      extra_pairs = c.get("extra_verification_pairs", [])
      tpair = c["test"][0]
      st_list = by_tid_states.get(tid, [])
      att_recs: list[dict[str, Any]] = []

      any_l1 = False
      any_l12 = False
      any_l123 = False

      for st in st_list:
        rec = st.finalize_record()
        raw_transcript = st.transcript
        code_str = extract_last_submitted_code(raw_transcript)
        l1_ok = bool(rec.get("train_passed"))
        l2_ok = bool(rec.get("exact_match"))
        l3_ok, l3_rate, n_l3 = verify_level3_extra_pairs(code_str, extra_pairs)
        all_3_ok = bool(l1_ok and l2_ok and l3_ok)

        rec["level1_demo_pass"] = l1_ok
        rec["level2_heldout_pass"] = l2_ok
        rec["level3_extra_pass"] = l3_ok
        rec["level3_extra_pass_rate"] = l3_rate
        rec["num_level3_extra_pairs"] = n_l3
        rec["all_3_levels_verified"] = all_3_ok
        rec["submitted_solve_code"] = code_str

        any_l1 = any_l1 or l1_ok
        any_l12 = any_l12 or (l1_ok and l2_ok)
        any_l123 = any_l123 or all_3_ok
        att_recs.append(rec)

        sft_entry = {
            "deepseek_task_id": tid,
            "sample_id": c["sample_id"],
            "dataset": c["dataset"],
            "family_id": c["family_id"],
            "learning_tier": c["learning_tier"],
            "learning_signal_score": c["learning_signal_score"],
            "attempt_idx": rec["attempt_idx"],
            "model": active_model_name,
            "engine_backend": active_backend,
            "precision": "full_precision_bf16_dequant_fp32_mxu",
            "reasoning_effort": FLAGS.reasoning_effort,
            "level1_demo_pass": l1_ok,
            "level2_heldout_pass": l2_ok,
            "level3_extra_pass": l3_ok,
            "level3_extra_pass_rate": l3_rate,
            "num_level3_extra_pairs": n_l3,
            "all_3_levels_verified": all_3_ok,
            "turns": rec["turns"],
            "tool_calls": rec["tool_calls"],
            "output_tokens": rec["output_tokens"],
            "thinking_tokens": rec["thinking_tokens"],
            "submitted_solve_code": code_str,
            "thought_text": rec.get("thought_text"),
            "pred_grid": rec.get("pred_grid"),
            "transcript": raw_transcript,
        }
        if all_3_ok:
          verified_rows.append(sft_entry)
          verified_trajectories_saved += 1
        else:
          failed_rows.append(sft_entry)
          failed_trajectories_saved += 1

      puzzles_done += 1
      if any_l1:
        l1_pass_puzzles += 1
      if any_l12:
        l12_pass_puzzles += 1
      if any_l123:
        l123_verified_puzzles += 1

      pred_rows.append({
          "task_id": tid,
          "sample_id": c["sample_id"],
          "dataset": c["dataset"],
          "family_id": c["family_id"],
          "learning_tier": c["learning_tier"],
          "learning_signal_score": c["learning_signal_score"],
          "pair_idx": 0,
          "pass_at_1": bool(att_recs[0]["all_3_levels_verified"]) if att_recs else False,
          "pass_at_2": any_l123,
          "level1_demo_solved": any_l1,
          "level12_demo_heldout_solved": any_l12,
          "level123_all_verified_solved": any_l123,
          "attempts": [
              {k: v for k, v in a.items() if k not in ("transcript", "cross_pair_cands")}
              for a in att_recs
          ],
      })

    # Flush batch results to CNS
    with gfile.Open(preds_path, "a") as pf:
      for r in pred_rows:
        pf.write(json.dumps(r) + "\n")
    if verified_rows:
      with gfile.Open(verified_sft_path, "a") as vf:
        for r in verified_rows:
          vf.write(json.dumps(r) + "\n")
    if failed_rows:
      with gfile.Open(failed_sft_path, "a") as ff:
        for r in failed_rows:
          ff.write(json.dumps(r) + "\n")

    wall_s = time.time() - t_eval_start
    tok_s = cum_gen_tokens / max(cum_gen_seconds, 1e-6)
    shard_summary = {
        "exp_name": FLAGS.exp_name,
        "shard_idx": FLAGS.shard_idx,
        "num_shards": FLAGS.num_shards,
        "model": active_model_name,
        "engine_backend": active_backend,
        "precision": "full_precision_bf16_dequant_fp32_mxu",
        "reasoning_effort": FLAGS.reasoning_effort,
        "tpu_hardware": f"{num_tpus}x {device_kind}",
        "puzzles_completed_in_shard": puzzles_done,
        "puzzles_total_in_shard": len(shard_candidates),
        "level1_demo_pass_pct": round(100.0 * l1_pass_puzzles / max(1, puzzles_done), 2),
        "level12_heldout_pass_pct": round(100.0 * l12_pass_puzzles / max(1, puzzles_done), 2),
        "level123_3level_verified_pct": round(100.0 * l123_verified_puzzles / max(1, puzzles_done), 2),
        "verified_sft_trajectories_saved": verified_trajectories_saved,
        "failed_or_overfit_trajectories_saved": failed_trajectories_saved,
        "total_generated_tokens": cum_gen_tokens,
        "decode_tokens_per_sec": round(tok_s, 2),
        "wall_elapsed_s": round(wall_s, 1),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    try:
      with gfile.Open(shard_summary_path, "w") as sf:
        json.dump(shard_summary, sf, indent=2)
      if FLAGS.shard_idx == 0:
        with gfile.Open(global_summary_path, "w") as gsf:
          json.dump(shard_summary, gsf, indent=2)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    pbar.update(len(batch_cands))
    pbar.set_postfix(
        done=f"{puzzles_done}/{len(shard_candidates)}",
        L1=f"{shard_summary['level1_demo_pass_pct']:.1f}%",
        L12=f"{shard_summary['level12_heldout_pass_pct']:.1f}%",
        L123_verif=f"{shard_summary['level123_3level_verified_pct']:.1f}%",
        verif_traces=verified_trajectories_saved,
        tok_s=f"{tok_s:.0f}",
    )
    del wave_states
    gc.collect()

  pbar.close()
  print("=" * 80, flush=True)
  print(f"COMPLETED SHARD {FLAGS.shard_idx}/{FLAGS.num_shards} ({FLAGS.exp_name})", flush=True)
  print("=" * 80, flush=True)


if __name__ == "__main__":
  app_runner.run(main)
