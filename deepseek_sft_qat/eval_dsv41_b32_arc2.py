"""Full ARC-AGI-2 (120 tasks / 167 pairs) Milestone Evaluator for 2-bit DeepSeek-V4.1 E128 using Havu's Candidate B32 Harness.

Evaluates exported 2-bit Humming + FP8 block-32 milestone checkpoints (`0.25`, `0.50`, `0.75`, and `1.00` epoch)
on the full 120-task / 167-pair ARC-AGI-2 evaluation benchmark using Havu's Candidate B32 harness (`havu_b32_harness.py`):
  - Task-level multi-test rollouts (`ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`, Fix A5 `inputs_prelude`)
  - Retry-once AST-safe identifier typo auto-repair (`ARC_AUTO_REPAIR=1`, `havu_auto_repair.py`)
  - Token-exact DSML prompt history (`<｜begin▁of▁sentence｜><｜System｜>...<｜Assistant｜><think>`)
  - Uninterrupted thinking (`max_tokens=64000`, `80k` hard / `40k` soft `"gentle"`, `max_turns=40`)
  - Isolated child-process Python sandbox (`LocalExecutor`, `style="ipython"`, `A1-A11` session `answer`/`solve()` fallbacks)
  - 4-path zero-token post-attempt `train_check` (`expr` -> `cell` -> `fn` -> `transcript`) + `ranked_answers`
  - E058 `--grid_format=text+image --image_cell=0` (Havu's image harness): the JSON prompt + IMAGE_NOTE +
    one token-aligned PNG per grid in the first user message, run through the replicated ViT/aligner of an
    image checkpoint (`dsv41_img_e128_2bit_causal`) with image-aware prefix caching (`sha256` of the images)
    and a follower broadcast protocol that leaves the text harness byte-identical to round 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Sequence

from absl import app
from absl import flags
import numpy as np
import tokenizers
import torch
import torch.distributed as dist

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_b32_harness
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector
from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens
from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model
from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_2bit_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints/dsv41_e128_arc_agi2_full_sft_8xb200/export_2bit/epoch-0.25",
    "Path to the exported 2-bit Humming + FP8 block-32 checkpoint on CNS.",
)
flags.DEFINE_string(
    "base_engram_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "Path to the base dsv41_e128_2bit directory containing model-engram-*.safetensors and tokenizer.json.",
)
flags.DEFINE_string(
    "dataset_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "Path to the 120-task / 167-pair ARC-AGI-2 evaluation dataset on CNS.",
)
flags.DEFINE_string(
    "out_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/evals/dsv41_e128_b32_epoch_0.25",
    "CNS directory for Candidate B32 evaluation outputs.",
)
flags.DEFINE_string(
    "milestone_tag",
    "epoch-0.25",
    "Milestone identifier (`epoch-0.25`, `epoch-0.50`, `epoch-0.75`, `epoch-1.00`).",
)
flags.DEFINE_integer("num_attempts", 2, "Number of Candidate B32 attempts per task.")
flags.DEFINE_integer("shard_index", 0, "Shard index (0-based).")
flags.DEFINE_integer("num_shards", 1, "Total number of shards.")
flags.DEFINE_integer("max_tasks", 0, "If > 0, limit evaluation to first N tasks.")
flags.DEFINE_integer("num_gpus", 8, "Number of B200 GPUs per worker.")
flags.DEFINE_boolean(
    "enable_qat",
    True,
    "Enable exact Havu FP8/FP4 dynamic activation & KV quantization in evaluation forward pass.",
    allow_override=True,
)
flags.DEFINE_boolean(
    "load_compact_engram",
    True,
    "Load the 16 compact Engram shards (layers 1 & 14) across EP ranks.",
    allow_override=True,
)
flags.DEFINE_boolean(
    "poll_for_checkpoint",
    True,
    "If True, wait for EXPORT_COMPLETED.json in checkpoint_2bit_dir before starting evaluation.",
)
flags.DEFINE_integer(
    "poll_timeout_hours",
    12,
    "Maximum hours to poll for EXPORT_COMPLETED.json when poll_for_checkpoint=True.",
)
flags.DEFINE_enum(
    "grid_format",
    "json",
    ["json", "text+image"],
    "B32 prompt format: 'json' (text harness) or 'text+image' (E058: JSON prompt + one PNG per grid; "
    "requires a checkpoint with a vision tower).",
)
flags.DEFINE_integer(
    "image_cell", 0, "E058 pixels per grid cell; 0 = token-aligned 42*k px (`--image-cell 0`)."
)
flags.DEFINE_enum("image_note", "", ["", "strict"], "E058 IMAGE_NOTE variant.")
flags.DEFINE_integer(
    "max_images_per_prompt",
    32,
    "Refuse prompts with more images than the serving cap (`--limit-mm-per-prompt {\"image\": 32}`).",
)
flags.DEFINE_integer(
    "prefill_chunk_tokens",
    4096,
    "Prefill chunk size; image spans are never split across chunks (`--disable-chunked-mm-input`).",
)


# ------------------------------------------------------------------------------------------------
# Rank-0 -> follower broadcast protocol (`_run_distributed_cached_forward` / `_follower_worker_loop`).
# Every command is one int64[3] tensor `(op_code, a, b)` followed by op-specific payload tensors.
# The round-1 text protocol (ops 0..4) is unchanged byte for byte; the E058 VL extension only adds
# the three prefill ops 11/12/13, which are never emitted for text-only prompts.
#
#   OP_STOP      (0, _, _)                  follower exits.
#   OP_APPEND    (1, slot, L)   + ids[1,L]  text prefill chunk appended to `slot`.
#   OP_RESET     (2, slot, L)   + ids[1,L]  text prefill chunk into a fresh `slot`.
#   OP_DECODE    (3, B, steps)  + slots[B] (+ tok[B,1] per step) + valid_steps[B]
#   OP_EVICT     (4, slot, _)
#   OP_VL_APPEND (11, slot, L)  VL delta starts on an existing slot. Payload BEFORE the chunk ids:
#                               hdr int64[2] = (delta_len, n_images), delta_ids[1,delta_len],
#                               delta_types[1,delta_len], broadcast_object_list([png_bytes, ...]);
#                               then ids[1,L] of the first chunk. Every rank runs
#                               `embed_with_images` on the whole delta and slices it per chunk.
#   OP_VL_RESET  (12, slot, L)  same, into a fresh slot.
#   OP_VL_CONT   (13, slot, L)  + ids[1,L]  next chunk of the open VL delta (no extra payload).
# ------------------------------------------------------------------------------------------------
OP_STOP = 0
OP_APPEND = 1
OP_RESET = 2
OP_DECODE = 3
OP_EVICT = 4
VL_OP_OFFSET = 10
OP_VL_APPEND = OP_APPEND + VL_OP_OFFSET
OP_VL_RESET = OP_RESET + VL_OP_OFFSET
OP_VL_CONT = 13


def image_digest(blobs: Sequence[bytes]) -> str:
    """sha256 over an image list (count, order and bytes); `""` for text-only prompts."""
    if not blobs:
        return ""
    h = hashlib.sha256()
    for blob in blobs:
        h.update(len(blob).to_bytes(8, "little"))
        h.update(blob)
    return h.hexdigest()


def build_image_inputs(
    blobs: Sequence[bytes], device: torch.device
) -> list[dsv41_sft_model.ImageInput]:
    """PNG bytes -> `ImageInput`s via `dsv41_image_tokens.preprocess_image` (the training preprocessing)."""
    out: list[dsv41_sft_model.ImageInput] = []
    for blob in blobs:
        patches, plan = dsv41_image_tokens.preprocess_image(blob)
        out.append(
            dsv41_sft_model.ImageInput(
                patches=torch.from_numpy(patches).to(device),
                n_vit_h=plan.n_vit_h,
                n_vit_w=plan.n_vit_w,
            )
        )
    return out


def check_image_slots(
    token_types: Any,
    image_inputs: Sequence[dsv41_sft_model.ImageInput],
    downsample_ratio: int,
) -> None:
    """Asserts `#IMAGE slots in token_types == #aligner rows the images will produce` (clear error otherwise)."""
    types = np.asarray(
        token_types.detach().cpu().numpy() if isinstance(token_types, torch.Tensor) else token_types
    ).reshape(-1)
    n_slots = int((types == dsv41_image_tokens.IMAGE).sum())
    rows_per_image = [
        math.ceil(img.n_vit_h / downsample_ratio) * math.ceil(img.n_vit_w / downsample_ratio)
        for img in image_inputs
    ]
    n_rows = int(sum(rows_per_image))
    n_spans = len(dsv41_image_tokens.image_spans(types))
    if n_slots != n_rows or n_spans != len(image_inputs):
        raise ValueError(
            f"Image span / image mismatch: token_types has {n_spans} spans with {n_slots} IMAGE slots "
            f"but {len(image_inputs)} images yield {n_rows} aligner rows (per image: {rows_per_image}; "
            f"ViT grids: {[(i.n_vit_h, i.n_vit_w) for i in image_inputs]}, downsample={downsample_ratio})."
        )


def vl_prefill_chunk(
    model: dsv41_sft_model.Dsv41SftModel,
    tok_t: torch.Tensor,
    kv_cache: dict[str, Any],
    types_chunk: torch.Tensor,
    embeds_chunk: torch.Tensor,
) -> torch.Tensor:
    """One prefill chunk of a VL delta; identical on rank 0 and the followers.

    Chunks that contain an image position take the `inputs_embeds` route (slices of the
    `embed_with_images` result, so an image span may be split across chunks); chunks without
    image positions use the exact round-1 call.
    """
    if bool((types_chunk >= 0).any()):
        return model.forward_hidden_cached(
            tok_t, kv_cache, token_types=types_chunk, inputs_embeds=embeds_chunk
        )
    return model.forward_hidden_cached(tok_t, kv_cache)


class _FastTokenizerAdapter:
    """Wraps `tokenizers.Tokenizer` (`tokenizer.json`) into the interface expected by `havu_b32_harness`."""

    def __init__(self, tok_path: str):
        with gfile.Open(tok_path, "rb") as f:
            raw_bytes = f.read()
        self._tok = tokenizers.Tokenizer.from_buffer(raw_bytes)
        self.eos_token_id = self._tok.token_to_id(havu_b32_harness.DS41_EOS_TOKEN) or 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)


@dataclass
class _MockCompletionOutput:
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class _MockRequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[_MockCompletionOutput]
    finished: bool = False


@dataclass
class _ActiveSeq:
    request_id: str
    prompt_token_ids: list[int]
    generated_ids: list[int] = field(default_factory=list)
    generated_text: str = ""
    max_tokens: int = 64000
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_strings: list[str] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None
    slot_id: int = 0
    prefilled: bool = False
    processed_ids: list[int] = field(default_factory=list)
    pending_logits: torch.Tensor | None = None
    cache_overshoot: bool = False
    # E058 text+image prompts: per-position span types of the prompt prefix (`dsv41_image_tokens`,
    # -1 = text; positions beyond the list are text) and the PNG bytes in prompt order.
    token_types: list[int] | None = None
    image_blobs: list[bytes] = field(default_factory=list)
    image_digest: str = ""
    # Set when the request was rejected at `enqueue` (e.g. over `max_images_per_prompt`); `step()`
    # then reports it as finished with `finish_reason="error"` and never runs the model on it.
    error: str | None = None


@dataclass
class _CompatSamplingParams:
    """Lightweight drop-in for `vllm.SamplingParams` when running on PyTorch B200 without vLLM."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 64000
    stop: list[str] | None = None
    include_stop_str_in_output: bool = True
    skip_special_tokens: bool = False
    spaces_between_special_tokens: bool = False
    output_kind: str = "CUMULATIVE"
    stream_interval: int = 250


def _install_vllm_sampling_params_shim() -> None:
    """Ensures `vllm.SamplingParams` and `vllm.sampling_params.RequestOutputKind.CUMULATIVE` exist."""
    import types

    try:
        import vllm as vllm_mod  # type: ignore
    except Exception:
        vllm_mod = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm_mod

    req_kind = type("RequestOutputKind", (), {"CUMULATIVE": "CUMULATIVE"})
    sp_mod = types.ModuleType("vllm.sampling_params")
    sp_mod.RequestOutputKind = req_kind  # type: ignore[attr-defined]
    sp_mod.SamplingParams = _CompatSamplingParams  # type: ignore[attr-defined]
    sys.modules["vllm.sampling_params"] = sp_mod
    vllm_mod.__dict__["sampling_params"] = sp_mod
    vllm_mod.__dict__["SamplingParams"] = _CompatSamplingParams


class Dsv41DistributedStepEngine:
    """Distributed 8xB200 KV-cached batched step engine for 2-bit DeepSeek-V4.1 E128 that implements `havu_b32_harness`'s `llm` interface."""

    def __init__(
        self,
        model: dsv41_sft_model.Dsv41SftModel,
        tokenizer: _FastTokenizerAdapter,
        device: torch.device,
        max_model_len: int = 163840,
        prefill_chunk: int = 4096,
        max_images_per_prompt: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_config = type("Cfg", (), {"max_model_len": max_model_len})()
        self.llm_engine = self
        # Prefill chunk length (`--prefill_chunk_tokens`); tests shrink it to exercise chunking.
        self.prefill_chunk = max(1, int(prefill_chunk))
        # Serving cap on images per prompt (`--limit-mm-per-prompt {"image": N}`): prompts above it
        # are rejected with an error, never truncated.
        self.max_images_per_prompt = int(max_images_per_prompt)
        self._active: dict[str, _ActiveSeq] = {}
        self._req_counter = 0
        self._cache_slots: dict[int, dict[str, Any]] = {}
        # (slot_id, cached token ids, image digest of the images embedded in that slot's KV).
        self._prefix_bank: list[tuple[int, list[int], str]] = []
        self._next_slot_id = 1
        self._max_bank_size = 36
        # Engine-level throughput accounting (independent of harness turn completion)
        self._decode_steps = 0
        self._decode_tokens = 0
        self._decode_time = 0.0
        self._prefill_tokens = 0
        self._prefill_time = 0.0
        self._prefill_calls = 0
        self._vl_prefill_calls = 0
        self._vl_images = 0
        self._engine_t0 = time.time()
        self._last_engine_log = time.time()
        self._open_chunk: tuple[Any, int, list[_ActiveSeq]] | None = None
        self._rollback_streams = 0

    def get_tokenizer(self) -> _FastTokenizerAdapter:
        return self.tokenizer

    def enqueue(
        self,
        prompts: list[dict[str, Any]],
        sampling_params: list[Any],
        use_tqdm: bool = False,
    ) -> list[str]:
        del use_tqdm
        rids: list[str] = []
        for p, sp in zip(prompts, sampling_params):
            self._req_counter += 1
            rid = f"req_{self._req_counter}"
            stop_list = list(getattr(sp, "stop", None) or [])
            if havu_b32_harness.DS41_EOS_TOKEN not in stop_list:
                stop_list.append(havu_b32_harness.DS41_EOS_TOKEN)
            raw_types = p.get("token_types")
            token_types = [int(t) for t in raw_types] if raw_types is not None else None
            image_blobs = [bytes(b) for b in (p.get("images") or [])]
            seq = _ActiveSeq(
                request_id=rid,
                prompt_token_ids=list(p["prompt_token_ids"]),
                max_tokens=int(getattr(sp, "max_tokens", 64000)),
                temperature=float(getattr(sp, "temperature", 1.0)),
                top_p=float(getattr(sp, "top_p", 1.0)),
                top_k=int(getattr(sp, "top_k", -1)),
                stop_strings=stop_list,
                token_types=token_types,
                image_blobs=image_blobs,
                image_digest=image_digest(image_blobs),
            )
            # Harness contract: one image span in `token_types` per image, spans inside the prompt.
            n_spans = (
                len(dsv41_image_tokens.image_spans(np.asarray(token_types, dtype=np.int8)))
                if token_types is not None
                else 0
            )
            if n_spans != len(image_blobs):
                raise ValueError(
                    f"{rid}: token_types has {n_spans} image spans but {len(image_blobs)} images were given"
                )
            if token_types is not None and len(token_types) > len(seq.prompt_token_ids):
                raise ValueError(
                    f"{rid}: token_types ({len(token_types)}) longer than prompt ({len(seq.prompt_token_ids)})"
                )
            if len(image_blobs) > self.max_images_per_prompt:
                # Mirrors vLLM's `--limit-mm-per-prompt` rejection: the request fails, the shard goes on.
                seq.error = (
                    f"prompt has {len(image_blobs)} images > max_images_per_prompt="
                    f"{self.max_images_per_prompt}; request rejected (not truncated)"
                )
                seq.finished = True
                seq.finish_reason = "error"
                seq.prefilled = True
                print(f"[ENGINE] {rid}: {seq.error}", flush=True)
            self._active[rid] = seq
            rids.append(rid)
        return rids

    def has_unfinished_requests(self) -> bool:
        return bool(self._active)

    def _evict_slot(self, slot_id: int) -> None:
        self._cache_slots.pop(slot_id, None)
        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        if is_dist:
            cmd = torch.tensor(
                [OP_EVICT, slot_id, 0], dtype=torch.int64, device=self.device
            )
            dist.broadcast(cmd, src=0)

    def abort_request(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            seq = self._active.pop(rid, None)
            if seq is not None and seq.slot_id > 0:
                self._evict_slot(seq.slot_id)

    @staticmethod
    def _boundary_splits_image_span(token_types: list[int] | None, pos: int) -> bool:
        """True if prompt position `pos` lies strictly inside an image span (START excluded)."""
        if token_types is None or not 0 < pos < len(token_types):
            return False
        t = token_types[pos]
        return t >= 0 and t != dsv41_image_tokens.IMAGE_START

    @classmethod
    def _prefix_images_compatible(
        cls,
        cached_digest: str,
        c_len: int,
        token_types: list[int] | None,
        prompt_digest: str,
    ) -> bool:
        """Whether a cached prefix of `c_len` tokens may serve a prompt (ids already match).

        Image spans carry the same `IMAGE_TOKEN_ID` everywhere, so identical ids never prove the
        pixels agree: a cached slot that embedded images is only reused for the identical image list
        (same sha256 over count/order/bytes); a text-only cached prefix is reused only if the new
        prompt has no image position inside it; and the cached boundary must not split a span
        (the delta is embedded with `embed_with_images`, which needs complete spans).
        """
        if cls._boundary_splits_image_span(token_types, c_len):
            return False
        if cached_digest == prompt_digest:
            return True
        if cached_digest:
            return False  # cached KV holds other images
        if token_types is None:
            return True
        return not any(t >= 0 for t in token_types[:c_len])

    def _acquire_slot_for_prompt(
        self,
        prompt_ids: list[int],
        token_types: list[int] | None = None,
        prompt_digest: str = "",
    ) -> tuple[int, list[int], list[int], int]:
        """Finds a matching prefix KV-cache slot or allocates a new slot (`op_code`: 1=append, 2=reset+prefill)."""
        best_idx = -1
        best_len = 0
        for idx, (slot_id, cached_ids, cached_digest) in enumerate(self._prefix_bank):
            c_len = len(cached_ids)
            if (
                0 < c_len < len(prompt_ids)
                and c_len > best_len
                and prompt_ids[:c_len] == cached_ids
                and self._prefix_images_compatible(cached_digest, c_len, token_types, prompt_digest)
            ):
                best_idx = idx
                best_len = c_len
        if best_idx >= 0:
            slot_id, cached_ids, _ = self._prefix_bank.pop(best_idx)
            return slot_id, list(cached_ids), prompt_ids[best_len:], OP_APPEND

        active_slot_ids = {s.slot_id for s in self._active.values() if s.slot_id > 0}
        if len(self._prefix_bank) >= self._max_bank_size:
            evicted_slot, _, _ = self._prefix_bank.pop(0)
            if evicted_slot not in active_slot_ids:
                slot_id = evicted_slot
            else:
                slot_id = self._next_slot_id
                self._next_slot_id += 1
        else:
            slot_id = self._next_slot_id
            self._next_slot_id += 1
        self._cache_slots[slot_id] = {}
        return slot_id, [], list(prompt_ids), OP_RESET

    @staticmethod
    def _delta_vl_inputs(
        seq: _ActiveSeq, cached_len: int, delta_len: int
    ) -> tuple[list[int] | None, list[bytes]]:
        """`(token_types, images)` of the prompt tokens `[cached_len, cached_len + delta_len)`.

        Returns `(None, [])` when the delta holds no image position (text-only prefill). Positions
        past the end of `seq.token_types` are text (`-1`); only the images whose spans lie inside
        the delta are returned (spans inside the reused prefix are already in the KV cache).
        """
        if seq.token_types is None or not seq.image_blobs:
            return None, []
        end = cached_len + delta_len
        delta_types = [
            int(seq.token_types[i]) if i < len(seq.token_types) else dsv41_image_tokens.TEXT
            for i in range(cached_len, end)
        ]
        if not any(t >= 0 for t in delta_types):
            return None, []
        spans = dsv41_image_tokens.image_spans(np.asarray(seq.token_types, dtype=np.int8))
        blobs: list[bytes] = []
        for k, (s, e) in enumerate(spans):
            if e <= cached_len:
                continue  # fully inside the reused prefix
            if s < cached_len or e > end:
                raise ValueError(
                    f"{seq.request_id}: image span {k} [{s}, {e}) is split by the prefill range "
                    f"[{cached_len}, {end})"
                )
            blobs.append(seq.image_blobs[k])
        return delta_types, blobs

    def _run_distributed_cached_forward(
        self,
        op_code: int,
        slot_id: int,
        token_ids: list[int],
        token_types: list[int] | None = None,
        image_blobs: Sequence[bytes] | None = None,
    ) -> torch.Tensor:
        """Prefills `token_ids` into `slot_id` on every rank and returns the last-token logits.

        Text-only (`token_types` without image positions): per chunk, broadcasts
        `(op_code, slot_id, seq_len)` + `token_ids` to Ranks 1..7 exactly like round 1.
        VL deltas: `embed_with_images` runs ONCE over the whole delta on every rank (the ViT and
        aligner are replicated), the delta ids/types/images are broadcast once with the first
        chunk (`OP_VL_*`), and each chunk feeds its `inputs_embeds` slice, so chunking may split a
        span. The chunk computation itself is `vl_prefill_chunk` on all ranks.
        """
        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        if op_code == OP_RESET:
            self._cache_slots[slot_id] = {}
        kv_cache = self._cache_slots.setdefault(slot_id, {})
        n_tokens = len(token_ids)

        types_arr: np.ndarray | None = None
        if token_types is not None:
            types_arr = np.full(n_tokens, dsv41_image_tokens.TEXT, dtype=np.int64)
            m = min(n_tokens, len(token_types))
            types_arr[:m] = np.asarray(token_types[:m], dtype=np.int64)
            if not bool((types_arr >= 0).any()):
                types_arr = None
        is_vl = types_arr is not None

        t_pf0 = time.time()
        h_full: torch.Tensor | None = None
        types_t: torch.Tensor | None = None
        ids_full_t: torch.Tensor | None = None
        blobs: list[bytes] = []
        if is_vl:
            if self.model.vision is None:
                raise ValueError(
                    "Prompt contains image spans but the loaded checkpoint has no vision tower "
                    "(text-only config); run with --grid_format=json or an image checkpoint."
                )
            blobs = [bytes(b) for b in (image_blobs or [])]
            image_inputs = build_image_inputs(blobs, self.device)
            check_image_slots(types_arr, image_inputs, self.model.cfg.vision_downsample_ratio)
            ids_full_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            types_t = torch.from_numpy(types_arr).unsqueeze(0).to(self.device)
            h_full = self.model.embed_with_images(ids_full_t, types_t, image_inputs)
            self._vl_prefill_calls += 1
            self._vl_images += len(blobs)

        hidden_last: torch.Tensor | None = None
        prefill_chunk = self.prefill_chunk
        for start in range(0, n_tokens, prefill_chunk):
            end = min(n_tokens, start + prefill_chunk)
            sub_ids = token_ids[start:end]
            if is_vl:
                sub_op = (op_code + VL_OP_OFFSET) if start == 0 else OP_VL_CONT
            else:
                sub_op = op_code if start == 0 else OP_APPEND
            cmd = torch.tensor(
                [sub_op, slot_id, len(sub_ids)],
                dtype=torch.int64,
                device=self.device,
            )
            if is_dist:
                dist.broadcast(cmd, src=0)
                if is_vl and start == 0:
                    assert ids_full_t is not None and types_t is not None
                    hdr = torch.tensor(
                        [n_tokens, len(blobs)], dtype=torch.int64, device=self.device
                    )
                    dist.broadcast(hdr, src=0)
                    dist.broadcast(ids_full_t, src=0)
                    dist.broadcast(types_t, src=0)
                    dist.broadcast_object_list([blobs], src=0, device=self.device)
            tok_t = torch.tensor([sub_ids], dtype=torch.long, device=self.device)
            if is_dist:
                dist.broadcast(tok_t, src=0)
            if is_vl:
                assert h_full is not None and types_t is not None
                hidden = vl_prefill_chunk(
                    self.model,
                    tok_t,
                    kv_cache,
                    types_t[:, start:end],
                    h_full[:, start:end],
                )
            else:
                hidden = self.model.forward_hidden_cached(tok_t, kv_cache)
            hidden_last = hidden[:, -1, :]
        del h_full

        assert hidden_last is not None
        logits = self.model.head(hidden_last.to(self.model.head.weight.dtype)).float()[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._prefill_time += time.time() - t_pf0
        self._prefill_tokens += len(token_ids)
        self._prefill_calls += 1
        return logits

    def _maybe_log_engine_stats(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_engine_log < 60.0:
            return
        self._last_engine_log = now
        dec_tps = self._decode_tokens / max(1e-6, self._decode_time)
        step_ms = 1000.0 * self._decode_time / max(1, self._decode_steps)
        pf_tps = self._prefill_tokens / max(1e-6, self._prefill_time)
        wall = now - self._engine_t0
        print(
            f"[ENGINE] decode: {self._decode_tokens} tok in {self._decode_time:.1f}s "
            f"({dec_tps:.1f} tok/s, {step_ms:.0f} ms/step, {self._decode_steps} steps) | "
            f"prefill: {self._prefill_tokens} tok in {self._prefill_time:.1f}s "
            f"({pf_tps:.0f} tok/s, {self._prefill_calls} calls, "
            f"{self._vl_prefill_calls} with images / {self._vl_images} images) | "
            f"active={len(self._active)} slots={len(self._cache_slots)} "
            f"bank={len(self._prefix_bank)} rollbacks={self._rollback_streams} | wall={wall/60:.1f}m",
            flush=True,
        )

    def _sample_token(self, logits: torch.Tensor, seq: _ActiveSeq) -> int:
        return int(self._sample_tokens_batch_gpu(logits.unsqueeze(0), [seq])[0, 0].item())

    def _sample_tokens_batch_gpu(
        self, logits_b: torch.Tensor, seqs: list[_ActiveSeq]
    ) -> torch.Tensor:
        """Samples `[B, 1]` next token IDs entirely on GPU without any CPU `.item()` synchronization."""
        bsz = logits_b.size(0)
        if bsz == 0:
            return torch.empty((0, 1), dtype=torch.long, device=self.device)

        first = seqs[0]
        uniform_params = all(
            s.temperature == first.temperature
            and s.top_p == first.top_p
            and s.top_k == first.top_k
            for s in seqs
        )
        if uniform_params:
            if first.temperature <= 0.0:
                return torch.argmax(logits_b, dim=-1, keepdim=True)
            scaled = logits_b / first.temperature
            if 0 < first.top_k < scaled.size(-1):
                topk_vals, _ = torch.topk(scaled, first.top_k, dim=-1)
                scaled = scaled.masked_fill(
                    scaled < topk_vals[:, -1:], -torch.inf
                )
            probs = torch.softmax(scaled, dim=-1)
            if 0.0 < first.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(
                    probs, dim=-1, descending=True
                )
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = (cum_probs - sorted_probs) > first.top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-12)
                sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
                return torch.gather(sorted_idx, -1, sampled_pos)
            return torch.multinomial(probs, num_samples=1)

        out_list: list[torch.Tensor] = []
        for b, seq in enumerate(seqs):
            out_list.append(
                self._sample_tokens_batch_gpu(logits_b[b : b + 1], [seq])
            )
        return torch.cat(out_list, dim=0)

    def _run_distributed_batched_decode_chunk(
        self, seqs: list[_ActiveSeq], chunk_steps: int
    ) -> list[list[int]]:
        """Runs `chunk_steps` batched 1-token forward passes across `B = len(seqs)` active streams on GPU with 1 CPU sync at the end."""
        bsz = len(seqs)
        if bsz == 0 or chunk_steps <= 0:
            return [[] for _ in seqs]

        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        slot_ids = [s.slot_id for s in seqs]
        kv_caches = [self._cache_slots.setdefault(sid, {}) for sid in slot_ids]
        head_dtype = self.model.head.weight.dtype

        # Streams whose last sampled token has not been fed to the KV cache yet
        # (`pending_logits is None`): feed them in ONE batched step (op_code 3, 1 step)
        # instead of B single-stream forwards.
        need_init = [s for s in seqs if s.pending_logits is None]
        if need_init:
            init_tok = torch.tensor(
                [[s.generated_ids[-1]] for s in need_init],
                dtype=torch.long,
                device=self.device,
            )
            init_caches = [
                self._cache_slots.setdefault(s.slot_id, {}) for s in need_init
            ]
            if is_dist:
                cmd = torch.tensor(
                    [3, len(need_init), 1], dtype=torch.int64, device=self.device
                )
                dist.broadcast(cmd, src=0)
                init_slots_t = torch.tensor(
                    [s.slot_id for s in need_init],
                    dtype=torch.int64,
                    device=self.device,
                )
                dist.broadcast(init_slots_t, src=0)
                dist.broadcast(init_tok, src=0)
            hidden_init = self.model.forward_hidden_cached_batch(init_tok, init_caches)
            logits_init = self.model.head(
                hidden_init[:, -1, :].to(head_dtype)
            ).float()
            if is_dist:
                # Protocol: every op_code-3 chunk is followed by a `valid_steps` broadcast
                dist.broadcast(
                    torch.ones(len(need_init), dtype=torch.int64, device=self.device),
                    src=0,
                )
            for b, s in enumerate(need_init):
                s.pending_logits = logits_init[b]
                s.processed_ids.append(s.generated_ids[-1])

        init_logits = torch.stack(
            [s.pending_logits for s in seqs], dim=0  # type: ignore[arg-type]
        )
        for s in seqs:
            s.pending_logits = None

        cur_tok_b = self._sample_tokens_batch_gpu(init_logits, seqs)  # [B, 1]
        sampled_steps: list[torch.Tensor] = [cur_tok_b]

        fwd_steps = max(0, chunk_steps - 1)
        profiler_ctx: Any = None
        if fwd_steps > 0:
            if is_dist:
                cmd = torch.tensor(
                    [3, bsz, fwd_steps], dtype=torch.int64, device=self.device
                )
                dist.broadcast(cmd, src=0)
                slots_t = torch.tensor(
                    slot_ids, dtype=torch.int64, device=self.device
                )
                dist.broadcast(slots_t, src=0)

            t_chunk0 = time.time()
            self._chunk_counter = getattr(self, "_chunk_counter", 0) + 1
            prof_chunk = int(os.environ.get("DSV41_PROFILE_DECODE_CHUNK", "0") or 0)
            if prof_chunk > 0 and self._chunk_counter == prof_chunk and self.device.type == "cuda":
                from torch.profiler import ProfilerActivity, profile  # pylint: disable=g-import-not-at-top

                profiler_ctx = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                )
                profiler_ctx.__enter__()
            session = (
                self.model.begin_batched_decode(kv_caches, max_steps=fwd_steps)
                if bsz > 1
                else None
            )
            try:
                for step_i in range(fwd_steps):
                    if is_dist:
                        dist.broadcast(cur_tok_b, src=0)
                    if session is not None:
                        hidden_b = self.model.step_batched_decode(cur_tok_b, session)
                    else:
                        hidden_b = self.model.forward_hidden_cached(
                            cur_tok_b, kv_caches[0]
                        )
                    logits_b = self.model.head(
                        hidden_b[:, -1, :].to(head_dtype)
                    ).float()
                    next_tok_b = self._sample_tokens_batch_gpu(logits_b, seqs)
                    sampled_steps.append(next_tok_b)
                    cur_tok_b = next_tok_b
            except Exception:
                if session is not None:
                    self.model.end_batched_decode(session)
                raise
            # The session stays open: `step()` decides per stream how many of this chunk's
            # tokens are valid (EOS / stop-string / max_tokens may cut a stream mid-chunk) and
            # then rolls streams back exactly instead of evicting their prefix cache.
            self._open_chunk = (session, fwd_steps, list(seqs))
            self._decode_steps += fwd_steps
            self._decode_tokens += fwd_steps * bsz

        all_sampled_cpu: list[list[int]] = (
            torch.cat(sampled_steps, dim=1).tolist()
        )
        if fwd_steps > 0:
            self._decode_time += time.time() - t_chunk0
            if profiler_ctx is not None:
                torch.cuda.synchronize(self.device)
                profiler_ctx.__exit__(None, None, None)
                ka = profiler_ctx.key_averages()
                cpu_total = sum(e.self_cpu_time_total for e in ka) / 1e3
                cuda_total = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)) for e in ka) / 1e3
                print(
                    f"[PROFILE] chunk={self._chunk_counter} B={bsz} steps={fwd_steps} "
                    f"wall={time.time()-t_chunk0:.2f}s self_cpu_total={cpu_total:.0f}ms "
                    f"self_cuda_total={cuda_total:.0f}ms (per step: cpu={cpu_total/fwd_steps:.1f}ms cuda={cuda_total/fwd_steps:.1f}ms)",
                    flush=True,
                )
                try:
                    print(ka.table(sort_by="self_cuda_time_total", row_limit=35), flush=True)
                except Exception:  # pylint: disable=broad-exception-caught
                    print(ka.table(sort_by="self_cpu_time_total", row_limit=35), flush=True)
                print(ka.table(sort_by="self_cpu_time_total", row_limit=25), flush=True)
        # Note: tokens 0 .. fwd_steps-1 were fed into the KV caches; token `fwd_steps` (the last
        # sampled token) has NOT been fed yet (matching `pending_logits is None`). `step()`
        # extends `processed_ids` with exactly the kept+fed tokens and finalizes the session.
        return all_sampled_cpu

    def _finalize_open_chunk(self, kept_new_tokens: dict[str, int]) -> None:
        """Closes the open batched session, rolling back streams that stopped mid-chunk."""
        open_chunk = getattr(self, "_open_chunk", None)
        self._open_chunk = None
        if open_chunk is None:
            return
        session, fwd_steps, seqs = open_chunk
        valid_steps = [
            min(fwd_steps, kept_new_tokens.get(s.request_id, fwd_steps)) for s in seqs
        ]
        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        if is_dist:
            dist.broadcast(
                torch.tensor(valid_steps, dtype=torch.int64, device=self.device), src=0
            )
        if session is not None:
            self.model.end_batched_decode(session, valid_steps)
            self._rollback_streams += sum(1 for k in valid_steps if k < fwd_steps)
        else:
            # Single-stream (no session) path cannot roll back: the KV cache holds all
            # `fwd_steps` tokens, so a cut stream's slot must not be banked for prefix reuse.
            for s_, k in zip(seqs, valid_steps):
                if k < fwd_steps:
                    s_.cache_overshoot = True

    @torch.no_grad()
    def step(self) -> list[_MockRequestOutput]:
        if not self._active:
            return []
        eos_id = self.tokenizer.eos_token_id
        outputs: list[_MockRequestOutput] = []

        # 0. Requests rejected at `enqueue` (over `max_images_per_prompt`): report them finished with
        #    `finish_reason="error"` (the harness records an errored attempt) without touching the model.
        for seq in list(self._active.values()):
            if seq.error is not None and seq.finished:
                self._active.pop(seq.request_id, None)
                outputs.append(
                    _MockRequestOutput(
                        request_id=seq.request_id,
                        prompt_token_ids=seq.prompt_token_ids,
                        outputs=[_MockCompletionOutput(text="", token_ids=[], finish_reason="error")],
                        finished=True,
                    )
                )

        # 1. Prefill any newly enqueued sequences so they join the active decode batch immediately
        for seq in list(self._active.values()):
            if not seq.prefilled:
                slot_id, cached_ids, delta_ids, op_code = self._acquire_slot_for_prompt(
                    seq.prompt_token_ids, seq.token_types, seq.image_digest
                )
                seq.slot_id = slot_id
                seq.processed_ids = cached_ids
                # Only the tokens past the reused prefix are embedded/prefilled; their span types
                # and the images inside that range come from the prompt's `token_types`.
                delta_types, delta_blobs = self._delta_vl_inputs(
                    seq, len(cached_ids), len(delta_ids)
                )
                seq.pending_logits = self._run_distributed_cached_forward(
                    op_code, slot_id, delta_ids, delta_types, delta_blobs
                )
                seq.processed_ids.extend(delta_ids)
                seq.prefilled = True

        active_seqs = [
            s for s in self._active.values() if s.prefilled and not s.finished
        ]
        if not active_seqs:
            return outputs

        min_rem = min(
            max(1, s.max_tokens - len(s.generated_ids)) for s in active_seqs
        )
        chunk_steps = min(32, min_rem)
        sampled_batch = self._run_distributed_batched_decode_chunk(
            active_seqs, chunk_steps
        )

        fwd_steps = max(0, chunk_steps - 1)
        kept_new: dict[str, int] = {}
        for seq, new_tokens in zip(active_seqs, sampled_batch):
            prev_len = len(seq.generated_ids)
            hit_eos_idx = -1
            for idx_t, tid in enumerate(new_tokens):
                if tid == eos_id:
                    hit_eos_idx = idx_t
                    break

            if hit_eos_idx >= 0:
                kept_tokens = new_tokens[: hit_eos_idx + 1]
                seq.generated_ids.extend(kept_tokens)
                seq.generated_text = self.tokenizer.decode(
                    seq.generated_ids, skip_special_tokens=False
                )
                seq.finished = True
                seq.finish_reason = "stop"
            else:
                seq.generated_ids.extend(new_tokens)
                # Fast tail decode (`last 64 tokens`) to check `stop_strings` without $O(T^2)$ full decode
                tail_ids = seq.generated_ids[max(0, prev_len - 32) :]
                tail_text = self.tokenizer.decode(
                    tail_ids, skip_special_tokens=False
                )
                matched_stop = any(s in tail_text for s in seq.stop_strings if s)
                if matched_stop:
                    # Find the exact token prefix where the stop string completed
                    trunc_ids = seq.generated_ids[:prev_len]
                    for tid in new_tokens:
                        trunc_ids.append(tid)
                        t_tail = self.tokenizer.decode(
                            trunc_ids[-48:], skip_special_tokens=False
                        )
                        if any(t_tail.endswith(s) for s in seq.stop_strings if s):
                            break
                    seq.generated_ids = trunc_ids
                    seq.generated_text = self.tokenizer.decode(
                        seq.generated_ids, skip_special_tokens=False
                    )
                    seq.finished = True
                    seq.finish_reason = "stop"
                elif len(seq.generated_ids) >= seq.max_tokens:
                    seq.generated_ids = seq.generated_ids[: seq.max_tokens]
                    seq.generated_text = self.tokenizer.decode(
                        seq.generated_ids, skip_special_tokens=False
                    )
                    seq.finished = True
                    seq.finish_reason = "length"
                else:
                    # Append tail or refresh full text periodically for `GuardedStream` loop monitor
                    if (
                        len(seq.generated_ids) < 128
                        or (len(seq.generated_ids) // 512)
                        != (prev_len // 512)
                        or "</think>" in tail_text
                        or "<｜DSML｜" in tail_text
                    ):
                        seq.generated_text = self.tokenizer.decode(
                            seq.generated_ids, skip_special_tokens=False
                        )
                    else:
                        seq.generated_text += self.tokenizer.decode(
                            new_tokens, skip_special_tokens=False
                        )

            # Tokens of this chunk that are both kept (not cut by EOS/stop/max_tokens) and were
            # fed to the KV cache (the last sampled token is never fed inside the chunk).
            n_kept = len(seq.generated_ids) - prev_len
            kept_new[seq.request_id] = n_kept
            seq.processed_ids.extend(new_tokens[: min(n_kept, fwd_steps)])

            ro = _MockRequestOutput(
                request_id=seq.request_id,
                prompt_token_ids=seq.prompt_token_ids,
                outputs=[
                    _MockCompletionOutput(
                        text=seq.generated_text,
                        token_ids=list(seq.generated_ids),
                        finish_reason=seq.finish_reason,
                    )
                ],
                finished=seq.finished,
            )
            outputs.append(ro)

        # Close the batched session: streams cut mid-chunk are rolled back exactly, so their
        # prefix caches stay reusable for the next turn (no full re-prefill).
        self._finalize_open_chunk(kept_new)

        for seq in active_seqs:
            if seq.finished:
                self._active.pop(seq.request_id, None)
                if seq.slot_id > 0:
                    # Only tokens actually committed to the KV cache (`len(seq.processed_ids)`) are valid if `seq` stopped early mid-chunk
                    valid_cached_len = min(
                        len(seq.processed_ids),
                        len(seq.prompt_token_ids) + len(seq.generated_ids),
                    )
                    if valid_cached_len == len(seq.processed_ids) and not seq.cache_overshoot:
                        self._prefix_bank.append(
                            (seq.slot_id, list(seq.processed_ids), seq.image_digest)
                        )
                        active_slot_ids = {
                            s.slot_id
                            for s in self._active.values()
                            if s.slot_id > 0
                        }
                        while len(self._prefix_bank) > self._max_bank_size:
                            old_slot, _, _ = self._prefix_bank.pop(0)
                            if (
                                old_slot != seq.slot_id
                                and old_slot not in active_slot_ids
                            ):
                                self._evict_slot(old_slot)
                    else:
                        self._evict_slot(seq.slot_id)
        self._maybe_log_engine_stats()
        return outputs


@dataclass
class _FollowerVlDelta:
    """Follower-side state of the VL delta currently being prefilled (`OP_VL_*` ops)."""

    slot_id: int
    delta_len: int
    types_t: torch.Tensor  # [1, delta_len]
    h_full: torch.Tensor  # [1, delta_len, dim] from `embed_with_images`
    offset: int = 0


def _follower_worker_loop(
    model: dsv41_sft_model.Dsv41SftModel,
    device: torch.device,
    cache_slots: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Ranks 1..7 participate in KV-cached expert-parallel all-reduces until Rank 0 broadcasts cmd[0] == 0.

    `cache_slots` may be injected (tests) to inspect the follower's KV caches afterwards.
    """
    if cache_slots is None:
        cache_slots = {}
    vl_delta: _FollowerVlDelta | None = None
    with torch.no_grad():
        while True:
            cmd = torch.zeros(3, dtype=torch.int64, device=device)
            dist.broadcast(cmd, src=0)
            op_code = int(cmd[0].item())
            if op_code == OP_STOP:
                break
            if op_code == OP_EVICT:
                slot_id = int(cmd[1].item())
                cache_slots.pop(slot_id, None)
                continue
            if op_code == OP_DECODE:
                bsz = int(cmd[1].item())
                fwd_steps = int(cmd[2].item())
                slots_t = torch.empty(bsz, dtype=torch.int64, device=device)
                dist.broadcast(slots_t, src=0)
                slot_ids = slots_t.tolist()
                kv_caches = [cache_slots.setdefault(int(sid), {}) for sid in slot_ids]
                tok_b = torch.empty((bsz, 1), dtype=torch.long, device=device)
                session = (
                    model.begin_batched_decode(kv_caches, max_steps=fwd_steps)
                    if bsz > 1
                    else None
                )
                try:
                    for _ in range(fwd_steps):
                        dist.broadcast(tok_b, src=0)
                        if session is not None:
                            _ = model.step_batched_decode(tok_b, session)
                        else:
                            _ = model.forward_hidden_cached(tok_b, kv_caches[0])
                    # Rank 0 decides per stream how many of the chunk's steps are valid
                    valid_t = torch.empty(bsz, dtype=torch.int64, device=device)
                    dist.broadcast(valid_t, src=0)
                    if session is not None:
                        model.end_batched_decode(session, valid_t.tolist())
                        session = None
                finally:
                    if session is not None:
                        model.end_batched_decode(session)
                continue
            slot_id = int(cmd[1].item())
            seq_len = int(cmd[2].item())
            if op_code in (OP_VL_APPEND, OP_VL_RESET, OP_VL_CONT):
                # E058 VL prefill: the delta payload arrives once (first chunk); every rank runs the
                # replicated ViT + aligner itself and then feeds per-chunk `inputs_embeds` slices.
                if op_code != OP_VL_CONT:
                    if op_code == OP_VL_RESET:
                        cache_slots[slot_id] = {}
                    hdr = torch.zeros(2, dtype=torch.int64, device=device)
                    dist.broadcast(hdr, src=0)
                    delta_len, n_images = int(hdr[0].item()), int(hdr[1].item())
                    delta_ids = torch.empty((1, delta_len), dtype=torch.long, device=device)
                    dist.broadcast(delta_ids, src=0)
                    delta_types = torch.empty((1, delta_len), dtype=torch.int64, device=device)
                    dist.broadcast(delta_types, src=0)
                    obj: list[Any] = [None]
                    dist.broadcast_object_list(obj, src=0, device=device)
                    blobs = [bytes(b) for b in (obj[0] or [])]
                    if len(blobs) != n_images:
                        raise RuntimeError(
                            f"follower received {len(blobs)} images, header announced {n_images}"
                        )
                    if model.vision is None:
                        raise ValueError(
                            "Prompt contains image spans but the loaded checkpoint has no vision tower."
                        )
                    image_inputs = build_image_inputs(blobs, device)
                    check_image_slots(delta_types, image_inputs, model.cfg.vision_downsample_ratio)
                    vl_delta = _FollowerVlDelta(
                        slot_id=slot_id,
                        delta_len=delta_len,
                        types_t=delta_types,
                        h_full=model.embed_with_images(delta_ids, delta_types, image_inputs),
                    )
                elif vl_delta is None or vl_delta.slot_id != slot_id:
                    raise RuntimeError(
                        f"OP_VL_CONT for slot {slot_id} without an open VL delta "
                        f"(open: {None if vl_delta is None else vl_delta.slot_id})"
                    )
                assert vl_delta is not None
                kv_cache = cache_slots.setdefault(slot_id, {})
                tok_t = torch.empty((1, seq_len), dtype=torch.long, device=device)
                dist.broadcast(tok_t, src=0)
                off = vl_delta.offset
                if off + seq_len > vl_delta.delta_len:
                    raise RuntimeError(
                        f"VL chunk [{off}, {off + seq_len}) exceeds delta length {vl_delta.delta_len}"
                    )
                _ = vl_prefill_chunk(
                    model,
                    tok_t,
                    kv_cache,
                    vl_delta.types_t[:, off : off + seq_len],
                    vl_delta.h_full[:, off : off + seq_len],
                )
                vl_delta.offset += seq_len
                if vl_delta.offset >= vl_delta.delta_len:
                    vl_delta = None  # frees `h_full`
                continue
            if op_code == OP_RESET:
                cache_slots[slot_id] = {}
            kv_cache = cache_slots.setdefault(slot_id, {})
            tok_t = torch.empty((1, seq_len), dtype=torch.long, device=device)
            dist.broadcast(tok_t, src=0)
            _ = model.forward_hidden_cached(tok_t, kv_cache)


def _wait_for_exported_checkpoint(ckpt_dir: str, timeout_hours: int) -> dict[str, Any]:
    """Blocks until the 2-bit checkpoint at `ckpt_dir` is complete.

    For trainer milestone exports (`.../export_2bit/<tag>`) only `EXPORT_COMPLETED.json` counts: the trainer
    writes it after every rank's shard has been flushed (barrier), whereas the individual files become
    visible on CNS while still being written. The marker-less fallback is reserved for static, externally
    produced checkpoint directories (e.g. chankhavu's base `dsv41_e128_2bit`) and requires the full file set.
    (v12 epoch_0p50 shard 3 died with `SafetensorError: incomplete metadata` after the old fallback
    accepted a half-written export.)
    """
    marker = os.path.join(ckpt_dir, "EXPORT_COMPLETED.json")
    is_trainer_export = "/export_2bit/" in (ckpt_dir.rstrip("/") + "/")
    # Backbone shards come from `model.safetensors.index.json` (legacy `model-0000N-of-00002` text
    # checkpoints as well as the 3-shard `model-0000N.safetensors` image checkpoints).
    backbone_shards = dsv41_sft_model._list_backbone_shards(ckpt_dir)  # pylint: disable=protected-access
    static_required = [os.path.join(ckpt_dir, s) for s in backbone_shards] + [
        os.path.join(ckpt_dir, "model.safetensors.index.json"),
        os.path.join(ckpt_dir, "config.json"),
        os.path.join(ckpt_dir, "tokenizer.json"),
    ] + [os.path.join(ckpt_dir, "gptq", f"layer_{i:02d}.safetensors") for i in range(40)]
    t0 = time.time()
    timeout_s = timeout_hours * 3600.0
    while True:
        if gfile.Exists(marker):
            with gfile.Open(marker, "r") as f:
                return json.load(f)
        if not is_trainer_export and all(gfile.Exists(p) for p in static_required):
            return {"status": "ready_existing_2bit_dir", "path": ckpt_dir}
        elapsed = time.time() - t0
        if not FLAGS.poll_for_checkpoint or elapsed >= timeout_s:
            raise TimeoutError(
                f"Timed out after {elapsed/60.0:.1f}m waiting for 2-bit checkpoint at {ckpt_dir}"
            )
        print(
            f"[eval_dsv41_b32_arc2] Waiting for milestone checkpoint {ckpt_dir} "
            f"(elapsed={elapsed/60.0:.1f}m)...",
            flush=True,
        )
        time.sleep(30.0)


def _worker_main(rank: int, world_size: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        import datetime

        os.environ.setdefault("MASTER_ADDR", "::1")
        os.environ.setdefault("MASTER_PORT", "29521")
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET6")
        store_path = os.environ.get(
            "TORCH_DIST_INIT_FILE", "/tmp/eval_dsv41_b32_c10d_filestore"
        )
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
            device_id=device if device.type == "cuda" else None,
            timeout=datetime.timedelta(minutes=90),
        )
    try:
        if world_size == 1:
            export_and_quantize.ensure_cns_quota_dir(FLAGS.out_dir)
            ckpt_info = _wait_for_exported_checkpoint(
                FLAGS.checkpoint_2bit_dir, FLAGS.poll_timeout_hours
            )
            print(
                f"[eval_dsv41_b32_arc2] Milestone {FLAGS.milestone_tag} checkpoint ready: {ckpt_info}",
                flush=True,
            )

        cfg_path = os.path.join(FLAGS.checkpoint_2bit_dir, "config.json")
        if not gfile.Exists(cfg_path):
            cfg_path = os.path.join(FLAGS.base_engram_dir, "config.json")
        if gfile.Exists(cfg_path):
            with gfile.Open(cfg_path, "r") as f:
                cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
        else:
            cfg = dsv41_sft_model.Dsv41E128Config()
        cfg.routed_expert_lora_r = 0
        cfg.enable_qat = FLAGS.enable_qat
        cfg.qat_trainable_expert_scales = False
        cfg.load_compact_engram = FLAGS.load_compact_engram
        cfg.max_seq_len = 163840
        # E058 text+image needs the vision tower (`vision_config` present and not `e043_pruning.text_only`).
        # `from_json_dict` keeps it enabled for the image base model and disabled for the text-only E043
        # export; refuse to start (on every rank, before loading 10+ GB) instead of silently running the
        # image spans on the placeholder text embedding.
        if FLAGS.grid_format == "text+image" and not cfg.vision_enabled:
            raise RuntimeError(
                f"--grid_format=text+image requires a checkpoint with a vision tower, but {cfg_path} is "
                "text-only (no `vision_config` or `e043_pruning.text_only=true`). Use the image base model "
                "(e.g. /cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal) or --grid_format=json."
            )
        if rank == 0:
            print(
                f"[eval_dsv41_b32_arc2] prompt format: grid_format={FLAGS.grid_format} "
                f"image_cell={FLAGS.image_cell} image_note={FLAGS.image_note!r} "
                f"max_images_per_prompt={FLAGS.max_images_per_prompt} "
                f"prefill_chunk_tokens={FLAGS.prefill_chunk_tokens} | vision_enabled={cfg.vision_enabled} "
                f"(vision_n_layers={cfg.vision_n_layers}) from {cfg_path}",
                flush=True,
            )

        with torch.device(device):
            model = dsv41_sft_model.Dsv41SftModel(
                cfg=cfg,
                ep_rank=rank,
                ep_world_size=world_size,
                gradient_checkpointing=False,
                ep_replicated_input=True,
            )
        load_stats = dsv41_sft_model.load_dequantized_dsv41_e128(
            model_dir=FLAGS.checkpoint_2bit_dir,
            model=model,
            device=device,
            engram_model_dir=FLAGS.base_engram_dir,
        )
        if rank == 0:
            print(
                f"[eval_dsv41_b32_arc2] Loaded checkpoint {FLAGS.checkpoint_2bit_dir}: {load_stats}",
                flush=True,
            )
        model.eval()
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            tok_path = os.path.join(FLAGS.base_engram_dir, "tokenizer.json")
            tok = _FastTokenizerAdapter(tok_path)
            engine = Dsv41DistributedStepEngine(
                model=model,
                tokenizer=tok,
                device=device,
                max_model_len=cfg.max_seq_len,
                prefill_chunk=FLAGS.prefill_chunk_tokens,
                max_images_per_prompt=FLAGS.max_images_per_prompt,
            )
            base_cfg = prompts_and_selector.PILOT2_HYPOTHESES["havu_b32"]
            is_image_run = FLAGS.grid_format == "text+image"
            exp_suffix = "_img" if is_image_run else ""
            title_suffix = (
                f" + E058 images (cell={FLAGS.image_cell or 'aligned'}, note={FLAGS.image_note or 'default'})"
                if is_image_run
                else ""
            )
            eval_cfg = prompts_and_selector.Pilot2HypothesisConfig(
                **{
                    **base_cfg.__dict__,
                    "exp_id": f"dsv41_e128_b32_{FLAGS.milestone_tag}{exp_suffix}",
                    "title": (
                        f"DeepSeek-V4.1 E128 QAT SFT ({FLAGS.milestone_tag}) + Havu Candidate B32"
                        + title_suffix
                    ),
                    "num_attempts": FLAGS.num_attempts,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    # E058 prompt format -> `havu_b32_harness._create_attempt_state` (prompt builder)
                    # and the per-shard config/summary json.
                    "grid_format": FLAGS.grid_format,
                    "image_cell": FLAGS.image_cell,
                    "image_note": FLAGS.image_note,
                }
            )
            try:
                _install_vllm_sampling_params_shim()
                havu_b32_harness.run_havu_b32_shard(
                    llm=engine,
                    cfg=eval_cfg,
                    dataset_dir=FLAGS.dataset_dir,
                    out_dir=FLAGS.out_dir,
                    shard_index=FLAGS.shard_index,
                    num_shards=FLAGS.num_shards,
                    max_tasks=FLAGS.max_tasks,
                    model_path=FLAGS.checkpoint_2bit_dir,
                )
            finally:
                if world_size > 1:
                    cmd = torch.zeros(3, dtype=torch.int64, device=device)
                    dist.broadcast(cmd, src=0)
        else:
            _follower_worker_loop(model, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _serialize_eval_flags_for_children() -> list[str]:
    flag_names = [
        "checkpoint_2bit_dir",
        "base_engram_dir",
        "dataset_dir",
        "out_dir",
        "milestone_tag",
        "num_attempts",
        "shard_index",
        "num_shards",
        "max_tasks",
        "num_gpus",
        "enable_qat",
        "load_compact_engram",
        "poll_for_checkpoint",
        "poll_timeout_hours",
        # E058 text+image prompt format + serving caps (every rank needs them: the followers
        # replicate the vision tower and rank 0 builds the harness cfg from them).
        "grid_format",
        "image_cell",
        "image_note",
        "max_images_per_prompt",
        "prefill_chunk_tokens",
    ]
    args: list[str] = [
        "--undefok=cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
        "--cfs_d_cid_enable=true",
        "--cfs_d_resource_user_override=deepmind-resources-colossus",
        "--cfs_default_traffic_class=AF1",
        "--cfs_max_write_bytes_inflight=33554432",
    ]
    for name in flag_names:
        if name not in FLAGS:
            continue
        val = FLAGS[name].value
        if isinstance(val, bool):
            args.append(f"--{name}={'true' if val else 'false'}")
        else:
            args.append(f"--{name}={val}")
    return args


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Too many command-line arguments: {argv[1:]}")

    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world_size is not None:
        _worker_main(rank=int(env_rank), world_size=int(env_world_size))
        return

    world_size = FLAGS.num_gpus if torch.cuda.is_available() else 1
    if world_size > 1:
        export_and_quantize.ensure_cns_quota_dir(FLAGS.out_dir)
        ckpt_info = _wait_for_exported_checkpoint(
            FLAGS.checkpoint_2bit_dir, FLAGS.poll_timeout_hours
        )
        print(
            f"[eval_dsv41_b32_arc2] Milestone {FLAGS.milestone_tag} checkpoint ready in parent: {ckpt_info}",
            flush=True,
        )
        exe_path = sys.argv[0]
        child_args = _serialize_eval_flags_for_children()
        master_port = os.environ.get("MASTER_PORT", "29521")
        store_file = f"/tmp/eval_dsv41_b32_c10d_filestore_{os.getpid()}_{int(time.time())}"
        if os.path.exists(store_file):
            os.remove(store_file)
        print(
            f"Launching {world_size} Candidate B32 eval worker subprocesses via {exe_path} "
            f"(init_method=file://{store_file})...",
            flush=True,
        )
        procs: list[subprocess.Popen[bytes]] = []
        for r in range(world_size):
            child_env = os.environ.copy()
            child_env["RANK"] = str(r)
            child_env["LOCAL_RANK"] = str(r)
            child_env["WORLD_SIZE"] = str(world_size)
            child_env["MASTER_ADDR"] = "::1"
            child_env["MASTER_PORT"] = master_port
            child_env["NCCL_SOCKET_FAMILY"] = "AF_INET6"
            child_env["TORCH_DIST_INIT_FILE"] = store_file
            procs.append(subprocess.Popen([exe_path, *child_args], env=child_env))

        exit_codes: list[int] = []
        for r, p in enumerate(procs):
            rc = p.wait()
            exit_codes.append(rc)
            if rc != 0:
                print(
                    f"Eval worker rank {r} exited with non-zero code {rc}; terminating remaining workers.",
                    flush=True,
                )
                for other in procs:
                    if other.poll() is None:
                        other.terminate()
                break

        if any(rc != 0 for rc in exit_codes):
            raise RuntimeError(
                f"Candidate B32 eval worker subprocesses failed with exit codes: {exit_codes}"
            )
    else:
        _worker_main(rank=0, world_size=1)


if __name__ == "__main__":
    g3_multiprocessing.handle_main(main)

