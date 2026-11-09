"""DiffusionGemma ARC20 response canvases and official random corruption."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ARC_GRID_BYTES = b"0123456789\n"
ARC_GRID_TRANSLATION = bytes.maketrans(ARC_GRID_BYTES, bytes(range(11)))
ARC_VOCAB_SIZE = 20
PAD_ID = 19
BOS_ID = 13
START_ID = 14
END_TURN_ID = 15
USER_ID = 11
MODEL_ID = 12
NEWLINE_ID = 10
CHANNEL_START_ID = 16
THOUGHT_ID = 17
CHANNEL_END_ID = 18
EOS_ID = 19
CANVAS_LENGTH = 256
DATA_ADAPTER_VERSION = 5
CACHE_MODES = ("materialized", "lazy")
NVARC_SAMPLE_MODES = ("response", "conversation")


def encode_arc_grid(content: str) -> bytearray:
    if not isinstance(content, str):
        raise TypeError("ARC message content must be a string")
    try:
        raw = content.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("ARC content contains a non-ASCII character") from exc
    invalid = raw.translate(None, ARC_GRID_BYTES)
    if invalid:
        raise ValueError(
            f"ARC content contains unsupported byte 0x{invalid[0]:02x}; "
            "only digits and newline are accepted"
        )
    return bytearray(raw.translate(ARC_GRID_TRANSLATION))


def encode_complete_message(role: str, content: str) -> bytearray:
    if role == "user":
        role_id = USER_ID
    elif role == "assistant":
        role_id = MODEL_ID
    else:
        raise ValueError(f"Unsupported ARC message role: {role!r}")
    result = bytearray((START_ID, role_id, NEWLINE_ID))
    result.extend(encode_arc_grid(content))
    result.extend((END_TURN_ID, NEWLINE_ID))
    return result


def expand_saved_messages(
    messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    max_seq_length: int,
    min_physical_tokens: int = 0,
    canvas_length: int = CANVAS_LENGTH,
    instruction_token_ids: Iterable[int] = (),
    sample_mode: str = "response",
) -> list[dict[str, Any]]:
    """Create atomic diffusion rows from one saved ARC conversation.

    ``response`` preserves the historical behavior: every assistant response
    becomes one row whose prompt contains all preceding complete turns.
    ``conversation`` emits only the final assistant response, so every saved
    conversation becomes at most one row and all preceding training examples
    remain in its prompt. In either mode a target is EOS-padded to an integral
    number of 256-token canvases and admitted or rejected atomically. The
    collator later samples one canvas uniformly for diffusion supervision while
    the causal encoder sees the clean full prompt and response.
    """
    if canvas_length <= 0:
        raise ValueError("canvas_length must be positive")
    if max_seq_length <= canvas_length:
        raise ValueError("max_seq_length must exceed the diffusion canvas")
    if not messages or len(messages) % 2:
        raise ValueError("NVARC messages must contain complete user/assistant pairs")
    sample_mode = str(sample_mode).lower()
    if sample_mode not in NVARC_SAMPLE_MODES:
        raise ValueError(
            f"unsupported NVARC sample mode {sample_mode!r}; "
            f"expected one of {NVARC_SAMPLE_MODES}"
        )
    instruction_token_ids = tuple(int(value) for value in instruction_token_ids)
    if any(value < 0 or value > 255 for value in instruction_token_ids):
        raise ValueError("compact instruction token IDs must fit uint8")

    prefix = bytearray((BOS_ID,))
    examples: list[dict[str, Any]] = []
    example_index = 0
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        if role != expected_role:
            raise ValueError(
                f"message {index} has role {role!r}; expected {expected_role!r}"
            )
        content = message.get("content")
        if role == "user":
            encoded_user = encode_complete_message(role, content)
            if index == 0 and instruction_token_ids:
                # Official framing is ``<|turn>user\n`` followed by the fixed
                # instruction and then the first ARC grid.
                encoded_user[3:3] = bytearray(instruction_token_ids)
            prefix.extend(encoded_user)
            continue

        emit_response = sample_mode == "response" or index == len(messages) - 1
        if emit_response:
            assistant_prefix = bytearray(
                (
                    START_ID,
                    MODEL_ID,
                    NEWLINE_ID,
                    CHANNEL_START_ID,
                    THOUGHT_ID,
                    NEWLINE_ID,
                    CHANNEL_END_ID,
                )
            )
            prompt = prefix + assistant_prefix
            real_target = encode_arc_grid(content)
            real_target.append(END_TURN_ID)
            num_canvases = math.ceil(len(real_target) / canvas_length)
            padded_target_length = num_canvases * canvas_length
            target = real_target + bytearray(
                [EOS_ID] * (padded_target_length - len(real_target))
            )
            physical_tokens = len(prompt) + len(target)
            if min_physical_tokens <= physical_tokens <= max_seq_length:
                examples.append(
                    {
                        "input_ids": list(prompt),
                        "target_ids": list(target),
                        "target_length": len(real_target),
                        "num_canvases": num_canvases,
                        "prompt_tokens": len(prompt),
                        "physical_tokens": physical_tokens,
                        "example_index": example_index,
                    }
                )
        example_index += 1
        prefix.extend(encode_complete_message(role, content))
    return examples


def _expand_saved_batch(
    batch: dict[str, list[Any]],
    *,
    max_seq_length: int,
    min_physical_tokens: int,
    canvas_length: int,
    instruction_token_ids: tuple[int, ...],
    sample_mode: str,
) -> dict[str, list[Any]]:
    output = {
        "puzzle_name": [],
        "input_ids": [],
        "target_ids": [],
        "target_length": [],
        "num_canvases": [],
        "prompt_tokens": [],
        "physical_tokens": [],
        "example_index": [],
    }
    for puzzle_name, messages in zip(batch["puzzle_name"], batch["messages"]):
        for example in expand_saved_messages(
            messages,
            max_seq_length=max_seq_length,
            min_physical_tokens=min_physical_tokens,
            canvas_length=canvas_length,
            instruction_token_ids=instruction_token_ids,
            sample_mode=sample_mode,
        ):
            output["puzzle_name"].append(str(puzzle_name))
            for key in output:
                if key != "puzzle_name":
                    output[key].append(example[key])
    return output


def diffusion_arc_features() -> Any:
    from datasets import Features, Sequence as ArrowSequence, Value

    return Features(
        {
            "puzzle_name": Value("string"),
            "input_ids": ArrowSequence(Value("uint8")),
            "target_ids": ArrowSequence(Value("uint8")),
            "target_length": Value("uint16"),
            "num_canvases": Value("uint16"),
            "prompt_tokens": Value("uint32"),
            "physical_tokens": Value("uint32"),
            "example_index": Value("uint16"),
        }
    )


def diffusion_arc_lazy_index_features() -> Any:
    """Return the compact metadata needed to regenerate responses on demand."""
    from datasets import Features, Value

    return Features(
        {
            "raw_index": Value("uint32"),
            "example_index": Value("uint16"),
            "target_length": Value("uint16"),
            "num_canvases": Value("uint16"),
            "prompt_tokens": Value("uint32"),
            "physical_tokens": Value("uint32"),
        }
    )


def _numpy_dataset_column(
    dataset: Any,
    name: str,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """Read one Arrow scalar column without invoking dataset formatters.

    Newer ``datasets`` NumPy formatters import torchvision's video stack even
    for integer columns. Kaggle's intentionally minimal/broken torchvision
    build then fails on ``VideoReader``. Direct Arrow access is both faster and
    keeps this text-only data path independent of all vision packages. Preserve
    a selected/shuffled dataset's logical row order through its Arrow indices.
    """
    source = getattr(dataset, "index_dataset", dataset)
    table = getattr(source, "data", None)
    if table is not None and name in getattr(table, "column_names", ()):
        values = table.column(name).to_numpy(zero_copy_only=False)
        indices_table = getattr(source, "_indices", None)
        if indices_table is not None:
            logical_indices = indices_table.column(0).to_numpy(
                zero_copy_only=False
            )
            values = values[logical_indices]
    else:
        values = source[name]
    return np.asarray(values, dtype=dtype)


def _index_saved_batch(
    batch: dict[str, list[Any]],
    indices: list[int],
    *,
    max_seq_length: int,
    min_physical_tokens: int,
    canvas_length: int,
    instruction_token_ids: tuple[int, ...],
    sample_mode: str,
) -> dict[str, list[int]]:
    """Expand only sample metadata; token bytes are regenerated at access."""
    output: dict[str, list[int]] = {
        "raw_index": [],
        "example_index": [],
        "target_length": [],
        "num_canvases": [],
        "prompt_tokens": [],
        "physical_tokens": [],
    }
    for raw_index, messages in zip(indices, batch["messages"], strict=True):
        examples = expand_saved_messages(
            messages,
            max_seq_length=max_seq_length,
            min_physical_tokens=min_physical_tokens,
            canvas_length=canvas_length,
            instruction_token_ids=instruction_token_ids,
            sample_mode=sample_mode,
        )
        for example in examples:
            output["raw_index"].append(int(raw_index))
            for name in output:
                if name != "raw_index":
                    output[name].append(int(example[name]))
    return output


def _dataset_cache_key(
    dataset: Any,
    *,
    max_seq_length: int,
    min_physical_tokens: int,
    canvas_length: int,
    split_name: str,
    instruction_token_ids: tuple[int, ...],
    sample_mode: str = "response",
) -> str:
    payload = {
        "adapter_version": DATA_ADAPTER_VERSION,
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "rows": len(dataset),
        "max_seq_length": max_seq_length,
        "min_physical_tokens": min_physical_tokens,
        "canvas_length": canvas_length,
        "split_name": split_name,
        "instruction_token_ids": list(instruction_token_ids),
    }
    # Keep the historical response-mode key stable so an unchanged run can
    # reuse its cache and resume checkpoints created before this option existed.
    if sample_mode != "response":
        payload["sample_mode"] = sample_mode
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _lazy_index_cache_key(
    dataset: Any,
    *,
    max_seq_length: int,
    min_physical_tokens: int,
    canvas_length: int,
    split_name: str,
    instruction_token_ids: tuple[int, ...],
    sample_mode: str = "response",
) -> str:
    materialized_key = _dataset_cache_key(
        dataset,
        max_seq_length=max_seq_length,
        min_physical_tokens=min_physical_tokens,
        canvas_length=canvas_length,
        split_name=split_name,
        instruction_token_ids=instruction_token_ids,
        sample_mode=sample_mode,
    )
    payload = {
        "materialized_key": materialized_key,
        "representation": "lazy-response-index-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


class DiffusionArcLazyResponseDataset(torch.utils.data.Dataset):
    """Regenerate compact token rows from a small sample metadata index.

    The source ``messages`` remain memory-mapped under read-only Kaggle input.
    Only response-to-source mapping and scalar length metadata are cached in
    writable storage. This avoids retaining every prompt/target byte in tmpfs
    while preserving the materialized dataset's exact row order and values.
    """

    def __init__(
        self,
        raw_dataset: Any,
        index_dataset: Any,
        *,
        max_seq_length: int,
        min_physical_tokens: int,
        canvas_length: int,
        instruction_token_ids: tuple[int, ...],
        sample_mode: str = "response",
    ) -> None:
        if len(index_dataset) == 0:
            raise ValueError("lazy DiffusionGemma sample index is empty")
        self.raw_dataset = raw_dataset
        self.index_dataset = index_dataset
        self.max_seq_length = int(max_seq_length)
        self.min_physical_tokens = int(min_physical_tokens)
        self.canvas_length = int(canvas_length)
        self.instruction_token_ids = tuple(instruction_token_ids)
        self.sample_mode = str(sample_mode).lower()
        if self.sample_mode not in NVARC_SAMPLE_MODES:
            raise ValueError(
                f"unsupported NVARC sample mode {self.sample_mode!r}; "
                f"expected one of {NVARC_SAMPLE_MODES}"
            )

    def __len__(self) -> int:
        return len(self.index_dataset)

    def __getitem__(self, index: int | str) -> Any:
        if isinstance(index, str):
            if index not in self.index_dataset.column_names:
                raise KeyError(index)
            return self.index_dataset[index]

        indexed = self.index_dataset[int(index)]
        raw = self.raw_dataset[int(indexed["raw_index"])]
        example_index = int(indexed["example_index"])
        matches = [
            example
            for example in expand_saved_messages(
                raw["messages"],
                max_seq_length=self.max_seq_length,
                min_physical_tokens=self.min_physical_tokens,
                canvas_length=self.canvas_length,
                instruction_token_ids=self.instruction_token_ids,
                sample_mode=self.sample_mode,
            )
            if int(example["example_index"]) == example_index
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "lazy DiffusionGemma sample index no longer matches source "
                f"row {indexed['raw_index']} response {example_index}"
            )
        example = matches[0]
        for name in (
            "target_length",
            "num_canvases",
            "prompt_tokens",
            "physical_tokens",
        ):
            if int(example[name]) != int(indexed[name]):
                raise RuntimeError(
                    "lazy DiffusionGemma response metadata changed for "
                    f"{name}: index={indexed[name]}, regenerated={example[name]}"
                )
        return {"puzzle_name": str(raw["puzzle_name"]), **example}


def tokenize_diffusion_arc_dataset(
    raw_dataset: Any,
    *,
    cache_root: str | Path,
    max_seq_length: int,
    min_physical_tokens: int,
    data_workers: int,
    split_name: str,
    instruction_token_ids: Iterable[int] = (),
    cache_mode: str = "materialized",
    sample_mode: str = "response",
) -> tuple[Any, dict[str, Any], Path]:
    """Expand and cache compact response- or conversation-level examples."""
    if len(raw_dataset) == 0:
        raise ValueError("NVARC dataset is empty")
    if data_workers <= 0:
        raise ValueError("data_workers must be positive")
    cache_mode = str(cache_mode).lower()
    if cache_mode not in CACHE_MODES:
        raise ValueError(
            f"unsupported DiffusionGemma cache mode {cache_mode!r}; "
            f"expected one of {CACHE_MODES}"
        )
    sample_mode = str(sample_mode).lower()
    if sample_mode not in NVARC_SAMPLE_MODES:
        raise ValueError(
            f"unsupported NVARC sample mode {sample_mode!r}; "
            f"expected one of {NVARC_SAMPLE_MODES}"
        )
    instruction_token_ids = tuple(int(value) for value in instruction_token_ids)
    cache_key = _dataset_cache_key(
        raw_dataset,
        max_seq_length=max_seq_length,
        min_physical_tokens=min_physical_tokens,
        canvas_length=CANVAS_LENGTH,
        split_name=split_name,
        instruction_token_ids=instruction_token_ids,
        sample_mode=sample_mode,
    )
    compact_vocab_size = max(
        ARC_VOCAB_SIZE,
        max(instruction_token_ids, default=ARC_VOCAB_SIZE - 1) + 1,
    )
    cache_dir = (
        Path(cache_root).resolve() / f"diffusion-gemma-arc{compact_vocab_size}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    workers = min(data_workers, len(raw_dataset))
    sample_label = "responses" if sample_mode == "response" else "conversations"
    map_kwargs = {
        "batched": True,
        "batch_size": 64,
        "num_proc": workers if workers > 1 else None,
        "remove_columns": raw_dataset.column_names,
        "fn_kwargs": {
            "max_seq_length": int(max_seq_length),
            "min_physical_tokens": int(min_physical_tokens),
            "canvas_length": CANVAS_LENGTH,
            "instruction_token_ids": instruction_token_ids,
            "sample_mode": sample_mode,
        },
        "load_from_cache_file": True,
    }
    if cache_mode == "lazy":
        index_key = _lazy_index_cache_key(
            raw_dataset,
            max_seq_length=max_seq_length,
            min_physical_tokens=min_physical_tokens,
            canvas_length=CANVAS_LENGTH,
            split_name=split_name,
            instruction_token_ids=instruction_token_ids,
            sample_mode=sample_mode,
        )
        cache_file = cache_dir / f"{split_name}-{index_key}-lazy-index.arrow"
        index_dataset = raw_dataset.map(
            _index_saved_batch,
            with_indices=True,
            features=diffusion_arc_lazy_index_features(),
            cache_file_name=str(cache_file),
            new_fingerprint=index_key,
            desc=f"Index DiffusionGemma ARC {sample_label} ({split_name})",
            **map_kwargs,
        )
        tokenized = DiffusionArcLazyResponseDataset(
            raw_dataset,
            index_dataset,
            max_seq_length=max_seq_length,
            min_physical_tokens=min_physical_tokens,
            canvas_length=CANVAS_LENGTH,
            instruction_token_ids=instruction_token_ids,
            sample_mode=sample_mode,
        )
        response_dataset = index_dataset
        representation_cache_key = index_key
    else:
        cache_file = cache_dir / f"{split_name}-{cache_key}.arrow"
        tokenized = raw_dataset.map(
            _expand_saved_batch,
            features=diffusion_arc_features(),
            cache_file_name=str(cache_file),
            new_fingerprint=cache_key,
            desc=f"Build DiffusionGemma ARC {sample_label} ({split_name})",
            **map_kwargs,
        )
        response_dataset = tokenized
        representation_cache_key = cache_key
    if not len(tokenized):
        raise RuntimeError(
            "No complete diffusion samples satisfy the context window: "
            f"physical={min_physical_tokens:,}..{max_seq_length:,}"
        )
    physical = _numpy_dataset_column(response_dataset, "physical_tokens")
    prompts = _numpy_dataset_column(response_dataset, "prompt_tokens")
    canvases = _numpy_dataset_column(response_dataset, "num_canvases")
    cache_files = [
        Path(item["filename"])
        for item in getattr(response_dataset, "cache_files", ())
        if item.get("filename")
    ]
    cache_bytes = sum(
        path.stat().st_size for path in cache_files if path.is_file()
    )
    cache_directory_bytes = sum(
        path.stat().st_size for path in cache_dir.glob("*.arrow") if path.is_file()
    )
    metadata = {
        "cache_mode": cache_mode,
        "sample_mode": sample_mode,
        # This content key deliberately remains representation-independent so
        # an exact checkpoint can resume after switching between materialized
        # and lazy storage. The on-disk representation has its own key below.
        "cache_key": cache_key,
        "representation_cache_key": representation_cache_key,
        "cache_file": str(cache_file),
        "cache_files": len(cache_files),
        "cache_bytes": cache_bytes,
        "cache_directory_bytes": cache_directory_bytes,
        "input_rows": len(raw_dataset),
        "responses": len(tokenized),
        "examples": len(tokenized),
        "max_seq_length": max_seq_length,
        "min_physical_tokens_requested": min_physical_tokens,
        "physical_tokens_min": int(physical.min()),
        "physical_tokens_max": int(physical.max()),
        "prompt_tokens_min": int(prompts.min()),
        "prompt_tokens_max": int(prompts.max()),
        "canvases_min": int(canvases.min()),
        "canvases_max": int(canvases.max()),
        "canvas_length": CANVAS_LENGTH,
        "workers": workers,
        "instruction_token_ids": list(instruction_token_ids),
        "compact_vocab_size": compact_vocab_size,
    }
    return tokenized, metadata, cache_file


class DiffusionArcCollator:
    """Sample one response canvas and apply uniform random-token corruption."""

    def __init__(
        self,
        *,
        seed: int,
        eps: float = 1e-3,
        self_conditioning_probability: float = 0.5,
        canvas_length: int = CANVAS_LENGTH,
    ):
        if not 0.0 < eps <= 1.0:
            raise ValueError("eps must be in (0, 1]")
        if not 0.0 <= self_conditioning_probability <= 1.0:
            raise ValueError("self_conditioning_probability must be in [0, 1]")
        if canvas_length <= 0:
            raise ValueError("canvas_length must be positive")
        self.eps = float(eps)
        self.self_conditioning_probability = float(
            self_conditioning_probability
        )
        self.canvas_length = int(canvas_length)
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset corruption/canvas sampling, primarily for repeatable evaluation."""
        self.generator.manual_seed(self.seed if seed is None else int(seed))

    def state_dict(self) -> dict[str, Any]:
        """Return the exact corruption RNG state for segmented training."""
        return {
            "version": 1,
            "seed": self.seed,
            "eps": self.eps,
            "self_conditioning_probability": self.self_conditioning_probability,
            "canvas_length": self.canvas_length,
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore corruption RNG state, failing on a changed data contract."""
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported DiffusionArcCollator state version")
        expected = {
            "seed": self.seed,
            "eps": self.eps,
            "self_conditioning_probability": self.self_conditioning_probability,
            "canvas_length": self.canvas_length,
        }
        actual = {name: state.get(name) for name in expected}
        if actual != expected:
            raise ValueError(
                "DiffusionArcCollator checkpoint contract changed: "
                f"saved={actual}, current={expected}"
            )
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state) or generator_state.device.type != "cpu":
            raise TypeError("collator generator_state must be a CPU tensor")
        self.generator.set_state(generator_state)

    def __call__(self, rows: Iterable[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = list(rows)
        if not rows:
            raise ValueError("cannot collate an empty DiffusionGemma batch")
        batch_size = len(rows)
        full_lengths = torch.tensor(
            [len(row["input_ids"]) + len(row["target_ids"]) for row in rows],
            dtype=torch.long,
        )
        prompt_lengths = torch.tensor(
            [len(row["input_ids"]) for row in rows], dtype=torch.long
        )
        max_full = int(full_lengths.max())
        input_ids = torch.full((batch_size, max_full), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_full), dtype=torch.long)
        clean_canvas = torch.empty(
            (batch_size, self.canvas_length), dtype=torch.long
        )
        real_target_mask = torch.zeros(
            (batch_size, self.canvas_length), dtype=torch.bool
        )
        selected_canvas_indices = torch.empty(batch_size, dtype=torch.long)
        noise_fractions = torch.empty(batch_size, dtype=torch.float32)
        self_conditioning_mask = torch.empty(batch_size, dtype=torch.bool)

        for row_index, row in enumerate(rows):
            prompt = torch.as_tensor(row["input_ids"], dtype=torch.long)
            target = torch.as_tensor(row["target_ids"], dtype=torch.long)
            target_length = int(row["target_length"])
            num_canvases = int(row["num_canvases"])
            if not 0 < target_length <= len(target):
                raise ValueError(f"invalid real target length {target_length}")
            if num_canvases <= 0 or len(target) != num_canvases * self.canvas_length:
                raise ValueError(
                    f"invalid padded target length {len(target)} for "
                    f"{num_canvases} canvas(es)"
                )
            if target[target_length - 1].item() != END_TURN_ID:
                raise ValueError("the real response must end with the end-turn token")
            eos_fill = target[target_length:]
            if eos_fill.numel() and not bool(torch.all(eos_fill == EOS_ID)):
                raise ValueError(
                    "response padding after the end-turn token must be EOS-filled"
                )
            full = torch.cat((prompt, target))
            input_ids[row_index, : len(full)] = full
            attention_mask[row_index, : len(full)] = 1
            selected = int(
                torch.randint(
                    0, num_canvases, (1,), generator=self.generator
                ).item()
            )
            selected_canvas_indices[row_index] = selected
            start = selected * self.canvas_length
            clean_canvas[row_index] = target[start : start + self.canvas_length]
            real_tokens_in_canvas = min(
                self.canvas_length,
                max(0, target_length - start),
            )
            real_target_mask[row_index, :real_tokens_in_canvas] = True
            t = self.eps + (1.0 - self.eps) * float(
                torch.rand((), generator=self.generator).item()
            )
            noise_fractions[row_index] = t
            self_conditioning_mask[row_index] = bool(
                torch.rand((), generator=self.generator).item()
                < self.self_conditioning_probability
            )

        decoder_input_ids = clean_canvas.clone()
        noise_mask = (
            torch.rand(
                decoder_input_ids.shape,
                generator=self.generator,
                dtype=torch.float32,
            )
            < noise_fractions[:, None]
        )
        random_tokens = torch.randint(
            0,
            ARC_VOCAB_SIZE,
            decoder_input_ids.shape,
            generator=self.generator,
            dtype=torch.long,
        )
        decoder_input_ids = torch.where(
            noise_mask, random_tokens, decoder_input_ids
        )

        position_ids = torch.arange(max_full, dtype=torch.long)[None, :].expand(
            batch_size, -1
        ).clone()
        position_ids.masked_fill_(attention_mask == 0, 0)
        decoder_position_ids = (
            prompt_lengths[:, None]
            + selected_canvas_indices[:, None] * self.canvas_length
            + torch.arange(self.canvas_length, dtype=torch.long)[None, :]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_position_ids": decoder_position_ids,
            "labels": clean_canvas,
            "real_target_mask": real_target_mask,
            "self_conditioning_mask": self_conditioning_mask,
            "noise_fractions": noise_fractions,
            "noise_mask": noise_mask,
            "prompt_lengths": prompt_lengths,
            "selected_canvas_indices": selected_canvas_indices,
        }


class DiffusionArcDocumentPackDataset(torch.utils.data.Dataset):
    """Length-bucket independent ARC documents into one batch-axis microbatch.

    This deliberately does *not* concatenate documents along the sequence axis.
    Generic sequence packing would let one ARC problem attend to another unless
    every DiffusionGemma encoder/decoder attention path consumed document-aware
    masks.  Instead, each item contains up to ``max_documents`` independent
    examples.  The collator places them on the batch axis and pads only within
    a narrow length bucket, preserving the exact per-document objective while
    amortizing one forward/backward over several samples.

    Groups are formed from neighboring examples after a stable length sort.
    ``token_budget`` bounds the *actual padded encoder tensor* (documents times
    the longest document), not merely the sum of unpadded lengths.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        token_budget: int,
        max_documents: int,
        validation: bool = False,
        max_groups: int = 0,
    ):
        if len(dataset) == 0:
            raise ValueError("cannot pack an empty DiffusionGemma dataset")
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if max_documents <= 0:
            raise ValueError("max_documents must be positive")
        if max_groups < 0:
            raise ValueError("max_groups cannot be negative")
        lengths = _numpy_dataset_column(
            dataset,
            "physical_tokens",
            dtype=np.int32,
        )
        if lengths.ndim != 1 or len(lengths) != len(dataset):
            raise ValueError("physical_tokens must contain one scalar per row")
        if int(lengths.max()) > token_budget:
            raise ValueError(
                f"document length {int(lengths.max())} exceeds pack token budget "
                f"{token_budget}"
            )

        order = np.argsort(lengths, kind="stable")
        # The upper bound is one group per document. int32 is sufficient for
        # the current NVARC corpus and avoids a multi-gigabyte Python int list.
        if len(dataset) >= np.iinfo(np.int32).max:
            raise ValueError("DiffusionGemma pack indices exceed int32 capacity")
        groups = np.full((len(dataset), max_documents), -1, dtype=np.int32)
        counts = np.zeros(len(dataset), dtype=np.uint16)
        padded_tokens = np.zeros(len(dataset), dtype=np.int32)
        group_index = 0
        current: list[int] = []
        current_max = 0

        def commit() -> None:
            nonlocal group_index, current, current_max
            if not current:
                return
            groups[group_index, : len(current)] = current
            counts[group_index] = len(current)
            padded_tokens[group_index] = len(current) * current_max
            group_index += 1
            current = []
            current_max = 0

        # In stable length order, the largest legal group ending at each row is
        # ``min(max_documents, token_budget // length)``. This capacity is
        # monotonically non-increasing and has at most ``max_documents``
        # distinct runs. Bulk-copy every complete constant-capacity run while
        # carrying at most one partial group across a boundary. This is exactly
        # the prior greedy document-by-document algorithm, but keeps the
        # 17-million-response full corpus out of a multi-minute Python loop.
        sorted_lengths = lengths[order]
        capacities = np.minimum(
            max_documents,
            token_budget // sorted_lengths,
        )
        negative_capacities = -capacities
        for capacity in range(max_documents, 0, -1):
            segment_start = int(
                np.searchsorted(negative_capacities, -capacity, side="left")
            )
            segment_end = int(
                np.searchsorted(negative_capacities, -capacity, side="right")
            )
            if segment_start == segment_end:
                continue
            segment_order = order[segment_start:segment_end]
            segment_lengths = sorted_lengths[segment_start:segment_end]
            offset = 0

            if current:
                if len(current) >= capacity:
                    commit()
                take = min(capacity - len(current), len(segment_order))
                if take:
                    current.extend(int(value) for value in segment_order[:take])
                    current_max = int(segment_lengths[take - 1])
                    offset = take
                if len(current) == capacity:
                    commit()

            remaining = len(segment_order) - offset
            complete_groups = remaining // capacity
            complete_values = complete_groups * capacity
            if complete_groups:
                destination = slice(group_index, group_index + complete_groups)
                source = slice(offset, offset + complete_values)
                groups[destination, :capacity] = segment_order[source].reshape(
                    complete_groups, capacity
                )
                counts[destination] = capacity
                padded_tokens[destination] = (
                    capacity
                    * segment_lengths[source].reshape(
                        complete_groups, capacity
                    )[:, -1]
                )
                group_index += complete_groups
                offset += complete_values

            if offset < len(segment_order):
                current = [int(value) for value in segment_order[offset:]]
                current_max = int(segment_lengths[-1])
        commit()
        groups = groups[:group_index]
        counts = counts[:group_index]
        padded_tokens = padded_tokens[:group_index]

        if max_groups and max_groups < group_index:
            # Cover the entire length range instead of retaining only the
            # shortest validation groups.
            selected = np.linspace(
                0, group_index - 1, num=max_groups, dtype=np.int64
            )
            groups = groups[selected].copy()
            counts = counts[selected].copy()
            padded_tokens = padded_tokens[selected].copy()

        selected_indices = groups[groups >= 0].astype(np.int64, copy=False)
        selected_lengths = lengths[selected_indices]
        self.dataset = dataset
        self.groups = groups
        self.counts = counts
        self.validation = bool(validation)
        self.max_documents = int(max_documents)
        self.token_budget = int(token_budget)
        self.metadata = {
            "mode": "document-isolated-batch-axis",
            "groups": int(len(groups)),
            "documents": int(counts.astype(np.int64).sum()),
            "max_documents": int(max_documents),
            "documents_per_group_mean": float(counts.mean()),
            "documents_per_group_min": int(counts.min()),
            "documents_per_group_max": int(counts.max()),
            "token_budget": int(token_budget),
            "padded_tokens_min": int(padded_tokens.min()),
            "padded_tokens_max": int(padded_tokens.max()),
            "padded_tokens_mean": float(padded_tokens.mean()),
            "padding_fill_ratio": float(
                selected_lengths.astype(np.int64).sum()
                / padded_tokens.astype(np.int64).sum()
            ),
            "validation": bool(validation),
        }

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        count = int(self.counts[index])
        indices = self.groups[index, :count]
        return {
            "_arc_documents": [self.dataset[int(value)] for value in indices],
            "_arc_pack_size": count,
            "_arc_pack_capacity": self.max_documents,
            "_arc_validation": self.validation,
        }


class DiffusionArcDocumentPackCollator:
    """Route packed train/eval items to deterministic diffusion collators."""

    def __init__(
        self,
        train_collator: DiffusionArcCollator,
        eval_collator: DiffusionArcCollator | None = None,
        *,
        training_documents_per_pack: float = 1.0,
    ):
        if training_documents_per_pack <= 0:
            raise ValueError("training_documents_per_pack must be positive")
        self.train_collator = train_collator
        self.eval_collator = eval_collator or train_collator
        self.training_documents_per_pack = float(training_documents_per_pack)

    def reset_eval_seed(self) -> None:
        self.eval_collator.reset_seed()

    def state_dict(self) -> dict[str, Any]:
        """Capture independent train/eval corruption streams."""
        return {
            "version": 1,
            "training_documents_per_pack": self.training_documents_per_pack,
            "train_collator": self.train_collator.state_dict(),
            "eval_collator": self.eval_collator.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore both corruption streams after validating pack weighting."""
        if int(state.get("version", -1)) != 1:
            raise ValueError(
                "unsupported DiffusionArcDocumentPackCollator state version"
            )
        saved_mean = float(state.get("training_documents_per_pack", -1.0))
        if not math.isclose(
            saved_mean,
            self.training_documents_per_pack,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "document-pack loss weighting changed across resume: "
                f"saved={saved_mean}, current={self.training_documents_per_pack}"
            )
        self.train_collator.load_state_dict(state["train_collator"])
        self.eval_collator.load_state_dict(state["eval_collator"])

    def __call__(
        self, packed_rows: Iterable[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        packed_rows = list(packed_rows)
        if not packed_rows:
            raise ValueError("cannot collate an empty document pack")
        validation_flags = {bool(row["_arc_validation"]) for row in packed_rows}
        if len(validation_flags) != 1:
            raise ValueError("cannot mix training and validation documents")
        documents: list[dict[str, Any]] = []
        actual = 0
        for row in packed_rows:
            rows = row["_arc_documents"]
            if int(row["_arc_pack_size"]) != len(rows):
                raise ValueError("document pack size metadata is inconsistent")
            documents.extend(rows)
            actual += len(rows)
        is_validation = validation_flags.pop()
        collator = self.eval_collator if is_validation else self.train_collator
        batch = collator(documents)
        batch["arc_document_count"] = torch.tensor(actual, dtype=torch.long)
        # Trainer averages a fixed number of outer packs. Weighting by the
        # corpus-wide mean pack occupancy makes every document contribute the
        # same amount while preserving an expected loss scale of one per pack.
        # Exact per-update normalization would require looking ahead across all
        # GA microbatches before any backward call.
        # Evaluation reports an ordinary unscaled loss; its deterministic
        # length-stratified groups are a monitoring signal rather than a source
        # of training gradients.
        batch["arc_loss_weight"] = torch.tensor(
            (
                1.0
                if is_validation
                else actual / self.training_documents_per_pack
            ),
            dtype=torch.float32,
        )
        return batch


__all__ = [
    "ARC_VOCAB_SIZE",
    "BOS_ID",
    "CANVAS_LENGTH",
    "CHANNEL_END_ID",
    "CHANNEL_START_ID",
    "END_TURN_ID",
    "EOS_ID",
    "MODEL_ID",
    "PAD_ID",
    "START_ID",
    "THOUGHT_ID",
    "USER_ID",
    "CACHE_MODES",
    "NVARC_SAMPLE_MODES",
    "DiffusionArcCollator",
    "DiffusionArcDocumentPackCollator",
    "DiffusionArcDocumentPackDataset",
    "DiffusionArcLazyResponseDataset",
    "encode_arc_grid",
    "expand_saved_messages",
    "tokenize_diffusion_arc_dataset",
]
