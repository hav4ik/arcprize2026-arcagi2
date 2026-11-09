#!/usr/bin/env python3
"""Atomically convert DiffusionGemma to a compact native-framed ARC vocabulary."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_ASSETS = SCRIPT_DIR.parent / "ARChitects" / "qwen3_configs"
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "arc_diffusion_gemma_vocab_conversion.json"
ARC_TOKENS = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "Ċ",
    "user",
    "model",
    "<bos>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "thought",
    "<channel|>",
    "<eos>",
)
ARC_OUTPUT_VOCAB_SIZE = len(ARC_TOKENS)
DEFAULT_ARC_INSTRUCTION = (
    "Infer the grid transformation from the examples. Reply only with the "
    "output grid.\n"
)
DEFAULT_SUDOKU_INSTRUCTION = (
    "Solve this Sudoku puzzle. 0 marks an empty cell. Reply with the completed "
    "9x9 grid.\n"
)
TOKENIZER_TEMPLATE_TOKENS = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "Ċ",
    "user",
    "assistant",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
)
TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "chat_template.jinja",
)
VOCABULARY_WEIGHT_KEYS = {
    "model.decoder.embed_tokens.weight",
    "model.encoder.language_model.embed_tokens.weight",
    "lm_head.weight",
}
VISION_WEIGHT_PREFIXES = (
    "model.encoder.vision_tower.",
    "model.encoder.embed_vision.",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"DiffusionGemma source text {text!r} must encode to one token; "
            f"got {token_ids}"
        )
    return int(token_ids[0])


def resolve_arc_source_row_ids(tokenizer: Any) -> list[int]:
    """Resolve semantic ARC rows from the source tokenizer and fail closed."""
    row_ids = [_single_token_id(tokenizer, str(value)) for value in range(10)]
    row_ids.extend(
        [
            _single_token_id(tokenizer, "\n"),
            _single_token_id(tokenizer, "user"),
            _single_token_id(tokenizer, "model"),
        ]
    )
    bos_id = tokenizer.bos_token_id
    turn_id = tokenizer.convert_tokens_to_ids("<|turn>")
    end_turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if bos_id is None:
        raise ValueError("Source tokenizer must define bos_token_id")
    if turn_id is None or turn_id == unknown_id:
        raise ValueError("Source tokenizer does not expose a distinct <|turn> token")
    if end_turn_id is None or end_turn_id == unknown_id:
        raise ValueError("Source tokenizer does not expose a distinct <turn|> token")
    row_ids.extend(
        [
            int(bos_id),
            int(turn_id),
            int(end_turn_id),
            _single_token_id(tokenizer, "<|channel>"),
            _single_token_id(tokenizer, "thought"),
            _single_token_id(tokenizer, "<channel|>"),
        ]
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Source tokenizer must define eos_token_id")
    row_ids.append(int(eos_id))
    if len(set(row_ids)) != len(ARC_TOKENS):
        raise ValueError(f"ARC source-row mapping is not one-to-one: {row_ids}")
    return row_ids


def resolve_instruction_contract(
    tokenizer: Any,
    instruction_text: str,
    *,
    base_row_ids: Sequence[int],
) -> dict[str, Any]:
    """Map a fixed natural-language prefix into appended compact rows.

    The first 20 compact IDs remain the ARC output alphabet.  Only source rows
    needed by the fixed instruction are appended, in first-use order.  The
    returned surface pieces are used as non-special AddedTokens so the compact
    tokenizer can reproduce the same source-token sequence for the exact
    instruction text.
    """
    if not instruction_text:
        return {
            "text": "",
            "source_token_ids": [],
            "compact_token_ids": [],
            "added_source_row_ids": [],
            "added_source_tokens": [],
            "added_compact_tokens": [],
        }
    source_token_ids = [
        int(value)
        for value in tokenizer.encode(instruction_text, add_special_tokens=False)
    ]
    if not source_token_ids:
        raise ValueError("ARC instruction must encode to at least one source token")
    source_to_compact = {
        int(source_id): compact_id
        for compact_id, source_id in enumerate(base_row_ids)
    }
    added_source_row_ids: list[int] = []
    added_source_tokens: list[str] = []
    added_compact_tokens: list[str] = []
    compact_token_ids: list[int] = []
    backend = getattr(tokenizer, "backend_tokenizer", None)
    for source_id in source_token_ids:
        compact_id = source_to_compact.get(source_id)
        if compact_id is None:
            compact_id = ARC_OUTPUT_VOCAB_SIZE + len(added_source_row_ids)
            source_to_compact[source_id] = compact_id
            added_source_row_ids.append(source_id)
            source_token = tokenizer.convert_ids_to_tokens(source_id)
            if source_token is None:
                raise ValueError(f"Source tokenizer has no token for ID {source_id}")
            if backend is not None:
                surface = backend.decode([source_id], skip_special_tokens=False)
            else:
                surface = tokenizer.decode(
                    [source_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            if not surface:
                raise ValueError(
                    f"Source instruction token {source_id} decodes to empty text"
                )
            if surface in ARC_TOKENS or surface in added_compact_tokens:
                raise ValueError(
                    "Instruction source rows do not have a one-to-one surface "
                    f"representation: ID {source_id} -> {surface!r}"
                )
            added_source_tokens.append(str(source_token))
            added_compact_tokens.append(str(surface))
        compact_token_ids.append(compact_id)
    return {
        "text": instruction_text,
        "source_token_ids": source_token_ids,
        "compact_token_ids": compact_token_ids,
        "added_source_row_ids": added_source_row_ids,
        "added_source_tokens": added_source_tokens,
        "added_compact_tokens": added_compact_tokens,
    }


def validate_tokenizer_assets(asset_root: Path) -> None:
    missing = [
        name for name in TOKENIZER_ASSET_NAMES if not (asset_root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"ARC tokenizer assets missing from {asset_root}: {missing}"
        )
    tokenizer_json = load_json(asset_root / "tokenizer.json")
    actual = {
        str(token): int(index)
        for token, index in tokenizer_json.get("model", {}).get("vocab", {}).items()
    }
    actual.update(
        {
            str(entry["content"]): int(entry["id"])
            for entry in tokenizer_json.get("added_tokens", ())
        }
    )
    expected = {token: index for index, token in enumerate(TOKENIZER_TEMPLATE_TOKENS)}
    if actual != expected:
        raise ValueError(f"ARC tokenizer mapping mismatch: {actual}")


def build_compact_config(
    source_config: dict[str, Any],
    *,
    vocab_size: int = ARC_OUTPUT_VOCAB_SIZE,
    drop_vision: bool = False,
) -> dict[str, Any]:
    if source_config.get("model_type") != "diffusion_gemma":
        raise ValueError(
            "Expected source model_type='diffusion_gemma', got "
            f"{source_config.get('model_type')!r}"
        )
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("DiffusionGemma config has no text_config object")
    result = json.loads(json.dumps(source_config))
    result["text_config"].update(
        {
            "vocab_size": int(vocab_size),
            "pad_token_id": 19,
            "bos_token_id": 13,
            "eos_token_id": 19,
            "tie_word_embeddings": True,
        }
    )
    result.update(
        {
            "tie_word_embeddings": True,
            "pad_token_id": 19,
            "bos_token_id": 13,
            "eos_token_id": 19,
            "image_token_id": None,
            "boi_token_id": None,
            "eoi_token_id": None,
        }
    )
    if drop_vision:
        # DiffusionGemma's configuration explicitly permits a missing vision
        # configuration.  The repository runtime shim handles Transformers
        # releases whose generated encoder constructor still instantiates the
        # tower unconditionally despite that contract.
        result["vision_config"] = None
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_directory(output: Path, *, overwrite: bool) -> Path:
    output = output.resolve()
    if output == Path(output.anchor) or output == Path.home().resolve():
        raise ValueError(f"Refusing unsafe output path: {output}")
    if output == Path("/kaggle/input") or output.is_relative_to(Path("/kaggle/input")):
        raise ValueError(f"Kaggle input mounts are read-only: {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {output}; pass --overwrite-output-dir"
            )
        if not (output / MANIFEST_NAME).is_file():
            raise ValueError(f"Refusing to replace unmarked output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir()
    return staging


def finalize_output_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    output = output.resolve()
    if output.exists():
        if not overwrite or not (output / MANIFEST_NAME).is_file():
            raise ValueError(f"Refusing to replace output: {output}")
        shutil.rmtree(output)
    os.replace(staging, output)


def copy_compact_assets(
    source: Path,
    assets: Path,
    destination: Path,
    *,
    compact_tokens: Sequence[str] = ARC_TOKENS,
    instruction_text: str = "",
    retain_vision_assets: bool = True,
) -> None:
    validate_tokenizer_assets(assets)
    compact_tokens = tuple(str(token) for token in compact_tokens)
    if compact_tokens[:ARC_OUTPUT_VOCAB_SIZE] != ARC_TOKENS:
        raise ValueError("compact vocabulary must preserve the ARC20 ID prefix")
    if len(set(compact_tokens)) != len(compact_tokens):
        raise ValueError("compact tokenizer token strings must be unique")
    for name in TOKENIZER_ASSET_NAMES:
        shutil.copy2(assets / name, destination / name)

    tokenizer_json_path = destination / "tokenizer.json"
    tokenizer_json = load_json(tokenizer_json_path)
    tokenizer_json["model"]["unk_token"] = "<eos>"
    tokenizer_json["model"]["vocab"] = {
        token: index for index, token in enumerate(compact_tokens[:13])
    }
    tokenizer_json["added_tokens"] = [
        {
            "id": index,
            "content": compact_tokens[index],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": index < ARC_OUTPUT_VOCAB_SIZE,
        }
        for index in range(13, len(compact_tokens))
    ]
    tokenizer_json_path.write_text(
        json.dumps(tokenizer_json, indent=2, ensure_ascii=False) + "\n"
    )

    def token_record(content: str, *, special: bool = True) -> dict[str, Any]:
        return {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": special,
        }

    tokenizer_config_path = destination / "tokenizer_config.json"
    tokenizer_config = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(index): {
                **token_record(
                    compact_tokens[index],
                    special=index < ARC_OUTPUT_VOCAB_SIZE,
                ),
            }
            for index in range(13, len(compact_tokens))
        },
        "additional_special_tokens": [
            "<|turn>",
            "<|channel>",
            "thought",
            "<channel|>",
        ],
        "bos_token": "<bos>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<eos>",
        "model_max_length": 12288,
        "pad_token": "<eos>",
        "padding_side": "left",
        "split_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<eos>",
    }
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n"
    )
    special_tokens_path = destination / "special_tokens_map.json"
    special_tokens = {
        "additional_special_tokens": [
            token_record("<|turn>"),
            token_record("<|channel>"),
            token_record("thought"),
            token_record("<channel|>"),
        ],
        "bos_token": token_record("<bos>"),
        "eos_token": token_record("<eos>"),
        "pad_token": token_record("<eos>"),
        "unk_token": token_record("<eos>"),
    }
    special_tokens_path.write_text(
        json.dumps(special_tokens, indent=2, ensure_ascii=False) + "\n"
    )
    (destination / "added_tokens.json").write_text(
        json.dumps(
            {
                compact_tokens[index]: index
                for index in range(13, len(compact_tokens))
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (destination / "vocab.json").write_text(
        json.dumps(
            {token: index for index, token in enumerate(compact_tokens[:13])},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    first_user_prefix = (
        "{{- " + json.dumps(instruction_text, ensure_ascii=False) + " }}"
        if instruction_text
        else ""
    )
    (destination / "chat_template.jinja").write_text(
        "{%- set native_role = {'assistant': 'model'} %}"
        "{{- bos_token }}"
        "{%- for message in messages %}"
        "{{- '<|turn>' + native_role.get(message.role, message.role) + '\\n' }}"
        "{%- if loop.first and message.role == 'user' %}"
        + first_user_prefix
        + "{%- endif %}"
        "{{- message.content + '<turn|>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|turn>model\\n<|channel>thought\\n<channel|>' }}"
        "{%- endif %}\n"
    )

    optional_assets = [
        "LICENSE",
        "README.md",
        "chat_template.json",
    ]
    if retain_vision_assets:
        optional_assets.extend(("preprocessor_config.json", "processor_config.json"))
    for name in optional_assets:
        if (source / name).is_file() and not (destination / name).exists():
            shutil.copy2(source / name, destination / name)


def convert_safetensors(
    source: Path,
    destination: Path,
    *,
    row_ids: Sequence[int],
    source_vocab_size: int,
    drop_vision: bool = False,
) -> dict[str, Any]:
    """Stream shards, compact tied vocabulary rows, and optionally drop vision."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index = load_json(source / INDEX_NAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{source / INDEX_NAME} has no weight_map")
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        keys_by_shard[str(shard_name)].append(str(key))
    selected = torch.tensor(list(row_ids), dtype=torch.long)
    output_weight_map: dict[str, str] = {}
    source_bytes = int(index.get("metadata", {}).get("total_size", 0))
    saved_bytes = 0
    dropped_tensor_bytes = 0
    dropped_tensor_count = 0
    sliced: dict[str, dict[str, Any]] = {}
    linked_shards = 0
    copied_shards = 0
    rewritten_shards = 0

    for shard_name in sorted(keys_by_shard):
        shard_keys = keys_by_shard[shard_name]
        dropped_keys = {
            key
            for key in shard_keys
            if drop_vision and key.startswith(VISION_WEIGHT_PREFIXES)
        }
        if not any(key in VOCABULARY_WEIGHT_KEYS for key in shard_keys) and not dropped_keys:
            source_shard = source / shard_name
            destination_shard = destination / shard_name
            try:
                os.link(source_shard, destination_shard)
                linked_shards += 1
            except OSError:
                shutil.copy2(source_shard, destination_shard)
                copied_shards += 1
            for key in shard_keys:
                output_weight_map[key] = shard_name
            continue

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensor = handle.get_tensor(key)
                tensor_bytes = int(tensor.numel() * tensor.element_size())
                if key in dropped_keys:
                    saved_bytes += tensor_bytes
                    dropped_tensor_bytes += tensor_bytes
                    dropped_tensor_count += 1
                    continue
                if key in VOCABULARY_WEIGHT_KEYS:
                    if tensor.ndim != 2 or tensor.shape[0] != source_vocab_size:
                        raise ValueError(
                            f"Vocabulary tensor {key} has shape {tuple(tensor.shape)}, "
                            f"expected [{source_vocab_size}, hidden]"
                        )
                    tensor = torch.index_select(tensor, 0, selected)
                    saved_bytes += tensor_bytes - int(
                        tensor.numel() * tensor.element_size()
                    )
                    sliced[key] = {
                        "source_shape": [source_vocab_size, int(tensor.shape[1])],
                        "output_shape": list(tensor.shape),
                    }
                tensors[key] = tensor.contiguous()
                output_weight_map[key] = shard_name
        if tensors:
            save_file(tensors, destination / shard_name, metadata={"format": "pt"})
        rewritten_shards += 1
        del tensors
        gc.collect()

    if not sliced:
        raise RuntimeError("No stored DiffusionGemma vocabulary tensor was found")
    if source_bytes <= 0:
        # Synthetic or nonstandard indices should still be rejected rather
        # than publishing an unverifiable byte-savings claim.
        raise ValueError("Source safetensors index must declare metadata.total_size")
    output_bytes = source_bytes - saved_bytes
    converted_index = {
        "metadata": {**dict(index.get("metadata", {})), "total_size": output_bytes},
        "weight_map": output_weight_map,
    }
    (destination / INDEX_NAME).write_text(
        json.dumps(converted_index, indent=2, sort_keys=True) + "\n"
    )
    return {
        "source_tensor_bytes": source_bytes,
        "output_tensor_bytes": output_bytes,
        "saved_tensor_bytes": saved_bytes,
        "source_shards": len(keys_by_shard),
        "hardlinked_unchanged_shards": linked_shards,
        "copied_unchanged_shards": copied_shards,
        "rewritten_shards": rewritten_shards,
        "output_tensor_count": len(output_weight_map),
        "sliced_vocabulary_tensors": sliced,
        "dropped_vision_tensor_count": dropped_tensor_count,
        "dropped_vision_tensor_bytes": dropped_tensor_bytes,
        "dropped_vision_prefixes": list(VISION_WEIGHT_PREFIXES) if drop_vision else [],
    }


def convert_model(
    source: Path,
    output: Path,
    *,
    tokenizer_assets: Path = DEFAULT_TOKENIZER_ASSETS,
    overwrite: bool = False,
    source_row_ids: Sequence[int] | None = None,
    instruction_text: str | None = None,
    drop_vision: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    tokenizer_assets = tokenizer_assets.resolve()
    if source == output:
        raise ValueError("Source and output must differ")
    for required in ("config.json", "tokenizer.json", INDEX_NAME):
        if not (source / required).is_file():
            raise FileNotFoundError(f"Source model is missing {required}: {source}")
    source_config = load_json(source / "config.json")
    instruction_text = str(instruction_text or "")
    if source_row_ids is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        base_row_ids = resolve_arc_source_row_ids(tokenizer)
        instruction = resolve_instruction_contract(
            tokenizer,
            instruction_text,
            base_row_ids=base_row_ids,
        )
        row_ids = base_row_ids + instruction["added_source_row_ids"]
    else:
        if instruction_text:
            raise ValueError(
                "instruction_text cannot be combined with explicit source_row_ids"
            )
        row_ids = [int(value) for value in source_row_ids]
        if len(row_ids) != len(ARC_TOKENS) or len(set(row_ids)) != len(ARC_TOKENS):
            raise ValueError(
                f"source_row_ids must contain {len(ARC_TOKENS)} unique values"
            )
        instruction = resolve_instruction_contract(
            tokenizer=None,
            instruction_text="",
            base_row_ids=row_ids,
        )
    compact_tokens = ARC_TOKENS + tuple(instruction["added_compact_tokens"])
    compact_config = build_compact_config(
        source_config,
        vocab_size=len(compact_tokens),
        drop_vision=drop_vision,
    )
    compact_config.update(
        {
            "arc_output_vocab_size": ARC_OUTPUT_VOCAB_SIZE,
            "arc_instruction_token_ids": instruction["compact_token_ids"],
            "arc_instruction_text": instruction_text,
        }
    )
    source_vocab_size = int(source_config["text_config"]["vocab_size"])
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"Compact source-row mapping is not one-to-one: {row_ids}")
    if min(row_ids) < 0 or max(row_ids) >= source_vocab_size:
        raise ValueError(
            f"Source row mapping {row_ids} exceeds vocabulary {source_vocab_size}"
        )

    staging = prepare_output_directory(output, overwrite=overwrite)
    try:
        (staging / "config.json").write_text(
            json.dumps(compact_config, indent=2, sort_keys=True) + "\n"
        )
        copy_compact_assets(
            source,
            tokenizer_assets,
            staging,
            compact_tokens=compact_tokens,
            instruction_text=instruction_text,
            retain_vision_assets=not drop_vision,
        )
        generation_path = source / "generation_config.json"
        if generation_path.is_file():
            generation = load_json(generation_path)
            generation.update(
                {"pad_token_id": 19, "bos_token_id": 13, "eos_token_id": 19}
            )
            (staging / "generation_config.json").write_text(
                json.dumps(generation, indent=2, sort_keys=True) + "\n"
            )
        stats = convert_safetensors(
            source,
            staging,
            row_ids=row_ids,
            source_vocab_size=source_vocab_size,
            drop_vision=drop_vision,
        )
        manifest = {
            "format": "arc-diffusion-gemma-native-thought-compact-v5",
            "source": str(source),
            "architecture": compact_config.get("architectures"),
            "arc_tokens": list(ARC_TOKENS),
            "compact_tokens": list(compact_tokens),
            "source_row_ids": row_ids,
            "source_vocab_size": source_vocab_size,
            "output_vocab_size": len(compact_tokens),
            "arc_output_vocab_size": ARC_OUTPUT_VOCAB_SIZE,
            "instruction": instruction,
            "vision_weights_retained": not drop_vision,
            "text_only": bool(drop_vision),
            "image_token_ids_disabled": True,
            **stats,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest["config_sha256"] = _sha256(staging / "config.json")
        manifest["tokenizer_sha256"] = _sha256(staging / "tokenizer.json")
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        finalize_output_directory(staging, output, overwrite=overwrite)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-assets", default=str(DEFAULT_TOKENIZER_ASSETS))
    parser.add_argument(
        "--instruction-preset",
        choices=("none", "infer-grid", "sudoku"),
        default="none",
        help="retain and inject a fixed natural-language ARC instruction",
    )
    parser.add_argument(
        "--instruction-text",
        default=None,
        help="custom fixed instruction; overrides --instruction-preset",
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--drop-vision",
        action="store_true",
        help=(
            "publish a text-only artifact: remove the vision tower/projector "
            "tensors, set vision_config=null, and omit processor image assets"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    instruction_text = args.instruction_text
    if instruction_text is None and args.instruction_preset == "infer-grid":
        instruction_text = DEFAULT_ARC_INSTRUCTION
    elif instruction_text is None and args.instruction_preset == "sudoku":
        instruction_text = DEFAULT_SUDOKU_INSTRUCTION
    manifest = convert_model(
        Path(args.source),
        Path(args.output),
        tokenizer_assets=Path(args.tokenizer_assets),
        overwrite=args.overwrite_output_dir,
        instruction_text=instruction_text,
        drop_vision=args.drop_vision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
