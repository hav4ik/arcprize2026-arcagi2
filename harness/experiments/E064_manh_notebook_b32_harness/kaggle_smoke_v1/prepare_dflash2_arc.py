"""Adapt a full-vocabulary Speculators DFlash2 draft to the ARC target vocabulary.

CPU-only, local files only. Never modifies the target or source checkpoint.
The draft keeps its pretrained body/RoPE; verifier-owned projections are loaded
from the actual ARC target by Speculators, not sliced from an unrelated base.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


CONVERSION_FILE = "arc_dflash2_conversion.json"
CODEBOOK_KEYS = (
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
)
VERIFIER_KEYS = {
    "embed_tokens.weight",
    "lm_head.weight",
    "verifier_lm_head.weight",
    "verifier_norm.weight",
    "t2d",
    "d2t",
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "chat_template.jinja",
    "chat_template.j2",
)


def copy_arc_tokenizer(source, destination):
    """Preserve ARC token IDs and its WordLevel backend through HF loading."""
    source, destination = Path(source), Path(destination)
    for name in TOKENIZER_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    # Qwen2Tokenizer can rebuild BPE instead of loading the cut WordLevel
    # backend, silently losing user/assistant tokens without BPE merges.
    path = destination / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    path.write_text(json.dumps(config, indent=2) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_target(path):
    """Fail closed on trained architecture sidecars that vanilla HF would ignore."""
    from pre_sft_unsloth_saved_data import _arc_tokenizer_backend

    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("vocab_size") != 16:
        raise ValueError(
            "This entry point requires a plain Qwen3 target with the 16-token ARC vocabulary"
        )
    for name in (
        "delta_mhar_config.json",
        "delta_mhar.safetensors",
        "arc_rope_3d.json",
        "arc_bidirectional_prompt_attention.json",
    ):
        if (path / name).exists():
            raise ValueError(
                f"Custom target architecture is not supported here: {name}; use the original cut Qwen3"
            )
    if any(
        config.get(k)
        for k in ("arc_rope_3d", "delta_mhar", "bidirectional_prompt_attention")
    ):
        raise ValueError(
            "Custom target architecture metadata cannot be ignored by the frozen HF target"
        )
    _arc_tokenizer_backend(path)
    return config


def token_mapping(target_path, source_tokenizer_path):
    """new ARC ID -> old Qwen ID, matched by token strings, never assumed IDs."""
    from tokenizers import Tokenizer

    target = Tokenizer.from_file(str(Path(target_path) / "tokenizer.json"))
    source_path = Path(source_tokenizer_path)
    if source_path.is_dir():
        source_path = source_path / "tokenizer.json"
    source = Tokenizer.from_file(str(source_path))
    vocab = target.get_vocab(with_added_tokens=True)
    if sorted(vocab.values()) != list(range(16)):
        raise ValueError(
            "ARC tokenizer must have exactly 16 contiguous IDs including special tokens"
        )
    by_id = sorted(vocab.items(), key=lambda item: item[1])
    old_ids = [source.token_to_id(token) for token, _ in by_id]
    if any(i is None for i in old_ids) or len(set(old_ids)) != 16:
        raise ValueError(
            "Every ARC token must map uniquely into the original Qwen tokenizer"
        )
    return old_ids, [token for token, _ in by_id], source_path


def safetensor_inventory(path):
    """Use explicit model shard indexes; do not accidentally read sidecar weights."""
    from safetensors import safe_open

    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted(set(mapping.values()))
    else:
        mapping = None
        files = ["model.safetensors"]
    result = {}
    for name in files:
        file = (path / name).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError("Model index shard path escapes checkpoint directory")
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in result:
                    raise ValueError(f"Duplicate tensor: {key}")
                result[key] = file
    if mapping is not None and set(mapping) != set(result):
        raise ValueError("Model shard inventory does not match index")
    return result


def prepare_draft(source, target, source_tokenizer, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, target, output = map(lambda p: Path(p).resolve(), (source, target, output))
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}; choose a new prepared-draft directory"
        )
    target_config = validate_target(target)
    config = json.loads((source / "config.json").read_text())
    if config.get("speculators_model_type") != "dflash2":
        raise ValueError(
            "Expected a Speculators DFlash2 checkpoint, such as AQ-MedAI/Qwen3-4B-DFlash2"
        )
    original_config = copy.deepcopy(config)
    layer_config = config["transformer_layer_config"]
    old_vocab = layer_config["vocab_size"]
    if config["draft_vocab_size"] != old_vocab:
        raise ValueError("Source DFlash2 must use its full verifier vocabulary")
    if layer_config["hidden_size"] != target_config["hidden_size"]:
        raise ValueError("Source draft and ARC target hidden sizes differ")
    layer_ids = config["aux_hidden_state_layer_ids"]
    if (
        not layer_ids
        or len(set(layer_ids)) != len(layer_ids)
        or any(
            not isinstance(i, int) or not 0 <= i <= target_config["num_hidden_layers"]
            for i in layer_ids
        )
    ):
        raise ValueError("Invalid source auxiliary layer IDs for this target")
    ids, tokens, tokenizer_file = token_mapping(target, source_tokenizer)
    if max(ids) >= old_vocab:
        raise ValueError("Source tokenizer IDs exceed draft vocabulary")
    # <|endoftext|> is unused by compact NVARC messages. No extra target row is
    # added and no answer/grid/EOS token is repurposed for masking.
    mask_id = tokens.index("<|endoftext|>")
    config.update(
        draft_vocab_size=16,
        mask_token_id=mask_id,
        selector_top_k=min(config["selector_top_k"], 16),
    )
    config["speculators_config"]["verifier"] = {
        "name_or_path": str(target),
        "architectures": target_config["architectures"],
    }
    layer_config.update(
        vocab_size=16,
        bos_token_id=target_config.get("bos_token_id"),
        eos_token_id=target_config.get("eos_token_id"),
        pad_token_id=None,
    )
    config.pop("auto_map", None)  # Use installed Speculators; do not execute Hub code.
    inventory = safetensor_inventory(source)
    if not set(CODEBOOK_KEYS).issubset(inventory):
        raise ValueError("Source checkpoint is missing DFlash2 selector codebooks")
    tensors = {}
    row_ids = torch.tensor(ids, dtype=torch.long)
    for file in sorted(set(inventory.values())):
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in VERIFIER_KEYS:
                    continue  # All frozen projections must come from this target.
                tensor = f.get_tensor(key)
                if key in CODEBOOK_KEYS:
                    if tuple(tensor.shape) != (old_vocab, config["selector_rank"]):
                        raise ValueError(
                            f"Unexpected source codebook shape: {key} {tensor.shape}"
                        )
                    tensor = tensor.index_select(0, row_ids)
                tensors[key] = tensor.contiguous()
    expected_fc = (
        layer_config["hidden_size"],
        len(layer_ids) * target_config["hidden_size"],
    )
    if tuple(tensors["fc.weight"].shape) != expected_fc:
        raise ValueError(
            "Draft fc.weight does not match the auxiliary hidden-state layout"
        )
    manifest = {
        "version": 1,
        "source_model": str(source),
        "target_model": str(target),
        "source_config": original_config,
        "source_tokenizer_sha256": sha256_file(tokenizer_file),
        "target_config_sha256": sha256_file(target / "config.json"),
        "target_tokenizer_sha256": sha256_file(target / "tokenizer.json"),
        "output_tokenizer_class": "PreTrainedTokenizerFast",
        "source_weight_sha256": {
            f.name: sha256_file(f) for f in sorted(set(inventory.values()))
        },
        "tokens": tokens,
        "arc_to_original_ids": ids,
        "mask_token_id": mask_id,
        "mask_embedding_changed": True,
        "verifier_weights": "borrow from target at load",
        "draft_rope": "preserved from pretrained draft; independent of target RoPE",
        "num_parameters_saved": sum(t.numel() for t in tensors.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        save_file(tensors, staging / "model.safetensors", metadata={"format": "pt"})
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / CONVERSION_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
        copy_arc_tokenizer(target, staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(
        f"[DFlash2 ARC] prepared {output}: vocab=16, mask={mask_id}, "
        f"block={config['block_size']}, layers={layer_ids}, "
        f"weights={(output / 'model.safetensors').stat().st_size / 1024**3:.3f} GiB",
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument(
        "--source-tokenizer-path",
        required=True,
        help="Original uncut Qwen3 tokenizer.json or its local directory",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prepare_draft(
        args.draft_model_path,
        args.target_model_path,
        args.source_tokenizer_path,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
