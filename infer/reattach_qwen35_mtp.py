#!/usr/bin/env python3
"""Copy an existing cut text model unchanged and reattach original native MTP.

CPU only; streams backbone shards rather than loading the 9B model into RAM.
No vision tensors, vocabulary expansion, optimizer or training-state changes.
The destination must not exist. Interrupted output stays in a marked staging
directory, never masquerading as a complete model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import os

from pre_sft_native_mtp import hf_tensor_inventory, load_native_state, native_mtp_config, validate_native_inventory


def reattach(cut_model, original_model, output, *, loss_weight=0.3, num_predictions=2):
    from safetensors.torch import save_file
    contract = native_mtp_config(loss_weight, num_predictions=num_predictions)
    cut, original, output = Path(cut_model).resolve(), Path(original_model).resolve(), Path(output).resolve()
    if output.exists() or output in (cut, original) or str(output).startswith('/kaggle/input/'):
        raise ValueError("Choose a new writable output directory; source models are never overwritten")
    config = json.loads((cut / "config.json").read_text())
    origin_config = json.loads((original / "config.json").read_text())
    text = origin_config.get("text_config", origin_config)
    if config.get("model_type") != "qwen3_5_text" or config.get("arc_mtp"):
        raise ValueError("Expected an existing non-MTP text-only cut Qwen3.5 model")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rms_norm_eps", "layer_types", "rope_parameters"):
        if config.get(key) != text.get(key):
            raise ValueError(f"Original/cut architecture mismatch: {key}")
    inventory = hf_tensor_inventory(cut)
    if any(k.startswith("mtp.") or "visual." in k for k in inventory):
        raise ValueError("Cut model contains unexpected MTP/vision weights")
    mtp_inventory = validate_native_inventory(original)
    mtp_bytes = sum((entry["data_offsets"][1] - entry["data_offsets"][0]) for entry in mtp_inventory.values())
    shards = sorted({v["file"] for v in inventory.values()})
    needed = sum((cut / s).stat().st_size for s in shards) + mtp_bytes + 16 * 1024**2
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < needed:
        raise OSError(f"Need {needed / 1024**3:.2f} GiB free at {output.parent}")
    staging = output.with_name(output.name + f".staging-{os.getpid()}")
    staging.mkdir(exist_ok=False)
    (staging / "arc_mtp_reattach_incomplete.json").write_text(json.dumps({"output": str(output)}))
    hashes = {}
    for name in shards:
        digest = hashlib.sha256()
        with (cut / name).open("rb") as source, (staging / name).open("wb") as dest:
            while block := source.read(8 * 1024**2):
                dest.write(block)
                digest.update(block)
        hashes[name] = digest.hexdigest()
        print(f"Copied unchanged backbone shard: {name}", flush=True)
    # Copy tokenizer/configuration assets, not optimizer.pt or unrelated files.
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                 "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json", "LICENSE",
                 "arc_qwen35_text_conversion.json"):
        if (cut / name).is_file():
            shutil.copyfile(cut / name, staging / name)
    state = load_native_state(original)
    mtp_file = "model-native-mtp.safetensors"
    save_file({"mtp." + k: v for k, v in state.items()}, str(staging / mtp_file), metadata={"format": "pt"})
    mapping = {k: entry["file"] for k, entry in inventory.items()}
    mapping.update({"mtp." + k: mtp_file for k in state})
    total = sum(entry["data_offsets"][1] - entry["data_offsets"][0] for entry in inventory.values()) + mtp_bytes
    (staging / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": mapping}, indent=2) + "\n")
    config.update(arc_mtp=contract, mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (staging / "pre_sft_mtp.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = {"version": 1, "original_mtp_source": str(original), "cut_backbone_source": str(cut),
                "backbone_shard_sha256": hashes, "native_mtp_tensors": len(state), "native_mtp_bytes": mtp_bytes,
                "backbone_and_tokenizer_unchanged": True, "vision_included": False,
                "original_conversion_manifest_is_backbone_provenance_only": True}
    (staging / "arc_native_mtp_reattach.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_native_inventory(staging)
    (staging / "arc_mtp_reattach_incomplete.json").unlink()
    staging.rename(output)
    print(f"Native MTP ready: {output}; {len(state)} tensors, {mtp_bytes / 1024**2:.2f} MiB", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-model", required=True)
    parser.add_argument("--original-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mtp-loss-weight", type=float, default=0.3)
    parser.add_argument("--mtp-num-predictions", type=int, default=2,
                        help="Training horizon including the main head; reuses the same one-layer native MTP")
    args = parser.parse_args()
    reattach(args.cut_model, args.original_model, args.output, loss_weight=args.mtp_loss_weight,
             num_predictions=args.mtp_num_predictions)
