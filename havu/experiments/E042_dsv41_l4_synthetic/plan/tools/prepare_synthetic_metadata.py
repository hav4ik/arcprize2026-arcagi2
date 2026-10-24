#!/usr/bin/env python3
"""Prepare explicitly synthetic E64/E72 metadata; never fetch model weights.

Default: entirely offline, using the bundled, checksum-verified source config.
Optional --download-metadata: fetch only four fixed config/tokenizer/encoding
assets from the pinned model revision. This does not initialize GPU weights or
implement the custom vLLM loader/compact Engram adapter described in the plan.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
ALLOWED_DOWNLOADS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "encoding/README.md",
    "encoding/encoding.py",
)
PER_FILE_CAP = 128 * 1024**2
TOTAL_DOWNLOAD_CAP = 256 * 1024**2
CHUNK_BYTES = 1024**2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source(raw):
    if sha256(raw) != CONFIG_SHA256:
        raise ValueError("Source config differs from the pinned, audited revision.")
    config = json.loads(raw)
    text = config["text_config"]
    expected = {
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
    }
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"Unexpected source field: text_config.{key}")
    return config


def derive_config(source, experts):
    if experts not in (64, 72):
        raise ValueError("Only the plan's E64 and E72 fixtures are supported.")
    result = copy.deepcopy(source)
    result["text_config"]["n_routed_experts"] = experts
    restored = copy.deepcopy(result)
    restored["text_config"]["n_routed_experts"] = 384
    if restored != source:
        raise ValueError("Synthetic derivation changed more than routed-expert count.")
    return result


def download_asset(name, destination, remaining_bytes):
    # This check is deliberately inside the download function. No arbitrary URL,
    # filename, model ID, revision, weight index or checkpoint suffix is accepted.
    if name not in ALLOWED_DOWNLOADS:
        raise ValueError(f"Not an allowlisted metadata asset: {name!r}")
    cap = min(PER_FILE_CAP, remaining_bytes)
    if cap <= 0:
        raise ValueError("Metadata download budget exhausted.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        BASE_URL + name, headers={"User-Agent": "deepseek-l4-synthetic-metadata/1"}
    )
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > cap:
            raise ValueError(f"Metadata asset exceeds byte cap: {name}")
        with destination.open("xb") as output:
            while True:
                chunk = response.read(min(CHUNK_BYTES, cap - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                if count > cap:
                    raise ValueError(f"Metadata stream exceeds byte cap: {name}")
                output.write(chunk)
                digest.update(chunk)
    return {"path": name, "bytes": count, "sha256": digest.hexdigest(), "url": BASE_URL + name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are refused")
    parser.add_argument("--experts", type=int, choices=(64, 72), nargs="+", default=[64, 72])
    parser.add_argument(
        "--source-config", type=Path,
        default=Path(__file__).resolve().parent.parent / "metadata" / "source_config.json",
    )
    parser.add_argument("--download-metadata", action="store_true", help="download the fixed tokenizer/encoding allowlist only")
    args = parser.parse_args()

    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output already exists; choose a new directory.")
    source_raw = args.source_config.read_bytes()
    source = validate_source(source_raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="l4-metadata-", dir=output.parent) as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        shared = staged / "shared_metadata"
        shared.mkdir()
        (shared / "config.json").write_bytes(source_raw)
        license_path = Path(__file__).resolve().parent.parent / "metadata" / "SOURCE_MODEL_LICENSE"
        if license_path.is_file():
            shutil.copyfile(license_path, shared / "LICENSE")
        downloads = []
        if args.download_metadata:
            total = 0
            for name in ALLOWED_DOWNLOADS:
                record = download_asset(name, shared / name, TOTAL_DOWNLOAD_CAP - total)
                downloads.append(record)
                total += record["bytes"]
        fixtures = []
        for experts in dict.fromkeys(args.experts):
            directory = staged / f"synthetic-e{experts}"
            directory.mkdir()
            write_json(directory / "config.json", derive_config(source, experts))
            manifest = {
                "artifact_kind": "synthetic_metadata_only",
                "is_pruned_checkpoint": False,
                "contains_weights": False,
                "requires_custom_synthetic_loader": True,
                "custom_loader_implementation": None,
                "source_model_id": MODEL_ID,
                "source_revision": REVISION,
                "source_config_sha256": CONFIG_SHA256,
                "synthetic_config_sha256": sha256((directory / "config.json").read_bytes()),
                "changed_fields": {"text_config.n_routed_experts": {"from": 384, "to": experts}},
                "tensor_parallel_size": 4,
                "language_model_only": True,
                "speculative_decoding_enabled": False,
                "engram_constructor_hook_required": True,
                "original_engram_hash_geometry_preserved": True,
                "tokenizer_path_relative_to_fixture": "../shared_metadata" if args.download_metadata else None,
            }
            write_json(directory / "synthetic_manifest.json", manifest)
            (directory / "README.txt").write_text(
                "SYNTHETIC METADATA ONLY. There are no checkpoint weights here.\n"
                "The config changes only the backbone expert count. The custom\n"
                "format-aware initializer and pre-allocation compact Engram hook\n"
                "in AGENT_WORK_ORDERS.md are required before model construction.\n"
                "Stock --load-format dummy is not a validated path for this fixture.\n",
                encoding="utf-8",
            )
            fixtures.append({"directory": directory.name, "experts": experts, "config_sha256": manifest["synthetic_config_sha256"]})
        report = {
            "source_model": MODEL_ID,
            "source_revision": REVISION,
            "source_config_sha256": CONFIG_SHA256,
            "weights_downloaded": False,
            "network_requested": args.download_metadata,
            "downloaded_bytes": sum(item["bytes"] for item in downloads),
            "download_cap_bytes": TOTAL_DOWNLOAD_CAP,
            "downloads": downloads,
            "fixtures": fixtures,
        }
        write_json(staged / "metadata_manifest.json", report)
        # A complete fixture appears at the requested path only on success.
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staged.rename(output)
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
