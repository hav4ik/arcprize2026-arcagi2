#!/usr/bin/env python3
"""Build an ARC-targeted REAM calibration artifact from an NVARC dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arc-dataset", required=True)
    parser.add_argument("--ream-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--validation-rows", type=int, default=16)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--canvas-length", type=int, default=128)
    parser.add_argument("--fill-mode", choices=("eos", "repeat"), default="repeat")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _extract_rows(dataset, indices: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in indices:
        messages = dataset[int(index)]["messages"]
        users = [item["content"] for item in messages if item.get("role") == "user"]
        assistants = [
            item["content"] for item in messages if item.get("role") == "assistant"
        ]
        if not users or not assistants:
            continue
        rows.append(
            {
                "category": "arc",
                "prompt": str(users[-1]),
                "response": str(assistants[-1]),
            }
        )
    return rows


def build_calibration(args: argparse.Namespace) -> dict:
    from datasets import load_from_disk
    from transformers import AutoProcessor

    ream_root = Path(args.ream_root).expanduser().resolve()
    if not (ream_root / "ream" / "diffusion_gemma.py").is_file():
        raise FileNotFoundError(f"invalid --ream-root: {ream_root}")
    sys.path.insert(0, str(ream_root))
    from data.diffusion_gemma_calibration import (  # noqa: PLC0415
        SCHEMA_VERSION,
        _encode_split,
        _tokenizer_from_processor,
    )

    dataset_path = Path(args.arc_dataset).expanduser().resolve()
    dataset = load_from_disk(str(dataset_path))
    required = int(args.train_rows) + int(args.validation_rows)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    selected_indices: list[int] = []
    rows: list[dict[str, str]] = []
    for index in indices:
        extracted = _extract_rows(dataset, [index])
        if not extracted:
            continue
        selected_indices.append(int(index))
        rows.extend(extracted)
        if len(rows) == required:
            break
    if len(rows) != required:
        raise ValueError(f"ARC dataset yielded {len(rows)} rows; need {required}")

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = _tokenizer_from_processor(processor)
    train_rows = rows[: args.train_rows]
    validation_rows = rows[args.train_rows :]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "model": str(Path(args.model).expanduser().resolve()),
            "arc_dataset": str(dataset_path),
            "prompt_length": int(args.prompt_length),
            "canvas_length": int(args.canvas_length),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "categories": ["arc"],
            "noise_levels": [0.25, 0.5, 0.75, 1.0],
            "self_conditioning_fraction": 0.5,
            "seed": int(args.seed),
            "fill_mode": args.fill_mode,
            "eos_fill_is_context": args.fill_mode == "eos",
            "selected_indices_sha256": hashlib.sha256(
                json.dumps(selected_indices, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "train": _encode_split(
            train_rows,
            tokenizer,
            args.prompt_length,
            args.canvas_length,
            args.seed + 1,
            fill_mode=args.fill_mode,
        ),
        "validation": _encode_split(
            validation_rows,
            tokenizer,
            args.prompt_length,
            args.canvas_length,
            args.seed + 2,
            fill_mode=args.fill_mode,
        ),
    }
    return artifact


def main() -> None:
    args = parse_args()
    artifact = build_calibration(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "train_rows": artifact["metadata"]["train_rows"],
                "validation_rows": artifact["metadata"]["validation_rows"],
                "categories": artifact["metadata"]["categories"],
                "selected_indices_sha256": artifact["metadata"][
                    "selected_indices_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
