"""Prepare one shared BF16 PiSSA base before launching ARC workers."""

import argparse
import json

from arc_pissa_shared import prepare_shared_base
from run_config import normalize_lora_init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-method", choices=("lora", "rslora"), default="rslora")
    parser.add_argument("--lora-init", type=normalize_lora_init, default="pissa_niter_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = prepare_shared_base(
        args.model_path, args.output_dir, rank=args.lora_r, alpha=args.lora_alpha,
        method=args.lora_method, initialization=args.lora_init, seed=args.seed,
        device=args.device,
    )
    print(json.dumps({
        "status": "complete", "base_id": result["base_id"],
        "output_dir": args.output_dir, "rank": result["settings"]["rank"],
        "artifact_gib": sum(f["bytes"] for f in result["files"].values()) / 1024**3,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
