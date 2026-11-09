"""Reproducible real-GPU split-runtime smoke/accuracy gate (no package installs)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


TASKS = "0934a4d8,36a08778,981571dc,aa4ec2a5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction", choices=("new-stable", "stable-new"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="/kaggle/input/path-to-model")
    parser.add_argument(
        "--stable-python", default="/kaggle/working/stable_venv/bin/python"
    )
    parser.add_argument("--stable-site-root", default="/kaggle/working/stable_env")
    parser.add_argument("--new-python", default="/usr/bin/python3")
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--search-seconds", type=float, default=30)
    parser.add_argument("--task-seconds", type=float, default=180)
    parser.add_argument("--run-seconds", type=float, default=2400)
    parser.add_argument("--max-loras", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    app = Path(__file__).resolve().parent
    artifacts = Path(tempfile.mkdtemp(prefix=f"arc-{args.direction}-adapters-"))
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "UNSLOTH_"))
    }
    overrides = {
        "ARC_TASK_KEY": TASKS,
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "VLLM_CONFIGURE_LOGGING": "1",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "OMP_NUM_THREADS": "16",
        "ARC_AUTO_ACTIVATION_RECOMPUTE": "0",
        "ARC_OPTIM": "adamw_torch",
        "ARC_TRAIN_EPOCHS": "1",
        "ARC_SMALL_TASK_MAX_AVERAGE_GRID_AREA": "100",
        "ARC_SMALL_TASK_TRAIN_AUGMENTATIONS": str(args.augmentations),
        "ARC_TRAIN_AUGMENTATIONS": str(args.augmentations),
        "ARC_GRADIENT_CHECKPOINTING_INTERVAL": "1",
        "ARC_GRADIENT_CHECKPOINTING_PERCENT": "auto",
        "ARC_SELECTION_ALGORITHM": "kgmon",
        "ARC_ZERO_CANDIDATE_SHAPE_FALLBACK": "0",
        "ARC_SHAPE_FALLBACK_BEAM_WIDTH": "2",
        "ARC_TRAIN_SAMPLE_PRINT": "0",
        "ARC_TRAIN_TQDM": "1",
        "ARC_REUSE_BASE_MODEL": "1",
        "ARC_MODEL_PATH": args.model_path,
        "ARC_CHALLENGE_PATH": "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json",
        "ARC_VLLM_GPU_MEMORY_UTILIZATION": "0.89",
        "ARC_VLLM_LEAF_ROLLOUT": "1",
        "ARC_UNSLOTH_TEMP_DIR": f"/dev/shm/arc_split_validation_{os.getpid()}",
        "UNSLOTH_LORA_MLP_CACHE_XA": "0",
        "UNSLOTH_FLASH_ATTENTION_3": "0",
        "UNSLOTH_QWEN3_MLP_REPLAY_BYPASS": "1",
        "ARC_INFERENCE_OUTPUT_DIR": str(output / "inference_outputs"),
        "ARC_RUN_STATUS_PATH": str(output / "run_status.json"),
    }
    env.update(overrides)
    roles = (
        [
            "--trainer-python",
            args.new_python,
            "--vllm-python",
            args.stable_python,
            "--vllm-site-root",
            args.stable_site_root,
        ]
        if args.direction == "new-stable"
        else [
            "--trainer-python",
            args.stable_python,
            "--trainer-site-root",
            args.stable_site_root,
            "--vllm-python",
            args.new_python,
        ]
    )
    cmd = [
        sys.executable,
        str(app / "starter.py"),
        *roles,
        "--runtime-report-dir",
        str(output / "runtimes"),
        "--inference-backend",
        "vllm",
        "--multi-task-gpus",
        "2",
        "--vllm-gpus",
        "2",
        "--vllm-scale-after-training",
        "--inference-adapter-storage",
        "auto",
        "--vllm-max-loras",
        str(args.max_loras),
        "--vllm-auto-recover",
        "--vllm-max-adapter-recoveries",
        "3",
        "--no-vllm-enforce-eager",
        "--vllm-leaf-rollout",
        "--no-vllm-batch-invariant",
        "--vllm-async-scheduling",
        "--vllm-kv-resident-fraction",
        "0.6",
        "--kv-offloading-size",
        "0",
        "--vllm-rollout-max-tokens",
        "32",
        "--vllm-prefix-score",
        "--inference-batch-size",
        "24",
        "--vllm-max-num-seqs",
        "24",
        "--no-vllm-cumulative-nll-stop",
        "--vllm-rollout-branch-window",
        "1",
        "--lora-r",
        "128",
        "--gradient-checkpointing-mode",
        "unsloth",
        "--max-seq-length",
        "16384",
        "--inference-timeout-seconds",
        str(args.task_seconds),
        "--vllm-search-limit-seconds",
        str(args.search_seconds),
        "--rescore-views",
        "8",
        "--rescore-schedule",
        "prefix",
        "--vllm-prefix-score-tile-views",
        "4",
        "--score-batch-size",
        "16",
        "--vllm-max-pending-adapters",
        "64",
        "--dfs-mode",
        "auto",
        "--vllm-preserve-adapter-root",
        str(artifacts),
        "--end-time",
        str(time.time() + args.run_seconds),
    ]
    result = {
        "direction": args.direction,
        "command": cmd,
        "environment": overrides,
        "artifacts": str(artifacts),
        "augmentations": args.augmentations,
    }
    (output / "command.json").write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    with (output / "train_infer.log").open("w") as log:
        process = subprocess.run(
            cmd, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT
        )
    result.update(
        solver_exit=process.returncode, solver_seconds=time.monotonic() - started
    )
    if process.returncode == 0:
        with (output / "submit.log").open("w") as log:
            submit = subprocess.run(
                [sys.executable, str(app / "submit.py")],
                env=env,
                cwd=output,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result["submit_exit"] = submit.returncode
        if submit.returncode == 0:
            submission = json.loads((output / "submission.json").read_text())
            solutions = json.loads(
                Path(
                    overrides["ARC_CHALLENGE_PATH"].replace(
                        "challenges.json", "solutions.json"
                    )
                ).read_text()
            )
            result["correct_outputs"] = {
                k: [
                    any(row[name] == truth for name in ("attempt_1", "attempt_2"))
                    for row, truth in zip(submission[k], solutions[k], strict=True)
                ]
                for k in TASKS.split(",")
            }
            result["task_weighted_score"] = sum(
                sum(v) / len(v) for v in result["correct_outputs"].values()
            )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result.get("submit_exit", process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
