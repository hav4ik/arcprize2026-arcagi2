"""XManager launcher for Qwen3.8-27B Full-Parameter FSDP2 SFT on 8x B200 GPUs."""

from __future__ import annotations

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "qwen38_27b_arc_agi2_e031_full_sft_8xb200",
    "XManager experiment name.",
)
flags.DEFINE_integer("num_b200", 8, "Number of B200 GPUs per work unit.")
flags.DEFINE_integer("cpu_cores", 64, "CPU cores per work unit.")
flags.DEFINE_integer("ram_gb", 768, "System RAM (GB) per work unit.")
flags.DEFINE_string(
    "location",
    "viglobal",
    "Borg cell/location constraint (viglobal routes to any cell with 8x B200 capacity in the alloc).",
)
flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    "Path to base Qwen3.8-27B BF16 model on CNS.",
)
flags.DEFINE_string(
    "data_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/e031_pilot",
    "CNS directory containing train_tokens.npz and val_tokens.npz.",
)
flags.DEFINE_string(
    "output_cns_base",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_full_sft_v2",
    "Separate CNS base directory where XID-keyed Full-SFT checkpoints and logs will be saved.",
)
flags.DEFINE_float("learning_rate", 5e-6, "Peak learning rate.")
flags.DEFINE_float("min_lr_ratio", 0.1, "Min learning rate ratio at end of cosine decay (5e-7 / 5e-6 = 0.1).")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_float("warmup_ratio", 0.05, "Warmup ratio (0.05 = 5% of total steps).")
flags.DEFINE_float("max_grad_norm", 1.0, "Max gradient L2 norm for clipping.")
flags.DEFINE_integer("num_epochs", 1, "Number of epochs.")
flags.DEFINE_integer("max_steps", -1, "Optional max optimizer steps override (-1 = full num_epochs).")
flags.DEFINE_integer("micro_batch_size", 1, "Per-rank microbatch size.")
flags.DEFINE_integer("grad_accum_steps", 1, "Gradient accumulation micro-steps per rank.")
flags.DEFINE_integer("max_seq_len", 32768, "Max token sequence length per trajectory.")
flags.DEFINE_integer("eval_every_steps", 25, "Validation evaluation interval in optimizer steps.")
flags.DEFINE_integer("max_eval_samples", 64, "Fixed validation sample count matching LoRA.")
flags.DEFINE_integer("save_every_steps", 25, "Distributed checkpoint save interval in optimizer steps.")
flags.DEFINE_bool(
    "save_optimizer_state",
    False,
    "Whether to save sharded FP32 AdamW optimizer states in checkpoints (False = model weights only, 5x smaller).",
)
flags.DEFINE_string(
    "save_weights_dtype",
    "bfloat16",
    "Dtype for saved model weights in checkpoints ('bfloat16' = 53.8 GB total, 'float32' = 107.6 GB total).",
)
flags.DEFINE_integer(
    "keep_last_n_checkpoints",
    1,
    "Retain latest N completed non-milestone checkpoints (1 = only keep latest).",
)
flags.DEFINE_integer(
    "milestone_every_steps",
    0,
    "Retain milestone checkpoints every N steps (0 = disabled, only keep latest).",
)
flags.DEFINE_integer(
    "export_fp8_every_steps",
    50,
    "Collaboratively export Block-128 FP8 checkpoint every N steps (and final) for full ARC-AGI-2 eval.",
)
flags.DEFINE_integer("num_ckpt_parts_per_rank", 8, "Number of shard files written per rank.")
flags.DEFINE_bool(
    "use_fast_deltanet",
    True,
    "Use validated custom Triton Gated DeltaNet forward/backward kernels + cuBLAS solve_triangular.",
)
flags.DEFINE_integer(
    "grad_ckpt_stride",
    1,
    "Selective activation checkpointing stride (1 = all 64 layers).",
)
flags.DEFINE_bool(
    "use_torch_compile_norms",
    True,
    "Compile Qwen3_5RMSNorm and Qwen3_5RMSNormGated with torch.compile(dynamic=True).",
)
flags.DEFINE_string(
    "attn_implementation",
    "flash_attention_2",
    "Attention implementation (flash_attention_2 or sdpa).",
)
flags.DEFINE_bool(
    "smoke_test_mode",
    False,
    "Enable pre-launch smoke test verification (save/reload/resume, partial batch, grad checks).",
)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown arguments: {argv[1:]}")

    with xm_abc.create_experiment(experiment_title=FLAGS.exp_name) as experiment:
        reqs = xm.JobRequirements(
            b200=FLAGS.num_b200,
            cpu=FLAGS.cpu_cores,
            ram=FLAGS.ram_gb * xm.GiB,
            tmp_ram_fs=128 * xm.GiB,
            location=FLAGS.location,
        )
        executor = xm_abc.Borg(requirements=reqs)

        bazel_args = list(xm_abc.bazel_args.for_requirements(reqs))

        [train_pkg] = experiment.package(
            [
                xm.bazel_binary(
                    label="//experimental/users/arazdai/kaggle/sft:train_full_sft",
                    executor_spec=xm_abc.Borg.Spec(),
                    bazel_args=bazel_args,
                )
            ]
        )

        output_dir = f"{FLAGS.output_cns_base}/xid_{experiment.experiment_id}_{FLAGS.exp_name}"

        async def run_full_sft_job(wu: xm.WorkUnit, **unused_kwargs) -> None:
            job_args = {
                "undefok": "cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
                "cfs_d_cid_enable": True,
                "cfs_d_resource_user_override": "deepmind-resources-colossus",
                "cfs_default_traffic_class": "AF1",
                "cfs_max_write_bytes_inflight": 33554432,
                "model_path": FLAGS.model_path,
                "data_dir": FLAGS.data_dir,
                "output_dir": f"{output_dir}/wu_{wu.work_unit_id}",
                "learning_rate": FLAGS.learning_rate,
                "min_lr_ratio": FLAGS.min_lr_ratio,
                "weight_decay": FLAGS.weight_decay,
                "warmup_ratio": FLAGS.warmup_ratio,
                "max_grad_norm": FLAGS.max_grad_norm,
                "num_epochs": FLAGS.num_epochs,
                "max_steps": FLAGS.max_steps,
                "micro_batch_size": FLAGS.micro_batch_size,
                "grad_accum_steps": FLAGS.grad_accum_steps,
                "max_seq_len": FLAGS.max_seq_len,
                "eval_every_steps": FLAGS.eval_every_steps,
                "max_eval_samples": FLAGS.max_eval_samples,
                "save_every_steps": FLAGS.save_every_steps,
                "save_optimizer_state": FLAGS.save_optimizer_state,
                "save_weights_dtype": FLAGS.save_weights_dtype,
                "keep_last_n_checkpoints": FLAGS.keep_last_n_checkpoints,
                "milestone_every_steps": FLAGS.milestone_every_steps,
                "export_fp8_every_steps": FLAGS.export_fp8_every_steps,
                "num_ckpt_parts_per_rank": FLAGS.num_ckpt_parts_per_rank,
                "use_fast_deltanet": FLAGS.use_fast_deltanet,
                "grad_ckpt_stride": FLAGS.grad_ckpt_stride,
                "use_torch_compile_norms": FLAGS.use_torch_compile_norms,
                "attn_implementation": FLAGS.attn_implementation,
                "smoke_test_mode": FLAGS.smoke_test_mode,
            }
            wu.add(
                xm.Job(
                    executable=train_pkg,
                    executor=executor,
                    args=job_args,
                    env_vars={
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "TOKENIZERS_PARALLELISM": "false",
                        "TORCHINDUCTOR_COMPILE_THREADS": "1",
                    },
                )
            )

        experiment.add(
            run_full_sft_job,
            args={
                "lr": FLAGS.learning_rate,
                "max_seq_len": FLAGS.max_seq_len,
                "num_b200": FLAGS.num_b200,
                "save_opt": FLAGS.save_optimizer_state,
                "keep_n": FLAGS.keep_last_n_checkpoints,
                "fast_deltanet": FLAGS.use_fast_deltanet,
                "ckpt_stride": FLAGS.grad_ckpt_stride,
                "smoke_test": FLAGS.smoke_test_mode,
            },
        )


if __name__ == "__main__":
    app.run(main)
