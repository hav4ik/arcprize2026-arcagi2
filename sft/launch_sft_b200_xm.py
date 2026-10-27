"""XManager launcher for Qwen3.8-27B ARC-AGI-2 Pilot SFT on B200 GPUs."""

from __future__ import annotations

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "qwen38_27b_arc_agi2_e031_pilot_sft_b200",
    "XManager experiment name.",
)
flags.DEFINE_integer("num_b200", 1, "Number of B200 GPUs per work unit.")
flags.DEFINE_integer("cpu_cores", 16, "CPU cores per work unit.")
flags.DEFINE_integer("ram_gb", 128, "System RAM (GB) per work unit.")
flags.DEFINE_string(
    "location",
    "viglobal",
    "Borg cell/location constraint (viglobal routes to any cell with B200 capacity in the alloc).",
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
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints",
    "Base CNS directory where XID-keyed SFT checkpoints and logs will be saved.",
)
flags.DEFINE_string(
    "resume_output_dir",
    "",
    "Optional existing CNS experiment output_dir (containing wu_1/checkpoint-latest) to resume in-place.",
)
flags.DEFINE_integer("lora_r", 64, "LoRA rank.")
flags.DEFINE_integer("lora_alpha", 128, "LoRA alpha.")
flags.DEFINE_float("learning_rate", 1e-4, "Peak learning rate.")
flags.DEFINE_integer("num_epochs", 1, "Number of epochs.")
flags.DEFINE_integer("max_steps", -1, "Optional max optimizer steps override (-1 = full num_epochs).")
flags.DEFINE_integer("grad_accum_steps", 8, "Gradient accumulation micro-steps.")
flags.DEFINE_integer("max_seq_len", 32768, "Max token sequence length per trajectory.")
flags.DEFINE_integer("eval_every_steps", 25, "Validation evaluation interval in optimizer steps.")
flags.DEFINE_integer("save_every_steps", 10, "Checkpoint save interval in optimizer steps.")
flags.DEFINE_bool(
    "use_fast_deltanet",
    True,
    "Replace sequential Python Gated DeltaNet loops with vectorized triangular solve + fused Triton kernel.",
)
flags.DEFINE_integer(
    "grad_ckpt_stride",
    1,
    "Selective activation checkpointing stride (1 = all 64 layers, 2 = every 2nd layer).",
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
    "use_multi_gpu_ddp",
    True,
    "Use multi-GPU DDP across all visible B200 GPUs when num_b200 > 1.",
)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown arguments: {argv[1:]}")

    with xm_abc.create_experiment(experiment_title=FLAGS.exp_name) as experiment:
        reqs = xm.JobRequirements(
            b200=FLAGS.num_b200,
            cpu=FLAGS.cpu_cores,
            ram=FLAGS.ram_gb * xm.GiB,
            tmp_ram_fs=32 * xm.GiB,
            location=FLAGS.location,
        )
        executor = xm_abc.Borg(requirements=reqs)

        bazel_args = list(xm_abc.bazel_args.for_requirements(reqs))

        [train_pkg] = experiment.package(
            [
                xm.bazel_binary(
                    label="//experimental/users/arazdai/kaggle/sft:train_sft",
                    executor_spec=xm_abc.Borg.Spec(),
                    bazel_args=bazel_args,
                )
            ]
        )

        output_dir = (
            FLAGS.resume_output_dir
            if FLAGS.resume_output_dir
            else f"{FLAGS.output_cns_base}/xid_{experiment.experiment_id}_{FLAGS.exp_name}"
        )

        async def run_sft_job(wu: xm.WorkUnit, **unused_kwargs) -> None:
            job_args = {
                "model_path": FLAGS.model_path,
                "data_dir": FLAGS.data_dir,
                "output_dir": f"{output_dir}/wu_{wu.work_unit_id}",
                "lora_r": FLAGS.lora_r,
                "lora_alpha": FLAGS.lora_alpha,
                "learning_rate": FLAGS.learning_rate,
                "num_epochs": FLAGS.num_epochs,
                "max_steps": FLAGS.max_steps,
                "grad_accum_steps": FLAGS.grad_accum_steps,
                "max_seq_len": FLAGS.max_seq_len,
                "eval_every_steps": FLAGS.eval_every_steps,
                "save_every_steps": FLAGS.save_every_steps,
                "use_fast_deltanet": FLAGS.use_fast_deltanet,
                "grad_ckpt_stride": FLAGS.grad_ckpt_stride,
                "use_torch_compile_norms": FLAGS.use_torch_compile_norms,
                "attn_implementation": FLAGS.attn_implementation,
                "use_multi_gpu_ddp": FLAGS.use_multi_gpu_ddp,
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
            run_sft_job,
            args={
                "lora_r": FLAGS.lora_r,
                "lr": FLAGS.learning_rate,
                "max_seq_len": FLAGS.max_seq_len,
                "num_b200": FLAGS.num_b200,
                "fast_deltanet": FLAGS.use_fast_deltanet,
                "ckpt_stride": FLAGS.grad_ckpt_stride,
            },
        )


if __name__ == "__main__":
    app.run(main)
