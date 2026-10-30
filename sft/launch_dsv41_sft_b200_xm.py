"""XManager launcher for Pruned DeepSeek-V4.1 (`Tool E128`) Full-SFT on 8x NVIDIA B200 GPUs."""

from __future__ import annotations

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "dsv41_e128_moe_only_qat_sft_iter2_8xb200",
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
    "dsv41_model_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "CNS directory containing chankhavu's pruned DeepSeek-V4.1 Tool E128 2-bit checkpoint.",
)
flags.DEFINE_string(
    "dsv41_data_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/dsv41_e128_short_clean",
    "CNS directory containing train_tokens.npz and val_tokens.npz with loss_weights.",
)
flags.DEFINE_string(
    "output_cns_base",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2",
    "CNS base directory where XID-keyed Pruned DeepSeek-V4.1 QAT SFT checkpoints and 2-bit exports are saved.",
)
flags.DEFINE_string(
    "override_output_dir",
    "",
    "Optional explicit CNS output directory override (if non-empty, overrides output_cns_base/xid_... so an existing eval sweep polling a path can consume the milestones immediately).",
)
flags.DEFINE_float("learning_rate", 8e-6, "Peak learning rate for AdamW.")
flags.DEFINE_float("min_lr_ratio", 0.1, "Min learning rate ratio at end of cosine decay.")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_float("warmup_ratio", 0.05, "Warmup ratio (0.05 = 5% of total steps).")
flags.DEFINE_float("max_grad_norm", 1.0, "Max gradient L2 norm for clipping.")
flags.DEFINE_integer("num_epochs", 1, "Number of epochs.")
flags.DEFINE_integer("max_steps", -1, "Optional max optimizer steps override (-1 = full num_epochs).")
flags.DEFINE_integer("grad_accum_steps", 4, "Gradient accumulation micro-steps per rank (8 ranks x 4 = 32 global batch).")
flags.DEFINE_integer("max_seq_len", 36864, "Max token sequence length per trajectory.")
flags.DEFINE_integer("logits_chunk_size", 2048, "Sequence chunk size for LM head cross-entropy.")
flags.DEFINE_integer("eval_every_steps", 28, "Validation evaluation interval in optimizer steps.")
flags.DEFINE_integer("max_eval_samples", 64, "Validation sample count per check.")
flags.DEFINE_integer("save_every_steps", 28, "Checkpoint save interval in optimizer steps.")
flags.DEFINE_integer("routed_expert_lora_r", 32, "LoRA rank on the 128 routed experts per layer (32 = train all 128 routed experts + dense shared_experts within 82 GiB/B200 and re-quantize to 2-bit Humming at export).")
flags.DEFINE_float("routed_expert_lora_alpha", 64.0, "LoRA alpha scaling when routed_expert_lora_r > 0.")
flags.DEFINE_bool("trainable_routed_experts_dense", False, "If True (and routed_expert_lora_r == 0), train dense BF16 routed experts.")
flags.DEFINE_bool("train_moe_only", True, "Freeze attention, ngrams (Engram), expert router (gate), mHC, norms, embed, and head; train ONLY MoE expert layers.")
flags.DEFINE_bool(
    "enable_qat",
    True,
    "Enable Nemotron-style Quantization-Aware Training (QAT) with exact Havu FP8 block-32, FP4 act_quant, and 2-bit Humming locked-z STE.",
)
flags.DEFINE_bool(
    "qat_trainable_expert_scales",
    True,
    "Train FP32 per-group 2-bit Humming scales (s) on all 128 routed experts.",
)
flags.DEFINE_bool(
    "qat_lock_gptq_codes",
    False,
    "If True, lock integer codes q_orig and only adapt per-group scales s; if False, allow LoRA + STE to cross 2-bit thresholds with locked GPTQ zero-points z_orig.",
)
flags.DEFINE_float(
    "qat_kl_weight",
    0.15,
    "Weight on Nemotron QAD forward-KL distillation loss D_KL(P_teacher || P_quant_student).",
)
flags.DEFINE_float(
    "qat_kl_temperature",
    1.0,
    "Temperature for Nemotron QAD forward-KL logit distillation.",
)
flags.DEFINE_bool(
    "load_compact_engram",
    True,
    "Load the 16 compact Engram shards (layers 1 & 14) sharded across EP ranks so QAT matches Havu's inference activations.",
)
flags.DEFINE_string(
    "milestone_epoch_fractions",
    "0.25,0.50,0.75,1.00",
    "Milestone epoch fractions at which to run validation, save checkpoint, and export 2-bit checkpoint.",
)
flags.DEFINE_bool("use_token_loss_weights", True, "Apply token-level loss_weights (WIC-alnum=10, final_submit=5, tool_call=2).")
flags.DEFINE_bool("length_bucketed_batching", True, "Length-bucket trajectories per step and balance tokens across ranks.")
flags.DEFINE_integer("bucket_mega_batch_multiplier", 8, "Mega-batch multiplier for length bucketing.")
flags.DEFINE_integer("grad_ckpt_stride", 1, "Gradient checkpointing stride (1 = checkpoint all 40 layers).")


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
                    label="//experimental/users/arazdai/kaggle/sft:train_dsv41_sft",
                    executor_spec=xm_abc.Borg.Spec(),
                    bazel_args=bazel_args,
                )
            ]
        )

        output_dir = (
            FLAGS.override_output_dir
            if FLAGS.override_output_dir
            else f"{FLAGS.output_cns_base}/xid_{experiment.experiment_id}_{FLAGS.exp_name}"
        )

        async def run_dsv41_sft_job(wu: xm.WorkUnit, **unused_kwargs) -> None:
            job_args = {
                "undefok": "cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
                "cfs_d_cid_enable": True,
                "cfs_d_resource_user_override": "deepmind-resources-colossus",
                "cfs_default_traffic_class": "AF1",
                "cfs_max_write_bytes_inflight": 33554432,
                "dsv41_model_dir": FLAGS.dsv41_model_dir,
                "dsv41_data_dir": FLAGS.dsv41_data_dir,
                "dsv41_output_dir": f"{output_dir}/wu_{wu.work_unit_id}",
                "learning_rate": FLAGS.learning_rate,
                "min_lr_ratio": FLAGS.min_lr_ratio,
                "weight_decay": FLAGS.weight_decay,
                "warmup_ratio": FLAGS.warmup_ratio,
                "max_grad_norm": FLAGS.max_grad_norm,
                "num_epochs": FLAGS.num_epochs,
                "max_steps": FLAGS.max_steps,
                "grad_accum_steps": FLAGS.grad_accum_steps,
                "max_seq_len": FLAGS.max_seq_len,
                "logits_chunk_size": FLAGS.logits_chunk_size,
                "eval_every_steps": FLAGS.eval_every_steps,
                "max_eval_samples": FLAGS.max_eval_samples,
                "save_every_steps": FLAGS.save_every_steps,
                "routed_expert_lora_r": FLAGS.routed_expert_lora_r,
                "routed_expert_lora_alpha": FLAGS.routed_expert_lora_alpha,
                "trainable_routed_experts_dense": FLAGS.trainable_routed_experts_dense,
                "train_moe_only": FLAGS.train_moe_only,
                "enable_qat": FLAGS.enable_qat,
                "qat_trainable_expert_scales": FLAGS.qat_trainable_expert_scales,
                "qat_lock_gptq_codes": FLAGS.qat_lock_gptq_codes,
                "qat_kl_weight": FLAGS.qat_kl_weight,
                "qat_kl_temperature": FLAGS.qat_kl_temperature,
                "load_compact_engram": FLAGS.load_compact_engram,
                "milestone_epoch_fractions": FLAGS.milestone_epoch_fractions,
                "use_token_loss_weights": FLAGS.use_token_loss_weights,
                "length_bucketed_batching": FLAGS.length_bucketed_batching,
                "bucket_mega_batch_multiplier": FLAGS.bucket_mega_batch_multiplier,
                "grad_ckpt_stride": FLAGS.grad_ckpt_stride,
            }
            wu.add(
                xm.Job(
                    executable=train_pkg,
                    executor=executor,
                    args=job_args,
                    env_vars={
                        "PYTHONUNBUFFERED": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "TOKENIZERS_PARALLELISM": "false",
                        "TORCHINDUCTOR_COMPILE_THREADS": "1",
                    },
                )
            )

        experiment.add(
            run_dsv41_sft_job,
            args={
                "lr": FLAGS.learning_rate,
                "grad_accum": FLAGS.grad_accum_steps,
                "max_seq_len": FLAGS.max_seq_len,
                "num_b200": FLAGS.num_b200,
                "expert_lora_r": FLAGS.routed_expert_lora_r,
                "enable_qat": FLAGS.enable_qat,
                "qat_kl_weight": FLAGS.qat_kl_weight,
                "max_steps": FLAGS.max_steps,
            },
        )


if __name__ == "__main__":
    app.run(main)
