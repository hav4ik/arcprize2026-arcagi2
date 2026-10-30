"""XManager launcher for evaluating 2-bit DeepSeek-V4.1 E128 SFT milestones (0.25, 0.50, 0.75, 1.00 epoch) on full ARC-AGI-2 using Havu's Candidate B32 harness."""

from __future__ import annotations

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "dsv41_e128_moe_only_qat_iter2_b32_eval",
    "XManager experiment name.",
)
flags.DEFINE_string(
    "sft_output_root",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/r1/wu_1",
    "Root CNS directory of the QAT SFT run containing export_2bit/epoch_{0p25,0p50,0p75,1p00}.",
)
flags.DEFINE_string(
    "base_engram_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "CNS directory containing base Engram shards and tokenizer.json.",
)
flags.DEFINE_string(
    "dataset_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "Full 120-task / 167-pair ARC-AGI-2 evaluation dataset on CNS.",
)
flags.DEFINE_string(
    "eval_out_root",
    "/cns/lu-d/home/arazdai/kaggle/sft/evals/dsv41_e128_qat_iter2_b32",
    "CNS output root directory for milestone evaluations.",
)
flags.DEFINE_string(
    "milestones",
    "0.25,0.50,0.75,1.00",
    "Comma-separated epoch fractions to evaluate.",
)
flags.DEFINE_string(
    "explicit_checkpoints",
    "",
    "Optional comma-separated `tag=/cns/.../2bit_checkpoint_dir` list; when set, evaluates exactly "
    "these directories (e.g. `base=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit`) instead of "
    "sft_output_root/export_2bit/<milestone>.",
)
flags.DEFINE_integer("num_attempts", 2, "Candidate B32 attempts per task.")
flags.DEFINE_integer("max_tasks", 0, "If > 0, cap tasks per milestone (for smoke test).")
flags.DEFINE_integer("num_gpus", 8, "Number of B200 GPUs per milestone worker.")
flags.DEFINE_integer("num_shards", 1, "Number of task shards per milestone.")
flags.DEFINE_bool(
    "enable_qat",
    True,
    "Enable exact Havu FP8/FP4 dynamic activation & KV quantization in evaluation forward pass.",
)
flags.DEFINE_bool(
    "load_compact_engram",
    True,
    "Load the 16 compact Engram shards (layers 1 & 14) across EP ranks.",
)
flags.DEFINE_string("cell", "viglobal", "Borg cell.")
flags.DEFINE_integer(
    "profile_decode_chunk",
    0,
    "If > 0, rank 0 runs torch.profiler on this (1-based) batched decode chunk and prints op tables.",
)
flags.DEFINE_integer("priority", 200, "Borg priority.")


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with xm_abc.create_experiment(experiment_title=FLAGS.exp_name) as experiment:
        reqs = xm.JobRequirements(
            b200=FLAGS.num_gpus,
            cpu=64,
            ram=768 * xm.GiB,
            tmp_ram_fs=128 * xm.GiB,
            location=FLAGS.cell,
        )
        executor = xm_abc.Borg(requirements=reqs)
        bazel_args = list(xm_abc.bazel_args.for_requirements(reqs))

        [executable] = experiment.package(
            [
                xm.bazel_binary(
                    label="//experimental/users/arazdai/kaggle/sft:eval_dsv41_b32_arc2",
                    executor_spec=xm_abc.Borg.Spec(),
                    bazel_args=bazel_args,
                )
            ]
        )

        milestone_ckpts: list[tuple[str, str]] = []
        if FLAGS.explicit_checkpoints:
            for item in [x.strip() for x in FLAGS.explicit_checkpoints.split(",") if x.strip()]:
                tag, ckpt_dir = item.split("=", 1)
                milestone_ckpts.append((tag.strip(), ckpt_dir.strip()))
        else:
            for frac_s in [x.strip() for x in FLAGS.milestones.split(",") if x.strip()]:
                m_tag = f"epoch_{float(frac_s):.2f}".replace(".", "p")
                milestone_ckpts.append((m_tag, f"{FLAGS.sft_output_root}/export_2bit/{m_tag}"))

        for m_tag, ckpt_2bit in milestone_ckpts:
            out_dir = f"{FLAGS.eval_out_root}/{m_tag}"
            for shard_idx in range(FLAGS.num_shards):
                job_suffix = (
                    f"{m_tag}_s{shard_idx}" if FLAGS.num_shards > 1 else m_tag
                )

                async def run_milestone_eval(
                    wu: xm.WorkUnit,
                    ckpt_2bit: str = ckpt_2bit,
                    out_dir: str = out_dir,
                    m_tag: str = m_tag,
                    shard_idx: int = shard_idx,
                    job_suffix: str = job_suffix,
                    **unused_kwargs,
                ) -> None:
                    wu.add(
                        xm.Job(
                            executable=executable,
                            executor=executor,
                            name=f"eval_{job_suffix}",
                            args={
                                "undefok": "cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
                                "cfs_d_cid_enable": True,
                                "cfs_d_resource_user_override": "deepmind-resources-colossus",
                                "cfs_default_traffic_class": "AF1",
                                "cfs_max_write_bytes_inflight": 33554432,
                                "checkpoint_2bit_dir": ckpt_2bit,
                                "base_engram_dir": FLAGS.base_engram_dir,
                                "dataset_dir": FLAGS.dataset_dir,
                                "out_dir": out_dir,
                                "milestone_tag": m_tag,
                                "num_attempts": FLAGS.num_attempts,
                                "max_tasks": FLAGS.max_tasks,
                                "num_gpus": FLAGS.num_gpus,
                                "shard_index": shard_idx,
                                "num_shards": FLAGS.num_shards,
                                "enable_qat": FLAGS.enable_qat,
                                "load_compact_engram": FLAGS.load_compact_engram,
                                "poll_for_checkpoint": True,
                                "poll_timeout_hours": 12,
                            },
                            env_vars={
                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                "NCCL_DEBUG": "WARN",
                                "OMP_NUM_THREADS": "4",
                                "DSV41_PROFILE_DECODE_CHUNK": str(FLAGS.profile_decode_chunk),
                            },
                        )
                    )

                experiment.add(
                    run_milestone_eval,
                    args={"milestone": m_tag, "shard_idx": shard_idx},
                )


if __name__ == "__main__":
    app.run(main)

