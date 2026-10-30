"""XManager launcher for the before/after SFT sanity generation on 8x B200 (`gen_dsv41_before_after`)."""

from __future__ import annotations

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "dsv41_e128_before_after_sanity", "XManager experiment name.")
flags.DEFINE_integer("num_gpus", 8, "Number of B200 GPUs.")
flags.DEFINE_string("cell", "viglobal", "Borg cell / location.")
flags.DEFINE_string(
    "gen_base_model_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "Original 2-bit base checkpoint directory.",
)
flags.DEFINE_string(
    "gen_sft_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/xid_305548873_dsv41_e128_moe_only_full_sft_8xb200_v4/wu_1/export_2bit/epoch_1p00",
    "Exported SFT 2-bit checkpoint directory.",
)
flags.DEFINE_string(
    "gen_val_npz",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/dsv41_e128_short_clean/val_tokens.npz",
    "Held-out val trajectories.",
)
flags.DEFINE_integer("gen_val_index", -1, "Val example index (-1 = shortest).")
flags.DEFINE_integer("gen_max_new_tokens", 640, "Greedy tokens for the ARC example.")
flags.DEFINE_integer("gen_sanity_max_new_tokens", 160, "Greedy tokens per sanity prompt.")
flags.DEFINE_string(
    "gen_out_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/evals/dsv41_e128_before_after_sanity",
    "CNS output directory for the JSON report.",
)
flags.DEFINE_bool("gen_enable_qat", False, "Enable fake-quant activations in the forward pass.")
flags.DEFINE_bool("gen_load_compact_engram", True, "Load compact Engram shards.")
flags.DEFINE_bool("gen_skip_base", False, "Skip the base checkpoint.")
flags.DEFINE_bool("gen_skip_sft", False, "Skip the SFT checkpoint.")


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
                    label="//experimental/users/arazdai/kaggle/sft:gen_dsv41_before_after",
                    executor_spec=xm_abc.Borg.Spec(),
                    bazel_args=bazel_args,
                )
            ]
        )

        async def run_job(wu: xm.WorkUnit, which: str, **unused_kwargs) -> None:
            wu.add(
                xm.Job(
                    executable=executable,
                    executor=executor,
                    name=f"gen_{which}",
                    args={
                        "undefok": "cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
                        "cfs_d_cid_enable": True,
                        "cfs_d_resource_user_override": "deepmind-resources-colossus",
                        "cfs_default_traffic_class": "AF1",
                        "cfs_max_write_bytes_inflight": 33554432,
                        "gen_mode": "generate",
                        "gen_base_model_dir": FLAGS.gen_base_model_dir,
                        "gen_sft_dir": FLAGS.gen_sft_dir,
                        "gen_val_npz": FLAGS.gen_val_npz,
                        "gen_val_index": FLAGS.gen_val_index,
                        "gen_max_new_tokens": FLAGS.gen_max_new_tokens,
                        "gen_sanity_max_new_tokens": FLAGS.gen_sanity_max_new_tokens,
                        "gen_out_dir": FLAGS.gen_out_dir,
                        "gen_num_gpus": FLAGS.num_gpus,
                        "gen_enable_qat": FLAGS.gen_enable_qat,
                        "gen_load_compact_engram": FLAGS.gen_load_compact_engram,
                        # Exactly one checkpoint per process (see gen_dsv41_before_after.py).
                        "gen_skip_base": which != "base",
                        "gen_skip_sft": which != "sft",
                    },
                    env_vars={
                        "PYTHONUNBUFFERED": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "NCCL_DEBUG": "WARN",
                        "OMP_NUM_THREADS": "4",
                    },
                )
            )

        if not FLAGS.gen_skip_base:
            experiment.add(run_job, args={"which": "base"})
        if not FLAGS.gen_skip_sft:
            experiment.add(run_job, args={"which": "sft"})


if __name__ == "__main__":
    app.run(main)
