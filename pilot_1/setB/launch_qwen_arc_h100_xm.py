"""XManager launcher for Qwen3.8-27B ("Qwen 28B") + Tools on ARC-AGI-2 (H100 + vLLM).

Launch command (from google3/):
  /google/bin/releases/xmanager/cli/xmanager.par launch \
    //experimental/users/arazdai/kaggle:launch_qwen_arc_h100_xm -- \
    --noxm_monitor_on_launch \
    --xm_resource_pool=research-mm \
    --xm_resource_alloc=group:research-mm/ml-gemini-code-bridging-gap
"""

import time
from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "qwen28b_tools_arc_agi_2_h100",
    "Name of the XManager experiment.",
)
flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    "CNS path to BF16 Qwen3.8-27B weights (54GB native on 80GB H100).",
)
flags.DEFINE_string(
    "fp8_model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "CNS path to Qwen3.8-27B-FP8 weights.",
)
flags.DEFINE_string(
    "data_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "CNS directory containing the 120 ARC-AGI-2 evaluation tasks.",
)
flags.DEFINE_string(
    "output_dir",
    "",
    "CNS output directory for predictions.jsonl, summary.json, and kaggle_submission.json. "
    "If empty, auto-generates a timestamped path under /cns/lu-d/home/arazdai/kaggle/arc_qwen_h100_runs/.",
)
flags.DEFINE_string(
    "recipe",
    "e020_k3_v4",
    "Collaborator recipe: 'e012_tools_full' or 'e020_k3_v4'.",
)
flags.DEFINE_integer(
    "num_shards",
    4,
    "Number of parallel H100 work units (shards) to run across the 167 test pairs.",
)
flags.DEFINE_integer(
    "num_attempts",
    2,
    "Number of attempts per test pair (2 for pass@1 and pass@2).",
)
flags.DEFINE_integer(
    "total_token_budget",
    48000,
    "Hard token budget per attempt.",
)
flags.DEFINE_integer(
    "soft_token_target",
    24000,
    "Soft token pacing target per attempt.",
)
flags.DEFINE_string(
    "cell",
    "viglobal",
    "Borg cell or virtual cell for H100 placement (viglobal auto-places in available H100 cells).",
)


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  run_ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
  output_dir = (
      FLAGS.output_dir
      or f"/cns/lu-d/home/arazdai/kaggle/arc_qwen_h100_runs/{FLAGS.exp_name}_{FLAGS.recipe}_{run_ts}"
  )

  with xm_abc.create_experiment(experiment_title=FLAGS.exp_name) as experiment:
    experiment.context.annotations.add_tags(
        "qwen3.8-27b", "vllm", "h100", "arc-agi-2", FLAGS.recipe
    )
    experiment.context.annotations.set_notes(
        f"CNS Output Directory: {output_dir} | Recipe: {FLAGS.recipe}"
    )

    requirements = xm.JobRequirements(
        h100=1,
        ram=80 * xm.GiB,
        cpu=16 * xm.vCPU,
        location=FLAGS.cell,
        service_tier=xm.ServiceTier.PROD,
    )
    executor = xm_abc.Borg(requirements=requirements)

    bazel_args = list(xm_abc.bazel_args.gpu()) + [
        "--config=cuda",
        "--action_env=NEXUS_CUDA_VERSION=12.8",
        "-c",
        "opt",
        "--nocheck_visibility",
    ]

    [executable] = experiment.package([
        xm.bazel_binary(
            label="//experimental/users/arazdai/kaggle:eval_qwen_vllm_h100",
            executor_spec=executor.Spec(),
            bazel_args=bazel_args,
        )
    ])

    for shard_idx in range(FLAGS.num_shards):
      job_args = {
          "model_path": FLAGS.model_path,
          "fp8_model_path": FLAGS.fp8_model_path,
          "data_dir": FLAGS.data_dir,
          "output_dir": output_dir,
          "recipe": FLAGS.recipe,
          "num_attempts": FLAGS.num_attempts,
          "total_token_budget": FLAGS.total_token_budget,
          "soft_token_target": FLAGS.soft_token_target,
          "num_shards": FLAGS.num_shards,
          "shard_idx": shard_idx,
      }
      experiment.add(
          xm.Job(
              executable=executable,
              executor=executor,
              args=job_args,
              name=f"qwen28b_eval_shard_{shard_idx}",
          )
      )

    print(f"Launched XManager experiment: http://xid/{experiment.experiment_id}")
    print(f"CNS output directory: {output_dir}")


if __name__ == "__main__":
  app.run(main)
