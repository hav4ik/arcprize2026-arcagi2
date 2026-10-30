"""Multi-TPU XManager Launcher for DeepSeek-V4.1-Flash-1017 Full-Precision SFT Reasoning Traces on Top 10K ARC Candidates.

Launches parallel TPU slices (`ghostfish=2x2x1`, 8 TensorCores per slice) in the
LTI Lighthouse pool (`--xm_resource_pool=research-mm --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks`)
with mandatory `AutopilotParams(enabled=True)`.
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_flags
from xmanager.contrib.internal import requirements_flag
from xmanager.xm_abc import restricted_borg_overrides

FLAGS = flags.FLAGS
_PLATFORM = requirements_flag.DEFINE_requirements(
    "platform", "ghostfish=2x2x1", "TPU platform/topology per work unit (8 TensorCores)."
)
flags.DEFINE_string(
    "experiment_title",
    "arc_sft_top10k_deepseek_v41_flash_1017_fp",
    "XManager experiment title.",
)
flags.DEFINE_integer(
    "num_shards",
    16,
    "Number of parallel TPU slices (work units) in lti lh pool.",
)
flags.DEFINE_string(
    "run_tag",
    "ds41_fp_top10k",
    "Run tag for experiment output directory.",
)
flags.DEFINE_string(
    "engine_backend",
    "auto",
    "Backend: 'auto', 'ds41_full_precision', or 'vllm'.",
)
flags.DEFINE_string(
    "candidates_jsonl",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/top_10k_deepseek_candidates.jsonl",
    "CNS path to Top 10,000 candidates JSONL.",
)
flags.DEFINE_string(
    "out_root",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/deepseek_v41_traces_top10k",
    "CNS output root directory for verified and raw reasoning trajectories.",
)
flags.DEFINE_integer("limit_tasks", 0, "Optional limit on tasks (0 = all 10,000).")
flags.DEFINE_integer("num_attempts", 2, "Trajectories per candidate puzzle.")
flags.DEFINE_integer("pair_batch_size", 4, "Puzzles per batch wave per shard.")
flags.DEFINE_integer("max_turns", 5, "Maximum turns per trajectory (5 turns max).")
flags.DEFINE_integer("max_repair_rounds", 0, "Additional repair rounds after max_turns (0 = strict 5-turn cap).")
flags.DEFINE_integer("max_model_len", 16384, "Max sequence length for static CED prefill & decode.")
flags.DEFINE_integer("total_token_budget", 16384, "Hard output tokens per trajectory.")
flags.DEFINE_integer("soft_token_target", 8192, "Soft output tokens per trajectory.")
flags.DEFINE_integer("cpu", 64, "Host CPUs per TPU slice.")
flags.DEFINE_integer("ram_gb", 220, "Host RAM GB per TPU slice.")
flags.DEFINE_string("location", "viglobal", "Borg location.")
flags.DEFINE_boolean(
    "enable_xm_autopilot",
    True,
    "Enable XManager Autopilot (mandatory in genai-lti-lh-tasks).",
)
flags.DEFINE_list(
    "wim_tags",
    ["lti-lh-tasks", "data", "enable_xm_autopilot=true"],
    "WIM tags added to experiment and work units.",
)


def main(argv):
  del argv
  with xm_abc.create_experiment(
      experiment_title=FLAGS.experiment_title,
      settings=xm_abc.ExecutionSettings(admin_users=[flags.FLAGS.borguser]),
  ) as experiment:
    experiment.context.annotations.add_tags(
        "arc-agi-2",
        "sft-top10k",
        "deepseek-v4.1-flash-1017",
        FLAGS.run_tag,
        *FLAGS.wim_tags,
    )
    reqs = xm.JobRequirements(
        _PLATFORM.value,
        cpu=FLAGS.cpu,
        ram=FLAGS.ram_gb * xm.GiB,
        location=FLAGS.location,
        tmp_ram_fs=16 * xm.GiB,
    )
    executor = xm_abc.Borg(
        requirements=reqs,
        borg_user=flags.FLAGS.borguser,
        logs_read_access_roles=["all"],
        scheduling=xm_abc.BorgScheduling(
            max_task_failures=10, max_per_task_failures=10
        ),
        borg_overrides=restricted_borg_overrides.RESTRICTED_BorgOverrides(
            health_check_grace_period="3600s"
        ),
        # Autopilot is mandatory in group:research-mm/genai-lti-lh-tasks
        autopilot_params=xm_abc.AutopilotParams(enabled=FLAGS.enable_xm_autopilot),
    )
    extra = ["--norun_validations", "--nocheck_visibility", "--dynamic_mode=off"]
    if xm_flags.EXPERIMENTAL_SKIP_PYTYPE_CHECK.value:
      extra.append("--define=PYTYPE=FALSE")
    bazel_args = xm_abc.bazel_args.for_requirements(reqs) + tuple(extra)
    [executable] = experiment.package([
        xm.bazel_binary(
            label="//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:deepseek_v41_tpu_vllm_traces",
            executor_spec=executor.Spec(),
            bazel_args=bazel_args,
        )
    ])
    env_vars = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MODEL_IMPL_TYPE": "vllm",
        "SKIP_JAX_PRECOMPILE": "1",
    }
    exp_name = f"{FLAGS.experiment_title}_{FLAGS.run_tag}"
    out_dir = f"{FLAGS.out_root}_{FLAGS.run_tag}"
    common = [
        f"--engine_backend={FLAGS.engine_backend}",
        f"--candidates_jsonl={FLAGS.candidates_jsonl}",
        f"--output_dir={out_dir}",
        f"--exp_name={exp_name}",
        f"--num_attempts={FLAGS.num_attempts}",
        f"--pair_batch_size={FLAGS.pair_batch_size}",
        f"--max_turns={FLAGS.max_turns}",
        f"--max_repair_rounds={FLAGS.max_repair_rounds}",
        f"--max_model_len={FLAGS.max_model_len}",
        f"--total_token_budget={FLAGS.total_token_budget}",
        f"--soft_token_target={FLAGS.soft_token_target}",
        f"--limit_tasks={FLAGS.limit_tasks}",
        f"--num_shards={FLAGS.num_shards}",
    ]
    for s in range(FLAGS.num_shards):
      args = common + [f"--shard_idx={s}"]
      experiment.add(
          xm.Job(
              executable=executable,
              executor=executor,
              env_vars=env_vars,
              args=args,
          ),
          identity=f"ds41_fp_s{s:02d}",
      )
    print(
        f"Launched {FLAGS.num_shards} TPU slices (ghostfish=2x2x1, 8 TensorCores each = "
        f"{FLAGS.num_shards * 8} TensorCores total): http://xid/{experiment.experiment_id}"
    )


if __name__ == "__main__":
  app.run(main)
