"""Multi-TPU XManager Launcher for Full DeepSeek-V4.1-Flash-1017 (384 Experts, BF16->FP32) E065 Iterative Reasoning Correction (up to 4 retries) + Clean Prefix-Free Solution Rewrite."""

from __future__ import annotations

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_flags
from xmanager.contrib.internal import requirements_flag
from xmanager.xm_abc import restricted_borg_overrides

FLAGS = flags.FLAGS

_PLATFORM = requirements_flag.DEFINE_requirements(
    "platform",
    "ghostfish=2x2x1",
    "TPU platform/topology per work unit (8 TensorCores).",
)
flags.DEFINE_string(
    "experiment_title",
    "deepseek_v41_e065_correct_and_rewrite",
    "XManager experiment title.",
)
flags.DEFINE_string(
    "e065_output_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten",
    "CNS directory where corrected & rewritten E065 SFT Parquet and JSONL shards are saved.",
)
flags.DEFINE_string(
    "non_test_passing_parquet_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/non_test_passing/e065_images_non_test_passing.parquet",
    "CNS path to e065_images_non_test_passing.parquet.",
)
flags.DEFINE_string(
    "non_test_passing_tasks_jsonl_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/non_test_passing/e065_images_non_test_passing_tasks.jsonl",
    "CNS path to e065_images_non_test_passing_tasks.jsonl.",
)
flags.DEFINE_string(
    "difficulty_rankings_jsonl_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/difficulty_rankings/gemini38_flash_non_test_passing_ranked.jsonl",
    "CNS path to Gemini Flash 3.8 difficulty rankings JSONL.",
)
flags.DEFINE_integer(
    "num_shards",
    64,
    "Number of parallel TPU slices sharding the 1,429 uncontaminated failed E065 rollouts (~22 tasks/shard).",
)
flags.DEFINE_integer(
    "start_shard",
    0,
    "First shard index to launch (inclusive).",
)
flags.DEFINE_integer(
    "end_shard",
    -1,
    "Exclusive upper bound on shard index to launch (-1 = launch up to num_shards).",
)
flags.DEFINE_integer(
    "pair_batch_size",
    2,
    "Tasks decoded per wave per TPU slice (2 tasks = 12 rows <= 16-row single MXU tile).",
)
flags.DEFINE_integer(
    "max_correction_attempts",
    4,
    "Maximum number of full DeepSeek reasoning-correction attempts per failed rollout.",
)
flags.DEFINE_integer(
    "max_rewrite_attempts",
    2,
    "Maximum number of full DeepSeek clean-rewrite tries once a correct solution is found.",
)
flags.DEFINE_float(
    "temperature",
    1.0,
    "Stochastic sampling temperature (matching v8 / Vu E030 temperature=1.0).",
)
flags.DEFINE_float(
    "top_p",
    0.95,
    "Top-p nucleus sampling threshold.",
)
flags.DEFINE_integer(
    "top_k",
    20,
    "Top-k sampling candidates per step.",
)
flags.DEFINE_integer(
    "max_turns",
    40,
    "Maximum turns per trajectory (v8 / Vu E030 sub80s40 = 40).",
)
flags.DEFINE_integer(
    "max_turn_tokens",
    64000,
    "Max tokens per single assistant turn (v8 / Vu E030 sub80s40 = 64000).",
)
flags.DEFINE_integer(
    "max_reasoning_tokens",
    0,
    "Separate cap on <think> tokens (0 = disabled, governed by total_token_budget=80000).",
)
flags.DEFINE_integer(
    "max_model_len",
    65536,
    "Max sequence length for static CED prefill & decode (64K context).",
)
flags.DEFINE_integer(
    "compact_at_tokens",
    60000,
    "Prompt token threshold at which non-destructive transcript compaction triggers.",
)
flags.DEFINE_integer(
    "total_token_budget",
    80000,
    "Hard total output tokens per trajectory across turns (v8 / Vu E030 sub80s40 = 80000).",
)
flags.DEFINE_integer(
    "soft_token_target",
    40000,
    "Soft output token target per trajectory (v8 / Vu E030 sub80s40 = 40000).",
)
flags.DEFINE_integer(
    "turns_note_threshold",
    5,
    "Warn when N or fewer turns remain (v8 / Vu E030 --turns-note 5).",
)
flags.DEFINE_boolean(
    "token_exact",
    True,
    "Preserve exact sampled token_ids across turns (v8 / Vu E043 --token-exact).",
)
flags.DEFINE_boolean(
    "rescue_code",
    True,
    "Rescue Python code written outside DSML tool tags (v8 / Vu E043 --rescue-code).",
)
flags.DEFINE_float(
    "sandbox_timeout_s",
    30.0,
    "Per-cell timeout in seconds for LocalExecutor(style='ipython').",
)
flags.DEFINE_integer(
    "max_tasks",
    0,
    "Optional cap on tasks per shard (0 = all tasks in shard).",
)
flags.DEFINE_integer("cpu", 64, "Host CPUs per TPU slice.")
flags.DEFINE_integer("ram_gb", 220, "Host RAM GB per TPU slice.")
flags.DEFINE_string("location", "viglobal", "Borg location.")
flags.DEFINE_boolean(
    "enable_xm_autopilot",
    True,
    "Enable XManager Autopilot.",
)
flags.DEFINE_list(
    "wim_tags",
    ["lti-lh-tasks", "data", "enable_xm_autopilot=true"],
    "WIM tags added to experiment and work units.",
)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    num_shards = max(1, FLAGS.num_shards)
    start_s = max(0, FLAGS.start_shard)
    end_s = num_shards if FLAGS.end_shard < 0 else min(num_shards, FLAGS.end_shard)

    with xm_abc.create_experiment(
        experiment_title=FLAGS.experiment_title,
        settings=xm_abc.ExecutionSettings(admin_users=[flags.FLAGS.borguser]),
    ) as experiment:
        experiment.context.annotations.add_tags(
            "arc-agi-2",
            "sft-rollouts",
            "e065-hardest-rollouts",
            "deepseek-v4.1-flash-1017",
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
            autopilot_params=xm_abc.AutopilotParams(
                enabled=FLAGS.enable_xm_autopilot
            ),
        )
        extra = ["--norun_validations", "--nocheck_visibility", "--dynamic_mode=off"]
        if xm_flags.EXPERIMENTAL_SKIP_PYTYPE_CHECK.value:
            extra.append("--define=PYTYPE=FALSE")
        bazel_args = xm_abc.bazel_args.for_requirements(reqs) + tuple(extra)

        [executable] = experiment.package(
            [
                xm.bazel_binary(
                    label="//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:deepseek_v41_e065_correct_and_rewrite",
                    executor_spec=executor.Spec(),
                    bazel_args=bazel_args,
                )
            ]
        )

        env_vars = {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "MODEL_IMPL_TYPE": "vllm",
            "SKIP_JAX_PRECOMPILE": "1",
        }
        common = [
            "--engine_backend=ds41_full_precision",
            f"--e065_output_dir={FLAGS.e065_output_dir}",
            f"--non_test_passing_parquet_path={FLAGS.non_test_passing_parquet_path}",
            f"--non_test_passing_tasks_jsonl_path={FLAGS.non_test_passing_tasks_jsonl_path}",
            f"--difficulty_rankings_jsonl_path={FLAGS.difficulty_rankings_jsonl_path}",
            f"--exp_name={FLAGS.experiment_title}",
            f"--pair_batch_size={FLAGS.pair_batch_size}",
            f"--max_correction_attempts={FLAGS.max_correction_attempts}",
            f"--max_rewrite_attempts={FLAGS.max_rewrite_attempts}",
            f"--temperature={FLAGS.temperature}",
            f"--top_p={FLAGS.top_p}",
            f"--top_k={FLAGS.top_k}",
            f"--max_turns={FLAGS.max_turns}",
            f"--max_turn_tokens={FLAGS.max_turn_tokens}",
            f"--max_reasoning_tokens={FLAGS.max_reasoning_tokens}",
            f"--max_model_len={FLAGS.max_model_len}",
            f"--compact_at_tokens={FLAGS.compact_at_tokens}",
            f"--total_token_budget={FLAGS.total_token_budget}",
            f"--soft_token_target={FLAGS.soft_token_target}",
            f"--turns_note_threshold={FLAGS.turns_note_threshold}",
            f"--token_exact={str(FLAGS.token_exact).lower()}",
            f"--rescue_code={str(FLAGS.rescue_code).lower()}",
            f"--sandbox_timeout_s={FLAGS.sandbox_timeout_s}",
            f"--limit_tasks={FLAGS.max_tasks}",
            f"--num_shards={num_shards}",
        ]

        for shard_idx in range(start_s, end_s):
            wu_args = common + [f"--shard_idx={shard_idx}"]
            experiment.add(
                xm.Job(
                    executable=executable,
                    executor=executor,
                    args=wu_args,
                    env_vars=env_vars,
                ),
                identity=f"ds41_e065_cr_s{shard_idx:03d}of{num_shards:03d}",
            )
        print(
            f"Launched {end_s - start_s} TPU slices (ghostfish=2x2x1): "
            f"http://xid/{experiment.experiment_id}",
            flush=True,
        )
        print(f"CNS Output Directory: {FLAGS.e065_output_dir}", flush=True)


if __name__ == "__main__":
    app.run(main)
