"""Multi-TPU XManager Launcher for DeepSeek-V4.1-Flash-1017 Pass@1 Rollouts on Top-10K SFT Candidates (`v8` Best Setting).

Generates 3-level verified SFT rollouts on the Top 10,000 ranked ARC candidates
(`/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/top_10k_deepseek_candidates.jsonl`)
using the best-scoring Pass@1 configuration (`v8`, `http://xid/304081446`, 59.28% single-attempt
Pass@1 / 61.68% ranked Pass@1 / 70.66% Pass@2 on ARC-AGI-2 Eval):
  - Platform: `ghostfish=2x2x1` (8 TensorCores per TPU v7x slice)
  - Engine: `ds41_full_precision` (`BF16 -> FP32` MXU accumulation, lossless speculative
    rejection + residual distribution sampling with `temperature=1.0, top_k=20, top_p=0.95`)
  - Prompt & Budget (`E030` `sub80s40` + `E043` `--token-exact --rescue-code`):
    `ARC_PRIZE_TEMPLATE` + `HAVU_LEAN2SUB_SUFFIX`, `HAVU_EFFORT_HEADER_MINIMAL` (`reasoning_effort=0`),
    `max_turns=40`, `max_turn_tokens=64000`, `max_reasoning_tokens=0`, `total_token_budget=80000`,
    `soft_token_target=40000`, `max_model_len=65536`, `compact_at_tokens=60000`
  - Sandbox: Isolated persistent child-process IPython session (`HavuSubprocessSession`, `style="ipython"`)
    with 3-tier fallback and in-session 3-level verification (`train` + `test` + `extra_verification_pairs`)
  - Output Format: Exact 24-column `E031` Parquet + JSONL rollout schema (`rollouts_shard_XXXofYYY.parquet`,
    `verified_rollouts_shard_XXXofYYY.parquet`, `.jsonl`) directly compatible with
    `//experimental/users/arazdai/kaggle/sft:audit_and_prepare_data` and `data_pipeline.py`.
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
    "platform",
    "ghostfish=2x2x1",
    "TPU platform/topology per work unit (8 TensorCores).",
)
flags.DEFINE_string(
    "experiment_title",
    "deepseek_v41_top10k_sft_rollouts_v8_pass1",
    "XManager experiment title.",
)
flags.DEFINE_integer(
    "num_shards",
    64,
    "Number of parallel TPU slices across the 10,000 candidates (~156 candidates/shard).",
)
flags.DEFINE_string(
    "run_tag",
    "deepseek_v41_rollouts_top10k_v8_pass1",
    "Output folder name under --out_root on CNS.",
)
flags.DEFINE_string(
    "batch_label",
    "top10k_v8_pass1",
    "Batch label written to the E031 `batch` column.",
)
flags.DEFINE_string(
    "candidates_jsonl",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/top_10k_deepseek_candidates.jsonl",
    "CNS path to the Top 10,000 candidate tasks JSONL.",
)
flags.DEFINE_string(
    "eval_contamination_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "CNS path to official ARC-AGI-2 120 evaluation tasks for contamination filtering.",
)
flags.DEFINE_string(
    "out_root",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline",
    "CNS output root directory.",
)
flags.DEFINE_integer(
    "limit_tasks", 0, "Optional limit on tasks per shard (0 = all)."
)
flags.DEFINE_integer(
    "num_attempts",
    1,
    "Pass@1 rollout generation (1 trajectory per candidate puzzle).",
)
flags.DEFINE_integer(
    "pair_batch_size",
    2,
    "Candidate puzzles per batch wave per shard (2 puzzles x 1 attempt = 2 slots = 12 rows <= 16-row single MXU tile).",
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
    "parquet_flush_every_batches",
    5,
    "Write atomic Parquet snapshots every N batches (and at shard completion).",
)
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
        "sft-rollouts",
        "top-10k",
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
        autopilot_params=xm_abc.AutopilotParams(
            enabled=FLAGS.enable_xm_autopilot
        ),
    )
    extra = ["--norun_validations", "--nocheck_visibility", "--dynamic_mode=off"]
    if xm_flags.EXPERIMENTAL_SKIP_PYTYPE_CHECK.value:
      extra.append("--define=PYTYPE=FALSE")
    bazel_args = xm_abc.bazel_args.for_requirements(reqs) + tuple(extra)
    [executable] = experiment.package([
        xm.bazel_binary(
            label="//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:deepseek_v41_top10k_rollouts",
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
    out_dir = f"{FLAGS.out_root}/{FLAGS.run_tag}"
    common = [
        "--engine_backend=ds41_full_precision",
        f"--candidates_jsonl={FLAGS.candidates_jsonl}",
        f"--eval_contamination_dir={FLAGS.eval_contamination_dir}",
        f"--output_dir={out_dir}",
        f"--exp_name={exp_name}",
        f"--batch_label={FLAGS.batch_label}",
        f"--num_attempts={FLAGS.num_attempts}",
        f"--pair_batch_size={FLAGS.pair_batch_size}",
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
        f"--parquet_flush_every_batches={FLAGS.parquet_flush_every_batches}",
        f"--limit_tasks={FLAGS.limit_tasks}",
        f"--num_shards={FLAGS.num_shards}",
    ]
    for s_idx in range(FLAGS.num_shards):
      wu_args = common + [f"--shard_idx={s_idx}"]
      experiment.add(
          xm.Job(
              executable=executable,
              executor=executor,
              args=wu_args,
              env_vars=env_vars,
          ),
          identity=f"ds41_top10k_v8_s{s_idx:03d}of{FLAGS.num_shards:03d}",
      )
    print(
        f"Launched {FLAGS.num_shards} TPU slices (ghostfish=2x2x1, 8 TensorCores each = "
        f"{FLAGS.num_shards * 8} TensorCores total): http://xid/{experiment.experiment_id}",
        flush=True,
    )
    print(f"CNS Output Directory: {out_dir}", flush=True)


if __name__ == "__main__":
  app.run(main)
