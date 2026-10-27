"""Multi-TPU XManager Launcher for DeepSeek-V4.1-Flash-1017 Full-Precision Evaluation on ARC-AGI-2 Public Eval (120 Tasks, 167 Pairs).

Replicates collaborator Vu (`havu`)'s `E030` (`sub80s40` + `effort_header="minimal"`, `74.5%` Pass@1, `84.9%` Pass@2)
and his Kaggle notebook (`arc2-qwen38-e032-sft-minimal.ipynb` / `havu/evals/arc_harness/executor.py`) while keeping
100% Full-Precision (`BF16 -> FP32` MXU accumulation) on `DeepSeek-V4.1-Flash-1017`:
  - Platform: `ghostfish=2x2x1` (8 TensorCores per TPU v7x slice)
  - Engine: `ds41_full_precision` (FP8 E4M3FN & MXFP4 E2M1 dequantized by E8M0 scales to BF16, IEEE-754 FP32 MXU accumulation,
    Two-Phase CED + Bounded SWA Replay Prefill, 196.6B Engram conditional memory at L1/L14, 3-layer DSpark 1+5 speculative decode)
  - Sandbox: Isolated persistent child-process IPython session (`HavuSubprocessSession` / `LocalExecutor(style="ipython")`)
    with `IPython.lib.pretty`, IPython plain tracebacks with `<ipython-input-N-hash>` and failing source lines in `stdout`,
    `<output cut>` at `16,000` chars, `RLIMIT_AS=4096MB`, and two-stage `SIGINT -> SIGKILL` timeout recovery (30s timeout)
  - Sampling: Stochastic target-side Gumbel-max sampling (`temperature=1.0, top_k=20, top_p=0.95`) so both Attempt 0 and
    Attempt 1 run Vu's #1 `E030` config with independent random seeds
  - Prompt & Encoding: `ds4_encoding.encode_messages` with `TOOLS_TEMPLATE` (`run_python` and `submit_answer`),
    `ARC Prize` prompt verbatim + `lean2sub` suffix, `HAVU_EFFORT_HEADER_MINIMAL` with `reasoning_effort=0` (`low` adds no
    conflicting numeric header), `--cap-nudge` at `max_turn_tokens=16000`, non-destructive transcript compaction at `36,000` tokens,
    and 3-tier fallback (`answer` -> `solve(test_input)` -> `salvage_answer(session)`) + `train_check`
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
    "arc_agi2_eval_deepseek_v41_flash_1017_fp",
    "XManager experiment title.",
)
flags.DEFINE_integer(
    "num_shards",
    16,
    "Number of parallel TPU slices (work units) across the 167 ARC-AGI-2 eval pairs (~10 pairs/shard).",
)
flags.DEFINE_string(
    "run_tag",
    "ds41_fp_havu_e030_sub80s40_lossless_v8",
    "Run tag for experiment output directory.",
)
flags.DEFINE_string(
    "engine_backend",
    "ds41_full_precision",
    "Backend: 'ds41_full_precision', 'auto', or 'vllm'.",
)
flags.DEFINE_string(
    "cns_data_path",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data",
    "CNS path to ARC-AGI-2 dataset.",
)
flags.DEFINE_string(
    "out_root",
    "/cns/lu-d/home/arazdai/kaggle/arc_deepseek_v41_eval",
    "CNS output root directory for predictions, reasoning traces, and summaries.",
)
flags.DEFINE_integer(
    "limit_tasks", 0, "Optional limit on tasks (0 = all 120 tasks / 167 pairs)."
)
flags.DEFINE_integer(
    "num_attempts", 2, "Trajectories per evaluation test pair."
)
flags.DEFINE_integer(
    "attempts_per_wave",
    2,
    "Attempts per wave (2 = run both Attempt 0 and Attempt 1 of Vu's E030 config in parallel with independent seeds).",
)
flags.DEFINE_integer(
    "pair_batch_size",
    1,
    "Test pairs per batch wave per shard (1 pair x 2 attempts = 2 slots = 12 rows <= 16-row single MXU tile).",
)
flags.DEFINE_float(
    "temperature",
    1.0,
    "Stochastic sampling temperature (matching Vu E029/E030 temperature=1.0).",
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
    "Maximum turns per trajectory (Vu E029/E030 sub80s40 = 40).",
)
flags.DEFINE_integer(
    "max_turn_tokens",
    64000,
    "Max tokens per single assistant turn (Vu E029/E030 sub80s40 = 64000).",
)
flags.DEFINE_integer(
    "max_reasoning_tokens",
    0,
    "Optional separate cap on <think> tokens (0 = disabled, governed by total_token_budget=80000 matching Vu E030 sub80s40).",
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
    "Hard total output tokens per trajectory across turns (Vu E029/E030 sub80s40 = 80000).",
)
flags.DEFINE_integer(
    "soft_token_target",
    40000,
    "Soft output token target per trajectory (Vu E029/E030 sub80s40 = 40000).",
)
flags.DEFINE_integer(
    "turns_note_threshold",
    5,
    "Warn when N or fewer turns remain (Vu E029/E030 --turns-note 5).",
)
flags.DEFINE_boolean(
    "token_exact",
    True,
    "Preserve exact sampled token_ids across turns (Vu E043 --token-exact).",
)
flags.DEFINE_boolean(
    "rescue_code",
    True,
    "Rescue Python code written outside DSML tool tags (Vu E043 --rescue-code).",
)
flags.DEFINE_float(
    "sandbox_timeout_s",
    30.0,
    "Per-cell timeout in seconds for LocalExecutor(style='ipython') (matching Vu's 30s timeout).",
)
flags.DEFINE_integer(
    "reasoning_effort",
    0,
    "DeepSeek-V4.1 numeric effort prefix (0 = 'none', matching Vu's low effort so HAVU_EFFORT_HEADER_MINIMAL is not duplicated).",
)
flags.DEFINE_boolean(
    "drop_prior_turn_thinking",
    False,
    "Keep full <think>...</think> across multi-turn tool calls until compact_at_tokens.",
)
flags.DEFINE_boolean(
    "compact_grid_prompts",
    False,
    "Format grids with standard json.dumps(grid) matching Vu ARC Prize prompt.",
)
flags.DEFINE_boolean(
    "shard_by_task",
    True,
    "Assign all test pairs of the same task_id to the same shard.",
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
        "arc-agi-2-eval",
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
            label="//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:deepseek_v41_arc2_eval",
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
        f"--engine_backend={FLAGS.engine_backend}",
        f"--cns_data_path={FLAGS.cns_data_path}",
        f"--output_dir={out_dir}",
        f"--exp_name={exp_name}",
        f"--num_attempts={FLAGS.num_attempts}",
        f"--attempts_per_wave={FLAGS.attempts_per_wave}",
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
        f"--reasoning_effort={FLAGS.reasoning_effort}",
        f"--drop_prior_turn_thinking={str(FLAGS.drop_prior_turn_thinking).lower()}",
        f"--compact_grid_prompts={str(FLAGS.compact_grid_prompts).lower()}",
        f"--shard_by_task={str(FLAGS.shard_by_task).lower()}",
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
          identity=f"ds41_eval_s{s:02d}",
      )
    print(
        f"Launched {FLAGS.num_shards} TPU slices (ghostfish=2x2x1, 8 TensorCores each = "
        f"{FLAGS.num_shards * 8} TensorCores total): http://xid/{experiment.experiment_id}",
        flush=True,
    )
    print(f"CNS Output Directory: {out_dir}", flush=True)


if __name__ == "__main__":
  app.run(main)
