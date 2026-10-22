"""XManager launcher for the ARC-AGI-2 round-2 controlled study.

One experiment, one package step, one Borg work unit per (config, shard).

  /google/bin/releases/xmanager/cli/xmanager.par launch \
    //experimental/users/arazdai/kaggle/arc_agi_2/tpu_vllm:launch_round2_xm -- \
    --noxm_monitor_on_launch --xm_resource_pool=research-mm \
    --xm_resource_alloc=group:research-mm/ml-gemini-code-bridging-gap \
    --configs=exp0_base16,exp1_turns40,exp2_autochk40,exp3_mixed,exp4_audit40,exp5_views16 \
    --num_shards=6 --run_tag=r2
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
    "platform", "ghostfish=2x2x1", "TPU platform/topology per work unit."
)
flags.DEFINE_string("experiment_title", "arc_agi2_round2", "Experiment title.")
flags.DEFINE_list(
    "configs",
    ["exp0_base16", "exp1_turns40", "exp2_autochk40", "exp3_mixed", "exp4_audit40", "exp5_views16"],
    "Round-2 configurations to launch.",
)
flags.DEFINE_integer("num_shards", 6, "Shards (work units) per config.")
flags.DEFINE_string("run_tag", "r2", "Run tag (output subdirectory suffix).")
flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "CNS model path.",
)
flags.DEFINE_string(
    "out_root", "/cns/lu-d/home/arazdai/kaggle/arc_round2", "Output root."
)
flags.DEFINE_integer("attempts_in_flight", 64, "Attempts in flight per worker.")
flags.DEFINE_integer("max_num_seqs", 64, "vLLM max concurrent sequences.")
flags.DEFINE_integer("total_token_budget", 48000, "Generated-token cap per attempt.")
flags.DEFINE_integer("max_model_len", 131072, "vLLM max_model_len.")
flags.DEFINE_integer("cpu", 32, "Host CPUs.")
flags.DEFINE_integer("ram_gb", 128, "Host RAM GB.")
flags.DEFINE_string("location", "viglobal", "Borg location.")
flags.DEFINE_multi_string("extra_arg", [], "Extra args for every job (e.g. smoke-test limits).")
flags.DEFINE_boolean(
    "enable_xm_autopilot",
    True,
    "Enable XManager Autopilot (mandatory in genai-lti-lh-tasks).",
)
flags.DEFINE_list(
    "wim_tags",
    ["data", "enable_xm_autopilot=true"],
    "WIM tags added to experiment and work units (e.g. 'data' for 200x floor multiplier).",
)


def main(argv):
  del argv
  with xm_abc.create_experiment(
      experiment_title=FLAGS.experiment_title,
      settings=xm_abc.ExecutionSettings(admin_users=[flags.FLAGS.borguser]),
  ) as experiment:
    experiment.context.annotations.add_tags(
        "arc-agi-2", "round2-2026-10-20", FLAGS.run_tag, *FLAGS.wim_tags
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
        # Resume-safe runner: allow a few restarts (completed pairs are skipped).
        scheduling=xm_abc.BorgScheduling(max_task_failures=3, max_per_task_failures=3),
        borg_overrides=restricted_borg_overrides.RESTRICTED_BorgOverrides(
            health_check_grace_period="3600s"
        ),
        autopilot_params=xm_abc.AutopilotParams(enabled=FLAGS.enable_xm_autopilot),
    )
    extra = ["--norun_validations", "--nocheck_visibility", "--dynamic_mode=off"]
    if xm_flags.EXPERIMENTAL_SKIP_PYTYPE_CHECK.value:
      extra.append("--define=PYTYPE=FALSE")
    bazel_args = xm_abc.bazel_args.for_requirements(reqs) + tuple(extra)
    [executable] = experiment.package([
        xm.bazel_binary(
            label="//experimental/users/arazdai/kaggle/arc_agi_2/tpu_vllm:round2_eval",
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
    common = [
        "--model_path", FLAGS.model_path,
        f"--total_token_budget={FLAGS.total_token_budget}",
        f"--max_model_len={FLAGS.max_model_len}",
        f"--max_num_seqs={FLAGS.max_num_seqs}",
        f"--attempts_in_flight={FLAGS.attempts_in_flight}",
        f"--num_shards={FLAGS.num_shards}",
    ] + list(FLAGS.extra_arg)
    n = 0
    for cfg in FLAGS.configs:
      for s in range(FLAGS.num_shards):
        args = common + [
            f"--r2_config={cfg}",
            f"--shard_idx={s}",
            f"--output_dir={FLAGS.out_root}_{FLAGS.run_tag}/{cfg}",
        ]
        experiment.add(
            xm.Job(executable=executable, executor=executor, env_vars=env_vars, args=args),
            identity=f"{cfg}_s{s}",
        )
        n += 1
    print(
        f"Launched {n} work units ({len(FLAGS.configs)} configs x {FLAGS.num_shards} shards):"
        f" http://xid/{experiment.experiment_id}"
    )


if __name__ == "__main__":
  app.run(main)
