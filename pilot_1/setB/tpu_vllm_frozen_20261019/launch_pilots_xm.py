"""Multi-arm, multi-shard XManager launcher for the ARC-AGI-2 TPU pilots.

One experiment, one package step, one Borg job per (arm, shard). Example:

  /google/bin/releases/xmanager/cli/xmanager.par launch \
    //experimental/users/arazdai/kaggle/arc_agi_2/tpu_vllm_frozen_20261019:launch_pilots_xm -- \
    --noxm_monitor_on_launch --xm_resource_pool=research-mm \
    --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks \
    --arms=b0,p1,p2,p3 --num_shards=6 --run_tag=fast6
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
flags.DEFINE_string("experiment_title", "arc_agi2_pilots", "Experiment title.")
flags.DEFINE_list("arms", ["b0", "p1", "p2", "p3"], "Arms to launch.")
flags.DEFINE_integer("num_shards", 6, "Shards (work units) per arm.")
flags.DEFINE_string("run_tag", "fast6", "Suffix for exp_name/output_dir.")
flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "CNS model path.",
)
flags.DEFINE_string(
    "split_file",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/dev_holdout_split.json",
    "Frozen DEV/HOLDOUT split file.",
)
flags.DEFINE_string("split_key", "dev", "Key in split file.")
flags.DEFINE_string(
    "out_root", "/cns/lu-d/home/arazdai/kaggle/arc_qwen_tpu_runs", "Output root."
)
flags.DEFINE_integer("pair_batch_size", 14, "Pairs per lockstep wave per shard.")
flags.DEFINE_integer("max_num_seqs", 28, "vLLM max concurrent sequences.")
flags.DEFINE_integer("total_token_budget", 48000, "Hard output tokens/attempt.")
flags.DEFINE_integer("soft_token_target", 24000, "Soft output tokens/attempt.")
flags.DEFINE_integer("max_model_len", 114688, "vLLM max_model_len.")
flags.DEFINE_integer("cpu", 32, "Host CPUs.")
flags.DEFINE_integer("ram_gb", 128, "Host RAM GB.")
flags.DEFINE_string("location", "viglobal", "Borg location.")
flags.DEFINE_multi_string("extra_vllm_arg", [], "Extra args for every job.")

ARM_ARGS = {
    "b0": ["--auto_check_on_define=false", "--commit_after_tokens=0", "--max_turns=16"],
    "p1": ["--auto_check_on_define=true", "--commit_after_tokens=0", "--max_turns=16"],
    "p2": ["--auto_check_on_define=true", "--commit_after_tokens=12000", "--max_turns=16"],
    "p3": ["--auto_check_on_define=true", "--commit_after_tokens=0", "--max_turns=40"],
}


def main(argv):
  del argv
  with xm_abc.create_experiment(
      experiment_title=FLAGS.experiment_title,
      settings=xm_abc.ExecutionSettings(admin_users=[flags.FLAGS.borguser]),
  ) as experiment:
    experiment.context.annotations.add_tags("arc-agi-2", "pilot-2026-10-19", FLAGS.run_tag)
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
        scheduling=xm_abc.BorgScheduling(max_task_failures=0, max_per_task_failures=0),
        borg_overrides=restricted_borg_overrides.RESTRICTED_BorgOverrides(
            health_check_grace_period="3600s"
        ),
        # Autopilot is mandatory in genai-lti-lh-tasks (WIM pruner deletes
        # non-autopilot workloads). Keep CPU/RAM fixed for the vLLM host.
        autopilot_params=xm_abc.AutopilotParams(
            enabled=True, fixed_cpu=True, fixed_ram=True, fixed_replicas=True
        ),
    )
    extra = ["--norun_validations", "--nocheck_visibility", "--dynamic_mode=off"]
    if xm_flags.EXPERIMENTAL_SKIP_PYTYPE_CHECK.value:
      extra.append("--define=PYTYPE=FALSE")
    bazel_args = xm_abc.bazel_args.for_requirements(reqs) + tuple(extra)
    [executable] = experiment.package([
        xm.bazel_binary(
            label="//experimental/users/arazdai/kaggle/arc_agi_2/tpu_vllm_frozen_20261019:qwen_tpu_vllm_eval",
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
        "--mode=tools", "--num_attempts=2",
        f"--total_token_budget={FLAGS.total_token_budget}",
        f"--soft_token_target={FLAGS.soft_token_target}",
        "--turn_think_cap=8000", f"--max_model_len={FLAGS.max_model_len}",
        "--audit_on_submit=false", "--freeze_first_verified=true",
        "--keep_candidates=true", "--save_transcripts=true",
        "--submission_policy=primary_first",
        f"--max_num_seqs={FLAGS.max_num_seqs}",
        f"--pair_batch_size={FLAGS.pair_batch_size}",
        "--attempts_per_wave=2",
        f"--task_ids_file={FLAGS.split_file}", f"--task_ids_key={FLAGS.split_key}",
        f"--num_shards={FLAGS.num_shards}",
    ] + list(FLAGS.extra_vllm_arg)
    n = 0
    for arm in FLAGS.arms:
      exp_name = f"pilot_{arm}_dev60_n2_48k_{FLAGS.run_tag}"
      for s in range(FLAGS.num_shards):
        args = common + ARM_ARGS[arm] + [
            f"--exp_name={exp_name}",
            f"--shard_idx={s}",
            f"--output_dir={FLAGS.out_root}/{exp_name}",
        ]
        experiment.add(
            xm.Job(executable=executable, executor=executor, env_vars=env_vars, args=args),
            identity=f"{arm}_s{s}",
        )
        n += 1
    print(f"Launched {n} work units ({len(FLAGS.arms)} arms x {FLAGS.num_shards} shards): http://xid/{experiment.experiment_id}")


if __name__ == "__main__":
  app.run(main)
