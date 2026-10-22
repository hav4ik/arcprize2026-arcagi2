"""Standalone XManager launcher for the 6 controlled pilot_2 ARC-AGI-2 experiments on TPU."""

from __future__ import annotations

import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_flags
from xmanager.contrib.internal import requirements_flag
from xmanager.xm_abc import restricted_borg_overrides

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector


FLAGS = flags.FLAGS

_PLATFORM = requirements_flag.DEFINE_requirements(
    "platform",
    "viperfish=2x2x1",
    "TPU platform/topology per work unit (viperfish=2x2x1 = 4 TPU v6e chips).",
)
flags.DEFINE_string(
    "experiment_title",
    "arc_agi2_pilot2_controlled_6hyp",
    "XManager experiment title.",
)
flags.DEFINE_list(
    "exp_ids",
    list(prompts_and_selector.PILOT2_HYPOTHESES.keys()),
    "Comma-separated list of pilot_2 hypothesis IDs to launch.",
)
flags.DEFINE_integer(
    "num_shards",
    6,
    "Number of parallel TPU shards per hypothesis across the 167 eval pairs (~28 pairs/shard).",
)
flags.DEFINE_string(
    "cns_output_root",
    prompts_and_selector.FROZEN_BASELINE_CONFIG["cns_output_root"],
    "CNS root directory where each hypothesis writes its reasoning and code traces.",
)
flags.DEFINE_integer("cpu", 32, "Host CPUs per work unit.")
flags.DEFINE_integer("ram_gb", 128, "Host RAM (GiB) per work unit.")
flags.DEFINE_string("location", "viglobal", "Borg location.")
flags.DEFINE_boolean(
    "enable_xm_autopilot",
    True,
    "Enable XManager Autopilot (mandatory in genai-lti-lh-tasks).",
)
flags.DEFINE_list(
    "wim_tags",
    ["data", "enable_xm_autopilot=true"],
    "WIM tags added to experiment and work units.",
)
flags.DEFINE_integer(
    "max_tasks",
    0,
    "Optional cap on number of tasks (0 = full 120 tasks / 167 test pairs).",
)


def main(argv: list[str]) -> None:
  del argv
  for exp_id in FLAGS.exp_ids:
    if exp_id not in prompts_and_selector.PILOT2_HYPOTHESES:
      raise ValueError(
          f"Unknown hypothesis {exp_id!r}. Valid IDs: "
          f"{sorted(prompts_and_selector.PILOT2_HYPOTHESES.keys())}"
      )
    prompts_and_selector.PILOT2_HYPOTHESES[exp_id].validate()

  with xm_abc.create_experiment(
      experiment_title=FLAGS.experiment_title,
      settings=xm_abc.ExecutionSettings(admin_users=[flags.FLAGS.borguser]),
  ) as experiment:
    experiment.context.annotations.add_tags(
        "arc-agi-2", "pilot-2-2026-10-21", *FLAGS.wim_tags
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
            max_task_failures=3, max_per_task_failures=3
        ),
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
            label="//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:pilot2_eval",
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

    total_wus = 0
    for exp_id in FLAGS.exp_ids:
      exp_out_dir = os.path.join(FLAGS.cns_output_root, exp_id)
      for shard_idx in range(FLAGS.num_shards):
        args = [
            f"--exp_id={exp_id}",
            f"--output_dir={exp_out_dir}",
            f"--shard_index={shard_idx}",
            f"--num_shards={FLAGS.num_shards}",
            f"--max_tasks={FLAGS.max_tasks}",
        ]
        experiment.add(
            xm.Job(
                executable=executable,
                executor=executor,
                env_vars=env_vars,
                args=args,
            ),
            identity=f"{exp_id}_s{shard_idx}",
        )
        total_wus += 1

    print(
        f"[pilot_2] Launched {total_wus} work units ({len(FLAGS.exp_ids)} hypotheses x "
        f"{FLAGS.num_shards} shards): http://xid/{experiment.experiment_id}",
        flush=True,
    )


if __name__ == "__main__":
  app.run(main)
