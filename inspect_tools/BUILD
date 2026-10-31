load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

py_binary(
    name = "inspect_bagz",
    srcs = ["inspect_bagz.py"],
    deps = [
        "//learning/gemini/proto:gemini_example_py_pb2",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/bagz",
    ],
)

py_binary(
    name = "compute_evicted_prefix_lengths",
    srcs = ["compute_evicted_prefix_lengths.py"],
    deps = [
        "//learning/gemini/proto:gemini_example_py_pb2",
        "//learning/gemini/rex/projects/logs_based_iteration/enrichment/behavior_judges:example_rendering",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/bagz",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "check_trainable",
    srcs = ["check_trainable.py"],
    deps = [
        "//learning/gemini/proto:gemini_example_py_pb2",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/bagz",
    ],
)

py_binary(
    name = "export_top100",
    srcs = ["export_top100.py"],
    deps = [
        "//learning/gemini/proto:gemini_example_py_pb2",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/bagz",
    ],
)

py_binary(
    name = "check_hydra_access",
    srcs = ["check_hydra_access.py"],
    deps = [
        "//assistant/lamda/bard/data/hydra/client:hydra_client",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)

py_binary(
    name = "diff_safetensors",
    srcs = ["diff_safetensors.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "lora_delta_vs_2bit_step",
    srcs = ["lora_delta_vs_2bit_step.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "lora_flip_census",
    srcs = ["lora_flip_census.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "safetensors_param_census",
    srcs = ["safetensors_param_census.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)

py_binary(
    name = "shared_expert_update_survival",
    srcs = ["shared_expert_update_survival.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/torch:pytorch",
    ],
)
