load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "sandbox",
    srcs = ["sandbox.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_library(
    name = "prompts_and_voting",
    srcs = ["prompts_and_voting.py"],
    deps = [
        ":sandbox",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "qwen_tpu_vllm_eval",
    srcs = ["qwen_tpu_vllm_eval.py"],
    deps = [
        ":prompts_and_voting",
        ":sandbox",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/tpu_inference",
        "//third_party/py/tqdm",
        "//third_party/py/vllm",
    ],
)

# One XManager experiment with one work unit per (arm, shard) for the pilots.
py_binary(
    name = "launch_pilots_xm",
    srcs = ["launch_pilots_xm.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)
