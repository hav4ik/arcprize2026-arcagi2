load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "arc_dsl_and_summary",
    srcs = ["arc_dsl_and_summary.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_library(
    name = "sandbox_verifier",
    srcs = ["sandbox_verifier.py"],
    deps = [
        ":arc_dsl_and_summary",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_library(
    name = "prompts_and_selector",
    srcs = ["prompts_and_selector.py"],
    deps = [
        ":arc_dsl_and_summary",
        ":sandbox_verifier",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "pilot2_eval_lib",
    srcs = ["pilot2_eval.py"],
    deps = [
        ":prompts_and_selector",
        ":sandbox_verifier",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/tpu_inference",
        "//third_party/py/vllm",
    ],
)

py_test(
    name = "pilot2_test",
    srcs = ["pilot2_test.py"],
    deps = [
        ":arc_dsl_and_summary",
        ":pilot2_eval_lib",
        ":prompts_and_selector",
        ":sandbox_verifier",
        "//testing/pybase",
    ],
)

py_binary(
    name = "pilot2_eval",
    srcs = ["pilot2_eval.py"],
    deps = [
        ":prompts_and_selector",
        ":sandbox_verifier",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/tpu_inference",
        "//third_party/py/vllm",
    ],
)

py_binary(
    name = "launch_pilot2_xm",
    srcs = ["launch_pilot2_xm.py"],
    deps = [
        ":prompts_and_selector",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
        "//third_party/py/xmanager/xm_abc:restricted_borg_overrides",
    ],
)
