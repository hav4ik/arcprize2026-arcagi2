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
    name = "havu_auto_repair",
    srcs = ["havu_auto_repair.py"],
    data = ["auto_repair_names.json"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

# Stdlib-only pre-TPU fork server; must stay dependency-free so the zygote is tiny.
py_library(
    name = "havu_sandbox_zygote",
    srcs = ["havu_sandbox_zygote.py"],
)

py_library(
    name = "havu_b32_harness",
    srcs = ["havu_b32_harness.py"],
    deps = [
        ":havu_auto_repair",
        ":havu_sandbox_zygote",
        ":prompts_and_selector",
        "//pyglib:gfile",
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/vllm",
    ],
)

py_library(
    name = "pilot2_eval_lib",
    srcs = ["pilot2_eval.py"],
    deps = [
        ":havu_b32_harness",
        ":havu_sandbox_zygote",
        ":prompts_and_selector",
        ":sandbox_verifier",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:prompts_and_voting",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:sandbox",
        "//experimental/users/arazdai/kaggle/sft:export_and_quantize_lib",
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
        ":havu_auto_repair",
        ":havu_b32_harness",
        ":pilot2_eval_lib",
        ":prompts_and_selector",
        ":sandbox_verifier",
        "//testing/pybase",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "havu_b32_harness_test",
    size = "medium",
    srcs = ["havu_b32_harness_test.py"],
    deps = [
        ":havu_b32_harness",
        ":havu_sandbox_zygote",
        ":prompts_and_selector",
        "//testing/pybase",
    ],
)

py_binary(
    name = "pilot2_eval",
    srcs = ["pilot2_eval.py"],
    deps = [
        ":havu_b32_harness",
        ":havu_sandbox_zygote",
        ":prompts_and_selector",
        ":sandbox_verifier",
        "//experimental/users/arazdai/kaggle/sft:export_and_quantize_lib",
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
