load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

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

py_library(
    name = "ds4_full_precision_lib",
    srcs = [
        "decode_megakernel.py",
        "ds4_encoding.py",
        "ds4_engine_fns.py",
        "ds4_layout.py",
        "ds4_load.py",
        "ds4_programs.py",
        "ds4_ref.py",
    ],
    deps = [
        "//pyglib:gfile",
        "//third_party/py/absl/logging",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:pallas",
        "//third_party/py/jax/experimental:pallas_tpu",
        "//third_party/py/ml_dtypes",
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

py_binary(
    name = "deepseek_v41_tpu_vllm_traces",
    srcs = [
        "deepseek_v41_tpu_vllm_traces.py",
        "qwen_tpu_vllm_eval.py",
    ],
    deps = [
        ":ds4_full_precision_lib",
        ":prompts_and_voting",
        ":sandbox",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorstore",
        "//third_party/py/tokenizers",
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

py_binary(
    name = "deepseek_v41_arc2_eval",
    srcs = [
        "deepseek_v41_arc2_eval.py",
        "deepseek_v41_tpu_vllm_traces.py",
        "qwen_tpu_vllm_eval.py",
    ],
    deps = [
        ":ds4_full_precision_lib",
        ":prompts_and_voting",
        ":sandbox",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorstore",
        "//third_party/py/tokenizers",
        "//third_party/py/tpu_inference",
        "//third_party/py/tqdm",
        "//third_party/py/vllm",
    ],
)

py_binary(
    name = "launch_deepseek_v41_traces_xm",
    srcs = ["launch_deepseek_v41_traces_xm.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)

py_binary(
    name = "launch_deepseek_v41_arc2_eval_xm",
    srcs = ["launch_deepseek_v41_arc2_eval_xm.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)

py_binary(
    name = "deepseek_v41_top10k_rollouts",
    srcs = [
        "deepseek_v41_arc2_eval.py",
        "deepseek_v41_top10k_rollouts.py",
        "deepseek_v41_tpu_vllm_traces.py",
        "qwen_tpu_vllm_eval.py",
    ],
    deps = [
        ":ds4_full_precision_lib",
        ":prompts_and_voting",
        ":sandbox",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/pyarrow",
        "//third_party/py/scipy",
        "//third_party/py/tensorstore",
        "//third_party/py/tokenizers",
        "//third_party/py/tpu_inference",
        "//third_party/py/tqdm",
        "//third_party/py/vllm",
    ],
)

py_binary(
    name = "launch_deepseek_v41_top10k_rollouts_xm",
    srcs = ["launch_deepseek_v41_top10k_rollouts_xm.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)

py_test(
    name = "top10k_rollouts_test",
    size = "medium",
    srcs = [
        "deepseek_v41_arc2_eval.py",
        "deepseek_v41_top10k_rollouts.py",
        "deepseek_v41_tpu_vllm_traces.py",
        "qwen_tpu_vllm_eval.py",
        "top10k_rollouts_test.py",
    ],
    deps = [
        ":ds4_full_precision_lib",
        ":prompts_and_voting",
        ":sandbox",
        "//experimental/users/arazdai/kaggle/sft:data_pipeline",
        "//pyglib:gfile",
        "//testing/pybase",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/pyarrow",
        "//third_party/py/scipy",
        "//third_party/py/tensorstore",
        "//third_party/py/tokenizers",
        "//third_party/py/tpu_inference",
        "//third_party/py/tqdm",
        "//third_party/py/vllm",
    ],
)
