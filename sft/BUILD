load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "data_pipeline",
    srcs = ["data_pipeline.py"],
    deps = [
        "//pyglib:gfile",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pyarrow",
        "//third_party/py/transformers:v5",
    ],
)

py_binary(
    name = "audit_and_prepare_data",
    srcs = ["audit_and_prepare_data.py"],
    deps = [
        ":data_pipeline",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pyarrow",
        "//third_party/py/transformers:v5",
    ],
)

py_library(
    name = "fast_deltanet",
    srcs = ["fast_deltanet.py"],
    deps = [
        "//third_party/py/torch:pytorch",
        "//third_party/py/transformers:v5",
        "//third_party/py/triton",
    ],
)

py_library(
    name = "train_sft_lib",
    srcs = ["train_sft.py"],
    deps = [
        ":data_pipeline",
        ":fast_deltanet",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/accelerate",
        "//third_party/py/numpy",
        "//third_party/py/peft",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
        "//third_party/py/transformers:v5",
    ],
)

py_binary(
    name = "train_sft",
    srcs = ["train_sft.py"],
    deps = [
        ":data_pipeline",
        ":fast_deltanet",
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/accelerate",
        "//third_party/py/einops",
        "//third_party/py/flash_attn",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
        "//third_party/py/peft",
        "//third_party/py/safetensors:torch",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        "//third_party/py/transformers:v5",
        "//third_party/py/triton",
    ],
)

py_library(
    name = "train_full_sft_lib",
    srcs = ["train_full_sft.py"],
    deps = [
        ":data_pipeline",
        ":export_and_quantize_lib",
        ":fast_deltanet",
        ":train_sft_lib",
        "//pyglib:gfile",
        "//pyglib/contrib/g3_multiprocessing",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
        "//third_party/py/transformers:v5",
    ],
)

py_binary(
    name = "train_full_sft",
    srcs = ["train_full_sft.py"],
    deps = [
        ":data_pipeline",
        ":export_and_quantize_lib",
        ":fast_deltanet",
        ":train_sft_lib",
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//pyglib/contrib/g3_multiprocessing",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/accelerate",
        "//third_party/py/einops",
        "//third_party/py/flash_attn",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
        "//third_party/py/safetensors",
        "//third_party/py/safetensors:torch",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        "//third_party/py/transformers:v5",
        "//third_party/py/triton",
    ],
)

py_library(
    name = "export_and_quantize_lib",
    srcs = ["export_and_quantize.py"],
    deps = [
        "//pyglib:gfile",
        "//tech/file/proto:types_py_pb2",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/safetensors",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "export_and_quantize",
    srcs = ["export_and_quantize.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//tech/file/proto:types_py_pb2",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
        "//third_party/py/peft",
        "//third_party/py/safetensors",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
        "//third_party/py/tqdm",
        "//third_party/py/transformers:v5",
    ],
)

py_binary(
    name = "launch_sft_b200_xm",
    srcs = ["launch_sft_b200_xm.py"],
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
    name = "launch_full_sft_b200_xm",
    srcs = ["launch_full_sft_b200_xm.py"],
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
    name = "sft_test",
    size = "medium",
    srcs = ["sft_test.py"],
    deps = [
        ":data_pipeline",
        ":export_and_quantize_lib",
        ":fast_deltanet",
        ":train_full_sft_lib",
        ":train_sft_lib",
        "//pyglib:gfile",
        "//testing/pybase",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/peft",
        "//third_party/py/pyarrow",
        "//third_party/py/torch:pytorch",
        "//third_party/py/transformers:v5",
    ],
)
