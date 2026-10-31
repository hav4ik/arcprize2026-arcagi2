load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "dsv41_image_tokens",
    srcs = ["dsv41_image_tokens.py"],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "data_pipeline",
    srcs = ["data_pipeline.py"],
    deps = [
        ":dsv41_image_tokens",
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
        ":dsv41_image_tokens",
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

py_library(
    name = "evaluate_difficulty_argon_lib",
    srcs = ["evaluate_difficulty_argon.py"],
    deps = [
        ":data_pipeline",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "evaluate_difficulty_argon",
    srcs = ["evaluate_difficulty_argon.py"],
    deps = [
        ":data_pipeline",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "inspect_and_download_chankhavu_e128",
    srcs = ["inspect_and_download_chankhavu_e128.py"],
    deps = [
        ":export_and_quantize_lib",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "download_hf_to_cns",
    srcs = ["download_hf_to_cns.py"],
    deps = [
        ":export_and_quantize_lib",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/huggingface_hub",
        "//third_party/py/requests",
    ],
)

py_binary(
    name = "inspect_e065_rollouts",
    srcs = ["inspect_e065_rollouts.py"],
    deps = [
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pyarrow",
    ],
)

py_library(
    name = "audit_and_split_e065_rollouts_lib",
    srcs = ["audit_and_split_e065_rollouts.py"],
    deps = [
        ":data_pipeline",
        ":export_and_quantize_lib",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/pyarrow",
    ],
)

py_binary(
    name = "audit_and_split_e065_rollouts",
    srcs = ["audit_and_split_e065_rollouts.py"],
    deps = [
        ":data_pipeline",
        ":export_and_quantize_lib",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/pyarrow",
    ],
)

py_library(
    name = "rank_e065_difficulty_gemini38_flash_lib",
    srcs = ["rank_e065_difficulty_gemini38_flash.py"],
    deps = [
        ":export_and_quantize_lib",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/pyarrow",
    ],
)

py_binary(
    name = "rank_e065_difficulty_gemini38_flash",
    srcs = ["rank_e065_difficulty_gemini38_flash.py"],
    deps = [
        ":export_and_quantize_lib",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/pyarrow",
    ],
)

py_library(
    name = "dsv41_sft_model",
    srcs = ["dsv41_sft_model.py"],
    deps = [
        ":export_and_quantize_lib",
        ":train_sft_lib",
        "//pyglib:gfile",
        "//third_party/py/numpy",
        "//third_party/py/safetensors",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "train_dsv41_sft_lib",
    srcs = ["train_dsv41_sft.py"],
    deps = [
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":export_and_quantize_lib",
        ":train_full_sft_lib",
        ":train_sft_lib",
        "//pyglib:gfile",
        "//pyglib/contrib/g3_multiprocessing",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "train_dsv41_sft",
    srcs = ["train_dsv41_sft.py"],
    deps = [
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":export_and_quantize_lib",
        ":train_full_sft_lib",
        ":train_sft_lib",
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//pyglib/contrib/g3_multiprocessing",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
        "//third_party/py/safetensors",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "launch_dsv41_sft_b200_xm",
    srcs = ["launch_dsv41_sft_b200_xm.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)

py_library(
    name = "eval_dsv41_b32_arc2_lib",
    srcs = ["eval_dsv41_b32_arc2.py"],
    deps = [
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":export_and_quantize_lib",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:havu_b32_harness",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:prompts_and_selector",
        "//pyglib:gfile",
        "//pyglib/contrib/g3_multiprocessing",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
        "//third_party/py/vllm",
    ],
)

py_binary(
    name = "eval_dsv41_b32_arc2",
    srcs = ["eval_dsv41_b32_arc2.py"],
    deps = [
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":export_and_quantize_lib",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:havu_b32_harness",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:prompts_and_selector",
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//pyglib/contrib/g3_multiprocessing",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/numpy",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
        "//third_party/py/vllm",
    ],
)

py_binary(
    name = "launch_dsv41_b32_eval_xm",
    srcs = ["launch_dsv41_b32_eval_xm.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/xmanager:xm_flags",
        "//third_party/py/xmanager/contrib/internal:requirements_flag",
        "//third_party/py/xmanager/xm",
        "//third_party/py/xmanager/xm_abc",
    ],
)

py_library(
    name = "load_dsv41_2bit_example_lib",
    srcs = ["load_dsv41_2bit_example.py"],
    deps = [
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":train_dsv41_sft_lib",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "load_dsv41_2bit_example",
    srcs = ["load_dsv41_2bit_example.py"],
    deps = [
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":train_dsv41_sft_lib",
        "//file/cns2",
        "//file/liball",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/g3lib",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "sft_test",
    size = "medium",
    srcs = ["sft_test.py"],
    deps = [
        ":audit_and_split_e065_rollouts_lib",
        ":data_pipeline",
        ":dsv41_image_tokens",
        ":dsv41_sft_model",
        ":eval_dsv41_b32_arc2_lib",
        ":evaluate_difficulty_argon_lib",
        ":export_and_quantize_lib",
        ":fast_deltanet",
        ":load_dsv41_2bit_example_lib",
        ":rank_e065_difficulty_gemini38_flash_lib",
        ":train_dsv41_sft_lib",
        ":train_full_sft_lib",
        ":train_sft_lib",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:havu_b32_harness",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:prompts_and_selector",
        "//pyglib:gfile",
        "//testing/pybase",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/peft",
        "//third_party/py/pyarrow",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
        "//third_party/py/transformers:v5",
        "//third_party/py/vllm",
    ],
)

