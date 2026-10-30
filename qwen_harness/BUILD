# Custom Colab kernel for ARC Prize 2026 model evaluation & Qwen/HF/vLLM on GPU/TPU.
#
# Launch on H100 via XManager (from google3/):
#   COLAB_CLI="/google/bin/releases/grp-ix-team/rapid/colab-cli/cli.par"
#   $COLAB_CLI launch //experimental/users/arazdai/kaggle:arc_colab_kernel \
#       --xm_resource_pool=research-mm \
#       --xm_resource_alloc=group:research-mm/ml-gemini-code-bridging-gap \
#       --accelerator=h100=1 --ram=64
#
# Launch on TPU (ghostfish v6e) via XManager (from google3/):
#   $COLAB_CLI launch //experimental/users/arazdai/kaggle:arc_colab_kernel \
#       --xm_resource_pool=research-mm \
#       --xm_resource_alloc=group:research-mm/ml-gemini-code-bridging-gap \
#       --accelerator=ghostfish=2x2x1 --ram=128
#
# Launch locally on CPU (from google3/):
#   $COLAB_CLI launch_local //experimental/users/arazdai/kaggle:arc_colab_kernel --use_free_port

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")
load("//learning/grp/pynball:build_defs.bzl", "pkg_colab_binary")
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:private"])

py_library(
    name = "vllm_cuda_shim",
    srcs = [
        "vllm_cuda_shim.py",
        "vllm_fla_triton_ops.py",
    ],
    strict_deps = False,
    tags = [
        "no-pyrefly",
        "no-pytype",
    ],
    deps = [
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//third_party/py/einops",
        "//third_party/py/flash_attn",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        "//third_party/py/triton",
        "//third_party/py/vllm",
    ],
)

# buildozer: disable=denylisted-library
pkg_colab_binary(
    name = "arc_colab_kernel",
    exec_properties = {
        "cpp_link.mem": "24g",
        "mem": "20g",
    },
    g3_multiprocessing = True,
    deps = [
        # Google3 / CNS / Filesystem & Internal Clients
        "//file/cns2",
        "//file/liball",
        "//learning/deepmind/boq/data/ditto/client:ditto",
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        "//tech/env:envelope_loader",
        "//third_party/py/etils/epath",
        "//third_party/py/g3lib",
        # PyTorch
        "//third_party/py/einops",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        # vLLM & TPU Inference Support + Pilot 1 & Pilot 2 Harnesses
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:prompts_and_voting",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:sandbox",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:arc_dsl_and_summary",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:pilot2_eval_lib",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:prompts_and_selector",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:sandbox_verifier",
        "//third_party/py/etils/ecolab",
        "//third_party/py/jax",
        "//third_party/py/tpu_inference",
        "//third_party/py/vllm",
        # Hugging Face Ecosystem (v5 transformers + v1 hub with CNS/g3lib support)
        "//third_party/py/accelerate",
        "//third_party/py/datasets",
        "//third_party/py/huggingface_hub",
        "//third_party/py/peft",
        "//third_party/py/safetensors:torch",
        "//third_party/py/sentencepiece",
        "//third_party/py/tokenizers",
        "//third_party/py/transformers:v5",
        # ARC-AGI-2 & ARC-AGI-3 Collaborator Notebook & Evaluation Dependencies
        "//third_party/py/PIL:pil",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/arc_agi",
        "//third_party/py/arcengine",
        "//third_party/py/fastapi",
        "//third_party/py/google/genai",
        "//third_party/py/httpx",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/openai",
        "//third_party/py/pandas",
        "//third_party/py/pydantic:pydantic_v2",
        "//third_party/py/requests",
        "//third_party/py/scipy",
        "//third_party/py/tqdm",
        "//third_party/py/uvicorn",
    ],
)

# buildozer: disable=denylisted-library
pkg_colab_binary(
    name = "arc_colab_kernel_gpu",
    exec_properties = {
        "cpp_link.mem": "24g",
        "mem": "20g",
    },
    g3_multiprocessing = True,
    deps = [
        ":vllm_cuda_shim",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:prompts_and_voting",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm:sandbox",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:arc_dsl_and_summary",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:pilot2_eval_lib",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:prompts_and_selector",
        "//experimental/users/arazdai/kaggle/arc_agi_2/pilot_2:sandbox_verifier",
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//file/cns2",
        "//file/liball",
        "//learning/deepmind/boq/data/ditto/client:ditto",
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        "//tech/env:envelope_loader",
        "//third_party/py/PIL:pil",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/accelerate",
        "//third_party/py/arc_agi",
        "//third_party/py/arcengine",
        "//third_party/py/datasets",
        "//third_party/py/einops",
        "//third_party/py/etils/ecolab",
        "//third_party/py/etils/epath",
        "//third_party/py/fastapi",
        "//third_party/py/flash_attn",
        "//third_party/py/g3lib",
        "//third_party/py/google/genai",
        "//third_party/py/httpx",
        "//third_party/py/huggingface_hub",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/openai",
        "//third_party/py/pandas",
        "//third_party/py/peft",
        "//third_party/py/pydantic:pydantic_v2",
        "//third_party/py/requests",
        "//third_party/py/safetensors:torch",
        "//third_party/py/scipy",
        "//third_party/py/sentencepiece",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        "//third_party/py/tqdm",
        "//third_party/py/transformers:v5",
        "//third_party/py/triton",
        "//third_party/py/uvicorn",
        "//third_party/py/vllm",
    ],
)

py_test(
    name = "vllm_h100_gpu_test",
    size = "large",
    srcs = ["vllm_h100_gpu_test.py"],
    strict_deps = False,
    tags = [
        "no-pyrefly",
        "no-pytype",
        "requires-gpu-sm90",
    ],
    deps = [
        ":vllm_cuda_shim",
        "//experimental/users/gaoxiao/deploy_qwen_h100:cuda_platform",
        "//third_party/py/PIL:pil",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/einops",
        "//third_party/py/flash_attn",
        "//third_party/py/safetensors:torch",
        "//third_party/py/tokenizers",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torchvision",
        "//third_party/py/transformers:v5",
        "//third_party/py/triton",
        "//third_party/py/vllm",
    ],
)


