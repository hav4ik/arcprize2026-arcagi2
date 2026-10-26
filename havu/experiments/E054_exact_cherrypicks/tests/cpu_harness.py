# SPDX-License-Identifier: Apache-2.0
"""CPU-only import harness for the fork tree (no GPU, no NVML, no vllm._C).

Import this module before anything from vllm. It must run with
CUDA_VISIBLE_DEVICES="" and TRITON_INTERPRET=1 in the environment (run_cpu_checks.sh sets
both), so that:
- importlib.metadata.version("vllm") answers the fork's version (the source tree has no
  dist-info, and vllm.triton_utils turns Triton off when the lookup raises);
- no platform plugin activates (the CUDA plugin would probe NVML and import vllm._C), so
  current_platform is UnspecifiedPlatform and the CPU is the only device;
- vllm.triton_utils keeps the real triton (its lenient path for an empty
  CUDA_VISIBLE_DEVICES), and TRITON_INTERPRET=1 makes every @triton.jit kernel run on CPU
  tensors through Triton's interpreter.
"""

import importlib.metadata as _md
import os
import sys
import types

FORK_VERSION = "0.28.1rc1.dev517+glm53.dsv41.vision11"

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES=''"
assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"

_orig_version = _md.version


def _version(name):
    if name == "vllm":
        return FORK_VERSION
    return _orig_version(name)


_md.version = _version

# The E054 files replace their fork-tree modules (PYTHONPATH = fork wheel source + our
# 12-file overlay, read-only), unless E054_OVERLAY=0.
import importlib.abc  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

E054_OVERLAY_DIR = Path(__file__).resolve().parent.parent / "overlay"
E054_MODULES = {
    "vllm.model_executor.kernels.attention.dsa.dsv41_indexer":
        E054_OVERLAY_DIR / "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py",
    "vllm.models.deepseek_v4_1.common.engram":
        E054_OVERLAY_DIR / "vllm/models/deepseek_v4_1/common/engram.py",
}


class _E054Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in E054_MODULES:
            return importlib.util.spec_from_file_location(fullname, E054_MODULES[fullname])
        return None


if os.environ.get("E054_OVERLAY", "1") != "0":
    sys.meta_path.insert(0, _E054Finder())


class NVMLError_LibraryNotFound(Exception):  # noqa: N801  (vLLM matches "nvml" in the name)
    pass


def _no_nvml():
    raise NVMLError_LibraryNotFound("E054 CPU harness: NVML disabled")


# vllm/__init__ resolves current_platform on import; with this stub the CUDA plugin sees no
# NVML (its handled path) and no other plugin activates: UnspecifiedPlatform, CPU only.
_fake_nvml = types.ModuleType("vllm.third_party.pynvml")
_fake_nvml.nvmlInit = _no_nvml
_fake_nvml.nvmlShutdown = lambda: None
sys.modules["vllm.third_party.pynvml"] = _fake_nvml

import numpy as _np  # noqa: E402
import torch  # noqa: E402
import triton.language as _tl  # noqa: E402
import triton.runtime.interpreter as _interp  # noqa: E402

assert not torch.cuda.is_available(), "a GPU is visible; this harness is CPU only"

# Triton 3.8's interpreter stores bf16 as uint16 and its create_dot hands those raw bit
# patterns to np.matmul (integer products). Upcast bf16 operands to fp32 first, which is
# what the GPU MMA computes (bf16 x bf16 products are exact in fp32; only the accumulation
# order differs). Both kernels under test go through the same dot, so this only matters
# for comparisons against an independent reference.
_orig_create_dot = _interp.InterpreterBuilder.create_dot


def _create_dot(self, a, b, d, *rest):
    if a.dtype.scalar == _tl.bfloat16 or b.dtype.scalar == _tl.bfloat16:
        def up(h):
            if h.dtype.scalar != _tl.bfloat16 or h.data.dtype != _np.uint16:
                return h.data
            return (h.data.astype(_np.uint32) << 16).view(_np.float32)  # bf16 bits -> fp32, exact

        return _interp.TensorHandle(
            _np.matmul(up(a), up(b), dtype=d.data.dtype) + d.data, d.dtype.scalar
        )
    return _orig_create_dot(self, a, b, d, *rest)


_interp.InterpreterBuilder.create_dot = _create_dot

from vllm.platforms import current_platform  # noqa: E402
from vllm.triton_utils import HAS_TRITON  # noqa: E402

assert HAS_TRITON, "vllm.triton_utils disabled Triton"
PLATFORM = type(current_platform).__name__
