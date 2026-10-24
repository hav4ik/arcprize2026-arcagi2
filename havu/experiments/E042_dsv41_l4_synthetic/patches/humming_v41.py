#!/usr/bin/env python3
"""E042 P12: opt-in Humming kernels for DeepSeek-V4.1 in the vision11 fork (speed tests on synthetic weights).

The fork whitelists V4.1's backends: experts only 'auto'/'b12x'/'marlin' (oracle/mxfp4.py), dense MXFP8 only
FlashInfer CUTLASS / AdaMxfp8 (kernels/linear/__init__.py; --linear-backend humming then silently falls back to Ada).
This adds Humming to both, only when asked for (--moe-backend humming / --linear-backend humming); defaults unchanged.
Humming quantizes expert activations itself, per VLLM_HUMMING_INPUT_QUANT_CONFIG (unset: BF16; MXFP8 K32 matches the
official boundary except its 1e-4 minimum amax), and routes through the generic MXFP4 path (router weight applied
after FC2 quant, unlike the fork's Marlin path). Idempotent; keeps .orig copies.
Run: python humming_v41.py <site-packages>/vllm
"""

import shutil
import sys
from pathlib import Path

V = Path(sys.argv[1])
PATCHES = {
    "model_executor/layers/fused_moe/oracle/mxfp4.py": (
        '''    elif runner_backend == "marlin":
        backends = [Mxfp4MoeBackend.MARLIN]
    else:
''',
        '''    elif runner_backend == "marlin":
        backends = [Mxfp4MoeBackend.MARLIN]
    elif runner_backend == "humming":
        # E042 opt-in (speed tests): Humming quantizes activations itself, per VLLM_HUMMING_INPUT_QUANT_CONFIG.
        backends = [Mxfp4MoeBackend.HUMMING]
    else:
''',
    ),
    "model_executor/kernels/linear/__init__.py": (
        '''            if kernel in (FlashInferCutlassMxfp8LinearKernel, AdaMxfp8LinearKernel)
        ]
''',
        '''            if kernel in (FlashInferCutlassMxfp8LinearKernel, AdaMxfp8LinearKernel)
        ]
        if _get_linear_backend() == "humming":  # E042 opt-in (speed tests)
            possible = [HummingMxfp8LinearKernel] + possible
''',
    ),
}

for rel, (old, new) in PATCHES.items():
    p = V / rel
    s = p.read_text()
    if "E042 opt-in" in s:
        print(f"{rel}: already patched")
        continue
    assert s.count(old) == 1, f"{rel}: anchor not found exactly once"
    orig = p.with_name(p.name + ".orig")
    if not orig.exists():
        shutil.copy2(p, orig)
    p.write_text(s.replace(old, new))
    print(f"{rel}: patched")
