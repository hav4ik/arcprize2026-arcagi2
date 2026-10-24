# SPDX-License-Identifier: Apache-2.0
"""E042 LB: low-bit routed experts for DeepSeek-V4.1 through vLLM's Humming MoE (synthetic throughput only).

Usage: python -I humming_experts.py <site-packages>/vllm

The fork serves V4.1's routed experts only as MXFP4 (Marlin, 4.25 bits). Humming (humming-kernels, already in the
venv) has a batched MoE path for integer weights at any bit width with group scales (GPTQ schema). With
``DSV41_EXPERT_HUMMING='{"quant_method": "gptq", "bits": 2, "group_size": 128, "sym": true, "desc_act": false}'``
the routed experts get ``HummingMoEMethod`` with that schema instead of ``DeepseekV41Mxfp4MoEMethod``: packed
weights are allocated at the low bit width, so E128 at 2 bits needs about E64's memory. Weights come from
``--load-format dummy`` (integer tensors stay zero), so this measures speed and memory, never quality. Unset, the
model is unchanged.

Also adds router shapes (96, 6), (104, 6), (112, 6), (128, 6), (160, 6) for those fixtures. Idempotent.

Per-layer bit widths come from Humming's ``dynamic`` overrides: a key ``"+:<regex>"`` matched against the experts'
prefix (``model.layers.<i>.ffn.experts``) merges its dict into the schema, e.g. ``{"bits": 3}``. One quant type
per MoE layer.
"""

import sys
from pathlib import Path

MARK = "DSV41_EXPERT_HUMMING"


def main():
    vllm = Path(sys.argv[1])
    qc = vllm / "models/deepseek_v4_1/quant_config.py"
    text = qc.read_text()
    if MARK not in text:
        old = "            return DeepseekV41Mxfp4MoEMethod(layer.moe_config)\n"
        assert text.count(old) == 1, "unexpected quant_config.py"
        new = (
            "            # E042 LB: DSV41_EXPERT_HUMMING=<humming weight-schema JSON> serves the routed experts\n"
            "            # through Humming's MoE at that bit width (synthetic throughput tests only).\n"
            "            import json as _json\n"
            "            import os as _os\n"
            "            _hcfg = _os.environ.get(\"DSV41_EXPERT_HUMMING\")\n"
            "            if _hcfg:\n"
            "                from vllm.model_executor.layers.quantization.humming import HummingConfig\n"
            "                return HummingConfig(full_config=_json.loads(_hcfg)).get_quant_method(layer, prefix)\n"
            + old
        )
        qc.write_text(text.replace(old, new))
        print(f"patched {qc}")
    else:
        print(f"already patched {qc}")

    router = vllm / "models/deepseek_v4_1/nvidia/router.py"
    lines = router.read_text().splitlines(keepends=True)
    i = next(i for i, l in enumerate(lines) if l.startswith("_SUPPORTED_ROUTING_SHAPES = frozenset({"))
    add = [s for s in ("(96, 6)", "(104, 6)", "(112, 6)", "(128, 6)", "(160, 6)") if s not in lines[i]]
    if add:
        assert lines[i].rstrip().endswith("})"), "unexpected router whitelist line"
        lines[i] = lines[i].rstrip()[:-2] + ", " + ", ".join(add) + "})\n"
        router.write_text("".join(lines))
        print(f"added {', '.join(add)}: {router}")


if __name__ == "__main__":
    main()
