#!/usr/bin/env python3
"""E043 low-bit round: call e043_obs.observe_x(hidden_states) at the MoE input (calibration dump for GPTQ).

Usage: apply_e043_dumpx.py <site-packages>/vllm   (idempotent, marker "E043X:"; needs apply_e043_patches.py first)
"""
import shutil
import sys
from pathlib import Path

MARK = "E043X:"


def main():
    vllm = Path(sys.argv[1])
    shutil.copy(Path(__file__).with_name("e043_obs.py"), vllm / "e043_obs.py")
    p = vllm / "models/deepseek_v4/nvidia/model.py"
    text = p.read_text()
    if MARK in text:
        print(f"already patched: {p}")
        return
    old = "        e043_obs.begin_moe(self.prefix)\n"
    assert text.count(old) == 1, text.count(old)
    text = text.replace(old, old + "        e043_obs.observe_x(hidden_states)  # E043X: MoE input for the calibration dump\n")
    p.write_text(text)
    print(f"patched: {p}")


if __name__ == "__main__":
    main()
