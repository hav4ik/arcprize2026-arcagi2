#!/usr/bin/env python3
"""Install the E043 hooks into an installed vLLM DeepSeek-V4.1 fork (vision11 wheel).

Usage: apply_e043_patches.py <site-packages>/vllm
Idempotent (marker "E043:"). Copies e043_obs.py next to the vllm package and adds calls to it.
Without $E043_CTRL the hooks only add a zero bias and an all-ones Engram bitmap, i.e. the original model.
Also applies the E042 router whitelist (64/72/48 experts) so physically pruned exports load.
"""
import shutil
import sys
from pathlib import Path

MARK = "E043:"


def edit(path: Path, pairs):
    text = path.read_text()
    if MARK in text:
        print(f"already patched: {path}")
        return
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {text.count(old)}x: {old!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched: {path}")


def main():
    vllm = Path(sys.argv[1])
    shutil.copy(Path(__file__).with_name("e043_obs.py"), vllm / "e043_obs.py")
    print(f"installed {vllm / 'e043_obs.py'}")

    # Model forward start: flush/reload control, allocate masks, start a step record.
    edit(vllm / "models/deepseek_v4_1/nvidia/model.py", [
        (
            "        if self.use_mega_moe:\n            input_ids = input_ids.to(torch.int64)\n\n"
            "        # Engram n-gram hashes for the whole (flattened) batch, computed once\n",
            "        # E043: step hook (masks, REAP observation).\n"
            "        from vllm import e043_obs\n"
            "        e043_obs.step_begin(input_ids, positions)\n"
            "        if self.use_mega_moe:\n            input_ids = input_ids.to(torch.int64)\n\n"
            "        # Engram n-gram hashes for the whole (flattened) batch, computed once\n",
        ),
        (
            "            shared_expert_weight=getattr(previous_router, \"shared_expert_weight\", 1.0),\n        )\n",
            "            shared_expert_weight=getattr(previous_router, \"shared_expert_weight\", 1.0),\n        )\n"
            "        # E043: layer id for runtime expert masks.\n"
            "        self.experts.router._e043_layer = extract_layer_index(prefix)\n",
        ),
    ])

    # MoE layer id for the observer (eager steps).
    edit(vllm / "models/deepseek_v4/nvidia/model.py", [
        (
            "        org_shape = hidden_states.shape\n        final_hidden_states = self.experts(\n"
            "            hidden_states=hidden_states,\n            router_logits=hidden_states,\n",
            "        org_shape = hidden_states.shape\n"
            "        from vllm import e043_obs  # E043: observer layer id\n"
            "        e043_obs.begin_moe(self.prefix)\n"
            "        final_hidden_states = self.experts(\n"
            "            hidden_states=hidden_states,\n            router_logits=hidden_states,\n",
        ),
    ])

    # Router: masked selection bias (pruned experts never selected; weights from the selected set).
    edit(vllm / "models/deepseek_v4_1/nvidia/router.py", [
        (
            "_SUPPORTED_ROUTING_SHAPES = frozenset({(128, 3), (384, 6)})",
            "# E043: (48, 6), (64, 6), (72, 6) for physically pruned exports.\n"
            "_SUPPORTED_ROUTING_SHAPES = frozenset({(128, 3), (384, 6), (64, 6), (72, 6), (48, 6), (96, 6),"
            " (384, 7), (384, 8), (72, 8), (64, 8), (48, 8)})",
        ),
        (
            "        topk_weights, topk_ids = deepseek_v41_topk(\n            router_logits,\n"
            "            self.e_score_correction_bias.data,\n",
            "        from vllm import e043_obs  # E043: runtime expert mask\n"
            "        _e043_li = getattr(self, \"_e043_layer\", -1)\n"
            "        topk_weights, topk_ids = deepseek_v41_topk(\n            router_logits,\n"
            "            e043_obs.masked_bias(self.e_score_correction_bias.data, _e043_li),\n",
        ),
        (
            "            correction_bias_vl=(\n                self.e_score_correction_bias_vl.data\n",
            "            correction_bias_vl=(\n"
            "                e043_obs.masked_bias(self.e_score_correction_bias_vl.data, _e043_li)\n",
        ),
    ])

    # Marlin experts: router weights + native weighted outputs before the top-k sum.
    edit(vllm / "model_executor/layers/fused_moe/experts/marlin_moe.py", [
        (
            "        ctx = self._lora_context\n        if self.router_weight_before_fc2_quant and ctx is not None:",
            "        from vllm import e043_obs  # E043: router weights for the observer\n"
            "        e043_obs.set_topk(topk_weights, topk_ids)\n"
            "        ctx = self._lora_context\n        if self.router_weight_before_fc2_quant and ctx is not None:",
        ),
        (
            "        expert_map: torch.Tensor | None,\n    ) -> None:\n        if expert_map is not None:\n"
            "            ops.moe_sum(input, output, topk_ids, expert_map)\n",
            "        expert_map: torch.Tensor | None,\n    ) -> None:\n"
            "        from vllm import e043_obs  # E043: per-expert native amplitudes\n"
            "        e043_obs.observe(input, topk_ids, expert_map)\n"
            "        if expert_map is not None:\n"
            "            ops.moe_sum(input, output, topk_ids, expert_map)\n",
        ),
    ])

    # Engram: dropped rows -> id -1 -> zero row before Wkv.
    edit(vllm / "models/deepseek_v4_1/common/engram.py", [
        (
            "        \"\"\"Gather this layer's rows on the main stream before decoder layers.\"\"\"\n"
            "        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])",
            "        \"\"\"Gather this layer's rows on the main stream before decoder layers.\"\"\"\n"
            "        from vllm import e043_obs  # E043: Engram row mask\n"
            "        hash_ids = e043_obs.engram_ids(self.layer_hash_index, hash_ids)\n"
            "        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])",
        ),
    ])


if __name__ == "__main__":
    main()
