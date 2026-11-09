#!/usr/bin/env python3
"""E068: port of upstream vLLM PR #57152 ("[Bugfix][Model] Restore causal image SWA for DeepSeek V4.1", merged
2026-09-16, merge commit 9f9e1dac) to the E043/E066 fork wheel (vllm-0.28.1rc1.dev517+glm53.dsv41.vision11).

Upstream's code is a later layout (vllm/models/deepseek_v41/, prefill_left_visible/right_visible widening inside the
SWA index kernels); the fork routes all image-span widening through mm-prefix ranges, which exist only when
model_config.is_mm_prefix_lm is true (v1/worker/gpu/model_states/default.py compute_mm_prefix_ranges, sparse_swa.py
swa_index_width, deepseek_v4/common/ops/cache_utils.py has_mm_prefix). So the PR's config hunk is the whole fix here:
  1. transformers_utils/configs/deepseek_v41.py: stop setting is_mm_prefix_lm and mm_prefix_clamp_sliding_window
     (verbatim PR hunk). Without the attribute the arch-config convertor falls back to its known-model list, which
     has no DeepSeek entry -> False.
  2. models/deepseek_v4_1/attention.py: max_image_tokens = 0 (PR: removed). In the fork it only sizes CED-prefill
     retained tokens (config.ced_prefill), not the SWA window.
The V4 (non-.1) config keeps its behaviour, as in the PR. Idempotent; refuses to run on unexpected source.
Serve with --disable-chunked-mm-input as in E068's `causal` arm (atomic image-span prefill, as DeepSeek's reference).

  apply_pr57152_port.py <site-packages>/vllm [--check]
"""
import sys
from pathlib import Path

CONFIG_OLD = (
    "        # The vision variant needs the mm-prefix plumbing (atomic image-span\n"
    "        # prefill + sparse-SWA window widening); the base arch-config\n"
    "        # convertor reads this config-level attribute.\n"
    "        self.is_mm_prefix_lm = vision_n_layers > 0\n"
    "        # The sparse-SWA index kernels widen the window within image spans\n"
    "        # in-kernel, so mm-prefix ranges longer than sliding_window must be\n"
    "        # kept (they are only consumed by those kernels).\n"
    "        self.mm_prefix_clamp_sliding_window = vision_n_layers > 0\n"
)
CONFIG_NEW = (
    "        # E068: port of vLLM PR #57152 -- V4.1 image tokens use the plain causal\n"
    "        # window (DeepSeek's reference get_window_topk_idxs); no mm-prefix /\n"
    "        # bidirectional image spans.\n"
)
ATTN_OLD = (
    "        self.max_image_tokens = (\n"
    "            getattr(config, \"vision_max_n_token\", 0)\n"
    "            if getattr(config, \"vision_n_layers\", 0) > 0\n"
    "            else 0\n"
    "        )\n"
)
ATTN_NEW = "        self.max_image_tokens = 0  # E068: port of vLLM PR #57152 (causal image SWA)\n"


def edit(path, old, new, check):
    text = path.read_text()
    if new in text and old not in text:
        return "already applied"
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected source not found once; not the vision11 fork?")
    if not check:
        path.write_text(text.replace(old, new))
    return "would apply" if check else "applied"


def main():
    vllm = Path(sys.argv[1])
    check = "--check" in sys.argv[2:]
    print("config:", edit(vllm / "transformers_utils/configs/deepseek_v41.py", CONFIG_OLD, CONFIG_NEW, check))
    print("attention:", edit(vllm / "models/deepseek_v4_1/attention.py", ATTN_OLD, ATTN_NEW, check))


if __name__ == "__main__":
    main()
