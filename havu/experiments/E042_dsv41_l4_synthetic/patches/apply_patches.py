#!/usr/bin/env python3
"""Apply the E042 patches to an installed vLLM DeepSeek-V4.1 SM89 fork (vision11 wheel).

Usage: apply_patches.py <site-packages>/vllm
Idempotent: each edit checks for its marker first. All behaviour changes are
env-gated except the router whitelist, which only adds shapes:
- router.py: allow (64, 6), (72, 6), (48, 6) and the cut DSpark drafters (64, 3), (32, 3), (16, 3) (the Triton kernel
  already masks lanes).
- engram.py: DSV41_ENGRAM_SYNTH_ROWS=N allocates at most N physical rows per
  rank and layer *before* the constructor would allocate the original table;
  original ids keep their hash geometry and TP ownership and are folded into
  the compact table with a modulo. Synthetic TPS runs only (no row semantics).
- dummy_loader.py: DSV41_SYNTH_INIT=1 runs one dummy pass over the root module
  (instead of once per module depth) and then synth_init.apply_synthetic_init.
- router.py: DSV41_ROUTE_STATS=N logs routing occupancy every N router calls
  (eager only; syncs the GPU).
"""

import shutil
import sys
from pathlib import Path

MARK = "E042"


def edit(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    if f"{MARK}:" in text:
        print(f"already patched: {path}")
        return
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {text.count(old)}x: {old!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched: {path}")


def main() -> None:
    vllm = Path(sys.argv[1])
    v41 = vllm / "models/deepseek_v4_1"

    edit(v41 / "nvidia/router.py", [
        (
            "_SUPPORTED_ROUTING_SHAPES = frozenset({(128, 3), (384, 6)})",
            "# E042: (48, 6), (64, 6) and (72, 6) for the pruned E48/E64/E72 fixtures.\n"
            "_SUPPORTED_ROUTING_SHAPES = frozenset({(128, 3), (384, 6), (64, 6), (72, 6), (48, 6)})\n"
            "_ROUTE_STATS_EVERY = int(__import__('os').environ.get('DSV41_ROUTE_STATS', '0'))\n"
            "_route_stats: dict = {}",
        ),
        (
            "            input_tokens=input_ids,\n        )\n\n        if self.num_fused_shared_experts > 0:",
            "            input_tokens=input_ids,\n        )\n"
            "        if _ROUTE_STATS_EVERY:\n"
            "            _record_route_stats(self, topk_ids)\n\n"
            "        if self.num_fused_shared_experts > 0:",
        ),
    ])
    router = v41 / "nvidia/router.py"
    text = router.read_text()
    if "(64, 6), (72, 6)})" in text and "(48, 6)" not in text:  # patched before E48 was added
        text = text.replace("(64, 6), (72, 6)})", "(64, 6), (72, 6), (48, 6)})")
        router.write_text(text)
        print(f"added (48, 6): {router}")
    if "(48, 6)})" in text and "(32, 3)" not in text:  # DS1: DSpark drafters cut to 64 / 32 / 16 experts (top-3)
        text = text.replace("(48, 6)})", "(48, 6), (64, 3), (32, 3), (16, 3)})")
        router.write_text(text)
        print(f"added (64, 3), (32, 3), (16, 3): {router}")
    if "def _record_route_stats" not in text:
        text += '''

def _record_route_stats(router, topk_ids: torch.Tensor) -> None:
    """E042: per-router expert load and unique experts per call (eager only)."""
    if torch.cuda.is_current_stream_capturing() or topk_ids.shape[0] == 0:
        return
    key = id(router)
    st = _route_stats.setdefault(
        key, {"layer": len(_route_stats), "calls": 0, "load": None, "uniq": {}}
    )
    ids = topk_ids[:, : router.top_k].reshape(-1).to(torch.int64)
    counts = torch.bincount(ids, minlength=router.global_num_experts)
    st["load"] = counts if st["load"] is None else st["load"] + counts
    bucket = 1 << (topk_ids.shape[0] - 1).bit_length()
    u = st["uniq"].setdefault(bucket, [0, 0])
    u[0] += int((counts > 0).sum())
    u[1] += 1
    st["calls"] += 1
    if st["layer"] == 0 and st["calls"] % _ROUTE_STATS_EVERY == 0:
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.logger import init_logger

        if get_tensor_model_parallel_rank() != 0:
            return
        log = init_logger(__name__)
        for s in sorted(_route_stats.values(), key=lambda s: s["layer"]):
            load = s["load"].float()
            share = load / load.sum()
            log.info(
                "E042 route layer=%d calls=%d max/mean load=%.2f top6 share=%.3f "
                "uniq experts by tokens-bucket=%s",
                s["layer"], s["calls"], float(share.max() * share.numel()),
                float(share.topk(6).values.sum()),
                {b: round(a / n, 1) for b, (a, n) in sorted(s["uniq"].items())},
            )
'''
        router.write_text(text)

    edit(v41 / "common/engram.py", [
        ("import weakref\n", "import os\nimport weakref\n"),
        ("    num_rows,\n", "    num_rows,\n    phys_rows,\n"),
        (
            "        local = tl.where(owned, index - vocab_start, 0)",
            "        # E042: fold into a compact synthetic table (no-op at full size).\n"
            "        local = tl.where(owned, (index - vocab_start) % phys_rows, 0)",
        ),
        (
            "        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx\n",
            "        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx\n"
            "        synth_rows = int(os.environ.get(\"DSV41_ENGRAM_SYNTH_ROWS\", \"0\"))\n"
            "        self.phys_rows = (\n"
            "            min(self.part_num_embeddings, synth_rows)\n"
            "            if synth_rows > 0\n"
            "            else self.part_num_embeddings\n"
            "        )\n",
        ),
        (
            "                self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn, **kwargs",
            "                self.phys_rows, dim, dtype=torch.float8_e4m3fn, **kwargs",
        ),
        (
            "            torch.empty(\n                self.part_num_embeddings,\n"
            "                dim // block_size,",
            "            torch.empty(\n                self.phys_rows,\n"
            "                dim // block_size,",
        ),
        (
            "                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,",
            "                self.phys_rows * (dim + dim // block_size) / 1024**3,",
        ),
        (
            "            rows,\n            indices.stride(0),",
            "            rows,\n            self.phys_rows,\n            indices.stride(0),",
        ),
    ])

    edit(vllm / "model_executor/model_loader/dummy_loader.py", [
        (
            "    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:\n"
            "        for layer in model.modules():\n",
            "    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:\n"
            "        # E042: one dummy pass over the root, then format-valid synthetic\n"
            "        # values for DeepSeek-V4.1 (see synth_init.py).\n"
            "        import os\n\n"
            "        if os.environ.get(\"DSV41_SYNTH_INIT\") == \"1\":\n"
            "            from vllm.models.deepseek_v4_1.synth_init import (\n"
            "                apply_synthetic_init,\n"
            "            )\n\n"
            "            for layer in model.modules():\n"
            "                info = get_layerwise_info(layer)\n"
            "                if info.can_load():\n"
            "                    self._process_online_quant_layer(layer, info)\n"
            "            initialize_dummy_weights(model, model_config)\n"
            "            apply_synthetic_init(model)\n"
            "            return\n"
            "        for layer in model.modules():\n",
        ),
    ])

    shutil.copy(Path(__file__).with_name("synth_init.py"), v41 / "synth_init.py")
    print(f"copied synth_init.py -> {v41}")


if __name__ == "__main__":
    main()
