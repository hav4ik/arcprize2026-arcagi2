"""Shape-only DeepSeek-V4.1 checkpoint for `vllm serve --load-format dummy` on Manh's fork (KV pool / memory checks of
pruned layouts without the real export).

The fork picks 2-bit Humming GPTQ for a layer's routed experts only if `gptq/layer_LL.safetensors` exists, and infers
the schema from the lowest expert's tensor shapes (deepseek_v41/expert_gptq.py, quant_config.py). Dummy loading
never reads checkpoint data, so each layer file holds just e0's w1/w2/w3 q/s/z (zeros) at the real shapes; every
other weight is dummy-initialized at its true size from config.json.

Usage:
  make_dummy_ckpt.py --config CONFIG.json --tokenizer-dir DIR --out OUT [--per-layer 72,72,...] [--bits 2]
      [--group13 128] [--group2 64] [--scale-dtype BF16] [--no-zero-points]
--per-layer writes text_config.n_routed_experts_per_layer (40 counts); without it the config's uniform count stays.
"""
import argparse
import json
import shutil
import struct
from pathlib import Path

DTYPE_BYTES = {"I32": 4, "BF16": 2, "F16": 2}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
                   "chat_template.jinja", "encoding")


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape)}; data is all zeros."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + offset)  # zeros


def expert_tensors(hidden, inter, bits, g13, g2, scale_dtype, zero_points):
    """Humming-packed GPTQ shapes: q [N, K*bits/32], s [N, K/group], z [N*bits/32, K/group]."""
    out = {}
    for shard, n, k, g in (("w1", inter, hidden, g13), ("w3", inter, hidden, g13), ("w2", hidden, inter, g2)):
        assert (k * bits) % 32 == 0 and (n * bits) % 32 == 0 and k % g == 0, (shard, n, k, g)
        out[f"e0.{shard}.q"] = ("I32", (n, k * bits // 32))
        out[f"e0.{shard}.s"] = (scale_dtype, (n, k // g))
        if zero_points:
            out[f"e0.{shard}.z"] = ("I32", (n * bits // 32, k // g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-layer", default="")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group13", type=int, default=128)
    ap.add_argument("--group2", type=int, default=64)
    ap.add_argument("--scale-dtype", default="BF16", choices=["BF16", "F16"])
    ap.add_argument("--no-zero-points", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    text = cfg.get("text_config", cfg)
    n_layers = text["num_hidden_layers"]
    if a.per_layer:
        counts = [int(x) for x in a.per_layer.split(",")]
        assert len(counts) == n_layers and min(counts) >= text["num_experts_per_tok"], counts
        text["n_routed_experts_per_layer"] = counts
    out = Path(a.out)
    (out / "gptq").mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    for name in TOKENIZER_FILES:
        src = Path(a.tokenizer_dir) / name
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, out / name)
    tensors = expert_tensors(text["hidden_size"], text["moe_intermediate_size"], a.bits, a.group13, a.group2,
                             a.scale_dtype, not a.no_zero_points)
    for layer in range(n_layers):
        write_safetensors(out / "gptq" / f"layer_{layer:02d}.safetensors", tensors)
    print(json.dumps({"out": str(out), "layers": n_layers, "experts_total": sum(text.get(
        "n_routed_experts_per_layer", [text["n_routed_experts"]] * n_layers)), "per_layer": bool(a.per_layer),
        "layer_file_tensors": {k: v[1] for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
