#!/usr/bin/env python3
"""E043: load a compact (row-pruned) Engram table in the vLLM DeepSeek-V4.1 fork.

Usage: apply_compact_engram.py <site-packages>/vllm      (idempotent, marker "E043C:")

Active only when the checkpoint's text_config has `engram_compact_kept`: for each Engram layer, 24 kept-row counts,
one per hash column (order-major: 2-gram heads 0-7, 3-gram, 4-gram), as written by export_pruned.py. Then
  layers.{L}.engram.embed.weight / .scale       compact kept rows in original-id order  [kept, 256] / [kept, 8]
  layers.{L}.engram.embed_tokens.row_map        int32 [num_embeddings]: original row id -> compact row, -1 dropped
Each TP rank owns complete hash columns, i.e. a contiguous original-id range and the matching contiguous compact
slice. The rank keeps its slice of the table and of the map (converted to rank-local rows) in pinned host memory
(with cpu_offload) and the lookup kernel reads the map: a dropped row is not "owned", so it reads as a zero row
before Wkv, the same as an id outside the rank's range. Without `engram_compact_kept` nothing changes.

Row-chunked tables (any Engram table, compact or not; `reshard_safetensors.py`, for <= 5 GB files): a table may
instead arrive as `layers.{L}.engram.embed.{weight,scale}.rows_<start>` tensors. The model's load_weights copies
each chunk's overlap with the rank's rows into the parameter and checks that every row of the rank was covered.
"""
import sys
from pathlib import Path

MARK = "E043C:"


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
    edit(vllm / "models/deepseek_v4_1/common/engram.py", [
        # Loader for the rank's slice of the original->compact map, converted to rank-local compact rows.
        (
            "@triton.jit\ndef _engram_lookup_kernel(",
            "def _engram_row_map_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:\n"
            "    \"\"\"E043C: rank slice of the original->compact row map, as rank-local compact rows (-1 dropped).\"\"\"\n"
            "    part = loaded_weight.narrow(0, param.engram_vocab_start, param.shape[0]).to(torch.int64)\n"
            "    local = torch.where(part >= 0, part - param.engram_compact_start, -1)\n"
            "    param.data.copy_(local.to(torch.int32))\n\n\n"
            "@triton.jit\ndef _engram_lookup_kernel(",
        ),
        (
            "    weight,\n    scales,\n    ids,\n    out,\n    vocab_start,\n",
            "    weight,\n    scales,\n    row_map,\n    ids,\n    out,\n    vocab_start,\n",
        ),
        (
            "    BLOCK_R: tl.constexpr,\n    GRID: tl.constexpr,\n):\n    \"\"\"Gather fp8 rows",
            "    BLOCK_R: tl.constexpr,\n    GRID: tl.constexpr,\n    HAS_MAP: tl.constexpr = False,\n"
            "):\n    \"\"\"Gather fp8 rows",
        ),
        (
            "        owned &= (index >= vocab_start) & (index < vocab_end)\n"
            "        local = tl.where(owned, index - vocab_start, 0)\n",
            "        owned &= (index >= vocab_start) & (index < vocab_end)\n"
            "        if HAS_MAP:  # E043C: compact table, dropped rows read as zeros\n"
            "            mapped = tl.load(row_map + tl.where(owned, index - vocab_start, 0), mask=owned, other=-1)\n"
            "            owned &= mapped >= 0\n"
            "            local = tl.where(owned, mapped, 0).to(tl.int64)\n"
            "        else:\n"
            "            local = tl.where(owned, index - vocab_start, 0)\n",
        ),
        # Constructor: compact allocation.
        (
            "        block_size: int = 32,\n        cpu_offload: bool = False,\n    ):",
            "        block_size: int = 32,\n        cpu_offload: bool = False,\n"
            "        compact_head_rows: tuple[int, ...] | None = None,\n    ):",
        ),
        (
            "        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx\n",
            "        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx\n"
            "        # E043C: compact table slice owned by this rank (contiguous: kept rows are in original-id order).\n"
            "        self.compact = compact_head_rows is not None\n"
            "        if self.compact:\n"
            "            assert len(compact_head_rows) == len(head_sizes)\n"
            "            self.compact_start = sum(compact_head_rows[: self.head_start])\n"
            "            table_rows = sum(compact_head_rows[: self.head_start + self.part_n_hash_cols]) - self.compact_start\n"
            "        else:\n"
            "            self.compact_start = 0\n"
            "            table_rows = self.part_num_embeddings\n",
        ),
        (
            "        self.weight = nn.Parameter(\n            torch.empty(\n"
            "                self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn, **kwargs\n",
            "        self.weight = nn.Parameter(\n            torch.empty(\n"
            "                table_rows, dim, dtype=torch.float8_e4m3fn, **kwargs\n",
        ),
        (
            "            torch.empty(\n                self.part_num_embeddings,\n                dim // block_size,\n",
            "            torch.empty(\n                table_rows,\n                dim // block_size,\n",
        ),
        (
            "                    \"weight_loader\": _engram_head_shard_weight_loader,\n"
            "                    \"engram_vocab_start\": self.vocab_start_idx,\n                },\n            )\n",
            "                    \"weight_loader\": _engram_head_shard_weight_loader,\n"
            "                    \"engram_vocab_start\": self.compact_start if self.compact else self.vocab_start_idx,\n"
            "                },\n            )\n"
            "        self.row_map = None\n"
            "        if self.compact:\n"
            "            self.row_map = nn.Parameter(\n"
            "                torch.empty(self.part_num_embeddings, dtype=torch.int32, **kwargs), requires_grad=False\n"
            "            )\n"
            "            set_weight_attrs(self.row_map, {\"weight_loader\": _engram_row_map_loader,\n"
            "                                            \"engram_vocab_start\": self.vocab_start_idx,\n"
            "                                            \"engram_compact_start\": self.compact_start})\n",
        ),
        # Storage views include the map.
        (
            "        if not self.cpu_offload:\n            return self.weight.data, self.weight_scale_inv.data\n"
            "        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())\n",
            "        if not self.cpu_offload:\n"
            "            rm = self.row_map.data if self.row_map is not None else self.weight_scale_inv.data\n"
            "            return self.weight.data, self.weight_scale_inv.data, rm\n"
            "        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())\n",
        ),
        (
            "                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),\n            )\n",
            "                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),\n"
            "                get_accelerator_view_from_cpu_tensor(self.row_map.data)\n"
            "                if self.row_map is not None else None,\n            )\n",
        ),
        (
            "        weight, scales = self._storage()\n",
            "        weight, scales, row_map = self._storage()\n"
            "        if row_map is None:\n            row_map = scales\n",
        ),
        (
            "        _engram_lookup_kernel[(grid,)](\n            weight,\n            scales,\n            indices,\n",
            "        _engram_lookup_kernel[(grid,)](\n            weight,\n            scales,\n            row_map,\n"
            "            indices,\n",
        ),
        (
            "            BLOCK_R=16,\n            GRID=grid,\n        )\n",
            "            BLOCK_R=16,\n            GRID=grid,\n            HAS_MAP=self.compact,\n        )\n",
        ),
        # Engram module: pass the per-column kept counts from the config.
        (
            "            cpu_offload=engram_config.cpu_offload if engram_config else True,\n        )\n",
            "            cpu_offload=engram_config.cpu_offload if engram_config else True,\n"
            "            compact_head_rows=(\n"
            "                tuple(config.engram_compact_kept[layer_hash_index])\n"
            "                if getattr(config, \"engram_compact_kept\", None) else None\n"
            "            ),  # E043C\n        )\n",
        ),
    ])
    edit(vllm / "models/deepseek_v4_1/nvidia/model.py", [
        # Row-chunked Engram tables: route each chunk into the rank's slice of the table parameter.
        (
            "def _make_deepseek_v4_weights_mapper(",
            "_E043_ENGRAM_ROWS = re.compile(r\"^(.*\\.engram)\\.embed\\.(weight|scale)\\.rows_(\\d+)$\")\n\n\n"
            "def _e043_engram_row_chunks(weights, params_dict, loaded_params):\n"
            "    \"\"\"E043C: Engram tables stored as row chunks `<L>.engram.embed.{weight,scale}.rows_<start>` (<= 5 GB\n"
            "    files). Copy each chunk's overlap with this rank's rows; other weights pass through unchanged.\"\"\"\n"
            "    covered: dict[str, int] = {}\n"
            "    for name, loaded_weight in weights:\n"
            "        m = _E043_ENGRAM_ROWS.match(name)\n"
            "        if m is None:\n"
            "            yield name, loaded_weight\n"
            "            continue\n"
            "        leaf = \"weight\" if m.group(2) == \"weight\" else \"weight_scale_inv\"\n"
            "        pname = f\"{m.group(1)}.embed_tokens.{leaf}\"\n"
            "        param = params_dict[pname]\n"
            "        if loaded_weight.dtype == torch.float8_e8m0fnu:\n"
            "            loaded_weight = loaded_weight.view(torch.uint8)\n"
            "        start, lo = int(m.group(3)), param.engram_vocab_start\n"
            "        a = max(start, lo)\n"
            "        b = min(start + loaded_weight.shape[0], lo + param.shape[0])\n"
            "        if a < b:\n"
            "            param.data[a - lo:b - lo].copy_(loaded_weight[a - start:b - start])\n"
            "            covered[pname] = covered.get(pname, 0) + (b - a)\n"
            "        loaded_params.add(pname)\n"
            "    for pname, rows in covered.items():\n"
            "        assert rows == params_dict[pname].shape[0], (\n"
            "            f\"engram row chunks cover {rows} of {params_dict[pname].shape[0]} rows of {pname}\"\n"
            "        )\n\n\n"
            "def _make_deepseek_v4_weights_mapper(",
        ),
        (
            "        params_dict = dict(self.named_parameters())\n        loaded_params: set[str] = set()\n\n"
            "        # TP for attention\n",
            "        params_dict = dict(self.named_parameters())\n        loaded_params: set[str] = set()\n"
            "        weights = _e043_engram_row_chunks(weights, params_dict, loaded_params)  # E043C: row chunks\n\n"
            "        # TP for attention\n",
        ),
    ])


if __name__ == "__main__":
    main()
