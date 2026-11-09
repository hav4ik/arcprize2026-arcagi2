---
license: mit
---
# DeepSeek-V4.1-Flash-1017 Engram tables in MXFP4 (E066)

Two MXFP4 versions of the 64 GiB tool Engram table that B32's model ships
(`chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`, the same table as the E112 repo and the Kaggle E128 model):

| folder | rows kept (layer 1 / 14) | share of the census lookups the 64 GiB table covers | format |
|---|---|---|---|
| `b32_tool64_mxfp4/` | 136,114,884 / 124,186,164 (all rows of the 64 GiB table) | 100% | MXFP4 |
| `b32_tool48_mxfp4/` | 101,674,506 / 93,551,280 (48 GiB budget) | 99.44% | MXFP4 |

**Format.** Compact Engram as in E043 `export_pruned.py`: `layers.{1,14}.engram.embed.weight` / `.scale` hold the kept
rows in original-id order (row chunks `.rows_<start>`, every file <= 5 GB) and `embed_tokens.row_map` maps each
original row id to its compact row (-1 = dropped). The values are **MXFP4 stored in the MXFP8 container**:
- every FP8 E4M3 value is an exact E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6, with sign);
- one E8M0 scale per 32 values; per group of 32, the lower-MSE of the two standard scale choices.

So the files serve unchanged on the fork with E043's `apply_compact_engram.py`, and give exactly MXFP4's numbers. To
save memory they must be repacked to 4.25 bits (vLLM PR 61530 layout: `[rows, 128]` uint8 nibbles + scales), which is a
lossless repacking of these values. Relative MSE vs the FP8 table: 1.39% (L2 11.8%).

**How they were built** (`scripts/`):
- `fp4_engram_chunked.py`: MXFP4 rounding, the same `requant()` as E043 `make_fp4_engram.py`, which is included.
- `engram_subset.py`: the 48 GiB table is the 195,225,786 most-used rows of the 64 GiB table. Rows are ranked by
  E066's census of the E065 training rollouts (151.3M text positions; prompt + generated counts), globally over both
  layers with ties broken by (layer, row), the same rule as E043 `engram_census.py select`. No row outside the 64 GiB
  table is added. The mask is `b32_tool48_mxfp4/engram_subset_48gib_mask.npz`.
- `assemble_engram_variant.py`: builds a full checkpoint from B32's export with a table swapped in. For the 48 GiB
  table it also rewrites the index and `text_config.engram_compact_kept`, which are in
  `b32_tool48_mxfp4/engram_subset.json`; the 64 GiB table keeps B32's values.

**Quality** (E066):
- Dev NLL with the C1 export: MXFP4 vs FP8 Engram changes generated-token and tool-call NLL by < 0.0005 on both
  harnesses.
- ARC-AGI-2 A/B on B32's model and harness (MXFP4, then 48 GiB + MXFP4, then the FP8 control; 120 tasks x 2): in
  progress. See `havu/experiments/E066_dsv41_image_pruning/README.md` in `hav4ik/arcprize2026-arcagi2`.
