# Synthetic E64/E72 budget verification

**CPU arithmetic only. No GPU allocations, performance measurements or provider quotes were produced.**

This file summarizes `budget_scenarios.json`, which retains the exact argument arrays and full JSON output from `tools/estimate_budget.py`. Run every recorded command from the package root. The helper SHA256 is `5904c7233660dc937c30c6232a9d227e6b76666620b874e3e4a41c99a27f5e02`.

## Assumptions

- TP4 on four hypothetical 22.3 GiB devices, at 90% utilization: **80.28 GiB aggregate planning capacity**.
- E64 or E72, all 40 layers, native packed Marlin experts, and the provisional **12.6 GiB aggregate non-routed weight allowance**.
- **65,536 total live tokens per request**, including the measured continuation; active B is 8, 16 or 32 and reserved request slots equal B.
- Native mixed cache, replicated on all four ranks; scheduled-token capacity 2048; CED reserve explicitly zero.
- **4 GiB aggregate additional residual reserve**, separate from known indexer/candidate/state buffers.
- `--engram-gib 0`: no additional GPU table/hot-cache allocation is explicitly included. Host Engram payload is excluded.

## Core results

| Experts/layer | B | Resident weights, GiB | Known nonweight state, GiB | Total including 4 GiB residual, GiB | Extra margin per balanced rank, GiB |
|---|---:|---:|---:|---:|---:|
| E64 | 8 | 60.06093750 | 4.15393066 | 68.21486816 | 3.01628296 |
| E64 | 16 | 60.06093750 | 5.97570801 | 70.03664551 | 2.56083862 |
| E64 | 32 | 60.06093750 | 9.61926270 | 73.68020020 | 1.64994995 |
| E72 | 8 | 65.99355469 | 4.15393066 | 74.14748535 | 1.53312866 |
| E72 | 16 | 65.99355469 | 5.97570801 | 75.96926270 | 1.07768433 |
| E72 | 32 | 65.99355469 | 9.61926270 | 79.61281738 | 0.16679565 |

The margin column is **beyond the residual reserve already counted in the total** and assumes balanced ranks. It does not represent an observed free-memory margin. A cache pool can reserve memory ahead of live requests; record pool capacity and live block occupancy separately.

## Native component checks

- Growing global/index history: **890 bytes per original live token per rank** for even lengths; **55.625 MiB per request per rank at 64K**.
- Logical SWA floor: **2.578125 MiB/request/rank**; actual paging and prefill admission can be larger.
- Compressor rings: **96 KiB/request/rank** with speculation disabled.
- Default shared indexer scratch: **577.03125 MiB/rank**, counted once per rank, not once per layer.
- Candidate/top-k outputs at capacity 2048: **20 MiB/rank**.

## Engram table and map scenarios

Optional **8 GiB and 32 GiB aggregate host payloads** are synthetic capacity experiments only. They are not the selected pruning budget, quality-validated masks, or GPU table allocations. Generate and touch the intended pages in bounded chunks; record physical host RSS, pinned pages and initialization peaks.

A dense uint32 map over all **768,022,850 original rows** has the following hypothetical placement costs:

| Placement | Aggregate GiB | Average GiB per TP rank |
|---|---:|---:|
| One full map, sharded by original TP ownership | 2.861108072 | 0.715277018 |
| Full map replicated on every GPU | 11.444432288 | 2.861108072 |

Ownership can be imbalanced and the adapter can use a different representation. These are capacity calculations, not implemented map layouts. Actual GPU map, gather and staging bytes must be assigned to measured allocation categories. If they exceed the residual allowance available to those categories, the excess consumes remaining margin; do not count the same allocation both inside and outside the reserve.

## Initial 85% setting and stop points

The proposed initial E64/B8 run can start at **85% utilization**, which gives **75.82 GiB aggregate** under the same hypothetical device size. This is a conservative starting setting, not an alternative hardware measurement.

At 64K, the additional modeled margin per balanced rank at 85% is:

| Candidate | B | Margin/rank, GiB |
|---|---:|---:|
| E64 | 8 | 1.90128296 |
| E64 | 16 | 1.44583862 |
| E64 | 32 | 0.53494995 |
| E72 | 8 | 0.41812866 |
| E72 | 16 | -0.03731567 |
| E72 | 32 | -0.94820435 |

Increase utilization or concurrency only after checking actual worst-rank initialization/repacking, prefill, graph and request-turnover peaks. **E72/B32 at 90% has only approximately 0.167 GiB/rank of extra modeled margin**, so aggregate arithmetic does not justify promoting it. E72/B16 already slightly exceeds the 85% hypothetical cap.

Synthetic weights reduce downloads and disk needs; they do not reduce the required full-shape GPU storage. Verify actual packed values/scales, fixed modules and unique physical tensor allocations before interpreting these totals.

## Reproduce

The three complete command argument arrays are stored under `reproducible_runs` in `budget_scenarios.json`. For example, the B8 command is:

```bash
python3 tools/estimate_budget.py \
  --experts 64 72 --packing marlin_fp4 --fixed-gib 12.6 --engram-gib 0 \
  --gpus 4 --gpu-gib 22.3 --utilization 0.90 --workspace-gib 4 \
  --context 65536 --batch 8 --cache-layout native --cache-replication 4 \
  --max-num-batched-tokens 2048 --max-num-seqs 8 --ced-window 0 --format json
```

Repeat with both `--batch` and `--max-num-seqs` set to 16, then 32. The corresponding E64 and E72 outputs are already retained in the JSON artifact. The 85% comparison is calculated by replacing only the utilization cap; it does not change the requested component allocations.
