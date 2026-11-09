# DeepSeek V4.1 Flash ARC implementation handoff

Prepared 19 October 2026 for serving a pruned DeepSeek-V4.1-Flash-1017 on four Kaggle L4 GPUs.

## First decision

Run **CPU Engram row census first**, select and validate a retained-row mask, then freeze that behavior before collecting fresh REAP statistics. Target **64 routed experts per backbone layer, top-k 6 (E64)**, with **E72** as the larger quality candidate. Provisional aggregate GPU backbone weights are **64.49 GB for E64** and **70.86 GB for E72**, excluding cache/workspace. Retained Engram tables live in host memory through the native pinned-UVA path plus a compact original-ID lookup adapter. The earlier 32–56 GB weight envelope and E48-first proposal are archived comparison cases, not current requirements.

Expert and Engram parameter budgets can be chosen separately, but their effects on hidden states are coupled. Validate Engram behavior before MoE scoring; if the Engram mask changes later, revalidate and recollect affected expert statistics. Neither quality nor L4 throughput has been demonstrated by this pack.

## Read and assign

| File | Purpose |
|---|---|
| [START_HERE.md](START_HERE.md) | Immediate agent assignments, concrete prompts and integration dependencies |
| [REAP_IMPLEMENTATION_CONTRACT.md](REAP_IMPLEMENTATION_CONTRACT.md) | Audited scoring/normalization rules, native-forward correction, distributed statistics and upstream integration defects |
| [CED_CALIBRATION.md](CED_CALIBRATION.md) | 8B/16B execution schedule, encoder/decoder score masks, first-target alignment and semantic Engram counts |
| [REAP_HARDWARE.md](REAP_HARDWARE.md) | One-card streamed and four/eight-card resident RTX calibration paths, long-indexer fix, memory and cost worksheet |
| MASTER_PLAN.md | Complete research, architecture, memory and batch arithmetic, methods, cost plan and decision gates |
| TASKS_ENGRAM_FIRST.md | CPU address census, global row selection, original-ID compaction, held-out validation and frozen-mask handoff |
| [DS4_REUSE.md](DS4_REUSE.md) | Reuse antirez/ds4 CPU hashing, bounded row I/O and export code; imatrix and GPU-runtime limitations |
| [COMMUNITY_PORTS.md](COMMUNITY_PORTS.md) | catonooka/Maya source check: useful grouping/I/O patterns and limits of the unreleased DeepSeek prototype |
| TASKS_SERVING.md | Terminal-agent tasks for environment, pinned kernels, pruned router, compact host Engram, TP4 and offline deployment |
| TASKS_PRUNING.md | Terminal-agent tasks for data provenance, full-position observers, REAP/HOPE, masks, selective loading, quality and recovery |
| KV_MEMORY_NOTES.md | Verified per-sequence TP4 cache payload, BF16 reference distinction and implementation workspace reservations |
| SOURCE_INDEX.md | Primary sources and evidence boundaries |
| tools/probe_environment.py | Read-only environment inventory; run inside Kaggle |
| tools/estimate_budget.py | Parameter, packed-weight, cache and optional generation-cost calculator |
| tools/compact_engram_reference.py | CPU synthetic contract for original-ID compaction and zero-sentinel mapping; not a model exporter or runtime adapter |
| tools/reap_stats_reference.py | CPU contract for conditional/weighted statistics, selected-set normalization, pair counts and distributed reduction invariants |
| tools/ced_phase_masks_reference.py | CPU input/target alignment and phase-count contract; not a model replay implementation |

Run environment/kernel checks alongside the CPU Engram census. Use a **proposed 10,000-task coverage pool**, verify root/generator lineage, and seed the pilot with existing compatible traces. Use tool-free traces as the required pruning/calibration workload. Tool-assisted calibration or comparison is optional only if requested later. API traces are enough for the census; expert importance requires an instrumented weight forward. After the Engram validation gate, start with **0.25–1M replayed tokens**, then target **1–8M scored positions** with their proper causal history if justified. Replaying every long trajectory from all 10,000 tasks is not a mandatory first pass. Export E64 first; compare E72 when measured headroom and held-out quality justify it.

[DS4_REUSE.md](DS4_REUSE.md) maps tested standalone Engram helpers from `antirez/ds4` into E01/E03. Its Linux Engram unit target passed during this research; the 1017 census/export adapter and GPU integration remain implementation work. Reuse those CPU components while retaining the current REAP and native SM89 serving bases.

## Commands available now

~~~bash
python3 tools/reap_stats_reference.py --self-test
python3 tools/ced_phase_masks_reference.py --self-test
python3 tools/compact_engram_reference.py
python3 tools/probe_environment.py > environment.json
python3 tools/estimate_budget.py --experts 64 72 --batch 16 --context 65536
python3 tools/estimate_budget.py --experts 64 72 --batch 32 --format json
~~~

The updated calculator defaults to E64/E72, 4 × 22.3 GiB GPUs, a 90% utilization ceiling and 12.6 GiB fixed GPU weights. It itemizes native global/SWA/compressor state, the default indexer and scheduled-token candidate buffers, and an optional 128-token CED tail reserved conservatively by default. **The 4 GiB workspace argument is now an additional residual reserve**, not a replacement for those explicit buffers. Use `--ced-window 0` only when CED state allocation is disabled and `--ced-window 1152` when the verified configuration retains that vision-aware tail. Host table bulk is outside VRAM; Wkv/projection weights, GPU maps, gathered outputs, staging and hot caches still count. Consult KV_MEMORY_NOTES.md and measure actual request-slot/prefill reservations before selecting engine limits.

The user reports approximately **80% pass@2 without tools and 84% with tools** for unpruned DeepSeek. These are supplied baseline context, not measurements or independently verified results from this research. The present task is to plan pruning for the **tool-free model behavior**; no Python-tool control or preservation of general coding capability is required. Compare quality loss against the original model under the same target encoding and solve policy.

## Status

The targeted primary-source and code audit is complete enough to assign the staged implementation now. It includes the REAP paper and canonical observer, native V4.1 calibration/export scripts, HOPE normalization, the exact 1017 forward, and the SM89/SM120 runtime. The audit found a native-forward shortcut in the upstream REAP collector that must be corrected and a 64K reference-indexer allocation that requires query tiling. These are explicit work orders in the new contract and hardware documents.

The environment and budget tools have been executed; compact-mapping, REAP-statistics and CED position references have CPU checks recorded under `verification/`. There is no working pruning or serving engine bundled here, and no full-weight, RTX or L4 performance claim. Native observer parity, long-context memory, distributed collection and pruning quality require execution. Code paths and commands marked proposed or illustrative in the task files must be implemented and validated by the assigned terminal agents.

The original unpruned checkpoint is approximately 510 GB. The full Engram table payload alone is 188.83 GiB; retained row count determines the compact host payload. Confirm effective host RAM, pinning limits, attachment capacity, source-download/conversion peaks and offline dependency compatibility. Hit coverage is a selection diagnostic; complete held-out ARC rollouts determine whether the frozen row mask is acceptable.
