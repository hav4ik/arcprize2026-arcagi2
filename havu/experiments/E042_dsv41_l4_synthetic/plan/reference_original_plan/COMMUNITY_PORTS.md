# Community port watch: catonooka

Checked 19 October 2026 after the user supplied [this X post](https://x.com/catonooka/status/2112042193204740142). Direct X retrieval failed. The author's current posts were retrieved through [the public timeline mirror](https://www.sotwe.com/catonooka), and linked GitHub artifacts were inspected independently. This is a source audit, not a local model benchmark.

## Current evidence and decision

The author reports a custom V4.1-Flash-1017 EXL3 engine at 11–13 tokens/s on two RTX 3090s. Their posts identify PCIe expert streaming and Engram-related workspaces as bottlenecks and request hot n-gram extraction. These are prototype reports; the public code inspected did not include a reviewable DeepSeek engine or a pruned 1017 checkpoint.

The author's linked [maya-s24-turbo](https://github.com/catonooka/maya-s24-turbo/tree/ebbd1904be97979bd3dc496e9a81578f99c4ae89) is a GLM-5.3-Flash tuning/benchmark repository. Its README says it includes no model weights. The short link in the public post resolves to this repository, grounding the account attribution. The [project-maya fork](https://github.com/catonooka/project-maya/tree/00936ecae44c1850d22211ef4b290b656c6fa008) has a public tuned branch, but the inspected archive remains GLM/Maya infrastructure.

Keep the native Marlin-based runtime and corrected REAP collector as the primary plan. Revisit a DeepSeek-specific release when source, weight revision, numerical checks and relevant hardware measurements are available. Do not delay CPU Engram census while waiting.

## Concrete patterns to reuse if needed

| Component | Reuse value | Required adaptation |
|---|---|---|
| [Grouped EXL3 experts](https://github.com/catonooka/project-maya/blob/00936ecae44c1850d22211ef4b290b656c6fa008/src/kernels/cuda/exl3.cu), `win_experts()` | Group token assignments by expert to amortize packed-weight reads and decoding | The inspected SM80+ MoE path rejects hidden widths above 4096; DeepSeek needs 5120. Also adapt scratch, padded stride-eight plans and native gating/quantization. |
| [Grouped call integration](https://github.com/catonooka/project-maya/blob/00936ecae44c1850d22211ef4b290b656c6fa008/src/core/glm_window.cu) | Concrete grouping, per-token coefficients and output wiring | GLM scheduling and cache semantics do not implement DeepSeek CED. |
| [PLE row reader](https://github.com/catonooka/project-maya/blob/00936ecae44c1850d22211ef4b290b656c6fa008/include/strata/ngram/ple_reader.hpp) and [implementation](https://github.com/catonooka/project-maya/blob/00936ecae44c1850d22211ef4b290b656c6fa008/src/ngram/ple_reader.cpp) | Issue reads once IDs are known, collect before consumption; bounded cache, page deduplication and byte/wait counters | This is a different 90-byte-row table, not DeepSeek Engram. Reuse I/O structure only; preserve our own hashes, 264-byte rows, bounds and compact-map/miss contract. |
| [Benchmark toolkit](https://github.com/catonooka/maya-s24-turbo/tree/ebbd1904be97979bd3dc496e9a81578f99c4ae89/benchkit) | Context, concurrency and functional logging | Add full target-harness quality and native parity; smoke tests alone cannot validate pruning. |

The published [GLM speed profile](https://github.com/catonooka/project-maya/blob/00936ecae44c1850d22211ef4b290b656c6fa008/EXL3-CURRENT-FINDINGS.md) also enables lossy skipping of selected nonresident experts with `TRUNK_KEEP=6`. That differs from globally selecting a retained bank and refilling native top-k. Keep such approximate execution disabled in the reference calibration baseline.

Frequency pruning and exact row caching can share a census but have different miss behavior. A cache retrieves missing original rows; our compact pruned table uses the declared zero-before-Wkv rule for deleted rows. A smaller host table alone does not reduce the original lookup count per consumed token. Measure whether a retained working set fits a faster memory tier, as well as table capacity.
