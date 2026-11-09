# First 4×L4 synthetic experiment: decision report

**Report status:** implementation pending. This template contains no measured GPU result.

Fill this report from `result_record.template.json` records and their raw evidence. A blank or `null` value means unmeasured. Keep planned settings separate from the effective configuration.

## 1. Decision

- **Decision:** [advance to selected real tensors / run one targeted synthetic test / fix before more GPU time / stop this configuration]
- **Scope of the decision:** [exact model fixture, Engram mode, batch/context and execution mode]
- **Main evidence:** [one or two concrete findings with result links]
- **Blocking issue, if any:** [symptom, failing configuration and minimal reproducer]
- **Next action and owner:** [specific work that this result enables]
- **What remains unestablished:** [trained-model throughput, final Engram locality, real generation lengths, ARC quality and DSpark acceptance remain unestablished by synthetic tests]

Use these truth labels consistently:

| Label | Meaning |
|---|---|
| Implementation pending | Required implementation or execution is incomplete. |
| CPU passed | Named CPU checks passed; GPU behavior is unestablished. |
| GPU blocked | A GPU attempt was blocked or failed; identify the cause. |
| GPU passed | Named checks passed on the recorded hardware and fixture. |

A GPU pass for a fixture is not a model-quality pass.

## 2. Configuration and provenance

| Item | Planned baseline | Actually used / evidence |
|---|---|---|
| Hardware | 4×L4 on one host | [inventory, driver, effective RAM and PCIe/P2P topology] |
| Model | Synthetic E64, all 40 original layers; E72 optional | [manifest and initializer revision] |
| Shape | Hidden 5120, intermediate 2304, top-6, original 64 query heads | [resolved shapes and dtype audit] |
| Parallelism | TP4; EP off initially | [resolved TP/EP and actual local expert ownership] |
| Precision/cache | Native packed experts and mixed native cache | [actual dispatched kernels, values/scales/cache dtypes] |
| Optional features | CED, DSpark and prefix caching off | [effective normalized configuration] |
| Scheduler | 2048 scheduled-token cap; initial max-seqs 8 | [effective settings] |
| GPU memory setting | 0.85 initially; up to 0.90 after measurement | [setting per case and measured peaks] |
| Engram | E0 smoke → E1 semantics → E2 host-capacity fixture | [actual mode, payload, mapping and traffic] |
| Seed and request schedule | Deterministic and recorded | [model seed, token seed, workload hash] |
| Software | Pinned vLLM/FlashInfer pair plus recorded patches | [artifact hashes, source revisions and lock] |

Confirm that all intended layers/expert banks have distinct physical storage, replicated tensors agree across TP ranks, and shards correspond to one consistent logical model. List intentional cache-sharing relationships separately from unexpected weight aliases.

## 3. Allocation accounting

The initial allocation is **four hours core + up to two conditional hours**: **16 GPU-hours core, 24 GPU-hours total cap**. Count all four reserved cards during one-card fixtures, compilation and idle periods.

| Field | Actual |
|---|---|
| Allocation start/end | [UTC timestamps] |
| Reserved wall-clock hours | [measured] |
| Reserved GPUs | 4 |
| Allocated GPU-hours | [4 × reserved wall-clock hours] |
| Active kernel time | [reported separately] |
| Compilation / initialization / idle time | [breakdown] |
| Reason for conditional extension | [specific unresolved question, or none] |

If a gate fails, preserve the smallest reproducer and identify whether another paid attempt can answer a concrete question. A spending cap does not imply that unfinished engineering passed.

## 4. Stage decisions

| Stage | Truth status | Pass/fail evidence | Advance / stop |
|---|---|---|---|
| Environment and native launch | Implementation pending | [ABI, real CUDA launch, memory and host limits] | [decision] |
| Attention, indexer, MoE/router, dense/mHC fixtures | Implementation pending | [existing tests at unchanged tolerances; actual planned shapes] | [decision] |
| Full E64 initialization/repacking | Implementation pending | [all 40 layers, legal packed values/scales, no accidental whole-model expansion] | [decision] |
| Eager topology and E0/E1 state | Implementation pending | [source/reuse relationships, finite outputs, request isolation] | [decision] |
| E2 physical table and true prefill | Implementation pending | [host pages touched/pinned, map bytes, real cache construction] | [decision] |
| Near-64K decoding | Implementation pending | [actual context interval and maximum-rank memory] | [decision] |
| Graph comparison | Implementation pending | [actual replay path and changing-input/state checks] | [decision] |
| Continuous batching | Implementation pending | [request turnover, overlapping prefill/decode and no stale state] | [decision] |
| Conditional EP/E72/seed/boundary test | Implementation pending | [why this comparison was needed] | [decision] |

Do not substitute configured maximum context for actual live context. Directly filling synthetic native caches is valid for an isolated decode/indexer benchmark; label it and exclude it from claims about true prefill or complete request latency.

## 5. Memory envelope

For **every rank**, collect current and peak CUDA allocated bytes, current and peak reserved bytes, device-free bytes, and sampled NVML used/free/peak bytes at these phases:

**Initialization → repacking → short eager → true prefill → graph capture/replay → steady decode → continuous request turnover.**

Record whether CUDA peaks were reset at each phase and how NVML sampling might miss brief peaks. Include other GPU processes. The decision uses the worst rank and worst relevant phase.

| Phase / configuration | Worst rank | Peak allocated | Peak reserved | NVML peak used / minimum free | Evidence |
|---|---:|---:|---:|---:|---|
| Initialization / repacking | [ ] | [ ] | [ ] | [ ] | [ ] |
| True prefill | [ ] | [ ] | [ ] | [ ] | [ ] |
| Graph capture/replay | [ ] | [ ] | [ ] | [ ] | [ ] |
| Steady decode | [ ] | [ ] | [ ] | [ ] | [ ] |
| Continuous batching | [ ] | [ ] | [ ] | [ ] | [ ] |

Also report **cache-pool capacity/reserved bytes versus live blocks/payload**. A full reservation shown by NVML does not mean every request consumes that pool. Account for expert values/scales, fixed modules, replicated caches, indexer scratch, graph pools and Engram maps/staging without double-counting.

For the host, record effective cgroup RAM/swap limits, current/peak memory, RSS by process, PSS or equivalent shared-mapping accounting, measured pinned bytes, physical table payload, and host/GPU mapping bytes. Summed process RSS can double-count shared pages; it is not proof of separate physical copies.

**Capacity decision:** [configuration that fits, retained per-rank operational margin, measured phase that limits it, and missing allocation evidence]

## 6. Engram evidence

| Mode | Provisioned storage | Observed traffic | Permitted interpretation |
|---|---|---|---|
| E0 all-miss/zero sentinel | [bytes] | [misses and whether the module was bypassed] | Initial smoke only. |
| E1 semantic fixture | [rows/bytes by bank] | [retained and missing rows, changed-input tests] | Lookup and state semantics. |
| E2 host-capacity fixture | [physical/touched/pinned payload and mapping] | [hit/miss counts, unique rows/pages, working set and reuse] | Synthetic capacity and lookup-cost sensitivity. |

The table size is not the observed working set. Give the hit-rate denominator and exclude invalid/padded lookups consistently. A tiny hot set in a large table and a broadly accessed table are different experiments.

**Engram conclusion:** [what was physically allocated, what was actually accessed, which behavior is synthetic and what the final row mask must revalidate]

## 7. Timing protocol and results

Prefer **R=3 nonoverlapping timed windows**, **W=64 warmup decode steps** and **M=256 measured decode steps per window** when the allocation permits. Record actual W, M and R, including any shorter run.

For consecutive windows, leave room for **W + R×M** positions below the context limit. Record each window's actual starting/ending context interval. State whether cache/model state was reset between windows; consecutive windows are not independent seed replicates.

- **Token/output policy:** [input source, fixed output budget, EOS and special-token handling]
- **Termination policy:** [length, EOS, cancellation, timeout or error; counts by reason]
- **True prefill:** [executed or bypassed; included in which timing interval]
- **Timing method:** [CUDA/synchronization method and client timestamp method]
- **ITL source:** [per-token timestamps or explicitly labeled chunk timestamps]
- **Profiling:** [separate run, or measured timing overhead]
- **Replicates:** [window count, independent launches and independent model seeds]
- **Requests:** [submitted, admitted, completed, failed, cancelled and unfinished]

| Case | E / B | Actual context interval | Engram mode | Actual graph mode | Output TPS, window range | p50/p95 ITL | p50/p95 TTFT | Failures |
|---|---|---|---|---|---|---|---|---|
| [case ID] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Give output-token counts and elapsed seconds, not TPS alone. Report queue waiting and completion latency separately. Define end-to-end throughput from its exact interval and report both completed requests/sec and output tokens/sec. Keep cold initialization/JIT, prefill and steady decoding distinct.

For continuous batching, show at least two waves of request replacement, admission while other requests decode, overlapping prefill/decode work and reused cache/request slots. Attach the scheduler trace. Use the declared engineering length distribution; do not call it an observed ARC workload without matching data.

## 8. Routing and graph interpretation

**Routing:** [native router or explicit stress assignment; per-layer histograms, unique experts per batch and maximum load/mean]

Spread-out and concentrated expert assignments are stress scenarios. Neither is automatically a mathematical throughput bound. Natural random routing is also not evidence of final retained-model routing.

**Graphs:** [requested, resolved and actually used modes; capture/replay counts; eager fallbacks and reasons]

Compare identical inputs/state against eager execution. Include changing tokens, positions, valid/invalid rows, Engram hashes/lookups and request-slot reuse. A configured graph mode without observed replay is not a graph result.

Preserve existing test tolerances. Any added full-shape or full-model comparison needs its own same-quantization reference and rationale; do not promote a small operator test's tolerance into a blanket 40-layer acceptance threshold.

## 9. Handoff to the real-weight stage

- **Accepted runtime contract:** [shapes, original layer ownership, dtypes, maps and artifact format]
- **Working synthetic configuration:** [exact case/result reference]
- **Measured capacity constraints:** [maximum-rank weights/state/graph peaks and cache capacity]
- **Main bottleneck:** [profile evidence]
- **Remaining implementation blockers:** [bounded list]
- **Next real-weight input:** [selected expert tensors first, then full retained E64 and frozen Engram]
- **Required reruns:** retained-tensor/native-arithmetic checks; loader/repack peaks; final map/table footprint and locality; route concentration; the selected runtime point and one justified neighbor.
- **Final validation:** the candidate's own tool-free ARC generations, output lengths, stopping, timeouts and quality/runtime tradeoff.

**No ARC accuracy, trained-model throughput or DSpark acceptance conclusion follows from this synthetic report.**

