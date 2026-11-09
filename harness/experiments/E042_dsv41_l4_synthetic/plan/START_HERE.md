# Start here: local-agent handoff

**Packaged 21 October 2026.** This archive contains the complete inference-experiment planning handoff for DeepSeek-V4.1-Flash-1017, pruned toward E64/top-6 or E72/top-6, on four NVIDIA L4 GPUs for tool-free ARC-AGI-2.

## Current instruction

The user's first rental is **4×L4 together**. Start with **synthetic weights and metadata only**, preserving all 40 backbone layers and the native architecture/quantization. Do not make a full DeepSeek checkpoint download a prerequisite for the initial experiment.

Begin with E64, TP4, EP off, eager execution, language-only, and DSpark/CED/prefix caching off. The four-hour core and optional two-hour extension are spending caps after preparation, not promises that unfinished runtime code can be implemented within those hours. Detailed conditions and stop rules are in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

## Reading order

1. **[LOCAL_AGENT_PROMPT.md](LOCAL_AGENT_PROMPT.md)** — ready-to-use coordinating-agent task.
2. **[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)** — hardware, preparation, experiment sequence, metrics, stop rules and real-weight transition.
3. **[AGENT_WORK_ORDERS.md](AGENT_WORK_ORDERS.md)** — exact implementation work, source files and acceptance gates.
4. **[synthetic_contract.json](synthetic_contract.json)** and **[software_lock.json](software_lock.json)** — architecture/loading requirements and intended source/environment pins.
5. **[experiment_matrix.json](experiment_matrix.json)** — adaptive cases and dependency order; it is a specification, not an executable runner.
6. **[PACKAGE_INDEX.md](PACKAGE_INDEX.md)** — helpers, result templates and the full original reference pack.

## Document precedence

| Material | Role |
|---|---|
| User's latest instructions | Governs the work and any subsequently changed constraints |
| Root START_HERE.md, EXPERIMENT_PLAN.md and AGENT_WORK_ORDERS.md | Current initial-inference plan and implementation requirements |
| Root JSON specifications | Machine-readable version of the current plan; record any discrepancy before implementing |
| reference_original_plan/ | Original detailed source/serving/pruning references; preserves their internal links |
| PARALLEL_L4_EXPERIMENTS.md | Redirect marking the older experiment addendum as superseded |

The original reference pack's START_HERE.md concerns the separate pruning/calibration pipeline, including real weights and RTX PRO 6000 hardware. Its real-checkpoint serve commands are not the initial synthetic launch instructions. Root documents supersede the older first-rental/loading sequence while preserving the pruning quality order: validated Engram behavior, frozen mask, fresh REAP, then expert export/evaluation.

## What already exists versus what must be built

| Component | Status |
|---|---|
| Pinned original config and license | Supplied in metadata/ |
| E64/E72 metadata preparation | Supplied CPU helper; offline derivation and download guards checked |
| Environment inventory and memory arithmetic | Supplied CPU helpers |
| Detailed operator/source audit | Supplied plans and source links |
| Native E64/E72 router patch | Implementation work; GPU validation required |
| Bounded, format-aware synthetic GPU initializer | Implementation work; stock dummy loading is insufficient |
| Compact Engram constructor/lookup adapter | Implementation work; must prevent original-table allocation |
| Integrated workload and phase-memory instrumentation | Implementation work, reusing existing engine tooling where suitable |
| Actual L4 benchmarks and pruned-model quality results | Not executed during preparation |

Source pins and package identities are included; engine repositories, compiled GPU binaries and model weights are not vendored. This is a self-contained planning handoff, not an offline runtime image.

## Immediate CPU-only action

From the extracted package root:

```bash
python3 tools/prepare_synthetic_metadata.py --output fixtures/metadata-offline
```

This produces clearly labeled E64/E72 metadata without network access or a GPU. To download only the allowlisted tokenizer/encoding assets, use a fresh destination:

```bash
python3 tools/prepare_synthetic_metadata.py --output fixtures/metadata-serving --download-metadata
```

Inside the engine repositories, use their prescribed environment and Python runner. There is no validated full-model synthetic launch command in this archive: implement and validate the loader/Engram path first.

## Suggested parallel ownership

| Owner | Scope | First output |
|---|---|---|
| Coordinator | Source/environment choice, interface agreement, integration and gate decisions | Resolved lock and ordered task list |
| Runtime agent | Router shapes, bounded native initialization, standard finalization, tensor/storage audit | Reviewable patch plus focused fixtures |
| Engram agent | Pre-allocation backing hook, original-ID mapping, zero-before-Wkv misses, request state and E0/E1/E2 modes | Adapter patch plus semantic/memory fixtures |
| Harness agent | Inventory, native test invocation, workload generation, phase-memory metrics, graphs and continuous batching | Bounded harness and empty run records |

Use separate worktrees or clearly assigned files where edits would overlap. Coordinate constructor/loader interfaces before integrating. The coordinator merges the pieces and runs the dependency-ordered GPU gates on the rented host.

## Required final handoff

Return actual code/patch commits, the resolved software lock and launch commands, operator results, full-depth storage audit, per-rank phase memory, workload/traffic statistics, filled case records, continuous-batching timeline and a concise decision report. Preserve failures and skipped cases. Use the supplied result/report templates and distinguish CPU checks, GPU passes, blocked work and later real-weight evidence.
