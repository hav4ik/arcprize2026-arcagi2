# Task for the local coordinating agent

Implement and execute the inference experiments described in this archive for **DeepSeek-V4.1-Flash-1017 on 4×NVIDIA L4**, while model pruning proceeds separately. Read root START_HERE.md, EXPERIMENT_PLAN.md, AGENT_WORK_ORDERS.md, synthetic_contract.json, software_lock.json and experiment_matrix.json before editing code. The reference_original_plan/ directory provides the complete original supporting pack; the root synthetic-first plan controls the first rental.

## Objective and fixed starting point

The user will rent four L4s together. Prepare the code and environment before starting paid device validation where possible. The initial experiment must not require a full DeepSeek checkpoint download or checkpoint storage on the local PC. Use metadata and synthetic weights, generated in bounded allocations. First establish a correct, full-depth E64 inference path and continuous batching, then replace synthetic contents with real pruned tensors when available.

Preserve all 40 original backbone layers, original layer numbering, hidden size 5120, expert width 2304, full vocabulary, top-6, shared experts, native attention/cache topology, mHC, native quantization/activation boundaries and original Engram hash geometry. Start TP4, EP off, PP1, eager, language-only, DSpark/CED/prefix caching off. E72 and EP are conditional follow-ups. Use the supplied source pins and explicitly record the selected editable or wheel installation route; those sources are distinct.

## Required implementation

Delegate independent work with explicit file ownership or separate worktrees:

1. **Runtime:** add E64/E72 top-6 router support and reference tests. Implement a V4.1-aware synthetic initializer keyed by global tensor identity/coordinates and seed. Initialize valid packed values and scales in bounded tiles, then run normal post-load finalizers. Keep replicas consistent and all layers physically materialized. Stock dummy loading is not sufficient.
2. **Engram:** intercept backing allocation before the original table is created. Preserve original hash/TP ownership, map retained original IDs to compact rows, and return zero embeddings on misses before Wkv. Implement E0 all-miss smoke, E1 semantic hit/miss fixtures and E2 physically initialized/touched/pinned host-capacity tests. Start E2 around 8 GiB aggregate and treat 32 GiB as optional. Account for GPU maps separately and measure actual lookup traffic/locality.
3. **Harness:** reuse existing native operator tests and engine benchmark tooling. Add only the target-shape tests, model/storage audit, phase-memory recording and mixed continuous queue needed by this plan. Build the actual synthetic launch only after the interfaces exist; no supplied JSON file is a drop-in engine launch config.

Prepare CPU metadata immediately with tools/prepare_synthetic_metadata.py. The supplied tools/probe_environment.py and tools/estimate_budget.py are inventory/arithmetic helpers. The compact Engram CPU reference is available in reference_original_plan/tools/compact_engram_reference.py. Existing verification files describe CPU work; they are not L4 results.

## Device execution order

Follow the matrix dependencies: environment/native launch → native operator correctness → one full-shape synthetic layer → full 40-layer E64 allocation/finalization → short eager prefill/decode → E2 memory/traffic → 32K and near 64K → graph comparison → continuous batching. A one-card attention fixture uses 16 local query heads corresponding to a TP4 shard; do not treat a full 64-head TP1 model as that fixture.

Start B8, scheduled-token cap 2048 and memory utilization 0.85. Promote B16 and conditionally B32 only after measuring the worst rank. Check initialization/repacking, cache setup, prefill, graph capture and steady-state peaks. Record reserved cache pools separately from live blocks. Keep unknown allocations and assumptions visible.

Prioritize S1/S2/S3/S6 over optional larger-batch or E72 work. The paid session has a four-hour core and a two-hour conditional extension, counting all four reserved GPUs. If a prerequisite fails, capture its minimal reproducer and skip dependent cases; do not silently bypass model modules or download the full checkpoint as a workaround.

## Measurement requirements

Full-system cases execute real prefill and real decode. Use declared fixed output budgets/EOS policy and the full LM head. Account for warmup and all measured windows in the actual context length. For W64/M256, initial context is 65,216 for one window or 64,704 for three consecutive windows finishing at 65,536. Record actual intervals and distinguish consecutive windows from independent-seed repeats.

The continuous queue uses at least 3B requests for an initial batch plus two replacement-wave equivalents. Vary prompt/output lengths within the context limit. Prove actual completions, slot reuse and an interval of new prefill while older requests decode. Record TTFT, ITL, aggregate output rate, queue wait, end-to-end workload time, preemptions and the active/queued timeline.

Report native routing concentration and actual Engram hit/locality statistics. Allocated table size does not prove the accessed working set. Forced routing or injected address stress must be labeled separately from native execution. Graph flags alone do not establish graph replay or state correctness.

## Outputs

Return a compact run bundle using result_record.template.json, decision_report.template.md and handoff_contract.template.json. Include source/patch commits, actual package/import provenance, model/tensor/storage manifests, actual launch commands, operator tests, per-rank phase memory, host/mapping memory, workload hashes, routing/lookup distributions and mixed-batching evidence. Record failures, skipped cases and unresolved risks as clearly as successes.

Recommend whether to proceed to real E64, use a smaller measured operating point, fix a specific runtime blocker or revisit the hardware/model target. Synthetic runs establish engineering behavior under their declared distributions. Later real-weight work must recheck loading peaks, trained routing, Engram locality, representative throughput, termination and ARC quality. Preserve the original pruning sequence: validate/freeze Engram behavior, collect fresh REAP statistics, then select/export/evaluate experts.
