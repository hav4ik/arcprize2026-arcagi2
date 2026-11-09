# Start here: executable work assignments

Updated 23 October 2026, retaining the 19 October paper/source audit. The user corrected the rental to **eight RTX 6000 GPUs and 1 TB disk**. Start with [RENTED_NODE_START.md](RENTED_NODE_START.md) and [DISK_1TB_PLAN.md](DISK_1TB_PLAN.md) for the current machine and use [LOCAL_AGENT_PROMPT.md](LOCAL_AGENT_PROMPT.md) to launch the coordinating agent. This packet contains research, work orders and CPU reference helpers. The model loader, native observers, compact-table runtime adapter and production exporter still need to be implemented and validated on real weights.

## Decisions already made

Use the tool-free ARC harness. Preserve the exact 1017 tokenizer/encoding and native computation. First count Engram row addresses on CPU, select and validate a compact row set, and freeze its original-ID mapping and zero-before-Wkv miss rule. Then collect fresh expert statistics with all 384 original experts and that frozen Engram behavior. E64 is the primary deployment candidate; E72 is the larger comparison. Keep top-6, shared experts and native routing normalization unchanged.

Verify that the rented GPUs are RTX PRO 6000 Blackwell 96 GB and use the existing eight-card rental. Start a short TP4 baseline on a subset, then validate TP8/EP as the intended resident collector. Two independent TP4 workers require separate aggregate host-memory and throughput checks; they can share one immutable checkpoint. A corrected single-GPU packed streamer remains an optional reference/debugging implementation. It does not establish that the native FlashInfer serving backend can run the full model at TP1, and launching that streamer under torchrun does not implement a distributed collector. Read [REAP_HARDWARE.md](REAP_HARDWARE.md) for placement, long-context workspaces and cost accounting; the current 1 TB disk contract takes precedence over its earlier 2 TB recommendation.

Read [REAP_IMPLEMENTATION_CONTRACT.md](REAP_IMPLEMENTATION_CONTRACT.md) before adapting any upstream REAP code. It specifies the conditional denominator, quantization-sensitive native forward, tensor-parallel reductions, optional pair statistics and source defects found in this audit.

Read [CED_CALIBRATION.md](CED_CALIBRATION.md) for exact encoder/decoder position masks. Execute all required causal state, but score encoder prompt/continuation and decoder generation separately. The decoder's last prompt input predicts the first completion token. Count Engram on consumed encoder inputs once; physical warmup/replay does not add semantic frequency.

## Assign these workstreams now

| Owner | Initial scope | Concrete deliverable |
|---|---|---|
| Source and data agent | P00, P01, E00 | Pinned source/tensor manifest; exact tokenization; disjoint root/generator partitions; a small compatible trace sample |
| Engram agent | E01–E03 | CPU address parity, resumable global counts, retained-row manifest and bounded compact row exporter |
| Native calibration agent | P02a | Corrected packed reference, observer-on/off parity, valid-position masks, sparse sufficient statistics and query-tiled indexer |
| Runtime agent | Serving Tasks 0–2 adapted to SM120; original-model TP4 baseline then TP8/EP collector | Environment/backend report, native operator checks, E384 topology/observer parity and a concrete pre-sum observer patch; pruned-router work follows independently |

Workstream names are assignments, not existing installed commands. Agents can work independently until a listed dependency requires integration. The detailed outputs and acceptance criteria live in [TASKS_ENGRAM_FIRST.md](TASKS_ENGRAM_FIRST.md), [TASKS_PRUNING.md](TASKS_PRUNING.md) and [TASKS_SERVING.md](TASKS_SERVING.md).

### Source and data agent prompt

Implement P00/P01 and E00 from the packet. Keep source weights immutable. Pin the exact 1017 revision and all code/tokenizer revisions. Inventory tensors without downloading or expanding unnecessary copies. Establish exact serialized token IDs, root/generator lineage and separate calibration, model-selection and final-audit partitions. Seed the prototype from existing compatible tool-free traces; do not require a billion-token generation job or the entire proposed 10,000-task pool. Deliver manifests, a small replay fixture and an explicit list of encoding/data gaps.

### Engram agent prompt

Implement E01–E03 using the pinned ds4 CPU helpers in [DS4_REUSE.md](DS4_REUSE.md). Preserve the original hash geometry and canonical token map. Count eligible lookups, retain original rows under a global payload budget, and export their original bytes/scales with a compact map. Do not resize the hash modulus or renormalize retained embeddings. Use the included compact-row helper as a CPU contract, including module bounds, valid off-rank IDs and padding masks. Produce coverage curves and candidate manifests; reserve model-quality decisions for E04.

### Native calibration agent prompt

Implement P02a from the pinned LibertAI streamer, applying the native-forward correction in REAP_IMPLEMENTATION_CONTRACT.md. Only `Expert(h, native_gate)` may advance the teacher state. Collect `reap_native_amp` cheaply and collect both native/reference amplitudes on a small diagnostic subset. Maintain explicit scored-position validity and merge sums/counts by immutable shard ID. Start with short whole-prefix forwards; tile the indexer's query axis before long prefixes. Prove observer-on/off parity, routing/quantization parity and per-layer counts. This agent supplies the model forward needed for Engram E04; it does not wait for E04 to begin.

### Runtime agent prompt

First adapt Serving Tasks 0–2 to the rented SM120 node and establish a short native TP4/EP forward with all 384 original experts. Confirm the backend and identify the selected-expert down-projection contribution before `moe_sum`. Start calibration eager with CED decoder skipping, DSpark and prefix caching off. Track global expert IDs and correct TP/EP output reductions before taking norms. Prove TP4 observer-on/off parity, then verify TP8 outputs, phase counts, statistics and per-rank memory. Use TP8 as the intended main collector once verified; compare two independent TP4 workers only after aggregate RAM and shared-storage checks. Retained-expert router support and physical L4 tests are later or parallel deployment tasks; they are not prerequisites for original-teacher scoring. Count the whole eight-card rental in all cost records.

## Integration sequence

1. **P02a + E03 → E04:** compare original full-table behavior with candidate compact Engram masks. Use complete held-out tool-free rollouts as well as fixed-prefix diagnostics. A 32–64-root panel is a cheap screen, not evidence of a tiny quality delta. Freeze an acceptable mask or report the memory/quality frontier without silently choosing a loss tolerance for the user.
2. **E04 → P02b:** repeat the final replay and observer checks under the exact frozen mask. Record its checksum in every statistics shard. Reject mixed-mask merges.
3. **P02b → P03:** run a 0.25–1M processed-token pilot. Measure valid scored positions, real processed history, memory, I/O, throughput and cost. Increase to the proposed 1–8M scored positions only when coverage/ranking or quality uncertainty justifies it.
4. **P03 → P04–P07:** derive named REAP/native-amplitude/contribution-total candidates, run inexpensive masked diagnostics, then use a retained-original-ID loader for complete candidate execution. Larger expert counts provide failure diagnostics; E64/E72 remain the deployment targets.
5. **P07 → P09 and runtime handoff:** export a viable candidate, verify every retained tensor/scale/router row against the keep manifest, and measure the real four-L4 memory and batching envelope. Optional HOPE, recovery, CED and DSpark are later experiments.

No numeric ARC quality threshold was specified. Infrastructure and measurement can proceed immediately. The quality decision must state its observed delta and uncertainty, or remain a conditional Pareto choice; it must not manufacture an acceptable-loss target.

## Checks and completion evidence

The following commands already exist in the packet:

```bash
python3 tools/reap_stats_reference.py --self-test
python3 tools/ced_phase_masks_reference.py --self-test
python3 tools/compact_engram_reference.py
python3 tools/probe_environment.py > environment.json
python3 tools/estimate_budget.py --experts 64 72 --batch 16 --context 65536
```

CPU arithmetic/mapping checks establish those contracts only. Real-weight native parity, long-context peaks, distributed observation, complete-task quality and target-L4 throughput remain execution gates. Each agent should deliver its implementation, commands actually run, versioned measurements and remaining blockers. A loaded checkpoint or a fluent short completion is not the final success criterion.
