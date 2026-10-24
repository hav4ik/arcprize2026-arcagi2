# Full experimentation plan: DeepSeek-V4.1-Flash-1017 on 4×L4

**Date:** 20 October 2026. **Goal:** start inference engineering in parallel with pruning, using the first four-L4 rental and no original checkpoint download. **Primary:** E64/top-6, all 40 original layers. **Alternative:** E72/top-6. **Deployment task:** text-serialized, tool-free ARC-AGI-2 with continuously batched decoding.

This is an execution plan backed by source inspection and CPU budget calculations. GPU work is proposed. There are no measured L4 throughput, memory-peak or model-quality results in this package. Reference IDs resolve through SOURCE_INDEX.md.

## 1. Recommendation and decisions to obtain

Generate **synthetic weights in native checkpoint-compatible packed formats**, using bounded per-rank tiles, then run the standard runtime finalizers. This preserves repacking and expansion costs. Start with the audited SM89 vLLM fork and its paired native V4.1 FlashInfer implementation. Use ordinary TP4, with expert parallelism disabled initially. Retain the original architecture, full vocabulary and all 40 layers. The synthetic model changes only the routed-expert count and weight contents; compact Engram backing is an explicit additional adapter.

The first rental should answer, in this order:

1. Do the actual L4 driver, native attention/indexer, router and packed MoE kernels execute correctly at the required shapes?
2. Can the complete E64 model initialize, repack and run without hidden full-table allocations or weight duplication?
3. How much per-rank memory remains after cache allocation, graph capture, Engram metadata and prefill peaks?
4. Does genuine prefill and decode work at 32K and near 64K context with correct request state?
5. What throughput/latency tradeoff appears at 8, 16 and conditionally 32 live requests under a declared synthetic workload?
6. Can completed requests be replaced while other requests continue decoding, including new prefill during decode?
7. Should the project proceed to real E64 weights, test E72, or fix a specific blocker first?

The outcome is a feasible runtime configuration and a measured bottleneck list. Model accuracy, trained routing, real Engram locality and ARC solve throughput require later real-weight runs.

### Engine scope and DeepInfer reuse

DeepInfer is useful as an official V4.1 reference and a source of scheduling, deterministic sampling and later DSpark/CED ideas. Its current release lists NVIDIA SM100 and Ascend 950 support and describes offline inference with additional cache/service infrastructure outside the release. Its published decode experiment uses 64 SM100 GPUs. These facts do not establish an L4 path. Use the existing SM89 native vLLM implementation for the first session. [D1]

SGLang remains a later comparison. It belongs in the first session only if a concrete commit already supports the same L4 kernels, compact Engram semantics, E64 routing, cache layout and quantization. Otherwise it is a separate integration project. Obtain one correct vLLM baseline first; reopen the comparison for a measured bottleneck or an available equivalent port. [V1, S1]

## 2. Initial rental and download policy

| Resource | Recommendation | Check / purpose |
|---|---|---|
| GPUs | **4×L4 on one host** | Target topology from the first rental; inspect actual PCIe/P2P behavior. NVIDIA lists 24 GB, 300 GB/s and PCIe Gen4 x16 per L4; use measured usable bytes for budgets. [H1] |
| Host RAM | **128 GB initially** | Four workers, software, bounded initialization and an 8 GiB aggregate synthetic Engram payload; inspect the effective cgroup limit. |
| Optional RAM | **192–256 GB** | Comfort for a 32 GiB aggregate host-table scenario, pinned pages and development. This is a planning recommendation, not a measured minimum. |
| CPU | **16–32 vCPUs**, or a fixed larger provider bundle | Scheduling/tokenization, four workers and initialization; inspect contention and NUMA. |
| SSD | **150–200 GB** | Software/container layers, CUDA dependencies, JIT/build caches, source and traces. A prepared environment may fit 100 GB; a broad source build can use more. |
| Duration | **4-hour core + up to 2 conditional hours** | **16–24 allocated GPU-hours**. All four GPUs count during single-GPU fixtures or CPU setup. |

These RAM/disk recommendations assume the bounded synthetic implementation. The original Engram constructor would allocate approximately **188.83 GiB of native host payload**, distributed across TP owners, before other host memory. It must be intercepted. [M1, V4]

**No DeepSeek checkpoint shard is needed in the first session.** The supplied 3,311-byte config can produce synthetic E64/E72 metadata offline. The optional metadata helper downloads only tokenizer.json, tokenizer_config.json and two encoding files, with 128 MiB/file and 256 MiB total caps. It never requests weights or a weight index. PyTorch, CUDA components and engines still involve software downloads.

Generate GPU weights and the host Engram payload in memory. Do not serialize a 60–66 GiB synthetic checkpoint. Save seeds, generator version, selected checksums and manifests. The local PC can prepare metadata, edit code and inspect results without room for any DeepSeek weights.

## 3. Fixed baseline and provenance

Use model revision **1bae0e1088993b9b82de08f59de5be5a34e42170**. The helper changes **text_config.n_routed_experts**, from 384 to 64 or 72, leaving every other configuration field unchanged. It labels the result synthetic metadata, not a pruned checkpoint. [M1]

| Area | Baseline |
|---|---|
| Model |40 layers, hidden 5120, routed intermediate 2304, 64 query heads, one KV head, full vocabulary 129280 |
| Routing |E64, top-6, one shared expert per layer; E72 conditional |
| Parallelism |TP4, EP off, PP1; local TP expert intermediate **576**, not a presumed 640 |
| Weights |Native packed MXFP4 experts, native MXFP8 dense inputs, normal Marlin/dense finalization including BF16 wo_a |
| Attention/state |Original KV sources 2/8/14/20; index refresh 2/8/14/20/24/28/32/36; local window 128; original compression schedule and mHC streams |
| Engram |Original layer IDs 1/14 and hash geometry; explicit bounded compact pinned-host backing and missing-row policy |
| Optional features |Language-model-only; DSpark, CED and prefix caching off |
| Scheduling |Start 8 live sequences; scheduled-token cap 2048; record chunked-prefill behavior |
| Cache |Native mixed FP8/FP4 path selected by kv-cache-dtype=fp8; block size 128 |
| Context |2K/8K smoke, then 32K and near 64K; long-case engine limit 65536 |
| GPU-memory utilization |Start 0.85; increase to 0.90 only after observed peaks allow it |
| Execution |Eager first; compare a supported graph mode and record actual replay/fallbacks |

The JSON files in this package are planning specifications. They are not unmodified vLLM launch configs. No working synthetic CLI flag or configurable dummy-loader interface is assumed. The implementing agent must create the launch script after the custom path exists and record the arguments actually used.

Pin the editable vLLM baseline to **3f3bbaea916707c851ca027b87b97fb6605998b8** and paired FlashInfer source to **d4eef3c44c70594ff18b9126381ae6a978694e3f**. The publisher's stack uses Python 3.12, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1 and Transformers 5.16.1. Record the real driver, import paths, package/wheel hashes and patch commit. [V1]

The released Vision11 wheel's Python source is 86d35745c8797c9d2524877fb235084a6a5f5a7d, distinct from the editable pin. Do not label one as the other. Prepare a compatible environment and tracked patch route before paid testing; a full source rebuild is not part of the four-hour assumption. Follow each repository's build/environment instructions.

Set VLLM_USE_V2_MODEL_RUNNER=1 and FLASHINFER_CUDA_ARCH_LIST=8.9. Keep VLLM_MARLIN_INPUT_DTYPE unset; the V4.1 path preserves specific activation quantization boundaries. Do not apply the SM120-specific TRITON_PTXAS_BLACKWELL_PATH recipe to L4. After dependencies and metadata are ready, use local paths and offline Hugging Face settings for startup. Verify a missing asset fails visibly instead of triggering a checkpoint download. [V1, V3]

## 4. Preparation before starting the rental clock

Prepare the following on the PC or a small CPU machine. CUDA correctness remains a rental gate. The four-hour core assumes code and packages are ready for device validation; it is not a promise to design and debug the complete adapter in four hours.

### A. Metadata, environment and workloads

Run tools/prepare_synthetic_metadata.py. Commit the source hash, derived manifests and intended software lock. Prepare deterministic text-token workloads with checksums. Use the exact tokenizer for realistic hash/serving exercises; standalone operator fixtures can start before tokenizer download. Keep arbitrary text inputs within valid text-token IDs, excluding image/control tokens unless deliberately testing them.

Prepare a container or wheel environment, implementation patch, native tests and bounded harness. Reuse existing vLLM request/benchmark tooling where possible; add only required metrics and model-specific fixtures. A new general-purpose benchmark framework is outside the critical path.

### B. E64/E72 router support

The inspected specialized whitelist contains (384,6) and (128,3). Add (64,6) and (72,6) with focused tests. Preserve sqrt(softplus) scoring, correction bias for selection only, unbiased-score normalization, scale 1.5 and tie handling. Mask padded lanes for E72. Retain the separate draft entry even with DSpark disabled. A whitelist edit alone does not prove tensor loading or expert dispatch. [V2]

### C. Format-aware synthetic initialization

**Stock --load-format dummy is insufficient.** It constructs the model before initialization, skips nonfloating CUDA tensors and uses a whole-parameter FP16 temporary for FP8 initialization. Packed weights/scales therefore remain zero or uninitialized in important paths, and Engram still requests its full backing. The dummy-loader constructor also rejects nonempty extra configuration. [L1–L3]

Implement a guarded V4.1-specific initializer or dedicated loader. Visit each unique parameter once, generating bounded tiles or owned slices. Derive values from global tensor name, logical coordinates and a global seed so replicas match, TP slices represent one model, chunk size does not change values, and different layers have distinct contents.

Use controlled finite values and existing native quantizers. Do not randomize raw scale bytes, use 255 as an ordinary UE8M0 scale or leave experts zero. Initialize norms/mHC deliberately, preserve semantic buffers and invalid padding, and verify nontrivial routing and finite state. Retain separate physical parameter storage for all 40 layers. Run the normal expert/dense/wo_a/mHC finalizers once and measure their allocations. Avoid a whole-model BF16 temporary. AGENT_WORK_ORDERS.md maps these requirements to source and fixtures. [V3, L1–L3]

### D. Compact Engram before allocation

ParallelEngramEmbedding directly creates pinned host FP8 values and uint8 scales in its constructor. A replacement after construction is too late. Add a constructor-level hook separating original logical hash dimensions from physical retained capacity. Misses return a **zero embedding before Wkv/projection**, followed by native module computation. Do not emulate row pruning by changing hash moduli, deleting Engram layers or bypassing their computation. [V4]

| Mode | Purpose | Valid inference |
|---|---|---|
|**E0: all-miss**|Tiny zero-sentinel path, native module still runs|Assembly/state smoke only |
|**E1: semantic**|Known retained original IDs; hits/misses, reset/chunk/interleaving fixtures|Mapping, TP ownership and missing-row arithmetic; not table latency |
|**E2: capacity/traffic**|**8 GiB aggregate** native host payload initially; optional 32 GiB; initialize, touch and pin pages|Declared memory/staging capacity and separately observed access pattern |

The 8/32 GiB values are experimental scenarios, not final retention budgets. They are totals across TP owners, not per worker. Initialize bounded chunks, including scales; record RSS, cgroup charges, pinned bytes and NUMA placement. Untouched virtual pages or unpinned stand-ins do not establish capacity.

**Allocated table size does not prove lookup traffic.** A 32 GiB table with almost all misses or accesses confined to a few rows is not a 32 GiB working set. Measure hits/misses, distinct rows, reuse, bytes staged/touched and latency. Use unchanged native hashes for semantic parity. For traffic sensitivity, use declared original-ID traces with spread/concentrated reuse. Any test-only injected address schedule must be separately labeled synthetic; retain real mapping/gather/dequantization operations and do not count it as native-hash or ARC evidence.

Account for GPU mapping explicitly. A dense uint32 map over all original rows is **2.861 GiB**. Replicated on each GPU, that is 2.861 GiB per rank. A TP-owner-sharded dense map averages 0.715 GiB/rank, with exact sizes determined by ownership. A sparse compact map has different memory and lookup costs. Measure the selected design before deciding batch capacity. [M1, V4]

## 5. Memory budget and initial batch sizes

The routed-expert runtime estimate is:

`40 × E × 3 × 5120 × 2304 × (0.5 + 2/32) bytes`.

This models packed four-bit values and BF16-expanded group 32 scales in the audited Marlin path. Add **12.6 GiB aggregate non-routed weights**, provisionally covering attention, shared experts, embeddings/LM head and wo_a expansion. It excludes host Engram payload, extra GPU table/hot cache or mapping assumptions, KV and workspaces. Replace this allowance with measurements. [V3; supplied pruning pack]

| Candidate | Routed weights | Provisional GPU weights | Average/rank |
|---|---:|---:|---:|
|**E64**|47.46 GiB|**60.06 GiB**|15.02 GiB|
|E72|53.39 GiB|65.99 GiB|16.50 GiB|

Top6 remains top-6: fewer stored experts do not proportionally reduce active expert operations per token. Batched reuse and routing concentration affect throughput.

Native TP4 growing global/index cache uses **890 bytes per original live token per request per rank**. At 65536 tokens this is 55.625 MiB/request/rank, plus a 2.578 MiB logical local-window floor and compressor rings. These caches are replicated under TP4. Default indexer scratch is 577.03125 MiB/rank; candidate/top-k arrays add 20 MiB/rank at cap 2048. Extra pages, graphs and Engram buffers require measurement. [V5]

The following planning table assumes 64K, CED off, cap 2048, reserved slots equal to batch, and a **separate 4 GiB aggregate residual allowance** for other workspace/maps/graphs/staging. Hypothetical usable memory is 4×22.3 GiB, with a 90% cap of 80.28 GiB. These are arithmetic totals, not measured peaks.

| Live requests | E64 modeled total | E72 modeled total | E64 extra margin/rank at90% | E72 extra margin/rank at90% |
|---:|---:|---:|---:|---:|
|8|68.21 GiB|74.15 GiB|3.02 GiB|1.53 GiB|
|16|70.04 GiB|75.97 GiB|2.56 GiB|1.08 GiB|
|32|73.68 GiB|79.61 GiB|1.65 GiB|**0.17 GiB**|

Margins are beyond the modeled 1 GiB/rank residual. A replicated 2.86 GiB/rank map alone exceeds that residual. Uncounted graph/prefill costs consume the apparent margin. The largest rank determines success; aggregate balance is an assumption. At 85%, E64/B32 has only about 0.54 GiB/rank beyond the residual, so start E64/B8. E72/B32 is especially fragile.

Measure the **reserved cache pool and live occupancy separately**. An engine can preallocate its cache, leaving NVML usage flat as occupancy grows. Record token capacity, free/occupied blocks and request preemption/recompute. Measure separate construction/init/repack, cache setup, prefill, graph-capture and steady-decode peaks. A post-load parameter count alone does not prove fit.

To reproduce a row, from this package root:

```bash
python3 tools/estimate_budget.py --experts 64 72 --packing marlin_fp4 \
  --fixed-gib 12.6 --engram-gib 0 --gpus 4 --gpu-gib 22.3 \
  --utilization 0.90 --workspace-gib 4 --context 65536 --batch 8 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 --ced-window 0 --format json
```

Repeat with batch/max-seqs 16 or 32. --engram-gib means additional **GPU** table/cache bytes, not the host 8/32 GiB. Account for the actual map against the residual without double-counting. Exact CPU results are in verification/.

## 6. First rental: time caps and correctness gates

These are spending caps, not runtime predictions. If an earlier gate fails, save a minimal reproducer, relevant logs and the source lock, then stop dependent tests. A diagnosed blocker is a useful outcome. Do not rescue a blocked synthetic path with a full checkpoint download.

| Elapsed node time | Work | Evidence / decision |
|---|---|---|
|0:00–0:15|Inventory and native CUDA launch|Actual four L4s, usable bytes, host/cgroup limits, driver, topology/P2P and provenance |
|0:15–1:00|Operator/shape gates|Attention/indexer, router, E64 packed MoE, EngramE1 and changed-input graph fixtures |
|1:00–2:00|Full 40-layer initialization and S0|Unique storage, finite state, eager B1 → B8 short decode, bounded E2 table/map allocation |
|2:00–3:00|S1/S2 long context and S3 graphs|B8 at 32K/near 64K; actual prefill/caches; B16 at 32K eager versus supported graphs if room |
|3:00–4:00|S6 continuous batching; conditional S4/S5|Queue turnover and concurrent prefill/decode; a larger-batch/long-context winner if time |
|4:00–6:00 conditional|One unresolved decision|Matched EP, E72 one point, larger Engram scenario or needed repeat; prioritize one question |

Independent small correctness fixtures may use separate cards; isolate performance measurements. Keep unrelated builds, row census and competing GPU kernels out of timing. Count setup/build/JIT time in the paid session, even when excluded from steady-state TPS.

### G1: native correctness on actual L4

Use a **16-query-head TP4-shard fixture**, not the full 64-head model at TP1. Reuse paired FlashInfer tests in tests/attention/test_deepseek_v41_mixed_sparse.py for packing, selected rows, candidate boundaries, inverse RoPE and changed-input graph replay. Exercise the actual indexer/candidate-selection path too. [V6]

Reuse tests/kernels/moe/test_deepseek_v41_marlin_moe.py and extend it to E64/E72, H5120, local I576 and top-6. The existing E4/top2 small case is insufficient alone. Keep native activation rounding and router-weight placement in the reference; check padded lanes and nonfinite/overflow behavior. Existing per-operator tolerances belong to those tests, not a made-up40-layer parity tolerance. [V7]

Pass requires actual kernel execution with correct formats/shapes/semantics. Imports, whitelist edits and trivial all-zero output do not pass. Isolated random-KV attention tests are useful but do not establish full-model prefill or serving.

### G2: full-depth model and memory

Instantiate all 40 layers at E64. Inspect unique storage bytes/pointers, actual dtypes and local shapes. Confirm no vision tower, speculative hidden buffer or unexpected CED allocation. Verify original source-layer sharing, mHC finalizers and compact Engram behavior. Record initialization tile size, original/final packed bytes, transient peaks, per-layer activation statistics and routing histograms.

Run real prefill plus decode eagerly, B1 then B8. Establish the declared E2 table and mapping before using later results as deployment capacity evidence. Fail on unexpected aliases, uninitialized scales, nonfinite state, invalid expert IDs, wrong cache ownership or hidden full-table allocation. If the custom path is unfinished, label full-model work blocked and retain only valid operator findings.

## 7. Adaptive experiment matrix

Avoid a Cartesian sweep. Keep unspecified controls fixed to Section3. Use E2 for capacity/performance claims and report its actual payload and lookup pattern.

| ID | Model / live requests | Context | Purpose | Dependency |
|---|---|---|---|---|
|**S0**|E64 B1 → B8|2K→8K|Eager full-model smoke; E0/E1 → E2|G1 and custom path |
|**S1**|E64 B8|32K|Eager baseline and memory|S0 |
|**S2**|E64 B8|near 64K|Long prefill, indexer/layer20, cache/decode peak|S1 |
|**S3**|E64 B16|32K|Matched eager versus supported graph mode|S1 and graph fixtures |
|S4|E64 B32|32K|Conditional throughput/capacity; B24 only to locate boundary|S3 and measured margin |
|S5|E64 best passing B|near 64K|Selected higher-concurrency long point|S2 and selected S3/S4 |
|**S6**|E64 selected B8/16/32|Mixed8K/32K/near 64K|Continuously queued requests and prefill/decode overlap|S1; use a completed safe setting |
|C1|E64 one selected B/context|Matched|EP disabled/enabled pair with same logical weights/workload|S6 and EP correctness |
|C2|E72 B8 orB16|One32K or near 64K point|Repack/allocation and one informative case|E64 passes; E72 kernel gate and headroom |

S1/S2/S3/S6 have priority over S4/C1/C2. Complete continuous batching at B8 before chasing B32 if necessary. Do not start B64, many graph modes, PP, a different cache layout, CED or DSpark in this session. For the EP comparison keep attention and dense-layer TP4, PP1 and DP1 fixed. EP4 gives 16 full-width local experts for E64 or 18 for E72, each with I2304; attention still has 16 local query heads. Verify that layout instead of reusing the TP-only I576 shape assumption. [V2, V3]

### Workloads and context accounting

For full-system tests, execute **actual model prefill and decode**. Randomly filling cache bytes is an isolated operator/decode fixture, not a prefill or end-to-end result.

Use fixed output budgets and a declared EOS/special-token policy so random EOS does not end requests early. Retain the full vocabulary and LM-head computation. Reuse a fixed-token replay path if already available; otherwise use deterministic seeded prompts and a fixed output budget, recording generated IDs and routing. Avoid building a complex replay framework. Graph correctness includes targeted internal state comparisons; long autoregressive text equality is not the sole criterion when small numeric differences can change token choices.

Account for warmup and all timed windows:

`initial_prompt_tokens = target_final_context - warmup_tokens - windows × tokens_per_window`.

Start with 64 warmup decode tokens and 256 measured tokens. For one near 64K window, prefill 65,216 tokens and finish at 65,536. For three contiguous windows from one prefill, start at 64,704. Report actual timed intervals: a configured 64K maximum with a 2K prompt is not a 64K result. Exclude warmup, JIT and capture from steady-decode timing, but include them in paid-session accounting. Report cold-start-inclusive elapsed time separately from the timed request workload, whose end-to-end interval runs from first timed submission to final timed completion.

Use one timed window for exploratory cases. Recheck the winning comparison with three windows or repeats, matching intervals, inputs, seed and Engram mode. Consecutive windows are not independent model-seed replicates. Do not claim improvement smaller than observed variation. Profile only a selected bottleneck case and keep profiler overhead out of unprofiled TPS.

### Routing and Engram sensitivity

The full model uses its native router on synthetic hidden states. Log selections per layer, unique experts per batch and concentration. Random weights do not guarantee balanced routing: repeated prompts or random-output collapse can skew traffic. If a conclusion depends on concentration, run a second seed and one MoE spread/hot routing stress fixture.

Forced spread/concentration are sensitivity cases, not guaranteed speed bounds; GEMM batch sizes, reuse and communications change. Engram size and hit/locality are likewise separate measurements. Revisit both with real weights and actual ARC histories before forecasting final throughput.

### Continuous batching

A static batch decoded for 256 steps is insufficient. Keep the server backlogged and submit at least 3B requests: an initial B plus two replacement waves, with different output lengths. Confirm actual slot turnover in the request timeline. Mix prompt lengths around 8K/32K/near 64K and output budgets such as 128/256/512/1024, always below 65,536 total tokens.

Observe admissions, completions and reuse of cache/Engram history slots. Show an interval where a new prompt is processed while older requests decode. Verify resetting one request cannot contaminate another's token/hash history, compressor rings, positions, candidate state or cache. Record any preemptions/recompute. Prefix caching stays off initially. The fixed 64+256-token timing formula applies to homogeneous decode cases; the mixed queue uses its declared per-request budgets and reports a separate common wall-clock measurement window.

Record aggregate output tokens/s, per-request p50/p95 inter-token latency, TTFT including queueing, queue wait, prefill service time, end-to-end completion rate and an active/queued timeline. Report sample counts; tail percentiles from few requests are exploratory. This short case establishes mechanics and an initial performance point, not production tail-latency certification.

## 8. Measurements and stop rules

Use result_record.template.json. Minimum evidence includes:

- Per-rank allocated/reserved/device-used/device-free and peak bytes during construction/init/repack, cache setup, prefill, graph capture and decode; cache pool capacity versus live occupancy.
- Host RSS/cgroup and pinned bytes, aggregate/per-worker table payload, mapping device/bytes, staging, measured hits/locality and swapping/page-fault behavior.
- Actual model/shard shapes, quantization, source-layer ownership, unique storage, graph mode actually executed and fallback counts.
- Actual prompt/output/context intervals, warmup/windows, generation policy, routing histograms, homogeneous decode metrics and mixed-queue results.
- Build/load/JIT time, rental wall time, allocated GPU-hours, package/patch provenance and a decision with limitations.

Promote only after correctness, measured peak/cache capacity and mixed-request state checks pass. Do not silently lower context, bypass Engram, change precision, force favorable routing or reduce depth after failure while retaining the same case label. Label modified fixtures and state what they establish.

| Finding | Action |
|---|---|
|Driver/native kernel fails|Save a minimal device reproducer; resolve the build before full-model work |
|Dummy/custom path leaves invalid weights or allocates full Engram|Fix initialization/constructor; retain only operator evidence |
|Startup/repack exceeds a rank's capacity|Inspect duplicates, temporary lifetimes and mapping; nominal weight bytes are insufficient |
|B8 works but larger batches fail|Record the boundary and retain B8 as valid progress |
|Graphs break state/correctness or have unexplained fallback|Keep eager baseline and isolate the transition |
|Static decode works but continuous batching fails|Serving remains blocked pending request-state/scheduler fixes |
|Favorable TPS depends on tiny/hot Engram or skewed routing|Report that condition; run one targeted sensitivity case |
|Four-hour core yields a useful answer|Use the extension only for a concrete decision, otherwise stop and hand off |

If the quoted whole-node hourly rate is H, the core costs 4H and the extended session 6H before separate storage/network charges. If quoted per L4 at g/hour, GPU cost is 16g or 24g. These are formulas, not current provider quotes. Allocated GPU-hours are distinct from timed kernel utilization.

## 9. Parallel pruning and real-weight transition

Serving can implement shapes, formats, lookup semantics and batching before final expert/row IDs exist. Keep the handoff stable:

| Workstream | Work that can proceed now | Later evidence |
|---|---|---|
|Serving|Synthetic model, kernels, E64/E72 routing, compact adapter, memory/graphs/batching|Trained routing, actual lookup locality, real load-time I/O/peaks and final throughput |
|Engram pruning|Tokenizer/hash parity, CPU row census, candidate masks and export ABI|Hidden-state contributions and held-out quality of row misses |
|Expert pruning|Exporter/scoring plumbing and immutable tensor contract|Fresh REAP on 384-expert real model under the validated frozen Engram behavior |

Do not collect final pruning statistics with random weights or choose a quality mask solely because a synthetic table fits. Preserve the original sequence: **validate Engram → freeze its behavior → fresh REAP → E64/E72 export → held-out ARC evaluation**. The first 4×L4 rental is inference feasibility work, not the original-model calibration machine.

Freeze revision, original layer numbering/dimensions, top-6, weight/activation quantization, expert/router/bias/scale remapping contract, original Engram hash IDs, missing-row behavior and cache formats. Keep actual retained IDs and row contents mutable until selection. A synthetic fixture should then be replaceable by the real checkpoint without changing the engine's interface.

Transition in this order:

1. **Small real tensor fixture when available:** one representative layer/expert slice, native scales, router/bias rows and captured hidden inputs. Compare against the native-format reference. It requires no whole checkpoint on the PC and establishes no 40-layer quality result.
2. **Provisional real E64 checkpoint:** same runtime, selected real tensors and declared compact Engram mask. Verify original-ID remapping, provenance, loading/repacking peaks and outputs/state. Add remote disk for the actual exported checkpoint and temporary staging at this stage.
3. **Repeat the winning point and one neighbor:** measure trained routing, Engram locality, real prefill and continuous batching. Revise the operating point if the synthetic distributions were misleading.
4. **Final validated checkpoint:** run frozen-mask/export checks and held-out tool-free ARC. Compare solve rate and complete harness wall time under actual reasoning-length, two-guess and deadline rules. Synthetic data cannot predict accuracy, termination, reasoning length or speculation acceptance.
5. **Then test CED, DSpark or another engine:** each is a separate correctness/quality/performance comparison; avoid combining them with the first pruning comparison.

No acceptable accuracy loss or hard final TPS threshold was supplied. Treat these as explicit later selection criteria. The initial gate is supported, measurable execution at intended shapes/context; final selection uses actual solutions per wall-clock budget and matched prompts/tool policy/hardware.

## 10. End-of-session deliverables

Return a compact results bundle: model/software/patch lock; actual launch commands; operator/startup logs; full model/storage manifest; per-rank phase memory table; filled case records; continuous-batching timeline; selected-case profile only if useful; and a decision report. Preserve failures and skipped cases.

The report should recommend **proceed to real E64**, **proceed at a smaller measured operating point**, **fix a specific engine/adapter blocker**, or **revisit the model/hardware target with measured evidence**. State whether E72 merits a follow-up, which Engram capacity/traffic was exercised, and which conclusions still require real weights.

This document supersedes the earlier real-weight-first experiment addendum. The supplied pruning pack remains authoritative for its pruning objective and quality sequence. All memory totals here are calculations from stated config/format assumptions unless explicitly identified as future measurements.
