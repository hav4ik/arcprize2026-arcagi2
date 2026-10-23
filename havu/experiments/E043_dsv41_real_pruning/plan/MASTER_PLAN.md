# DeepSeek V4.1 Flash 1017 on four L4 GPUs for ARC AGI 2

Research date: 19 October 2026. This is a source-backed engineering and experiment plan. No full model, pruning run, paid cloud job, or L4 throughput benchmark was executed during the research. The included tools provide environment inventory, budget arithmetic and a CPU synthetic compact-row mapping reference; they do not implement the full pruning pipeline.

**Implementation update:** begin with [START_HERE.md](START_HERE.md). The completed source audit in [REAP_IMPLEMENTATION_CONTRACT.md](REAP_IMPLEMENTATION_CONTRACT.md) resolves native-forward, normalization and upstream integration issues. [REAP_HARDWARE.md](REAP_HARDWARE.md) is the current RTX PRO 6000 Blackwell calibration recommendation and cost worksheet; it supersedes the older two-B200 prototype allocation below as the preferred starting path.

## Recommendation and decision

Use **CPU Engram row census → select and validate a retained-row mask → freeze that mask → collect fresh REAP statistics → export E64** as the current execution order. E64 retains **64 routed experts per layer with the original top-6 routing**, approximately **98.9B backbone parameters and 64.49 GB (60.06 GiB) of GPU-resident weights across four GPUs**, before KV cache and workspace. **E72** is the larger quality candidate at approximately **110.2B backbone parameters and 70.86 GB (65.99 GiB) of GPU weights**. These are packing estimates, not measured allocations.

The earlier **32–56 GB / E48-first** proposal is an archived smaller-envelope comparison, not a current hard requirement. Engram and expert parameter budgets are separate choices. Full Engram tables contain approximately **196.6B parameters and occupy 188.83 GiB**; the current plan retains domain-relevant original rows and stores the compact payload in host RAM. Native pinned-UVA CPU offload exists, but an original-ID-to-compact-row adapter is new work. Host table bulk is outside VRAM; Engram Wkv/projection weights, ID metadata, gathered outputs, staging and any GPU hot cache are not.

The most credible initial serving base is **yhfgyyf/vllm-deepseek-v4-sm89 with its paired native V4.1 FlashInfer build**. Source inspection finds Ada attention, packed FP4 expert execution and pinned-UVA Engram CPU offload. The exact combination of 1017 weights, E64/E72 pruning, a validated compact Engram mask and four L4s has not been demonstrated by the sources inspected.

The first spending decision should follow three gates: (1) real-L4 kernel correctness and latency, (2) held-out quality and host feasibility of the selected Engram rows, and (3) an ARC quality curve from expert statistics collected with that frozen Engram behavior. The budgets are orthogonal in storage but statistically coupled through hidden states and routing. Spend on recovery training only when a candidate retains useful long-rollout behavior.

## What is established and what remains uncertain

| Question | Evidence | Decision consequence |
|---|---|---|
| Is L4 SM86? | NVIDIA lists L4 under compute capability **8.9**, Ada. | Build and test SM89. |
| Does V4.1 have an older-GPU serving implementation? | Native mixed FP8/FP4 attention and Marlin expert paths exist in the audited fork. | Reuse and validate them. |
| Does the full model fit four L4s? | Approximately 552B backbone plus 196B Engram; original checkpoint about 510 GB. | Pruning plus separate Engram residency is necessary. |
| Do E64/E72 leave room for inference? | Estimated backbone weights are 60.06/65.99 GiB aggregate; native cache is small but replicated, and engine buffers are material. | Start E64 at batch 8–16 and measure; treat E72 as a candidate. |
| Will domain row pruning plus E64 retain ARC capability? | No identified published result establishes this exact combination on ARC long reasoning. | Validate Engram first, then measure expert pruning under that behavior. |
| Will it beat a dense 27B model in the Kaggle runtime? | Active parameter count does not establish batched throughput, interconnect efficiency or reasoning length. | Compare complete harness solves per wall-clock budget. |

Primary references: [NVIDIA compute capabilities](https://developer.nvidia.com/cuda-gpus), [1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), [SM89 fork](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89), [native FlashInfer branch](https://github.com/yhfgyyf/flashinfer/tree/deepseek-v4.1-native-sm89-sm120).

## Architecture that matters for implementation

### Backbone and routing

The exact configuration specifies **40 layers: 20 encoder and 20 decoder**, hidden dimension **5120**, routed-expert intermediate width **2304**, **384 routed experts plus one shared expert per layer**, and routed **top-k 6**. The published active counts are approximately **8B during optimized prefill and 16B during decode**. The 763B repository display includes additional modules; it is not the backbone size to use for expert arithmetic.

Routing is not generic softmax gating. Compute **sqrt(softplus(logit))**, add the text/image correction bias for selection only, choose experts, and normalize their original unbiased scores before multiplying the configured **1.5** scale. Preserve the checkpoint's SwiGLU clamps and activation quantization boundaries. Expert removal must retain the mapping among expert tensors, router rows, both correction-bias vectors and quantization scales. Keep shared experts, layer numbering, top-k and widths unchanged initially.

Sources: [exact config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), [reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), [reference kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py).

### CSA2 attention and Causal Encoder-Decoder

V4.1 uses **CSA2 and Causal Encoder-Decoder organization**, rather than simply inheriting V4's CSA/HCA layout. Every layer has a 128-token local sliding window. There are four global cache sources, at zero-based layers **2, 8, 14 and 20**. Encoder sources compress pairs of tokens; the decoder source has one global row per original token. Other layers reuse KV and, in some cases, sparse selections.

The first two encoder layers are local-only. The remaining encoder layers form three groups of six. Decoder layers share the global cache generated at layer 20 from the final encoder representation. Global index selection is refreshed at layers **2, 8, 14, 20, 24, 28, 32 and 36**. The first decoder indexer performs a full-context scan and builds a smaller candidate pool for subsequent reindexers. A sparse attention call combines up to 512 selected global rows with local-window rows; this does not imply that all indexing work is constant in context length.

There are 64 query heads and one shared 512-channel main latent. **The 512 already includes the 64 RoPE channels, and the latent serves as both K and V.** Do not add a second K/V factor or another 64 channels when sizing memory. Output projection, learned sinks and inverse RoPE must follow the reference. The model also uses four residual streams with single-pass mHC; replacing it with a standard residual path is a model change.

Sources: [technical report, architecture and inference sections](https://arxiv.org/html/2609.19969v1), [attention configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), [reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

CED can skip most decoder computation during prompt ingestion and replay a bounded suffix to rebuild local decoder state. The published bounded replay is approximate relative to running every decoder position. Test CED separately from pruning, particularly for long tool conversations. The minimal reference runs all layers during prefill, which is useful for calibration, but its Python expert loops, host synchronization, fake-quantized BF16 buffers and unoptimized index work make it unsuitable as a throughput target.

The source-audited reference indexer can allocate a 256 GiB raw BF16 score tensor for one 64K sequence before head reduction. Tile its query axis inside whole-prefix execution; small persistent KV does not guarantee a small calibration peak. The exact allocation derivation and parity requirements are in REAP_HARDWARE.md.

The official serving recipe states that the 1017 update keeps the architecture, tokenizer and configuration unchanged. The inspected reference model, Engram helper and kernel files were byte-identical between the base V4.1 and 1017 repositories. Reuse implementation work, but recompute pruning statistics on the actual **1017 weights**.

Source: [official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).

### Engram and optional modules

Engram operates at layers 1 and 14. Each module hashes the last 2/3/4 canonical compressed token IDs through eight heads per order: **48 table-row lookups per new token across both modules**. Addresses depend on the exact tokenizer normalization/mapping, original layer IDs, per-head prime table sizes, multipliers and offsets. CPU traces are sufficient to reproduce addresses. Engram gates and contributions depend on model hidden states, so address coverage alone cannot establish quality.

The two full tables have 384,006,168 and 384,016,682 rows. Each row stores 256 FP8 values and eight scale bytes. Their full native payload is **202,758,032,400 bytes = 188.83 GiB**, excluding auxiliary tensors and filesystem overhead. Domain row pruning reduces backing capacity; a hot cache alone would not. Preserve the original address function and use a compact retained-row mapping. The declared initial miss policy is a zero embedding row **before Wkv/projection**, followed by the native module computation; this is a model change requiring held-out validation.

Sources: [1017 Engram helper](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py), [exact table dimensions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json), [FP8 table and scale storage](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

For text-serialized ARC, disable vision loading using the selected engine's supported text-only path. Start without DSpark. The draft adds approximately 14B logical parameters, has three blocks and separate 128-expert/top-3 routing, and proposes multiple tokens. Its acceptance and scheduling benefit must be measured against the pruned target on ARC. Keep its configuration separate when editing backbone expert counts.

Sources: [model card and modules](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), [official serving options](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).

## How much to prune

One expert contains:

`3 × 5120 × 2304 = 35,389,440 parameters`.

For E retained routed experts per layer, all 40 layers contain:

`routed_parameters(E) = 1,415,577,600 × E`.

The non-routed backbone is approximately 8.3B parameters, so:

`backbone_parameters(E) ≈ 8.298B + 1.4155776B × E`.

This is a shape-derived estimate; the published backbone total is rounded to 552B. Engram is additional.

### GPU storage, including runtime scale expansion

The routed experts are already native FP4. Their checkpoint storage is approximately **0.53125 bytes/parameter**: half a byte for a value plus one scale byte per 32 values. The audited Ada Marlin path preserves packed 4-bit matrices but expands scales to BF16, requiring approximately **0.5625 bytes/parameter**.

Attention wo_a also expands to BF16 in this fork, adding approximately 1.29 GiB across the backbone relative to its original checkpoint representation. Therefore the table below adds a **provisional 12.6 GiB non-routed resident-weight allowance**. It is a planning reserve, not a measurement of the target engine. Replace it with per-rank post-load measurements. It excludes Engram tables/hot caches, DSpark, KV and workspaces.

Sources: [Marlin backend selection](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py), [FP4 repacking and scales](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py), [V4.1 quantization exceptions](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/quant_config.py).

| E per layer | Routed experts removed | Backbone parameters, approximately | Routed expert GPU GiB | Total GPU weights GiB | Total GPU weights, decimal GB |
|---:|---:|---:|---:|---:|---:|
| 24 | 93.75% | 42B | 17.80 | 30.40 | 32.64 |
| 32 | 91.67% | 54B | 23.73 | 36.33 | 39.01 |
| 40 | 89.58% | 65B | 29.66 | 42.26 | 45.38 |
| 48 | 87.50% | 76B | 35.60 | 48.20 | 51.75 |
| 56 | 85.42% | 88B | 41.53 | 54.13 | 58.12 |
| **64 — primary** | **83.33%** | **99B** | **47.46** | **60.06** | **64.49** |
| **72 — candidate** | **81.25%** | **110B** | **53.39** | **65.99** | **70.86** |
| 80 | 79.17% | 122B | 59.33 | 71.93 | 77.23 |
| 96 | 75.00% | 144B | 71.19 | 83.79 | 89.97 |

The current choices are E64 first and E72 if measured headroom and quality justify it. E24–56 remain archived smaller-envelope calculations; their inclusion is not a requirement to export or benchmark them. E80/E96 can provide higher-capacity quality controls when the evaluation hardware allows them.

E64 removes 83.33% of routed experts; E72 removes 81.25%. These percentages do not describe all backbone parameters or the separately retained Engram rows. Record expert and table retention independently in every manifest. A proportional Engram fraction, such as 64/384, would retain about 31.47 GiB of native row payload, but it is only an optional starting budget. Use the observed global row union, host limits and held-out quality to choose the actual table budget; equal fractions in every bank are not required.

Top-6 remains top-6 after pruning. The active routed matrices still contribute approximately **8.49B parameters per decode token**, and total decode activation remains around the original 16B. Pruning reduces residency and can improve weight reuse across a batch; it does not reduce active computation in proportion to the deleted weights. At high concurrency, many different experts execute across the batch, so a 16B-active MoE is not automatically faster than a dense 27B model.

## KV cache and batch capacity

The audited native path uses three cache row layouts:

| Cache | Bytes per stored row | Format |
|---|---:|---|
| Main global latent | 288 | 512 FP4 values plus 32 FP8 scales |
| Indexer key | 68 | 128 FP4 values plus four scale bytes |
| Local sliding window | 528 | 512 FP8 values plus 16 scale bytes |

With three half-rate global sources and one full-rate source:

`global_cache_per_token = (288 + 68) × (3/2 + 1) = 890 bytes`.

**This is per GPU under the inspected TP4 layout.** Query heads are sharded, but the single-head global cache and indexer caches are replicated. The global aggregate is 3560 bytes/token across the four cards. The 40 sliding windows add a logical floor of **2.578 MiB per request per GPU**. Padding, extra retained blocks, candidate arrays, prefix policy, prefill intermediates and replay state are additional.

Sources: [native cache ownership and attention](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/attention.py), [mixed attention implementation](https://github.com/yhfgyyf/flashinfer/blob/d4eef3c44c70594ff18b9126381ae6a978694e3f/flashinfer/mla/deepseek_v41.py).

At **65,536 total live tokens per request**, including prompt, reasoning and tool history:

The following includes native global/index + logical SWA + compressor rings, default shared indexer buffers, candidate/top-k arrays at 8192 scheduled-token capacity and a 128-token CED tail with reserved slots equal to concurrency. It then adds a **separate 4 GiB residual workspace reserve**. CED is a conservative planning assumption; the eager CED-off serving baseline can remove that component when its allocation is verified disabled.

| Concurrent requests | Known cache/state/shared buffers across TP4 | E64 weights + known buffers + 4 GiB residual | E72 weights + known buffers + 4 GiB residual |
|---:|---:|---:|---:|
| 8 | 4.54 GiB | 68.61 GiB | 74.54 GiB |
| 16 | 6.52 GiB | 70.58 GiB | 76.52 GiB |
| 32 | 10.48 GiB | 74.54 GiB | 80.47 GiB |
| 64 | 18.39 GiB | 82.45 GiB | 88.39 GiB |

The example cap is **4 × 22.3 GiB × 90% = 80.28 GiB**. E64/B32 lies below it with modeled margin; E72/B32 slightly exceeds it under these assumptions. This is not a safe batch-size certification. The native indexer reserves **577.03 MiB per rank** by default and shared candidate/top-k buffers add **80 MiB per rank at 8192 scheduled-token capacity**; these are already in the known-buffers column and must not be added again. Extra SWA pages, in-flight prefill tensors, graph pools and Engram gathers still require measured residual capacity. See KV_MEMORY_NOTES.md for component formulas and the conditional 5/45 MiB CED slot reserve. A maximum request-slot setting larger than live concurrency raises fixed state allocations.

Start at maximum sequences 8, then 16/32/64. Optimize the measured solve rate before pursuing B96/B128. A larger batch can reduce individual-request speed and make more trajectories hit the runtime cutoff. For different sequence lengths, use the included calculator.

The reference implementation's BF16 fake-quantized caches do not realize this footprint. A backend that expands caches changes the budget dramatically. The inspected native V4.1 path requires FP4 index rows; a generic FP8-indexer alternative in surrounding code is not an enabled V4.1 fallback. Its CLI's fp8 cache label refers to a mixed native layout.

## Serving roadmap and open-source choices

### Primary, backup and reference

| Project | Role and evidence boundary |
|---|---|
| [yhfgyyf SM89 vLLM](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) + [paired FlashInfer](https://github.com/yhfgyyf/flashinfer/tree/deepseek-v4.1-native-sm89-sm120) | First L4 target. Native mixed KV, BF16 tile arithmetic and Marlin W4A16. Actual source support exists, but the full proposed deployment is unvalidated. |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) | First fallback; publishes older-GPU V4.1 paths including SM89 packaging. Verify exact branch, formats and Engram behavior before switching. |
| [Official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) | High-end teacher/reference; current recipe requires v0.31.0+. A supported-model entry does not certify Ada kernels. |
| [SGLang V4.1 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx) | High-end reference; the cookbook currently requires the dev-dsv41 preview image because support has not shipped in a standard release. NVIDIA configurations focus H200/B200/B300/GB300. No verified stock L4 path was identified. |
| [LibertAI disk-Engram recipe](https://github.com/Libertai/dsv41-flash-vllm-2x-spark) and [underlying prestage implementation](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/main/patch/cudagraph-prestage) | Reuse exact row-access and staging ideas. Spark attention patches are a different backend, not native SM89 shape requirements. |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Do not pull latest as a presumed L4 solution. Current main changed formats and targets Blackwell datacenter hardware. |

Pin the vLLM source at **3f3bbaea916707c851ca027b87b97fb6605998b8** and the audited FlashInfer source at **d4eef3c44c70594ff18b9126381ae6a978694e3f** if implementing against this audit. The published Vision11 wheel has a different Python source SHA, **86d35745c8797c9d2524877fb235084a6a5f5a7d**, despite retaining a Vision9 release tag. Do not mix its automatic source archive with another wheel and assume they are identical. TASKS_SERVING.md records full versions and wheel checksums.

The packaged environment uses Python 3.12, CUDA 13.0 and PyTorch 2.13.0. NVIDIA's standard CUDA 13 compatibility baseline is driver R580; verify the real Kaggle driver before selecting a build. A supported forward-compatibility route or a backported build may help, but installing a toolkit does not upgrade the host driver. Package the accepted wheel/ABI/JIT combination for offline use.

Sources: [release](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/tag/v0.28.1rc1-vision9-sm89-sm120-cu130), [NVIDIA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

### Required modifications

1. **Router support.** The audited whitelist accepts only (384,6) and (128,3). Add explicit top-6 shapes for retained backbone counts. Preserve unbiased output weights, correction-bias selection, tie order and invalid padded lanes. The parameterized Triton kernel looks reusable; a whitelist edit alone is not evidence of correct loading or expert dispatch.
2. **Compact host Engram.** Start from native `cpu_offload:true` pinned-UVA lookup. Add an original-ID-to-compact-row map and the frozen miss policy; prevent the loader from allocating the full table first. Export retained FP8 rows and scales unchanged. If pinned host capacity is insufficient, a bounded disk stager is an optional fallback, not the default requirement. CPU work must execute outside CUDA graph replay.
3. **Pruned tensor loading/export.** Preserve original expert IDs for initial masked experiments; final compact exports must remap all expert-indexed tensors consistently. Keep original hash geometry and non-expert architecture untouched.
4. **Actual L4 operator validation.** The native attention kernel accepts 8/16/32 local query heads. A full 64-head TP1 model is not a valid smoke test. Use one L4 with a **16-head TP4-shard fixture**, then a topology-preserving integration fixture and full TP4.
5. **TP/EP and memory verification.** Compare TP4 with expert parallelism enabled/disabled where supported. E64/E72 give 16/18 local experts under EP4; inherited dispatch paths need actual tests. Check per-rank peak memory during loading, repacking, prefill and decode, including compact-table gather temporaries.

Source: [native router](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/router.py).

### Engram first: independent storage budget, coupled model behavior

For this stage, reuse the standalone CPU hashing and metadata components from **antirez/ds4**, pinned at `fc80bd695da76ee14bcb86820bf900a5cdfda806`, after checking their output against the exact 1017 reference. Its bounded native-row reader and exporter are also useful building blocks. See **DS4_REUSE.md** for specific functions and the remaining compaction work. The existing `make test-engram` suite passed on Linux CPU during this audit; actual 1017 tokenization, model quality and L4 execution remain untested. This component reuse does not change the primary SM89 vLLM serving path or imply that ds4's imatrix collector already computes REAP.

Follow TASKS_ENGRAM_FIRST.md. Before downloading full weights, run the exact tokenizer/hash helper for a proposed 10,000-task coverage pool, starting with existing compatible traces and verifying root/generator lineage. Use tool-free reasoning as the pruning/calibration objective. Tool-assisted data or comparisons are optional only if requested later. Split by immutable root and generator lineage before row selection. The CPU census needs token IDs only; it does not need model activations or a GPU. Count global `(bank, original_row_id)` occurrences, retaining separate diagnostics by harness, role, n-gram order and trajectory depth.

Keep every observed row if the global union fits the chosen host payload budget. Otherwise retain the most frequent rows globally with a deterministic tie rule. Because every native row is 264 bytes, row count is a direct payload budget; maps, pinning and gather buffers need extra space. Do not change hash moduli, prime sizes or original IDs. Nonuniform retained sizes across the 48 banks are allowed. Equal per-bank retention is only an optional baseline.

Held-out hit rate helps select candidates, but it cannot measure the physical model's quality: gates depend on hidden state, misses interact with later routing, and a rare row can matter. Validate the row mask with instrumented full-expert forwards and complete held-out ARC rollouts before freezing it. Retained rows preserve their original bytes; missing lookup rows become zero embeddings before Wkv/projection. Keep the rest of Engram computation native. Freeze both mask and miss semantics, then collect expert statistics with that behavior active. Reusing existing API teacher text is sufficient; regenerating every trajectory is not required.

The baseline serving path is native CPU offload with a compact-ID adapter. The full table would occupy 188.83 GiB, but only retained rows are deployed in this plan. Bulk CPU tables do not consume VRAM; Wkv and other projection weights, compact-ID metadata, GPU output/staging tensors and optional hot rows still do. Measure host RSS, pinned bytes, lookup latency and per-rank peaks. The previous full-table disk-prestage recipe remains an archived exact-reference or storage-fallback option, not a mandatory full-table deployment.

Primary implementation sources: [exact Engram hash helper](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py), [native pinned-UVA Engram](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/common/engram.py), and the [original Engram paper](https://arxiv.org/html/2601.07372v1). This paper's trained memory allocations do not prove post-hoc row-pruning quality for 1017.

## Pruning strategy

### Methods to compare

**REAP is the first baseline.** The official Cerebras implementation supports blockwise observation and quantized models. For each selected expert, accumulate its actual gate weight times the norm of its unweighted expert output. Divide by that expert's activation count. This conditional mean is different from summing importance over all tokens with zeros for inactivity. Record both if comparing variants, and name them differently.

**V4.1 requires a careful adapter:** its native Expert.forward multiplies the route weight into the SwiGLU intermediate before the down-projection's activation quantization. Consequently the actual weighted contribution q(h,w) is not exactly w times the unweighted Expert(h). For the reference REAP score, execute an additional unweighted down-projection path. For a cheaper native-contribution adaptation, observe norm(q(h,w)) directly and label it **native-amplitude REAP/HOPE**. Do not multiply by the gate twice or recover the unweighted output by dividing q by w. Compare the two score definitions on a small reservoir before choosing the pilot implementation. Counterfactual masks must recompute the weighted down-projection when normalized gate weights change. Source: [native Expert and linear implementations](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

**HOPE is the first method upgrade.** It incorporates pairwise coactivation amplitudes and solves a constrained quadratic selection problem. With 40 layers and 384 experts, a full FP64 pairwise sum and uint64 pairwise count tensor occupy approximately **90 MiB**. Only six experts execute per token, so update their 6×6 pairs rather than forming dense token-by-384-by-384 tensors. Match the released conditional coactivation normalization and validate the optimizer. HOPE's diagonal second moment is not generally the square of REAP's conditional mean.

**RAZOR is a targeted third experiment.** It considers expert-output direction and next-ranked replacement behavior. This can distinguish redundant-looking norms from actual replacement damage. Its evidence includes V4-family pruning, but a V4.1 adapter and extreme-pruning proof are absent. Additional candidate-expert execution increases calibration work.

Sources: [Cerebras REAP](https://github.com/CerebrasResearch/reap), [AWS HOPE](https://github.com/awslabs/HOPE), [HOPE paper](https://arxiv.org/abs/2609.18916), [RAZOR](https://github.com/nick7nlp/Razor), [RAZOR paper](https://arxiv.org/abs/2609.30465).

There is already a [native V4.1 REAP project](https://github.com/Libertai/deepseek-v41-flash-reap/tree/210ccebac92172fb58655c7b95f6a6776f57a641), with streaming, calibration, pruning, tensor-parallel conversion and shard verification scripts. Reuse its tensor/addressing machinery with the corrections in REAP_IMPLEMENTATION_CONTRACT.md. Its published text perplexity rises from 3.6552 at 384 experts to 4.1910 at 192 experts, approximately 14.7%. These are the publisher's results under its patched forward: the inspected collector/PPL path applies the gate after the unweighted expert instead of before native down-projection quantization. The shortcut is not generally equivalent, so the numbers do not establish native 1017 parity. Its released 256/272-expert models are integration references, far too large for the proposed GPU budget. They do not establish ARC performance at E64/E72 with a pruned Engram table.

Published pruning quality cannot be summarized as a universal percentage retained. For example, one RAZOR V4 comparison improves AIME while substantially reducing SWE-bench performance. The practical lesson is to evaluate the same long reasoning/tool harness used in the competition; text perplexity and short mathematics are insufficient proxies.

Expert merging, such as [REAM](https://github.com/SamsungSAILMontreal/ream), remains a later option. It introduces alignment, merging and requantization choices. It does not remove the need for end-to-end quality validation and is not the shortest initial path.

### Calibration trap: API text is not expert statistics

Use generated teacher rollouts as token sequences. They suffice for CPU Engram address census, but API access does not expose the expert outputs needed by REAP/HOPE. After the Engram mask passes its quality gate, replay a representative subset through the original full-expert 1017 weights **with the frozen Engram mask active**. This is fresh expert calibration under the chosen model behavior. Ordinary optimized serving prefill may skip most decoder positions, so merely attaching hooks to production prefill can leave half the model badly calibrated.

Build a **full-position, all-40-layer teacher-forced replay** adapter. Verify decoder observation counts and compare logits on short prefixes with genuine autoregressive decoding under the same formats. Fused experts often hide the individual outputs; instrument before expert reduction or use a reference/blockwise adapter. With tensor-parallel partial outputs, norms must reflect the complete expert output, not the sum of local norms.

Start with **0.25–1M replayed tokens**, including genuinely long contexts and shorter diverse traces, to validate statistics and measure cost. Then aim for **1–8M scored positions** only if the pilot passes. Their required causal replay can be larger. Stratify tool-free prompts, early/middle/late reasoning and final answers; include productive failures and cap repetitive-loop contributions. Keep root/harness strata separate so very long traces do not dominate by accident.

Use a proposed **10,000-task coverage pool**, verifying independence and immutable root/generator-lineage splits, not a requirement to teacher-force every long trajectory. A hypothetical one-trace-per-task corpus of 20–64K tokens contains 200–640M raw tokens; a complete replay of it is not the mandatory first experiment. Sampling late positions still requires their causal history, or a validated saved-state replay strategy. Log **raw generated, locally processed and actually scored tokens separately**, alongside checkpoint and frozen Engram mask checksums.

For NVARC roots, the [authors' repository](https://github.com/1ytic/NVARC) points to **sorokin/nvarc-synthetic-puzzles** (103K synthetic puzzles), **sorokin/nvarc-augmented-puzzles** (3.2M augmented puzzles) and **sorokin/nvarc-artifacts-puzzles** (puzzle-construction text). The artifacts are not automatically reasoning rollouts from your solver. Generate actual 1017 trajectories through the chosen harness, retain root provenance, and inspect bundled splits before treating any public evaluation set as an uncontaminated holdout.

Your 1B-token generation budget is more valuable after this pilot, for hard cases, recovery and distribution coverage. More repeated easy traces improve neither missing expert capacity nor long-horizon reliability automatically.

### A cheap quality curve before exporting many models

Collect one instrumented full-expert pass with the frozen Engram mask, construct survivor sets, then measure counterfactual routed outputs while leaving that reference state unchanged. Include E384 under the same Engram mask, E128/E96/E80 diagnostics, and **E72/E64** candidates. E48/E40/E32 are optional archived smaller-envelope stress cases. Use REAP first; consider HOPE at the plausible frontier rather than running every method at every size.

This fixed-input comparison is a **rejection test**, not generation quality. Record retained teacher gate mass, exact top-6 refill damage, output angle/norm changes and layerwise concentration. A mask that badly distorts local outputs deserves little training spend. A mask that looks good still needs end-to-end inference because errors change future activations and routing.

For the next gate, use a selective loader that keeps original IDs, loads only retained expert tensors and masks deleted router outputs before native top-k. This can avoid final reindexing and reduce source reads. Masking a fully loaded model saves no VRAM. Evaluate a few actual compressed candidates before producing permanent compact exports.

An optional small-reservoir diagnostic evaluates all surviving expert outputs and solves a nonnegative least-squares mixture whose coefficients sum to the native routed scale **1.5**, with any fixed shared-expert output subtracted consistently. Its optimum is a lower bound on top-6 reconstruction error within this fixed linear-mixture diagnostic. Weight-dependent activation quantization in the native MoE can break exact equivalence to that diagnostic. Low oracle error but high native-refill error suggests router recovery has an opportunity. High oracle error, depthwise drift and long-rollout collapse argue against a cheap router-only fix. Neither oracle nor gate mass alone proves downstream success.

### Evaluation and recovery

Split by original ARC problem and generator lineage before transformations; keep all augmentations and retries of a root in one partition. The user reports approximately **80% pass@2 without tools and 84% with tools** for unpruned DeepSeek; treat these as supplied context, not results established here. The required objective is **tool-free pruning**, with no mandatory Python control or general coding-retention target. Record exact-grid pass@1, the actual two-guess score, valid completions, reasoning length, loops and deadline failures. Include full-original Engram/full-expert, frozen Engram/full-expert and frozen Engram/E64 or E72 controls to attribute row-pruning and incremental expert-pruning damage. Use paired root-level uncertainty. Local candidate wall-clock comparisons require the same hardware; token-budget comparisons are diagnostics, and off-site teacher/API controls establish quality deltas rather than comparable serving speed.

First evaluate 32–64 diverse held-out problems with full intended reasoning lengths; expand only survivors. A short 2K-token generation smoke test cannot validate a 64K ARC solver. Freeze a final comparison split after choosing expert count and method.

If a candidate remains useful, attempt router-projection recovery with teacher KL and ARC outcomes, then limited LoRA/expert recovery if needed. Selection-only correction biases need a separate calibration/search rule or a validated surrogate: ordinary differentiation through hard top-k does not give them a useful selection gradient. A tiny trainable parameter count does not imply a tiny activation/backward footprint. Marlin serving kernels and forward-only sparse attention do not automatically provide a training stack. A frozen-quantized, differentiable implementation is a separate gate. Full optimizer training of a roughly 99–110B backbone is not a single inexpensive GPU task.

Keep the **validated Engram mask, original addressing, miss policy and arithmetic** fixed while measuring pruning/recovery. Changing its rows changes hidden-state and expert-use statistics; revalidate and recollect affected statistics instead of reusing stale scores. Also hold cache precision, CED mode and sampling policy fixed. Numeric reasoning-effort mappings differ between the local recipe and API labels; record numeric/local meaning, temperature, top-p and context/output limits explicitly.

Sources: [router recovery research](https://arxiv.org/abs/2603.02217), [official local reasoning-effort configuration](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).

## Compute and cost plan

**Current hardware choice:** use one 96 GB RTX PRO 6000 Blackwell for the corrected packed-streamer prototype and four/eight such GPUs for the resident collector after parity. The full Marlin backbone is provisionally about 319.29 decimal GB before workspaces, with Engram in host memory. Eight cards provide useful headroom; two B300s are an alternative whose value depends on measured throughput and the actual whole-node quote. REAP is forward-only. See REAP_HARDWARE.md for exact payloads, host/NVMe recommendations, the 64K indexer fix and cost scenarios. The table below retains other researched options for comparison.

Prices are public on-demand examples observed during this research, not reserved capacity or a quote. Availability, region, CPU/RAM/storage bundle, tax and startup time can change the bill.

| Resource | Example price | Suggested use |
|---|---:|---|
| One L4 | $0.59/hour | Native kernel and MoE fixture |
| Four L4s | $2.36/hour | Topology, memory and complete harness throughput |
| RTX PRO 6000 96 GB | $2.49/hour | Layer-streamed calibration pilot if storage/host RAM are suitable |
| Two B200s, Lambda 2-GPU rate | $13.78/hour total | Resident full teacher with Engram in sufficient host RAM |
| Four H200s | $21.16/hour total | Alternative full-teacher/calibration node |
| Eight H200s | $42.32/hour total | Optional, separately gated recovery training |

Sources: [Runpod pricing](https://www.runpod.io/pricing), [Lambda pricing](https://lambda.ai/pricing).

The official vLLM recipe demonstrates a full-teacher TP2 B200 configuration with Engram CPU offload. Check the host allocation: tables alone require about 189 GiB. Two H200s are too tight for the full original routed matrices plus dense weights in the intended representation. A 96 GB card can stream layer/expert tensors; it cannot hold the full teacher. Cheap hourly rent is not necessarily cheap calibration if the reader repeatedly streams hundreds of GiB.

The native REAP project reports around 19.4 GiB peak reference residency with streaming and approximately 269 GiB of routed weights read per full sweep. At its reported 0.65 GiB/s path, that alone is roughly seven minutes per sweep. Amortize a sweep over many valid tokens and reuse layer weights across microbatches where the CED dependency graph permits. Do not attempt to serve long autoregressive Kaggle rollouts by rereading the whole teacher on each token.

### Allocated prototype budget

This is the earlier two-B200/L4 allocation worksheet, retained as a comparison. Use the RTX spending sequence and measured pilot formula in REAP_HARDWARE.md for the current preferred plan. None of these allowances is a performance forecast.

| Stage | Illustrative allocation | Compute cost |
|---|---|---:|
| CPU hash/inventory work | Existing local CPU resources | $0 additional if available |
| L4 kernel/Engram integration | 8 hours single L4 + 8 hours four L4s | $23.60 |
| Full-teacher calibration and mask pilot | 4–12 hours on two B200s | $55–165 |
| E64/E72 deployment and harness pilot | 12–36 hours on four L4s | $28–85 |
| Optional recovery | 12–48 hours on eight H200s | $508–2,031 |

These are spend allocations, **not measured completion times**. A practical first-decision compute budget is approximately **$150–400**, with a **$300–600 cap** if integration retries, storage and larger validation are included. If the attention/build gate fails, stop before the full-teacher allocation. Engineering time can dominate: allow several working days for a successful adapter/integration path; missing backward kernels or a driver-incompatible stack can become a multi-week project.

Current DeepSeek API output pricing is $0.60/M tokens off-peak and $1.20/M peak: **10M outputs cost $6–12; 100M cost $60–120; 1B cost $600–1,200**, plus input charges. This buys trajectories, not local pruning statistics or recovery compute. Input cache-hit and miss prices differ; record actual billed usage and model version.

Source: [official DeepSeek API pricing](https://api-docs.deepseek.com/quick_start/pricing).

For any generation or replay job, estimate:

`node_hours = processed_tokens / measured_aggregate_tokens_per_second / 3600`.

`compute_cost = node_hours × node_hourly_price`.

Use processed tokens for calibration and training, output tokens for a decode-only estimate, and add loading, compilation, prefill, tools and retries separately. For example, a 12-hour run emitting one **64,000-output-token trajectory** for each of an illustrative 240 test cases requires about **356 aggregate output tokens/s before overhead**; two such trajectories require about 711. The actual competition test count and your harness lengths should replace this illustration. Sixteen active billion parameters is not a throughput forecast.

## Execution order and decision gates

### Working day 1: inexpensive feasibility

Run the environment probe inside Kaggle. Capture actual driver, SM89, memory, topology/P2P, Python/CUDA ABI, disk/attached-input capacity and `/dev/shm`. Confirm the live competition rules and offline execution behavior. The accessible competition material specifies a 12-hour notebook limit and disabled Internet during evaluation; host RAM, attachment limits and current driver were not reliably exposed by the public page and remain measured inputs.

In parallel, run CPU-only Engram census on existing traces and validate native attention/MoE fixtures on one actual L4. Establish original-ID parity, host payload budgets and candidate row masks before renting a large teacher node. Stop if kernels or compact-table host requirements cannot be made compatible with the environment.

### Working days 2–3: serving skeleton and pruning evidence

Patch E64/E72 routing, integrate compact host Engram and run a topology-preserving TP4 fixture. Validate candidate Engram masks with full-expert forwards and held-out rollouts, then freeze the chosen mask. Only then collect the 0.25–1M-token expert replay pilot on a suitable node or amortized streamed implementation. Grow scoring when stable statistics and a useful quality curve justify it. These working-day labels are sequencing goals, not measured duration guarantees.

### Working days 3–5: real E64 candidate and competition comparison

Use selective original-ID loading for E64/E72 quality, then export/reindex E64 first. Verify tensor bytes/scales, frozen Engram mask identity, matched-semantics logits and all 40 layer shapes. Measure TP4 resident/peak memory, host-table gather latency and complete ARC throughput. Measure retained DeepSeek quality under the fixed tool-free solve policy; evaluate E72 if its quality gain and measured headroom justify it.

### Only after a usable baseline

Try router recovery, adaptive expert budgets, smaller expert counts, CUDA graphs, prefix caching, CED and DSpark in separate controlled experiments. Nonuniform expert counts require per-layer loader/config support. Keep uniform E64 as the first deployment implementation. A changed Engram mask restarts its validation and expert-scoring dependency, even if its parameter budget is unchanged.

**Continue** if the model fits the real environment, completes long trajectories reliably and offers competitive solve rate within the notebook limit. **Revise the failing stage** if Engram row pruning damages held-out quality, expert pruning collapses capability, compact host access exceeds limits, or the harness remains too slow. Increase the affected capacity budget or return to distillation based on measured evidence. Do not fund a billion-token recovery run solely because a checkpoint loads.

Competition sources: [Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2), [rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules), [ARC Prize 2026 ARC AGI 2 track](https://arcprize.org/competitions/2026/arc-agi-2).

## Handoff files and first commands

START_HERE.md provides immediate assignments and avoids the E04/P02 dependency cycle by starting the original-model P02a scaffold before Engram validation. REAP_IMPLEMENTATION_CONTRACT.md specifies the audited native forward and exact statistics. REAP_HARDWARE.md supplies the RTX execution choices. TASKS_ENGRAM_FIRST.md provides the CPU census, retained-row selection, validation and freeze gate. TASKS_SERVING.md provides source pins, operator contracts and compact-host integration. TASKS_PRUNING.md provides the subsequent observer/statistics/export/quality tasks. KV_MEMORY_NOTES.md separates native payload from engine reservations. These describe code to implement; they do not claim an engine already exists in this pack.

The following tools are included and have been executed in this research environment:

~~~bash
python3 tools/probe_environment.py > environment.json
python3 tools/estimate_budget.py --experts 64 72 --context 65536 --batch 16
python3 tools/estimate_budget.py --experts 64 72 --context 65536 --batch 32 --format json
python3 tools/estimate_budget.py --experts 64 --tokens 10000000 --throughput 500 --hourly-usd 2.36
~~~

The 500 tokens/s in the last command is an explicit hypothetical input, not a benchmark. Replace GPU memory, fixed residency, workspace, cache policy and throughput with measurements. The calculator reports aggregate capacity and cannot prove balanced per-rank fit.

Begin with the CPU Engram census and real-L4 fixtures. Validate and freeze retained-row behavior, then collect fresh expert statistics under that behavior. E64 is the primary export; E72 is a measured quality/headroom comparison. This sequence is the current plan and supersedes the earlier E48-first, full-Engram-preservation proposal.
