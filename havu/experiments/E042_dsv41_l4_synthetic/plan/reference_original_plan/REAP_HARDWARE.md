# RTX PRO 6000 hardware plan for REAP

Updated 19 October 2026 UTC. Here **RTX 6000 means RTX PRO 6000 Blackwell 96 GB**, not RTX 6000 Ada 48 GB or RTX A6000. Prices below are a dated public reference, not a quote or a reservation. No GPU was rented or benchmarked in this research.

## Recommendation

Prioritize RTX PRO 6000 Blackwell for this project. Use **one 96 GB GPU for native-forward/observer development and the streamed pilot**. Use **four or eight 96 GB GPUs for a resident packed-model calibration implementation**, once its observers and replay schedule pass parity. If the user's eight-GPU offer is cheaper than two B300s, eight RTX cards are a credible main calibration choice; compare completed calibration work per dollar after a short pilot.

REAP is forward-only: it needs no gradients, optimizer moments or trainable master-weight copy. It needs the original full-expert computation, representative causal histories, individual selected-expert contribution measurements, and small sufficient-statistics arrays. After Engram validation, use all 384 experts with the chosen frozen row mask. Do not prune experts first merely to make the scoring model fit.

There is direct public serving precedent: the [SM89/SM120 fork](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) reports full V4.1 service on **4 × RTX PRO 6000 96 GB**, using TP4, expert parallelism and CPU Engram offload. Its latest focused SM120 tests are distinct from a full-model throughput rerun. This supports the hardware choice, but does not constitute 1017 REAP, observer, long-calibration or 8-GPU validation.

## 1. What must fit

These are derived payloads from the exact 1017 expert geometry; GiB means 2^30 bytes and GB means 10^9 bytes. Auxiliary modules and runtime peaks need a measured inventory.

| Component/representation | Payload | Implication |
|---|---:|---|
| One native packed expert including scales | 18,800,640 B = 17.9297 MiB | Native FP4 matrices plus one-byte group 32 scales |
| One layer's 384 native packed experts | 6.7236 GiB | Suitable for layer streaming on one 96 GB card |
| All 40 routed expert banks, native packed | 288.78 GB / 268.95 GiB | Full source expert capacity before nonexpert modules |
| All 40 routed banks after Marlin BF16 scale expansion | 305.76 GB / 284.77 GiB | The serving backend's runtime weight format |
| Marlin full backbone plus the packet's provisional 12.6 GiB fixed reserve | 319.29 GB / 297.37 GiB | Planning estimate; additional TP/EP replication and actual fixed tensors must be measured |
| Full original native Engram tables | 202.76 GB / 188.83 GiB | Host/disk or separately measured GPU placement; not intrinsically a required VRAM charge |
| Entire 552B backbone materialized at 2 bytes/parameter | About 1.104 TB, before Engram | Does not fit eight 96 GB GPUs; avoid generic BF16 expansion |

The native streamed collector preserves FP4 packed bytes, using an I8-to-FP4 **view**, and keeps native scale storage. It does not require unpacking a whole expert bank into BF16. Temporary activation/GEMM tiles are another matter. Some dense/reference modules deliberately use FP32/BF16, so preserve the native mixed-precision contract rather than globally casting the model.

**Four 96 GB cards:** memory-feasible precedent exists with host Engram, but long-prefill/observer headroom must be controlled. Ideal four-way splitting of Marlin routed weights alone is 71.19 GiB per rank. **Eight 96 GB cards:** lower per-rank routed residency and more room for calibration state. Eight cards could also have enough aggregate capacity for full original Engram plus packed backbone, but the baseline retains the host-table path to match the pruning/deployment adapter. Do not count device memory as one pool without a real placement plan.

## 2. Hardware by phase

| Phase | Starting allocation | Host/storage recommendation | Advancement condition |
|---|---|---|---|
| Engram address census | CPU only | 32–64 GB RAM, enough local disk for token/count shards | Exact hashes/counts and measured census throughput |
| Native forward, row lookup, observer and short replay | 1 × RTX PRO 6000 96 GB | 128–256 GB RAM with bounded disk Engram; 512 GB preferred if retaining the full original table in RAM; 1–2 TB local NVMe | Native parity and no accidental full-model/table allocation |
| Long streamed calibration pilot | Same 1 × 96 GB | Frozen compact Engram in RAM where feasible, fast source NVMe, bounded indexer and observer workspaces | Whole-prefix 32K/64K peak fits and statistics are complete |
| Resident packed calibration | 4 × 96 GB first if readily available; 8 × 96 GB for more headroom or the preferred quote | 512 GB host RAM is a useful starting spec; 1 TB helps source caching/multiple workers. Recompute from actual table/storage design | Distributed parity and measured valid replay positions per dollar |
| Final deployment validation | 4 × L4 in the target environment | Actual Kaggle limits | Required separately from RTX correctness/performance |

These host sizes are recommendations, not architectural minima. After row pruning, a 32–40 GiB table plus mapping, source-read staging and process memory can fit a much smaller host than the original table. The original full-table control can use bounded disk lookup through the audited Engram offload infrastructure when RAM is limited. Its latency is measurable; do not inherit the upstream comment that disk lookup is negligible.

The complete source is approximately 510 GB. Budget disk for source, selected export, intermediate native/Marlin conversion, package caches and atomic writes. One 1 TB volume can be tight if several full source/conversion copies coexist; 2 TB local NVMe is the safer development allocation. Stream native row/expert extents and avoid materializing every candidate checkpoint. Reading from a remote volume rather than local NVMe can change the cost ranking of GPU offers.

## 3. The 64K reference indexer must be fixed

The [official Indexer](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py) forms `einsum('bshd,btd->bsht', q, index_k)` before head reduction. At sequence length L, batch B, local index heads H and compression ratio r, its raw score tensor has approximately:

`B * L * H * floor(L/r) * score_element_bytes`.

For BF16, B 1, L 65,536 and 32 heads:

| Configuration | Raw score tensor alone |
|---|---:|
| One GPU, ratio 1 | 256 GiB |
| One GPU, ratio 2 | 128 GiB |
| TP4, ratio 1 per rank | 64 GiB |
| TP8, ratio 1 per rank | 32 GiB |

The subsequent weighted elementwise product can allocate another tensor of the same size, and reduced scores/candidate masks add more. The ratio 1 full query-key boolean candidate mask alone is 4 GiB. Persistent KV's small byte count does not bound these transient allocations. A consumer-Blackwell sparse-attention shared-memory patch does not solve this indexer tensor.

**Implement query tiling inside Indexer.forward while initially keeping a whole-prefix, start_pos 0 forward.** Start with query tile 128 and adjust to a declared workspace budget. At the worst ratio 1 source, one GPU's raw BF16 score tile then becomes 512 MiB rather than 256 GiB. This changes workspace, not which historical keys a query can see.

Preserve global query positions, ratio-dependent visible compressed length, source 20 candidate-block selection, per-query current-block pinning, later-layer candidate reuse, selected-index ordering and invalid sentinels. Test short untiled/tiled parity and boundary cases, then long prefixes. Compact candidate block IDs can replace the dense boolean mask if that additional optimization is necessary and validated. Do not allocate all tiles then concatenate their raw score tensors.

The official whole-prefix reference already executes all 40 backbone layers, making it useful for scoring. Its nonzero-start continuation branches assume one token. Arbitrary multi-token calls with increasing offsets are not a ready chunked implementation. Query tiling within whole-prefix execution avoids changing compressor scheduling initially. Do not create full-vocabulary logits for every position solely to collect REAP statistics; project only diagnostic positions when needed.

Apply the phase masks in [CED_CALIBRATION.md](CED_CALIBRATION.md): executing a position to construct causal state does not require including it in the decoder-generation score. Full replay uses both halves; the 8B optimized-prefill rate is not its throughput proxy.

## 4. Two legitimate calibration implementations

### A. Corrected single-GPU streaming: cheapest development path

Start with LibertAI's pinned packed streamer and its reference-kernel path, fix the native forward described in REAP_IMPLEMENTATION_CONTRACT.md, add validity masks and resumable sufficient statistics, and tile the indexer for long prefixes. The single-GPU recommendation is for this corrected reference path, conditional on its SM120 kernel check. It is not a claim that the native FlashInfer serving backend supports TP1: the inspected attention operators accept local head counts 8/16/32, while the 64-head model at TP1 would require 64. Keep original source experts on local NVMe and load only the active layer bank. Benchmark source-read, GPU-forward and observer time independently. A cold full routed-bank sweep reads 268.95 GiB; at B GiB/s its I/O floor is 268.95/B seconds, before compute and other tensors.

Begin with 2K/4K correctness sequences, then extend length. Short tests do not replace representative long histories in the eventual calibration. Amortize each layer load across valid tokens; do not use disk-streamed one-token generation as the main high-throughput plan.

Upstream sets device 0 and its patched MoE omits distributed reduction. Simply launching it with `torchrun --nproc-per-node8` creates an invalid collector. A single worker, independent data workers with explicit device ownership, and a distributed model are different execution modes. Eight independent streamers can saturate shared storage/CPU memory and are not automatically cheaper than one resident model.

### B. Instrumented native SM120 runtime: main resident-model path

Use the pinned vLLM/FlashInfer pair from TASKS_SERVING.md with its SM120 setup, not the L4-specific architecture environment unchanged. The inspected backend selects Marlin for V4.1 on SM120/SM89. It rejects a generic b12x path whose activation quantization lacks the required minimum amax 1e-4. Advertised native FP4 peak FLOPS therefore do not predict this adapter's throughput.

The routed down-projection in `_fused_marlin_moe` exposes a `[tokens*top_k,5120]` contribution tensor before `moe_sum`/top-k reduction. This is the concrete native-amplitude observer insertion point. Preserve dispatch-to-global-ID mapping, valid scored positions and the distinction between whole EP-owned outputs and TP partial vectors. Reduce chunk-local sufficient statistics; never retain a complete activation history.

Start with eager execution, DSpark off, CED decoder-prefill skipping off, and prefix caching off. Otherwise Python hooks may run at graph capture but not replay, speculative positions can be counted incorrectly, or cached prefixes can skip expected observations. Prove full-position teacher forcing and all 40 layer counts against the corrected reference before scaling.

Compare **one TP8/EP model** with **two independent TP4/EP workers on disjoint corpus shards** only after the four-card model is correct. The latter reuses a documented serving topology and avoids eight-way communication, but duplicates model state and possibly pinned Engram tables. The former gives more per-rank headroom; local attention head count 8 is supported by the inspected operators, but full 8-GPU calibration is not yet measured. Use actual PCIe/P2P/NUMA topology and host-memory placement to choose; do not assume eight-card linear scaling.

## 5. RTX versus B300

The [NVIDIA RTX specification](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) gives 96 GB GDDR7 and 1,597 GB/s memory bandwidth per RTX PRO 6000. [NVIDIA Blackwell Ultra documentation](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) gives up to 288 GB HBM3e, 8 TB/s memory bandwidth and 1.8 TB/s bidirectional NVLink bandwidth per GPU; confirm the rented B300 SKU and topology.

| Nominal comparison | 8 × RTX PRO 6000 | 2 × B300 |
|---|---:|---:|
| Advertised total GPU memory | 768 GB | 576 GB |
| Sum of device memory-bandwidth specifications | 12.78 TB/s | 16 TB/s |
| Model communication | PCIe topology/P2P dependent | NVLink if provided by the rented topology |
| Software gate | SM120-native kernels plus new collector | B300-supported kernels plus new collector |

The user's capacity comparison is correct for these SKUs. Aggregate memory bandwidth is not an end-to-end throughput forecast; collectives, expert batching, attention kernels, observer work and CPU row fetches matter. REAP has no backward pass, so B300 is not required merely because this is model compression.

For completed work N, whole-node hourly prices P_R/P_B, and measured throughput T_R/T_B, the RTX option is cheaper when:

`T_R / T_B > P_R / P_B`.

An illustrative $8/hour RTX node versus $18/hour B300 node can take 2.25 times as long for equal GPU rental cost. These are example quotes, not current advertised offers. If both execute the same correct workload, use the user's real offer in this calculation.

## 6. Dated cost worksheet and spending sequence

[Runpod's public price page](https://www.runpod.io/pricing), retrieved 19 October 2026 UTC, displayed $2.49/hour per RTX PRO 6000 and $8.99/hour per B300. Those list rates differ from the cheaper eight-GPU offer described by the user; inventory, marketplace and whole-node quotes can differ.

| Allocation at those displayed per-GPU rates | GPU rental/hour | GPU rental/24 hours |
|---|---:|---:|
| 1 × RTX PRO 6000 | 2.49 USD | 59.76 USD |
| 4 × RTX PRO 6000 | 9.96 USD | 239.04 USD |
| 8 × RTX PRO 6000 | 19.92 USD | 478.08 USD |
| 2 × B300 | 17.98 USD | 431.52 USD |

Do CPU census/implementation before renting a large node. A first **4–12 billable GPU-hour allowance on one RTX** is 9.96–29.88 USD at the displayed rate; this is a debugging budget, not a promise that all fixes finish within it. Rent a larger node for a short warm benchmark only after its collector is ready. Avoid paying for eight idle GPUs while agents edit Python or download the checkpoint.

For calibration, calculate `hours = processed_positions / measured_processed_positions_per_second /3600`. Keep processed positions, scored positions and generated output tokens separate. A sampled late position still needs its true causal history. Full 40-layer scoring cannot be priced using an 8B-active optimized-prefill throughput figure.

For **8M processed full-position tokens**, the following are scenarios, not measured speeds:

| Measured aggregate replay speed, if achieved | Runtime | Cost on 8 RTX at 19.92 USD/hour |
|---|---:|---:|
| 100 positions/s | 22.22 hours | 442.67 USD |
| 500 positions/s | 4.44 hours | 88.53 USD |
| 1,000 positions/s | 2.22 hours | 44.27 USD |

Add cold loading, compilation, parity/debugging, actual full-table/masked quality runs, storage and transfer. Replace the hourly price with the user's quote and the speed with the 0.25–1M-token pilot measurement. There is no honest fixed total-compute quote before that measurement. Start with that small pilot; enlarge the 1–8M-scored-position run only if coverage/rank/quality uncertainty warrants it.

## Agent acceptance record

Save GPU SKU/count/VRAM, driver/toolkit/package pins, kernel backend, P2P/NUMA topology, CPU RAM/pinning, local disk free bytes and measured read bandwidth. Save packed weight bytes and all peak allocations per rank. Record Engram placement/mask checksum, observer definition, prefix/CED/speculation/graph modes, valid per-layer positions, source-read bytes, warm/cold time, replay throughput and billed cost. The decision is the cheapest correct completed experiment, with the remaining target-L4 validation explicit.
