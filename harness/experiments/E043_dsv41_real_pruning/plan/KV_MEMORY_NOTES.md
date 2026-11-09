# DeepSeek V4.1 Flash 1017: cache payload and serving reservations

Verified source arithmetic as of 19 October 2026. These are byte/shape calculations from the pinned native SM89 vLLM and paired FlashInfer implementations. No L4 memory profile or benchmark was run. Ordinary TP4 is assumed, with no DSpark and no context parallelism.

## Native persistent history

There is one 512-dimensional latent KV head, including its 64 RoPE dimensions. The latent serves as both K and V; no second K/V multiplier applies. Query heads divide across TP ranks, but the single latent cache and index-key cache are **replicated on each rank**.

| Component | Stored bytes per row | Owners / retained history |
|---|---:|---|
| Global latent KV | 288: 256 packed FP4 bytes + 32 FP8 E4M3 scale bytes, group 16 | Layers 2/8/14 at ratio 2; layer 20 at ratio 1 |
| Index key | 68: 64 packed FP4 bytes + 4 UE8M0 scale bytes, group 32 | Same four persistent sources; later reindex layers share keys |
| Sliding-window KV | 528: 512 FP8 value bytes + 16 UE8M0 scale bytes, group 32 | 128-token logical window in each of 40 layers |

For a sequence containing `T` live prompt, reasoning and tool-history tokens:

`global_and_index_bytes_per_rank(T) = 356 * (3 * floor(T/2) + T)`.

This equals **890*T bytes for even T**. Multiply by four for aggregate TP4 GPU memory. Sparse selection of up to 512 global rows bounds an attention selection, not stored history; global history still grows with sequence length. Index scoring also has context-dependent work.

Logical SWA adds `40*128*528 = 2,703,360 bytes = 2.578125 MiB` per sequence per rank.

| Tokens per sequence | Global/index per rank | Global/index + logical SWA per rank | TP4 aggregate including logical SWA |
|---:|---:|---:|---:|
| 20,000 | 16.9754 MiB | 19.5535 MiB | 78.2141 MiB |
| 32,768 | 27.8125 MiB | 30.3906 MiB | 121.5625 MiB |
| 65,536 | 55.6250 MiB | 58.2031 MiB | 232.8125 MiB |

The CLI value `--kv-cache-dtype fp8` selects the native **mixed** layout in this backend. It is not uniform FP8 KV. Native V4.1 requires the FP4 index-key path; the generic 132-byte FP8-indexer branch is not an enabled fallback for this model.

## BF16 reference is a different representation

The official reference stores fake-quantized values in BF16 cache tensors. Its logical history is `2*(512+128)*(3*floor(T/2)+T)` bytes per rank, or **3200*T bytes for even T**, plus **5 MiB SWA**. At 20,000 / 32,768 / 65,536 tokens it needs **66.0352 / 105 / 205 MiB per rank**, or **264.1406 / 420 / 820 MiB aggregate TP4**. The reference may preallocate by maximum sequence length and maximum batch; its allocation is not a paged-engine occupancy estimate.

The native kernel dequantizes tiles when needed. It does not retain a second full BF16 historical cache. A hypothetical 1650-byte/token FP8 calculation is an arithmetic comparison, not proof of a supported native launch option.

## Bounded request state and page reservations

These are separate from the 890-byte growing-cache slope:

- **Compressor ring:** three ratio-two sources × eight state rows × two 512-dimensional FP32 vectors = **96 KiB/request/rank** without larger speculative buffering. The ring grows if speculative settings demand a larger power-of-two block.
- **SWA paging:** the admission bound is `ceil(min(127 + extra_retained_tokens + max_in_flight_tokens, max_model_len)/32) + 1` pages per layer, each 32 rows of 528 bytes. A one-token steady decode therefore reserves at least **160 rows**, or **3.222656 MiB across 40 layers**, rather than the 2.578125 MiB logical floor. Prefill chunks require additional in-flight rows. Use actual configured page grouping and admission behavior for capacity planning.
- **CED encoder replay tail, if enabled:** for each reserved request slot, `window * (4*5120*2 + 4*4 + 8)` bytes per rank stores BF16 hidden streams, FP32 coefficients and int64 token IDs. At window 128, this is **5.00293 MiB/slot/rank**. It is reserved by slot capacity, not just currently active requests.
- **Vision-aware CED caveat:** the source chooses `window = 128 + vision_max_n_token` when `vision_n_layers > 0`. Retaining 1024 image tokens produces **45.02637 MiB/slot/rank**. CED decoder SWA also retains 1024 additional rows in each of 20 decoder layers, another **10.3125 MiB/slot/rank** before paging. Verify the effective normalized text-only config; do not assume a CLI mode automatically removes the reserve.
- **mHC:** working residual streams are not an extra full-sequence KV cache. Their persistent replay-tail portion is included in the CED calculation; prefill activations remain transient working memory.

## Shared native engine buffers

| Buffer | Default or example reservation per rank | Accounting |
|---|---:|---|
| Sparse indexer logits | 512 MiB | Default `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512` |
| Indexer block scores, radix and lengths | 65.03125 MiB | 64 MiB + 1 MiB + 32 KiB |
| Combined indexer workspace | **577.03125 MiB** | Shared across indexer layers; do not multiply by eight |
| Candidate and top-k arrays | **80 MiB at 8192 scheduled-token capacity** | `max_num_batched_tokens*(2048+512)*4` bytes |
| FlashInfer split-attention scratch | Shape-dependent | FP32 LSE and BF16 partial outputs, local query heads and split count |

At TP4, the first two combined shared reservations are **2,755,788,800 bytes = 2.56653 GiB aggregate** at scheduled-token capacity 8192. These are not per-sequence KV. Smaller configured token capacity changes the candidate arrays; indexer default scratch remains a separate setting.

For illustration, **32 requests × 65,536 tokens**, TP4, 32 reserved CED slots at a 128-token tail and scheduled-token capacity 8192 give:

| Known component | Aggregate bytes | Decimal GB |
|---|---:|---:|
| Native global/index + logical SWA + compressor rings | 7,824,474,112 | 7.824474112 |
| Shared indexer + candidate/top-k buffers | 2,755,788,800 | 2.755788800 |
| CED 128-token tails | 671,481,856 | 0.671481856 |
| **Known nonweight subtotal** | **11,251,744,768** | **11.251744768** |

This subtotal excludes extra SWA pages, larger in-flight prefill storage, additional attention/quantization scratch, allocator fragmentation, CUDA graphs and Engram gathered/staging outputs. It assumes CED enabled with a 128-token tail; use zero CED reserve when that implementation truly disables its allocation, or the verified larger tail when needed. DSpark adds other state and is outside this baseline.

## Budget interpretation

Use the estimator's explicit component fields and then add a **residual workspace** reserve for the remaining unmodeled allocations. Do not count the known indexer/candidate/CED buffers twice by also treating them as included in that residual reserve. Older versions of this pack used a single 4 GiB workspace allowance to cover both known and unknown buffers; that aggregate shortcut is superseded by explicit components.

Bulk CPU-resident Engram table rows do not count as GPU weights. Engram Wkv/projection weights, GPU map indices, gathered embedding outputs, staging, transfer buffers and any GPU hot cache do. Measure host pinning separately. A GPU-memory-utilization cap does not guarantee every rank fits, nor does a large allocation reported by `nvidia-smi` imply that each request consumes the entire reserved cache pool.

Report measured weight bytes, active cache occupancy, reserved cache capacity, fixed buffers, maximum request slots, transient prefill peak and actual maximum rank separately. Begin E64 at low concurrency. E72 has less residual headroom and needs a measured quality advantage; no cache arithmetic here establishes throughput or a safe batch size.

## Primary sources

- [Exact checkpoint configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)
- [Official reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py)
- [Official V4.1 architecture/inference report](https://arxiv.org/html/2609.19969v1)
- [Pinned native attention and cache ownership](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/attention.py)
- [Compressor ring specification](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/compressor.py)
- [CED tail and replay capacity](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/ced.py)
- [Paged cache admission](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/v1/kv_cache_interface.py)
- [Sparse indexer workspace](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/layers/sparse_attn_indexer.py) and [environment defaults](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/envs.py)
- [Shared candidate buffers](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/model_state.py)
- [Native FlashInfer API and scratch sizing](https://github.com/yhfgyyf/flashinfer/blob/d4eef3c44c70594ff18b9126381ae6a978694e3f/flashinfer/mla/deepseek_v41.py)

These allocator statements apply to the inspected native SM89 implementation. Current SGLang allocation was not verified to have identical state, paging or scratch behavior.
