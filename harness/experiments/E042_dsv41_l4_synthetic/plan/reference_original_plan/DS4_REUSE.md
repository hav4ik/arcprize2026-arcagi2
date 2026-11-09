# Reusing antirez/ds4 for the Engram-first pruning pipeline

**Decision:** reuse its standalone CPU Engram hashing, bounded row reads and streaming export machinery. Keep native V4.1 REAP and the SM89 vLLM deployment as the primary bases. The inspected ds4 engine is not a drop-in E64/E72 server for four L4s, and Q2 conversion is not the default quantization choice for this plan.

Inspected [antirez/ds4 commit fc80bd695da76ee14bcb86820bf900a5cdfda806](https://github.com/antirez/ds4/tree/fc80bd695da76ee14bcb86820bf900a5cdfda806) is dated 7 October 2026, before the 1017 checkpoint. Use the actual [1017 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) and verify its configuration/tokenizer inputs. The converter documentation targets older base-V4.1 weights.

## Components to reuse

| Source path, relative to ds4 | Existing functionality | Required adaptation |
|---|---|---|
| `ds4_engram.c/.h` | CPU hash, explicit history, layout validation and original row IDs | Census wrapper, count arrays, lineage splits and exact 1017 parity fixtures |
| `gguf-tools/deepseek41_metadata.py` | `compressed_token_map()` and `engram_layout()` | Export a CPU layout from exact 1017 metadata without converting weights |
| `ds4_engram_read()` / `ds4_engram_read_batch()` | Native 264-byte row reads, deduplication, bounded scratch | Compact original-ID mapping and frozen missing-row semantics |
| `deepseek41_quantize.py::write_engram()` | Streaming FP8 values and original scales | Export selected rows rather than every row; compact offsets/manifest |
| `glm53_quantize.py::SourceDB.iter_read()` | Tensor byte-range reads | Coalesce retained row spans and preserve provenance/checksums |
| Sequential imatrix collector | Full-position observer plumbing in the Metal path | Different statistics and a native REAP observer; imatrix values are not REAP |

## CPU census: immediate practical value

The standalone interface is:

```c
bool ds4_engram_hash(const ds4_engram_layout *layout,
                    ds4_engram_history *history,
                    const int *tokens, const uint8_t *mask,
                    size_t count, uint32_t *rows);
```

Output is `[token][2 Engram layers][24 columns]`, the required 48 original row IDs per token. History stores the preceding three compressed token IDs, newest first; copying it snapshots hash state, and `ds4_engram_history_reset()` starts a sequence. The layout contains the token map, original row counts, multipliers and primes. Validate it before counting.

**The `mask` argument is a token eligibility/history mask, not the retained-row mask.** A zero breaks n-grams crossing that position, and the graph must suppress Engram at masked positions. Hash eligible tokens with unchanged original geometry, then count or map the output IDs separately. Do not implement row pruning by changing this mask or the hash moduli.

No GGUF, model weights or GPU is required for this stage. The metadata helper needs its Python dependencies and exact tokenizer/config files. A small C executable or extension can hash chunks and update counts without sending every row ID through Python callbacks. Retain the E00 lineage splits and E01 reset/chunk/BOS/normalization gates in TASKS_ENGRAM_FIRST.md. Existing compatible tool-free API traces remain sufficient.

Sources: [interface](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_engram.h), [CPU implementation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_engram.c), [metadata helper](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/deepseek41_metadata.py).

## Row reads and compact export

The native record is **256 FP8 E4M3 value bytes + eight original E8M0 scales = 264 bytes**. Batched reads deduplicate/sort requests and restore output order. Documented temporary scratch is bounded to 384 KiB, but caller-owned output scales with batch size and must be chunked separately.

The reader produces **F32 storage containing BF16-rounded values**. Transferring that buffer directly would use twice the bytes of BF16 staging. For native vLLM integration, preserve packed rows and use its GPU gather/dequantization path, or explicitly pack decoded values into BF16 staging after parity checks. The C reader is a useful reference and optional disk component, not the native compact pinned-UVA adapter itself.

`write_engram()` already streams 16,384 rows per chunk, approximately **4.125 MiB of native row payload**, not a total-RAM measurement. It currently exports **all** rows. Adapt it to selected original row spans via `SourceDB.iter_read()`, preserve FP8/scales exactly, and write compact bank offsets and original-ID mappings. Hash geometry must retain original row counts/moduli; physical compact storage needs separate dimensions. The full-table loader requires an explicit adapter.

The existing plan's miss policy remains **zero the missing embedding row before Wkv/projection**, followed by native projection/gating. Validate compact hit/miss arithmetic against a full-table reference executing the same mask. Freeze this behavior after E04 quality validation, then collect REAP with it active.

Sources: [row I/O](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_engram.c), [Engram writer](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/deepseek41_quantize.py), [source byte-range reader](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/glm53_quantize.py).

## Full-engine limitations for the proposed deployment

- **Expert shapes:** `DS4_SHAPE_FLASH41` fixes 384 routed experts; the CUDA router accepts 256/384. E64/E72 need shape, loader, routing and dispatch changes, not just a different export count.
- **Multi-GPU:** the inspected V4.1 runtime rejects CUDA tensor parallelism and more than one local GPU. Network TP is exactly two ranks. Neither is ordinary local TP4 on four L4s.
- **Cache representation:** `ds41_graph_alloc` stores main compressed history and index keys in F32. Three half-rate sources plus one full-rate source use **6400 bytes per original token per graph**, plus **10 MiB logical SWA per request** and small history-capacity rounding (10,240 bytes for an even token count). Numerical FP4/FP8 rounding does not change those F32 allocations. This is approximately 7.2× the native vLLM 890-byte growing-history slope. At 65,536 tokens it is **approximately 410 MiB per graph including SWA**, versus approximately 58.20 MiB per native vLLM cache copy. Do not transfer native vLLM memory formulas or infer a supported TP4 split for ds4.
- **Packed CUDA operations:** some V4.1 packed-indexer/tensor entry points are stubs. Existing functional paths do not establish equivalent packed-cache acceleration.
- **Different published demonstrations:** V4.1 CUDA Q2 SSD batching supports up to **eight decode rows**; this is a batching limit, not a tokens-per-second measurement. The documented CUDA network TP path serves sessions in order. The 126 tokens/s L40S demonstration is V4, not V4.1. These are publisher reports, not L4 forecasts or pruning results.

Sources: [engine shapes, allocation and validation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c), [CUDA implementation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu), [V4.1 CUDA paths](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_deepseek41_cuda.cuh), [publisher README](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/README.md).

## Observer plumbing is useful; imatrix is not REAP

`ds4_engine_collect_sequential_imatrix()` executes sequential full-position work across the 40-layer model, which is relevant to avoiding decoder-skipping during calibration. However, the inspected collector is Metal-only, and its imatrix accumulates squared input-channel statistics for quantization. Those values are not REAP's conditional mean of weighted expert-output norms or HOPE coactivation statistics.

Borrow observer placement/replay ideas after native-semantic validation. Keep [LibertAI's V4.1 REAP infrastructure](https://github.com/Libertai/deepseek-v41-flash-reap) as the initial scoring base, including TASKS_PRUNING.md's weight-dependent activation-quantization caveat. Adopting Q2 conversion would add another model change; the first pipeline continues to preserve selected native FP4 expert matrices and Engram row bytes.

## Agent handoff and test status

1. **Pin and extract.** Record ds4/model revisions and exact 1017 tokenizer/config checksums in `locks/engram.json`. Export hash layout using the metadata helper. If reusing conversion, pass the actual 1017 revision and correct the hard-coded source-model metadata URL.
2. **Wrap census.** Implement `tools/engram/census_rows` with the C hash, count arrays and state fixtures. Run E01 native parity before the proposed large census.
3. **Select/export.** Adapt bounded range reads and row packing for global keep-all-observed/frequency selection. Emit compact rows, mapping and manifest. Use `tools/compact_engram_reference.py` as a synthetic contract, not a production exporter.
4. **Integrate/validate/freeze.** Add compact lookup to native vLLM CPU offload, validate arithmetic and complete tool-free quality, freeze the mask, then run the REAP pilot and E64/E72 pruning.

**Executed:** `make test-engram` passed on Linux during this research, covering the repository's hash/history and bounded row-reading/code-scale fixtures. **Not executed:** an actual 1017 weight forward, pruned-table integration or GPU run. The census wrapper, selective exporter and native compact-table adapter remain proposed work.
