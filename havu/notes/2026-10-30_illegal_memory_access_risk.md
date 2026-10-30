# Illegal memory access (IMA) on Manh's fork: are V12 and B32-M exposed? (2026-10-30)

This note comes from reading code and logs, plus a search of public GitHub issues. Nothing ran on Kaggle, on pods or
on local GPUs, and nothing was pushed. **"est."** marks an estimate. Everything else is read in code (file:line, at the
named revision) or seen in a log.

## The incident

Manh ran a notebook on Kaggle 4x L4 on 2026-10-29 night:
- his own older harness, the Qwen3.8 image-mode REPL recipe;
- DeepSeek-V4.1-Flash E128 2-bit on his vLLM fork;
- 120 concurrent requests, max context "128k".

After about 3 h the engine died with `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`.
It was not an OOM. 156 of 300 requests never finished. **We do not have his logs**, so everything below about his run
is inferred from his launcher and recipe defaults.

**Sources:**
- **Fork clone:** `scratchpad/e062_port/manh_vllm`:
  - v11 = `8cf0c0664b`, Manh's V12;
  - B32-M = `5e1bccd07c`;
  - files extracted to `scratchpad/ima_audit/{v11,b5e1}`.
- **His launcher:** `scratchpad/audit_v12/src/dsv41_vllm.py`, plus `qwen38_recipe.py` and `dsv4_llama.py` from the same
  source tree.
- **His production cell:** cell 2m of `kaggle/dsv41_tools/out/manh_v12_363638159/dsv41-tools-submit-b32m-external-server.ipynb`.
- **Our notes:** `notes/2026-10-30_vllm_fork_acceleration_map.md` and `notes/2026-10-30_vllm_failure_paths.md`.
- **Server logs:** every fork `server.log` under `experiments/E062*`, `E064*` and `E066*`, scanned with
  `scratchpad/ima/inventory_server_logs.py`.

Three research forks did the kernel audits and the issue search.

## Bottom line

**What his run probably had that ours never had** (from his launcher and recipe):
- **120 decode rows.** `max-num-seqs` = `--concurrency` (`qwen38_pipeline.py:426-427`). The CUDA graph cap equals the
  batch (`dsv41_vllm.py:115-116`), so graphs of 72-120 rows were captured and replayed.
- **Large images:** vision recipe, `--vision-max-pixels 2097152`. Each image hits the 1024-token cap, and a prompt
  holds up to 16 of them.
- **Context 128,000**, the recipe default (`qwen38_recipe.py:19-20`).
- **top_k 64 sampling** (profile `v41-1017-l4`, `dsv4_llama.py:109`). This makes the fork's top-K exchange in
  batch-sharded sampling fire. It never fires with our solver.
- **Frequent stream aborts** from his gzip loop guard.
- **Possibly a newer wheel or DCP2:** unknown.

**No kernel on any path we read reaches an out-of-bounds address purely through 120 rows, a 128,000 context or
images.** Every one masks its loads, uses int64 offsets, clamps to the logits width, or checks its buffer sizes.
(Section 1.)

**What the audit did find** is a set of **unbounded index sinks**. They turn a bad index into an IMA:
- the top-k → slot gather `_compute_global_topk_indices_and_lens_kernel` (`deepseek_v41/common/ops/cache_utils.py:895-955`,
  v11 and 5e1b) checks only `local_idx >= 0`, with no upper bound against the block table;
- the Triton sparse MLA `_load_kv_tile` accepts any slot ≥ 0;
- the paged logits kernel does not check `page < num_blocks`.

So the likeliest explanation is **corrupted indices or NaN logits from a rare, data- or timing-dependent event**, not a
shape bug.

**Ranked hypotheses for Manh's IMA** (detail in section 3; the weights are est., from one event and no logs):

| # | hypothesis | weight (est.) | V12 exposure | B32-M exposure |
|---|---|---|---|---|
| H1 | NaN/Inf in a prefill indexer's logits → `top_k_per_row_prefill` (no NaN handling; public #49896, open, in neither wheel) writes garbage indices → unbounded slot gather / sparse MLA reads out of range. NaN sources his run had more of: 1024-token images, near-max contexts, heavy host restores | 30% | **yes**, same kernels; fewer NaN sources (text only) | yes, same sink (DCP branch); text only |
| H2 | A cross-stream race or use-after-free corrupts an index buffer (top-k buffer, block table, slot mapping); also the CPU-offload load stream, which has no compute fence (public #59768 class) | 20% | **yes** (host gather, AR overlap, Engram side stream, SimpleCPUOffload all present) | partly (Engram side stream; upstream OffloadingConnector instead) |
| H3 | Something only his config ran: graphs of 72-120 rows, NCCL Engram gather inside those graphs (> 85 tokens), top-K sampling exchange, DCP2, the vision encoder at full-size images | 20% | no (64 rows, text, no top_k, no DCP) | no (≤ 80 rows replayed, text, DCP4) |
| H4 | A wheel or commit newer than v11, or a local patch | 10% (unknown) | no, if V12 pins v11 | no |
| H5 | Positions at or above max_model_len: RoPE cache capped at max_model_len in v10+, unmasked cos/sin load. His harness runs requests to the limit; the scheduler caps positions (`scheduler.py:920-927`) | 5% | yes, rarely: our solver stops with 1024 tokens of room | no: 5e1b has no RoPE cap |
| H6 | Other: Humming MoE block-size mismatch at gaps in the tuning table, host-AR prefetch of freed memory, hardware | 15% | partly | little |

**Verdicts:**
- **Manh's V12 (TP4, 64 seqs, util 0.91, 163,840, text, vision recipe loaded):** **exposed to the same code, at a lower
  rate (est.).**
  - It runs every kernel, sink, side stream and offload path of his crash run except the H3 items.
  - Our clean record on this stack: ~14 h of no-DCP fork serving, including the exact-V12 pod replica at 3.07 h, ~7.8 h
    with images, and 0.7 h at 120 rows. That is about 30M generated tokens (est.), roughly 1.5 Kaggle runs. It shows the
    event is not common on our workload. It cannot show that the per-run risk is small.
  - If the cause is H3 (images, 120 rows, top_k), V12 is not exposed. If it is H1, H2 or H5, V12 can hit it.
  - An IMA kills the engine like an OOM does. His supervisor restarts on exit, but his solver (11bfaaaa, without our
    safety edits) loses a try for each attempt in flight.
- **B32-M (DCP4, 80 workers, 96-row graphs, util 0.945, 5e1bccd07c, Manh's flags off, OffloadingConnector, text):**
  **low exposure.**
  - It shares only the attention, indexer, Humming and Engram kernels, including the same sinks. It shares none of his
    host-comm, SimpleCPUOffload, vision or sampling paths.
  - It has ~31 h of clean DCP4 serving (~66M tokens est.). The exact-stack replica `b32m_x2t` ran 10.26 h with 73
    allocator retries.
  - Its Kaggle scored run did die at hour ~4-6 (LB 46.5) of an unobserved cause, so an IMA cannot be excluded there,
    but the OOM explanation still fits better: 1.1 GiB VRAM free, and the 5e1b wheel lacks the prefill-memory fix.
- **Note on `b32m-fix`:** our own candidate on Manh's v11 server (no DCP, SimpleCPUOffload) has V12's exposure, not
  B32-M's.

## 0. The configurations side by side

| | Manh's crash run (inferred) | V12 | B32-M | our coverage, 0 IMA |
|---|---|---|---|---|
| wheel | ? (v11 or newer) | v11 `8cf0c0664b` | `5e1bccd07c` | b9ad, 5e1b, f506 (v10), 8cf0c (v11) |
| TP / DCP | TP4 / ? (he reported DCP2 on 10-29) | TP4 / none | TP4 / DCP4 | both; DCP2 only 0.4 h at startup |
| max-num-seqs / graph cap / rows replayed | 120 / 120 / ≤ 120 | 64 / 64 / ≤ 64 | 96 / 96 / ≤ 80 | ≤ 80 for 44 h; 120 for 0.69 h (`ext_nodcp_w120`) |
| max-model-len | 128,000 (recipe default) or 131,072 | 163,840 | 163,840 | 163,840; 131,072 for 3.7 h |
| images | yes: 1024-token cap, ≤ 16 per prompt, `/inference/v1/generate` | vision loaded, text prompts | `--language-model-only` | small grid images ~7.8 h; full-size images only in the startup profile |
| sampling | T 1.0, top_p 1.0, **top_k 64** | T 1.0, top_p 1.0, no top_k | same as V12 | no top_k ever |
| stream aborts | frequent (gzip loop guard) | rare | rare | ~3 per 10 h |
| host AR / host gather / row shard / sparse logits / AR overlap | presumably on (his env) | on | **off** (not in 5e1b, or flags off) | on in all no-DCP runs |
| KV offload | SimpleCPUOffload (his default) | SimpleCPUOffload 10 GiB/rank | OffloadingConnector 32 GiB | both |
| Kaggle hardware | yes (50 W, PCIe Gen3) | yes | yes | pods (72 W, Gen4), plus 0.8 h of Kaggle smokes on 5e1b |

## Evidence of absence

Scan of every fork `server.log` in the repo: wheel, seqs, graph cap, max-model-len, DCP, mm limit, log span and
IMA/OOM/fatal counts. **There are 0 "illegal memory access" lines.** The fatal lines are our kill tests.

| config | wheel | seqs / graphs | ctx | images | running max | h |
|---|---|---|---|---|---|---|
| B32-M exact (`E064/l4e2e/runs/b32m_x2t`) | 5e1bccd07c | 96 / 96, DCP4 | 163,840 | no | 80 | 10.26 |
| `E062/l4/runs/manh_x2t_w80` | b9ad670a67 | 96 / 96, DCP4 | 163,840 | no | 80 | 10.46 |
| `acc48/acc_manh3`, `acc_manhflags` | b9ad670a67 | 64 / 64, DCP4 | 163,840 | no | 56 | 5.2 |
| e062b arms and crash test | b9ad / 95376a1 | 96 / 96, DCP4 | 163,840 | no | 80 | ~3.2 |
| B32-M Kaggle smokes V2, V3 | 5e1bccd07c | 96 / 96, DCP4 | 163,840 | no | 81 | 0.8 (Kaggle) |
| V12 replica (`phase_h/manh_v12`, still running) | 8cf0c0664b | 64 / 64 | 163,840 | vision loaded, text | 64 | 3.07 at the 04:56 copy |
| `phase_h/runs/r1_rerun_bec71fb1` | 8cf0c0664b | 64 / 64 | 163,840 | no | 64 | 2.0 |
| pod 3 run 3 | f506afa210 | 64 / 64 | 131,072 | yes | 64 | 3.7 |
| pod 3 run 2, `nb_p2_rerun` | f506afa210 | 80 / 80 | 163,840 | yes | 80 | 3.0 |
| `phase_h/runs/ext_nodcp_w120` | 5e1bccd07c | 120 / 120 | 163,840 | yes | 120 | 0.69 |
| `phase_h/runs/prod_nodcp_w80` | (5e1b-era kit) | 80 | 163,840 | yes | 80 | 0.43 |

Allocator-cap retries are relevant to H2 and H6, because the expandable-segment unmaps happen there:
- B32-M exact replica: 66-73 retries per rank by 10 h, the first at ~2.3 h. No IMA.
- Every no-DCP run (V12 replica, pod 3, w120): **0 retries.** So the retry path has never run together with the host
  gather, AR overlap and Engram side streams. The V12 replica extrapolated reaching the cap at ~5-7 h (acceleration map
  E1).

**How much the clean record says (est.):**
- With one event, his rate is uncertain by ~200x (Poisson 95% interval from one count).
- If the hazard were config-independent at his point rate, about 1 per 10M generated tokens (3 h at ~900 tok/s, est.):
  - our ~30M no-DCP tokens would expect ~3 events, P(0) ≈ 5%;
  - adding the ~66M DCP4 tokens, if it applied there too, gives P(0) ≈ 1e-4.
- So **a common, config-independent cause is unlikely.** A cause 10x rarer than his point estimate is fully compatible
  with our record.

## 1. Kernels and paths whose shapes depend on rows, tokens or context

"Ours" is the envelope our runs reached. "His" is what his run could reach and ours did not.

| path (v11 unless noted) | depends on | ours | his | bounds / size finding | IMA by shape? |
|---|---|---|---|---|---|
| **CUDA graphs** (`dsv41_vllm.py:115-116`; FULL → FULL_DECODE_ONLY, acceleration map F1) | decode rows padded to a capture size | sizes ≤ 64 (no DCP), ≤ 80 replayed (DCP4); 72-120 only in the 0.69 h w120 run | 72-120 replayed for 3 h | padded rows read zero-initialised positions and stale but valid block ids (`v1/worker/gpu/block_table.py:84-87, 312-327`; `input_batch.py:30`). No decode above the cap: the cap equals max-num-seqs. Decoder-replay graphs need piecewise graphs, so they are off here (`nvidia/model_state.py:259-275`) | no |
| **Mixed / prefill steps, eager** | ≤ 2048 tokens, any mix | all | all | every step with a prefill chunk is eager in every config | no |
| **Engram head gather in graphs** | 3 KiB/token/rank vs the 256 KiB host-gather limit | host gather in all ≤ 80-row graphs | **NCCL inside the 88-120-row graphs** (host gather covers ≤ 85 tokens) | a fallback chosen by size; NCCL is captured in the graph | no, but a config-only path (H3) |
| **Humming MoE**: `load_kernels` pre-run of profile ranges (b9ad670a67, `model_runner.py:1087`); tuning table (`fused_humming_moe.py:172-193, 699-800`) | m = tokens × 6 | decode m ≤ 480; mixed m ≤ 12,288 | decode m ≤ 720 | the tuning table starts at m > 1840, so decode is untouched. Every m ≤ 12,288 occurs in mixed steps in all configs. Risk: if Humming read a gap (w13 (3680, 3888], w2 > 8192) differently from vLLM's `valid_shape_m`, the padding block and the kernel block would disagree, and `expert_ids` would be read out of range. Humming's source is not local, so unverified. It would fire within minutes, and the m ranges are identical across configs | unlikely |
| **Decode indexer logits** (`triton_mqa_logits.py:159-222`) | [rows, max_model_len // ratio] | 64-96 × 81,920 / 163,840 | 120 × 64,000 / 128,000 | int64 rows and `page.to(int64) * stride_page` (:221); seq_len clamped (:193); masked block-table loads (:217) and stores; splits are only a grid dimension. Same in 5e1b (:190, :207, :340). The public int32 page overflow (yhfgyyf #51) is not present | no |
| **Prefill indexer logits and K gather** (`:28-60`; `indexer.py:1534-1585`; `csrc/cache_kernels.cu:776-830`) | chunk rows × context; workspace max_model_len × 8 × factor / ratio (factor 2, B5) | 163,840 | 128,000 | int64 offsets; the chunk splitter keeps Σseq within the workspace and M·N within 512 MiB/4; `token_idx < num_tokens` checked; slot −1 skipped; worst product 3.4e8 < 2^31 and computed in int64. v11 adds `del logits` between chunks (#63047); 5e1b lacks it (an OOM class, not an IMA) | no |
| **Decode top-k** (persistent kernel on SM89: cooperative needs SM90+, `indexer_topk.py:249-275`; `csrc/persistent_topk.cuh:670-679`) | rows; width | 8 rows per group | 15 rows per group | L4's 99 KB shared memory forces the persistent kernel for every row count; seq_len clamped; workspace size checked (`topk.cu:177`). NaN fix #63118 and race fixes #62936 / #63475 are in both wheels | no (hang risk only: spin barrier vs side-stream kernels) |
| **Prefill top-k `top_k_per_row_prefill`** (`sparse_attn_indexer.py:1005`; `csrc/sampler.cu:158-310, 405-420`) | rows, width | all | all | **no NaN handling** in the radix histogram. Public #49896: NaN logits make it emit uninitialised indices (open, in neither wheel) | **only with NaN input → H1** |
| **Candidate blocks / sparse consumer B1** (`candidate_blocks.py`, `sparse_mqa_logits.py:116`) | 2048 blocks × 8; width | 163,840 | 16,000 blocks at 128,000 | masked; candidates clamped to `0 ≤ cand < cdiv(ke-ks, 8)`; 64000 % 16 = 0, so the Triton specialisation is the same | no |
| **Decode row shard B2** (`indexer.py:1207-1231, 1306-1342`) | rows / 4 | ≤ 16 rows per rank | 30 | buffers hold `cdiv(max_num_seqs, 4)` rows; falls back on `num_decodes % 4`, < 8 rows, or overflow (:1321); the 61 KB gather fits under 256 KiB | no |
| **Slot gather `_compute_global_topk_indices_and_lens_kernel`** (`common/ops/cache_utils.py:895-955`; called at `flashinfer_sparse.py:861, 939` by the SM89 class) | topk indices → block table | — | — | **only `local_idx >= 0` is checked.** No bound against the block-table width or the request's blocks; `is_valid_token` only zeroes padding tokens' lengths (:972) and does not mask the block-table read. Public #55636 (PRs #55692 / #55744 unmerged): in neither wheel | **the sink for H1 / H2** |
| **Triton sparse MLA** (`nvidia/ops/triton_sparse_mla.py:48-88, 466-470, 692-699`) | rows, splits, indices | all | all | index loads masked by `j < len`; slots int64; `_load_kv_tile` accepts any slot ≥ 0 (no upper bound); split buffers per call (the 128 MiB FlashInfer workspace issue does not apply) | sink |
| **Compressor ring / SWA** (`compressor.py:91-95`, `sparse_swa.py:1119`) | positions | all | all | null block → slot −1 (#58560 and #60767 in both); SWA width = window | no |
| **DCP kernels** (B32-M; `dcp_indexer_topk.py`, `dcp_merge_partials`, `triton_sparse.py:347`) | rows, W | DCP4 31 h | DCP2? | fixed TOPK aranges; ids in fp32 exact below 2^24; int32 `chunk_bytes*chunk` ≤ 5e7 | no |
| **Host all-reduce** (MODE 556; `csrc/host_all_reduce*`; `host_all_reduce.py:206-217`) | bytes: 120 × 5120 × 2 = 1.17 MiB decode; 20 MiB per 2048-token chunk | ≤ 80 rows | 120 rows | ≤ 24 MiB checked in Python and C++; 2 MiB switches to the large instance; 64-bit indexing; a dead peer → 600 s `__trap()`, which reads as a launch failure, not an IMA | no |
| **Host all-gather** (`csrc/host_all_gather.cu:73-144, 240-258`) | ≤ 256 KiB/rank, 16-byte aligned | — | — | checked before launch; parity slots; an eager call on another stream right after a graph replay is not event-ordered (acceleration map A2, est.) | no (H2 candidate) |
| **AR overlap row chunks** (`nvidia/ops/ar_overlap.py:48-118, 165-239`) | prefill rows ≥ 512 | all | all | inputs and outputs stay referenced until the main stream waits (`live` list); row slices asserted | no |
| **mHC token blocks** (71dca8bc24, v11 only) | T 5-32 | all | all | partial blocks clamp loads and guard stores | no |
| **SimpleCPUOffloadConnector** (`v1/simple_kv_offload/worker.py:271-310`, `cuda_mem_ops.py:227-272`, `copy_backend.py:104-130`) | block ids | KV-bound hours with 19-98% host-restored prompts | more streams, more restores, many aborts | uint64 `base + id*stride`; request frees deferred while a step is in flight (`scheduler.py:192-198, 3179-3195`); SWA frees on processed tokens only (`kv_cache_manager.py:587-593`, also in 5e1b). **Loads are launched with no compute-stream fence** (stores wait on `_store_compute_done`). The copy thread has no try/except | no by shape; H2 |
| **OffloadingConnector** (B32-M; 5e1b `kv_offload/cpu/gpu_worker.py:40-65`) | ≥ 28 KiB pages → C++ copy | 31 h DCP4 | — | public #45475 / #53607: launch failures at ~100 concurrent prefill-heavy requests on DSv4, no fix; #52838 (degrade instead of kill) is in v11, not 5e1b | B32-M only |
| **Engram UVA** (`common/engram.py:660-760`) | tokens; hash rows | all | all + image placeholders | 64-bit row offsets, masked loads; `staged_rows` sized by max_num_batched_tokens and asserted; a missing compact row maps to −1, which is masked to a zero row; the token map covers all 129,280 ids incl. the image placeholder 129,264 | no |
| **embed_tokens UVA** (`offloader/uva.py:74-100`) | tokens | all | all | `F.embedding` on a UVA view; ids from the tokenizer | no |
| **RoPE cos/sin capped at max_model_len** (b1a14abb7a, v10+; `common/rope.py:50-56`) | positions | ≤ 163,840 − 1024 (our solver's room) | up to max_model_len | `_fused_indexer_q_rope_quant_kernel` cos/sin load is unmasked (`sparse_attn_indexer.py:~249`); scheduler caps positions below max_model_len (`scheduler.py:920-927`) | no, unless a scheduler bug (H5) |
| **Vision encoder** (FA2 varlen, int32 `cu_seqlens`; `vision.py:438-468`) | patches per step | small images | ~9.2k patches, 1024 tokens per image, ≤ 2 per step | encoder cache 2048 tokens, profiled at startup with 2 maximum-size images; DP gather pads; a count mismatch raises | no by shape; full-size images never ran after startup (H3) |
| **Batch-sharded sampling, top-K exchange** (`model_runner.py:1744-1766`; `sample/batch_shard.py`) | rows, K = 64 | never fired | every step | plain torch (`topk` / `scatter_` / `gather`), outside the graphs; a bad id would give "device-side assert", not an IMA | no |
| **MRV2 post_update** (`v1/worker/gpu/input_batch.py:570-625`) | total_len, token ids | — | — | no bounds on `all_token_ids[.., total_len + i]` or `output_bin_counts[token_id]`. Manh's own 0.23-era patch added them (`experiments/E064_.../kaggle_smoke_v1/patches/vllm_c6_arc_stability.patch`). Positions are capped by the scheduler, and penalties are unused | no |

**Images, specifically:**
- `DeepseekV41Config.max_image_tokens` resolves to 0 in `sparse_swa.py:495`, so no image-specific attention width
  exists.
- Images change token content and encoder work, not attention shapes.
- **For V12:** loading the vision recipe only costs KV memory (encoder weights and its profile). No encoder kernel runs
  for text prompts.

## 2. Public issues (2025-2026) and whether the fix is in the fork

From `gh api search/issues` over vllm, triton, DeepGEMM, FlashMLA, flashinfer and yhfgyyf/vllm-deepseek-v4-sm89
(~30 queries). Presence was checked with `git log --grep`, merge-base checks and code reads in both wheels. 5e1b's
upstream base is about `08a236fa43` (2026-10-28), est.

**No public report matches SM89 + TP4 + Triton sparse MLA + no spec decode + a fault after hours.**

| issue | trigger | fix | v11 / 5e1b | our path? |
|---|---|---|---|---|
| [yhfgyyf#51](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/51), [#46](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/46), [#69](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/69) | SM89 paged MQA logits: `block_idx*stride` in int32 past ~2k blocks | merged there | not vulnerable: Manh's kernel casts to int64 (v11 :221 / :376; 5e1b :207 / :340) | yes |
| [yhfgyyf#138](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/138) (E053's "fork #138") | B32's vision11 stack, FULL prefill graph warmup | open | n/a: other fork, startup only | no |
| [vllm#55636](https://github.com/vllm-project/vllm/issues/55636) / [#55692](https://github.com/vllm-project/vllm/pull/55692), [#55744](https://github.com/vllm-project/vllm/pull/55744) | the unbounded slot gather (section 1) | **unmerged** | **in neither** | **yes (the sink)** |
| [vllm#49896](https://github.com/vllm-project/vllm/issues/49896) / [#49897](https://github.com/vllm-project/vllm/pull/49897), [#50201](https://github.com/vllm-project/vllm/pull/50201) | NaN logits → `top_k_per_row_prefill` emits uninitialised indices → IMA in the gather (SM12x, prompts > ~2K) | open | **in neither** | **yes (H1)** |
| [vllm#60613](https://github.com/vllm-project/vllm/pull/60613), [#55299](https://github.com/vllm-project/vllm/pull/55299) | −1 sentinel in the prefill index workspace → NaN → gather IMA | merged | in both | yes |
| [vllm#63118](https://github.com/vllm-project/vllm/pull/63118), [#62936](https://github.com/vllm-project/vllm/pull/62936), [#63475](https://github.com/vllm-project/vllm/pull/63475) | persistent top-k: NaN IMA, candidate overflow, CTA race (> 32 rows) | merged | in both | yes |
| [vllm#57156](https://github.com/vllm-project/vllm/issues/57156); [#58560](https://github.com/vllm-project/vllm/pull/58560), [#60767](https://github.com/vllm-project/vllm/pull/60767) | null-block poisoning by capture dummies → NaN; compressor ring; null SWA entries | #57156 open; the two PRs merged | PRs in both | partly |
| [vllm#59768](https://github.com/vllm-project/vllm/issues/59768), [#47282](https://github.com/vllm-project/vllm/issues/47282); [#47653](https://github.com/vllm-project/vllm/pull/47653) (closed), [#60078](https://github.com/vllm-project/vllm/pull/60078) (open) | **SimpleCPUOffload load race:** Xid 13 IMA every 45-80 min at KV 0.92-0.98 with 74-141k-token restores in flight (sm120, MTP, MRV2) | unfixed upstream | in neither as a load fence. The fork defers request frees and frees SWA on processed tokens, which closes the two races I can see | **V12 and his run** (H2); not B32-M |
| [vllm#55085](https://github.com/vllm-project/vllm/pull/55085) | never-written tail of an async-loaded block keeps old bytes (possible NaN) | open | in neither | possible NaN source (H1) |
| [vllm#45475](https://github.com/vllm-project/vllm/issues/45475), [#53607](https://github.com/vllm-project/vllm/issues/53607); [#52838](https://github.com/vllm-project/vllm/pull/52838) | DSv4 + OffloadingConnector launch failure at ~100 concurrent prefill-heavy requests | no fix; #52838 merged 10-29 | #52838 in v11, not 5e1b | **B32-M** |
| [vllm#57030](https://github.com/vllm-project/vllm/issues/57030) / [#60736](https://github.com/vllm-project/vllm/pull/60736) | MRV2 int32 row offsets past 2^31 elements | merged | in both | our sizes ~2e7 |
| [vllm#56389](https://github.com/vllm-project/vllm/issues/56389) | dsv4_topk router IMA at > 256 seqs with DSpark padded rows | merged | n/a | no spec decode |
| [vllm#58910](https://github.com/vllm-project/vllm/issues/58910), [flashinfer#5608](https://github.com/flashinfer-ai/flashinfer/issues/5608) | SM120 FlashInfer sparse MLA prefill > 35K | open | n/a | no (SM89 uses Triton) |
| [vllm#61239](https://github.com/vllm-project/vllm/issues/61239), [flashinfer#6533](https://github.com/flashinfer-ai/flashinfer/issues/6533) | FlashInfer `plan()` pinned-workspace race | open | n/a | not the DSv4.1 builder; the vision encoder backend is unchecked |
| [vllm#46253](https://github.com/vllm-project/vllm/issues/46253), [#57785](https://github.com/vllm-project/vllm/pull/57785), [#64266](https://github.com/vllm-project/vllm/pull/64266) | IMAs during breakable-graph capture (host collectives, Engram UVA side stream) | open / closed | capture-time | no: our servers start fine |

## 3. Ranked hypotheses, and what each means for V12 and B32-M

Weights are est., from one event and no logs. Each line names the one log fact that would confirm or kill it (see
section 4).

**H1. NaN or garbage logits in a prefill indexer → `top_k_per_row_prefill` → the unbounded sinks (30%).**
- The decode top-k is NaN-safe since #63118; the prefill top-k is not (#49896).
- The gather and `_load_kv_tile` then follow whatever index they get.
- NaN sources his run had more of:
  - full-size images (1024-token embeddings, ~9.2k-patch encoder runs, never exercised by us after startup);
  - contexts running to max_model_len;
  - heavy CPU-offload restores (#55085 stale tails), with more streams and aborts.
- This fits a rare event after hours.
- **Confirms:** the failing step had prefill tokens (scheduler dump), the traceback is in or after the indexer or
  sparse MLA, or a core dump names `_compute_global_topk_indices_and_lens_kernel` / `_load_kv_tile` /
  `topKPerRowPrefill`.
- **V12:** same kernels; text-only NaN sources (rare; nothing in our logs). **B32-M:** same sink in its DCP branch.

**H2. A race or use-after-free corrupts an index buffer (20%).**
- Candidates:
  - the offload load stream (no compute fence; #59768 is the closest public match to "dies after hours at KV ≥ 0.9
    with big restores");
  - host gather vs graph replay ordering (acceleration map A2, est.);
  - the Engram side stream;
  - the shared `torch.empty` top-k buffer, cleared to −1 for the step's rows at `sparse_attn_indexer.py:805` unless
    `skip_topk_buffer_clear` is set.
- Timing differs on Kaggle (50 W, Gen3) from pods.
- **Confirms:** KV ≥ 0.9 with host restores in flight at the death; a fault in a decode graph replay; an Xid 31 address
  in the KV pool.
- **V12:** fully exposed (same streams and connector), at 64 rows and fewer restores per hour. **B32-M:** Engram side
  stream only; NCCL; upstream connector.

**H3. A path only his config ran (20%).**
- graph sizes 72-120 for hours;
- the Engram head gather over NCCL inside the 88-120 graphs;
- the top-K sampling exchange (torch ops, outside the graphs: unlikely to IMA);
- DCP2, if used (never tested beyond startup);
- the FA2 vision encoder on full-size images.
- **Confirms:** max-num-seqs 120 in his `non-default args`, a DCP2 flag, a fault during an image prefill.
- **Neither submission runs these.**

**H4. A different wheel (10%).** If his run used commits after `8cf0c0664b` or a local patch, our audit does not
cover them. **Confirms:** the `Initializing a V1 LLM engine (v...)` line. Neither submission is affected if pinned.

**H5. Positions at or above max_model_len (5%).**
- v10+ caps the RoPE cache at max_model_len, and one cos/sin load is unmasked.
- The scheduler caps positions, so this needs a scheduler bug.
- His harness runs requests to the limit; ours keeps 1024 tokens of room.
- **Confirms:** a request at the length cap in the dump.
- **V12:** low. **B32-M:** no RoPE cap in 5e1b.

**H6. Other (15%).**
- Humming block-size mismatch at gaps in the tuning table: would fire early, and the m ranges are config-independent.
- Host-AR L2 prefetch of memory freed later: nothing found that frees it.
- Hardware: ECC or a GPU off the bus give other error strings.

## 4. What to ask Manh for

1. **Every rank's traceback**, with timestamps:
   - the first `WorkerProc hit an exception` / `torch.AcceleratorError` block of each `Worker_TPn`, plus the
     EngineCore `EngineCore encountered a fatal error` block;
   - which rank raised first, and the Python frame where the error surfaced (`.tolist()`, `.item()`, a `copy_`, a graph
     `replay()`, a collective, the offload copy thread).
   - An IMA is asynchronous, so that frame only bounds where it happened.
2. **server.log** for the 10 min before the death and everything after, unfiltered:
   - `Dumping input data for V1 LLM engine`, `Dumping scheduler output for model execution`, `Dumping scheduler stats`.
     `log_error_detail` writes these on a step exception (`v1/engine/core.py:579-592`,
     `logging_utils/dump_input.py:56-85`). They give the failing step's decode rows, prefill chunks, request lengths,
     encoder inputs and KV usage;
   - the last `Running / Waiting / KV / prefix hit` lines; the per-rank `GPU memory:` lines (retries);
   - any `Triton kernel JIT compilation during inference`, `Exception in thread`, `never reduced`, `never posted`, or
     `KV fetch` line;
   - the startup block: `non-default args`, the `Initializing a V1 LLM engine (v...)` version, the capture sizes,
     `GPU KV cache size`, `Capped the PyTorch allocator`, `Host all-reduce enabled`, and the encoder profile lines.
3. **Exact launch:**
   - `launch.json` (command + environment overrides) and the full `VLLM_*`, `CUDA_*`, `NCCL_*`,
     `PYTORCH_CUDA_ALLOC_CONF` env;
   - the vllm-wheels dataset version and any local patch.
4. **Workload:**
   - the recipe name; `--concurrency` and `--server-max-num-seqs` (was max-num-seqs and the graph cap 120?);
   - `--context` (128,000 or 131,072?); DCP (none, or DCP2 with `--gpu-ids 0,2,1,3`?);
   - sampling (top_k 64?); images per prompt and pixel size; the gzip loop guard on?;
   - util, offload GiB/rank, watermark.
5. **State at the death:**
   - running / waiting, KV usage, host-restored tok/s, the longest live context;
   - whether the failing step was pure decode (a graph replay) or a mixed step, and whether an image prefill was in it;
   - his harness's requests in flight in the last 60 s, with prompt lengths and image counts.
6. **Hardware and driver:**
   - Kaggle or a pod; the power limit;
   - `dmesg` Xid lines if readable: Xid 31 gives the faulting address and READ/WRITE (a host-UVA address points at
     Engram, embed or host-AR buffers; a device address points at KV or workspaces). Xid 13 is a graphics exception.
7. **History:** has it happened before on this stack, at what concurrency, and was a rerun clean?

## 5. Pod repro plan

This needs a 4x L4 pod rented by the user; nothing runs on Kaggle. The crash took ~3 h at 120 streams, so an arm is
hours. Spend the first pod-hours on the closest copy of his run, with instrumentation that costs nothing until the
fault. Bisect only after it reproduces.

**Always-on instrumentation:**
- **CUDA core dump on the fault** ([vLLM blog, 2025-08-11](https://vllm.ai/blog/2025-08-11-cuda-debugging)). Put these
  in the server env:
  `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1`
  `CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'`
  `CUDA_COREDUMP_FILE=/root/dumps/cuda_coredump_%h.%p.%t`.
  - `cuda-gdb` → `target cudacore <file>` names the faulting kernel, grid, block and PC, also inside a graph replay.
    Triton kernels keep line info unless `TRITON_DISABLE_LINE_INFO=1`.
  - It adds a thread-exit check (some overhead), so time it on the A/A pair. Never use `skip_abort`.
- `--enable-logging-iteration-details` (`config/observability.py:99`; `v1/engine/core.py:595-640`): per-step context
  and generation request/token counts and encoder inputs, so the steps before the fault are on record.
- `dmesg -w | grep -i xid` if the pod allows it.
- `VLLM_GPU_MEMORY_LOG_INTERVAL=60`; `evals/serve_replay_bench.py --server-log server.log` for memory headroom.
- **Not** `CUDA_LAUNCH_BLOCKING=1` in long arms. It serialises every launch (several-fold slower) and does nothing
  inside a graph replay. Keep it for the short eager repro below.

**Arm R0, his run as closely as we can:**
- His own harness from `audit_v12/src` (vision-repl recipe), not the bench. The bench cannot send his 1024-token
  images through `/inference/v1/generate`, nor abort streams mid-turn.
- `--concurrency 120` (so max-num-seqs and graphs are 120), `--context 128000` (or his value), images on, top_k 64,
  his env from cell 2m, wheel v11 (or his).
- 120 eval tasks x2, for 4 h.
- If it dies, the core dump plus the iteration log name the kernel and the step.

**Arm R1, V12's own workload under stress** (in parallel on a second pod, or after R0):
- Manh's V12 cell 2m server through `phase_h/scripts/run_prod_cell.py`.
- Replay, tokens mode, the recorded b32m_x2t set:
  ```bash
  python serve_replay_bench.py --base-url http://127.0.0.1:18084 --traces dsv41_traces.tar.gz \
      --prep-cache b32m_x2t.tokens.json.gz --tools tools.json --concurrency 64 --order heavy-first \
      --temperature 1.0 --duration 14400 --warmup-s 300 --server-log server.log --out runs/v12_stress
  ```
- `heavy-first` reaches the KV-bound, restore-heavy phase within an hour. `--think-time 0` keeps more pressure than a
  real run.
- Run a second R1 at `--concurrency 120` with the same server (max-num-seqs 64), so the extra load queues instead of
  changing graph shapes. That separates "more restores and churn" (H2) from "bigger graphs" (H3).
- Chat-mode image traces (`pod3_runs/run3_data/dsv41_logs/traces`, 279 files) are an option for images. They are
  small images, though, not his.

**If R0 reproduces, bisect one variable at a time** (≤ 4 h each, stop at the first death):
- 120 → 64 rows;
- images off;
- top_k off;
- context 163,840;
- `--kv-offload-gib-per-rank 0`;
- `VLLM_TP_HOST_GATHER=0` + `VLLM_DSV41_AR_OVERLAP_CHUNKS=0`;
- `VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS=0`.

**Short repro (after the iteration log gives the composition of the failing steps):**
- Replay the same seed and order for ≤ 20 min with `CUDA_LAUNCH_BLOCKING=1`, once with graphs and once with
  `--enforce-eager`.
- If it does not reproduce under eager, the fault needs graph state (padded rows, persistent buffers).

**Kernel checks on one GPU, cheap, in parallel:**
- `compute-sanitizer --tool memcheck` on the sinks with fuzzed inputs:
  - `_compute_global_topk_indices_and_lens_kernel` and `_load_kv_tile` with indices ≥ max_model_len and NaN-derived
    indices;
  - `top_k_per_row_prefill` on logits rows containing NaN, at 2048 × {64,000, 81,920, 128,000, 163,840}.
- This confirms the H1 chain without a 4-h arm.

**Bench gaps:** `serve_replay_bench.py` has no mid-turn abort option. A `--abort-prob p` (close a stream after a random
number of tokens) would emulate the gzip loop guard.

## 6. Cheap hardening

None of this is built or tested. Each change turns an IMA into one bad answer and keeps the engine alive.
- **Bound the sinks:**
  - in `_compute_global_topk_indices_and_lens_kernel`, mask `local_idx < block_table_stride * block_size` and
    `is_valid_token` on the block-table load;
  - in `_load_kv_tile`, mask `slot < num_slots`.
- Both are one `tl.where` / mask each, in Python/Triton files, so they can ship as an overlay. Cost est. none (the loads
  are already masked).
- **Map NaN logits to −inf before the prefill top-k.** Doing it in the Triton logits kernel's store is the cheap place.
  A separate `nan_to_num` over 512 MiB chunks costs est. ~3-4 ms per indexer layer per chunk.
- **Recovery for V12:** if V12 is resubmitted, apply the failure-paths safety edits (#1 `CONN_ERRORS`/tries, #4 the
  pinned wheel). An IMA follows the same death and restart path as an OOM (`notes/2026-10-30_vllm_failure_paths.md`
  section 0).

## Files

- This note.
- Scratchpad (not synced):
  - `ima/inventory_server_logs.py` (the log scan);
  - `ima_audit/{v11,b5e1}/` (extracted fork files the audits read);
  - `ima/draft_*.md` (working drafts).
