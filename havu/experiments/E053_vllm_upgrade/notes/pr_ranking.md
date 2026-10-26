# E053 Q4: upstream vLLM PRs ranked for our workload (DSv4.1 E128, TP4 4xL4, tool use)

Written 2026-10-26 by the PR-ranking fork. CPU only; gh/web read-only.

## Sources and method

- Blog https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep: all numbers are on GB200/GB300/B300, P/D + wide-EP.
  Claims: #59894 "+19.9% at 32K new tokens (10.1% at 4K to 23.1% at 64K)"; #62359 "+2.8% at 32K"; #62240 "+15.6% at
  32K, 15.2% at 64K"; #61725 "460K -> 516K tok/s/GPU, P90 109 -> 153 tok/s/user" (GB300 7P1D, concurrency 4,800);
  #59908 + #62702 + #61724 + #62698 "855K prompt: 38.8 -> 6.0 ms of CPU per admission"; #61339 no number (tool).
  HiSparse and offloading are not discussed.
- Metadata: `gh api repos/vllm-project/vllm/pulls/N` (+ `/files`), saved in `e053/prs/meta/`. Diffs in `e053/prs/diffs/`.
- Apply test: `e053/forkcopy/` = fork wheel source (`src_vllm/vllm`, vision11) + our 12-file overlay laid over it,
  committed as a git baseline. `e053/prs/check.sh` rewrites `vllm/models/deepseek_v41/` -> `deepseek_v4_1/` (and the
  dotted module path), drops tests/AMD files, runs `git apply --check` and per-file `patch -p1 --dry-run -F3`.
  Sequential chain test in `e053/forkcopy2/` (patches actually applied with fuzz, rejects in `e053/prs/rej/`).
  "Applies" means with offset/fuzz <= 3; every fuzzed hunk still needs review.
- Fork snapshot check (grep of fork source): absent in the fork: `swa_bounded_replay`/`prefix_replay_tokens` (#56227),
  `decoder_replay*` (#58132/#59532/#59894), `thp_packing` (#56926), Engram `_engram_lookup_thresholds` (#59327),
  `null_head_len` (#61724), `max_num_active_seqs` (#56758), `deep_select` (#56464), `max_tp_shards` (#57652),
  `max_async_kv_loads_per_step` (#61725). The fork has its own approximate CED replay prototype
  (`models/deepseek_v4_1/nvidia/ced.py`, `--hf-overrides '{"ced_prefill":true}'`): decoder-layer SWA (layers >= 20)
  leaves the prefix cache, encoder SWA stays cached (`attention.py:454-461`), per-sequence tail buffers, MBT >= 128 x
  max_num_seqs; E042 measured ~1.9x prefill but rejected it (KV cost, +5-8% est., approximate).

## Ranked table (throughput for OUR workload; effort = port + CPU tests, excl. GPU validation)

| rank | PR | state | what it does | expected gain for us | port onto fork (hunks apply) | overlay overlap | deps | effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **#56227** encoder-side SWA bounded replay | merged 09-18, +1067/-38, 22 non-test files | SWA caches leave prefix caching (`SlidingWindowMLASpec(bounded_replay=True)`); a hit at H recomputes [H-128,H) | Paused-stream prefix costs 1 MLA block per 128 tokens in the LRU instead of 1+28 (interval 128) / ~1+7 (512); non-cached SWA blocks are LIFO-reused (fork `block_pool.py:723-747`), so SWA traffic stops evicting paused prefixes. Any surviving MLA prefix is a valid hit: the "hit falls to the 640-token system prefix" re-prefills (S3, ~2-4%) and the all-or-nothing collapse under KV pressure become partial tail re-prefills. Cost ~128 replay tokens per turn (~1-2% GPU), outputs not bit-identical by design (gsm8k parity upstream, TP4 GB200). The enabler for 44-56 streams (the 400 tok/s target), and it makes E051 offload need only the MLA+indexer group | 73/83 (sequential). Manual: `common/ops/cache_utils.py` 5/6 (upstream-only FlashInfer mixed-index Triton kernel; the fork's native path takes SWA slots from `sparse_swa.py`, whose 24/24 hunks apply, so these 5 can likely be dropped), `nvidia/model_state.py` 2/4, `nvidia/flashinfer_sparse.py` 1/3, `kv_cache_interface.py` 1/7, `model_states/mamba_hybrid.py` 1/1 | none | standalone on main; needs MRV2 (fork uses it), no PCP | 1.5-2 d |
| 2 | **#58132** decoder-side replay (+ **#59532**, **#59894**, fixes #61643/#62652) | merged 09-29 / 10-06 / 10-09 / 10-15 / 10-21 | layers 21-39 (and 20 with #59894) run only on each prefill's last 128 tokens in eager steps | Same idea as fork CED (E042 P12b: 826 -> 1,554 tok/s prefill on L4) without CED's per-seq tail buffer or MBT rule. For our mix (first prompts ~58% of uncached tokens, typical turn ~250 uncached) E042 estimated +5-8% e2e; also shortens the ~1.5 s 2048-token chunk stalls. #59894 adds ~+2-4% prefill on top (upstream: +3.6-5.5% B200, blog +19.9% at 32K). Our graphs are FULL_DECODE_ONLY (mixed steps eager), so #58132 alone trims; #59532 matters only for PIECEWISE but #59894 is written on top of it | #58132 13/17, #59532 16/18, #59894 16/16, #61643 1/1, #62652 1/1 (after #56227). Manual: `nvidia/model.py` forward loop split at the cut layer (2/6) and `model_state.py` (2/7, 2/8) | **yes: `nvidia/model.py`** (overlay file) | needs #56227; optional #58586 (SM90+ fused finalize, skip) | 1.5-2 d |
| 3 | **#58678** shard Engram `wkv` across TP | merged 09-26, +7/-3 | `ReplicatedLinear` -> `ColumnParallelLinear(gather_output=True)` | KV capacity: wkv is 6144 x 25600 MXFP8 = ~157 MB per Engram layer, 2 layers replicated -> 1/4 at TP4 saves ~0.22 GiB/GPU = **~+9-10% KV tokens** (Kaggle KV ~2.28 GiB/GPU). Decode ~neutral (-0.6 ms weight read vs +~0.6 ms all-gather of T x 25600 bf16); prefill chunk ~+2% (105 MB all-gather per layer per 2048 tokens at 4.9 GB/s) | 2/2 (fuzz) | **yes: `common/engram.py`** | none | 0.5 d |
| 4 | #61039 length-aware decode candidate selection | merged 10-14 | score only valid blocks; DeepSelect top-k | ~850 us/step at 1M max_model_len; at our 163,840 ~0.1-0.2 ms (~0.2%); DeepSelect part needs the vendored DeepSelect build (not in wheel) | 1/4 | no | DeepSelect (#56464) for the top-k part | 0.5 d, low value |
| 5 | #59908 / #62698 / #62702 / #61724 / #60269 admission + free-path CPU | #59908 open, #62698 merged 10-25, #62702 open, #61724 merged 10-15, #60269 merged 10-08 | int32 prompt transport, chunked staged-write kernel, skip null SWA head, faster free queue | We admit ~0.1-0.2 requests/s (36 streams, ~4-5 min per turn) with <=160K prompts under async scheduling: ~5-20 ms saved per admission, **<0.5%** | #59908 19/21, #62698 12/12, #61724 6/9, #60269 7/7; #62702 needs vllm-rs (rust/, not in fork) | no | #62702 stacked on #62698 | 0.5-1 d, skip |
| 6 | #61725 cap async KV loads per step | open | opt-in limit on new async KV loads per step | P/D-scale problem (90-140 loads/step); only relevant with offload on and refill bursts | 7/7 | no | none | 0.25 d, only with E051 |
| 7 | #57652 single-copy multi-group MLA offload layout (+ #57145 skip scratch groups) | merged 09-30 / 09-17 | rank-0-only host copy for replicated MLA/SWA groups; skip non-prefix-cacheable groups | Only with offload: 4x less pinned host RAM (matters next to the 94 GiB Engram). #57145 is reimplemented in E051's overlay | #57652 18/19; #57145 0/3 (relies on upstream group-mapping refactor) | no (E051 files) | #57652 builds on closed #50667 | 1 d (E051 phase 2) |
| 8 | #56926 / #59327 / #62359 Engram host-table paging | merged 09-25 / 10-01, #62359 open | THP for private offloaded tables; sorted lookups (Blackwell thresholds only); FABRIC cuMem 2 MB pages for DP-shared tables | E042: Engram table size does not change L4 speed; #62359 is DP-shared only (we are DP1); #59327 thresholds are off on non-Blackwell. <=1-3% at best, unmeasured | #56926 9/18, #59327 7/17, #62359 7/15 | **yes: `common/engram.py`** | #59327 moves code nvidia/engram.py -> common | 1-2 d, skip |
| 9 | #61339 `vllm bench forward` | merged 10-14 | exact-shape forward timing | no TPS; useful to A/B #56227/#58132/#58678 on the 4090 box | new files + 1-line hook; depends on MRV2 runner internals | no | - | 0.5-1 d, optional |
| - | correctness, cheap: #58560 (compressor ring out of the null block: NaN via 0 x NaN under FULL graphs on FlashInfer DSv4 decode, seen on SM120), #60767 (null SWA block-table entries below the window) | merged 10-05 / 10-19 | | no TPS; insurance (fork issue #138: FULL-graph illegal access on SM89 vision11) | #58560 clean 1/1 | no | - | 0.25 d |
| - | #56758 `--max-num-active-seqs` | merged 09-17 | server-side admission cap | no TPS; could back up the client KV gate | clean 7/7 | no | - | 0.25 d |
| - | #57613 retain one alignment block below SWA replay boundaries | open | sparse-retention same-turn divergence hit 0 -> hit | inert at our interval 128 (dense); moot with #56227 | clean 1/1 | no | - | skip |

### Irrelevant for us (hardware / topology)

- #62240 DeepSelect BF16 prefill ranking: SM100 DeepGEMM logits path, opt-in `deep_select`; the fork's SM89 prefill
  indexer is the Triton MXFP4 kernel. #56464 DeepSelect: compiled library via FetchContent, not in the wheel.
- #61725 (as motivated), #59197, #58504 (closed, superseded by #59197): P/D connector paths.
- #62359 (DP-shared Engram tables, FABRIC handles), #61767 (DP all-reduce), #62009/#61395 (EPLB), #61121 (NIXL push):
  DP / wide-EP / P/D.
- #61868/#61869 (GB300/B300 specialized decode), #56935/#56893 (FlashMLA mega-attn, NVFP4/MXFP8 KV), #56266/#56962
  (DeepGEMM Mega-Gate/Mega-mHC), #58586/#57643 (fused all-reduce; needs P2P/NVLink-class all-reduce, Kaggle uses
  PYNCCL over SHM), #58872/#58634 (SM90/SM100): need SM90/SM100 kernels.
- #53781 HiSparse [3/N]: DeepSeek-V4 support is the open #53592 [8/N]; SM90 FlashInfer backend open (#61277);
  rejects `cudagraph_mode=FULL` (#59688). Not usable on V4.1/SM89 now.

### Not upstream but worth a look

- Fork PR yhfgyyf/vllm-deepseek-v4-sm89#156 (open, 10-22): grid-stride context skip in `dsv41_indexer.py` (our overlay
  file), bit-identical, "all targets": under CUDA graphs the decode logits launch spans max_model_len // ratio columns
  regardless of context. Our indexer is ~21-23 ms of a decode step (E042 P3 / pod B profile); if the decode path
  launches full width (check: decode uses candidates in the overlay kernel), skipping tiles past each row's end could
  save several ms per step. Measure with the profiler first.

## Notes on the top three

- **#56227 port details.** The fork's V4.1 native attention takes explicit SWA slot lists from
  `DeepseekSparseSWAMetadata` (`v1/attention/backends/mla/sparse_swa.py`, `decode_swa_indices` /
  `prefill_swa_indices`, consumed in `nvidia/flashinfer_sparse.py:392-426`), so clamping the window at `replay_start`
  is a Triton/Python change; no FlashInfer rebuild. The FlashMLA combine-kernel change is SM90+ and not used.
  Remaining manual hunks are context drift (imports, docstrings) plus `_pad_replayed_slots_kernel` and
  `ModelSpecificAttnMetadata` wiring in `model_state.py`.
- **Validation (both replay PRs are approximate by design):** greedy parity is not expected; check with a
  replay-on/off A/B on the 120 eval tasks (pass@1 within the noise band) and the hit-rate/re-prefill counters on a
  multi-turn replay at 44-56 streams; `--no-swa-bounded-replay` keeps a kill switch.
- **#58678:** verify E128's `engram.wkv` is MXFP8 [32,32] blocks (6400 columns per rank divides 32) and that
  Humming's linear backend accepts the ColumnParallel shard; expected log: weights per GPU -0.2x GiB, KV +~9-10%.
