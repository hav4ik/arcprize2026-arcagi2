# E053: should we upgrade the vLLM we serve DeepSeek-V4.1 E128 with on Kaggle 4xL4?

Status: **done (CPU-only research, 2026-10-26).**

Recommendation:
- **Do not change the base stack.** There is no newer fork release to move to, and upstream cannot run V4.1 on SM89.
- **Stay on the vision11 fork and cherry-pick 3 changes into our overlay:**
  - fork PR #156: indexer grid-stride skip, exact;
  - upstream #58678: Engram `wkv` split across TP ranks;
  - upstream #56227: SWA bounded replay.
- **Validation:** each change needs its own validation, and each ships as Python/Triton files through the notebook's cell 0b.

Scope: CPU only. I used read-only `gh api` and web fetches, a shallow clone of upstream main (`af6665b5dd20`, 2026-10-26 19:34 UTC), the
fork's vision11 wheel source, and our overlay. No box, no Kaggle, no paid API.

Path conventions:
- `U:` = upstream main at that commit.
- `F:` = fork wheel `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11`, source commit 86d35745.
- `O:` = our overlay `kaggle/dsv41_tools/vllm_overlay_src/vllm/`.

Unless noted, line numbers refer to those trees.

## 0. Priority question: "KV cache offload for DeepSeek is supported in newer vLLM, right?"

**Partly. It is supported on upstream main, it is not supported on any stack that runs on our hardware, and nobody has tested it on V4.1.**

### (a) KV spec types and the OffloadingConnector

| cache | upstream main V4.1 spec | prefix-cacheable | our fork (vision11) |
|---|---|---|---|
| compressed MLA (kv sources 2/8/14/20) | `MLAAttentionSpec` (a `FullAttentionSpec`), `U:vllm/models/deepseek_v41/attention.py:1206-1233` | yes | `MLAAttentionSpec`, `F:models/deepseek_v4_1/attention.py:880-899` |
| indexer K (MXFP4) | `MLAAttentionSpec`, `U:.../deepseek_v41/attention.py:1274-1300` | yes | same, `F:attention.py:936-946` |
| SWA (every layer, 32-token blocks, window 128) | `SlidingWindowMLASpec(bounded_replay=...)`, `U:vllm/v1/attention/backends/mla/sparse_swa.py:120-130`; `prefix_cacheable = not bounded_replay`, `prefix_replay_tokens = sliding_window`, `U:vllm/v1/kv_cache_interface.py:1027-1037`; `swa_bounded_replay: bool = True` by default, `U:vllm/config/cache.py:367` | **no** (with replay on) | `SlidingWindowMLASpec`, always prefix-cacheable, `F:attention.py:94-123` |
| compressor ring (ratio-2 sources, 8 rows) | `CircularBufferSpec`, `U:.../deepseek_v41/compressor.py:247-255` | no | same, `F:compressor.py:135-166` |

**Upstream main's connector**
- **Only prefix-cacheable groups are offloaded.** `get_offloading_group_ids` returns `prefix_cacheable_group_ids`
  (`U:vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:36-43`; this is #57145, merged 09-17).
  - The `tokens_per_block % tokens_per_hash` assert (`config.py:112-116`) therefore sees only the MLA/indexer group.
  - So does `assert isinstance(kv_cache_spec, FullAttentionSpec)` (`config.py:83`).
  - The 8-row ring is never seen.
- **One host copy for TP4.** #57652 (merged 09-30) stores a single host copy for several replicated MLA groups.
- **Expected outcome.** With Model Runner V2 and bounded replay on, `--kv-offloading-size` should start for V4.1 and offload **only the
  compressed-MLA + indexer cache**.

**Our fork**
- It fails at startup, exactly as E042/E051 found. The divisibility check covers every group (`F:.../offloading/config.py:57-64`).
- `F:.../offloading/scheduler.py:125` asserts `FullAttentionSpec` against the ring.
- The fork's base is upstream `2a4e3cc3` (2026-09-02), which predates #57145, #56227 and #57652.

### (b) Who has tested it, and what happens to SWA state on a host restore

**Tests upstream.**
- The only DeepSeek offload test is `offloading-deepseek-v4-flash` (`U:tests/evals/gsm8k/test_gsm8k_offloading.py:158-172`): **V4, not V4.1**.
  It runs GSM8K on TP4+EP with 16 GiB of CPU, nightly on 4xH100.
- No V4.1 offload test exists. Open bugs on V4 offload:
  - [#54914](https://github.com/vllm-project/vllm/issues/54914): "Block not found" after days of uptime.
  - [#53569](https://github.com/vllm-project/vllm/issues/53569): the fs tier misses on multi-group caches.
- [#63718](https://github.com/vllm-project/vllm/issues/63718), opened 10-26, tracks which layouts offload supports.

**SWA state on a host restore (upstream): replayed, not stored.**
- A hit from a prefix-cache store "holds no window" (`U:vllm/v1/core/sched/scheduler.py:3383-3390`).
- `_mark_prefix_replay` sets `replay_start = H - 128` and recomputes the last 128 tokens of the hit (`scheduler.py:3392-3409`).
- Hits are rounded down to a block, and hits of 128 tokens or fewer are dropped.
- [#56227](https://github.com/vllm-project/vllm/pull/56227) says it plainly: "KV connectors transfer only the MLA and indexer caches". So a restore
  is **approximate by design**, not bit-identical; upstream checked GSM8K parity on GB200.
- Related PRs:
  - #58132 (decoder replay, merged 09-29) and #59894 (also trims layer 20, merged 10-09) cut the replay cost further.
  - [#58504](https://github.com/vllm-project/vllm/pull/58504), exact non-aligned hits, was closed unmerged.

**Fork draft PR [#118](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/118)** (09-20, unmerged)
- It filters offload groups by `prefix_cacheable`, so a non-CED server would probably get past the ring asserts.
- But on the fork the SWA groups are still prefix-cacheable, so they would be stored densely: ~100 KB/token of host memory (E051 DESIGN §5).
- Its replay logic works only with CED.
- It has no full-model or throughput validation.

**Community stacks with SWA-aware offload**
- [zenglanmu/vllm `dsv41-sm80`](https://github.com/zenglanmu/vllm/tree/dsv41-sm80), commit 60d85486 (10-21), "offloading: restore
  DeepSeek-V4.1 SWA replay windows from the host tier".
  - It stores one SWA window per prompt boundary and restores it, so the hit replays nothing. That is the same idea as E051's checkpoint
    policy, and its 709 lines of scheduler tests are worth reading for E051.
  - Untested on SM89; BF16 Triton sparse MLA.
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) (Docker images for sm80/86/89) offloads V4/V4.1 KV through its
  LMCache fork. It "requires a state checkpoint at every chunk boundary for the sliding-window groups" (README: `--chunk-size 1024`,
  `--prefix-cache-retention-interval 1024`).

### (c) HiSparse

**Status of the series**
- Merged 09-12:
  - [#53781](https://github.com/vllm-project/vllm/pull/53781) [3/N], host-resident sparse-MLA decode with a GPU hot buffer;
  - #56061 [4/N];
  - #56629 [5/N].
- **DeepSeek-V4 support is the open draft [#53592](https://github.com/vllm-project/vllm/pull/53592) [8/N].** It reports "a substantial
  high-concurrency regression and an allocator failure under saturation" on B300.

**Why it does not apply to us**
- Main refuses V4 outright: `raise ValueError("HiSparse does not support DeepSeek V4.")` for any
  `MLAAttentionSpec(model_version="deepseek_v4")`, which includes V4.1 (`U:vllm/v1/hisparse/layout.py:99-104`).
- It needs Model Runner V2, a compiled `csrc/hisparse_kernels.cu`, and a sparse-MLA backend for compute capability 9, 10 or 12.
- It rejects `cudagraph_mode=FULL` (#59688).
- **It does not apply to V4.1 or to SM89.**

### Bottom line for us

No upgrade gives working KV offload for V4.1 on L4. Two routes remain:
1. **Finish E051 on the fork.**
   - The ring skip, dense MLA storage and SWA checkpoints are already designed and tested on CPU.
   - Once #56227 is ported (§4), the SWA checkpoint part can be dropped: offload only the MLA+indexer group, with upstream's replay
     semantics.
   - Then port #57652 for one host copy instead of 4.
2. **Port the upstream pieces directly:** #57145 (ring skip), #56227 (bounded replay, scheduler replay hits) and #57652.

## 1. The community fork (yhfgyyf/vllm-deepseek-v4-sm89)

**No release newer than vision11 exists, and the fork has been dormant for 5 weeks.**

**Releases.** The newest tag is `v0.28.1rc1-vision9-sm89-sm120-cu130` (published 09-12).
- Its assets were updated in place on 09-15 to Vision11, built from fork commit
  [86d35745](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/commit/86d35745c8797c9d2524877fb235084a6a5f5a7d).
- Assets: vLLM wheel 247.2 MB, FlashInfer `0.6.18+glm53.dsv41.vision2` 24.0 MB.
- That is what we run (`experiments/E046_multi_test_rollout/scripts/box_install_sm120.sh:8-15`).

**Branches.**
- Last code commit on `main`: 86d35745 (09-15, #116), followed by docs only until 3f3bbaea (09-16).
- Last code on any branch: `feat/ced-kv-offloading` (09-20, the draft PR #118).
- Everything since then is dependabot.

**Upstream base.** The fork last merged upstream at 2a4e3cc3 (2026-09-02 11:09 UTC, #54759).
- Upstream main is now **3,088 commits ahead** of it (`gh api compare`).
- The fork's V4.1 is its own 19,396-line port (commit c897ad5a, 09-12), not upstream's #56214/#56228.

**Maintainer.**
- [#153](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/153) asks him to update upstream. His reply on 10-22: "会在合适的时候更新，最近没机器跑，无法验证" ("will update
  when appropriate; no machine to run or verify on lately").
- [#147](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/147), Flash-1017: "same architecture as V4.1-Flash, runs directly".

**Open community PRs touching our path**
- [#154](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/154), q predecode in the MXFP4 indexer: **already in our overlay**, env-gated by
  `DSV41_INDEXER_Q_PREDECODE=1` (`O:model_executor/kernels/attention/dsa/dsv41_indexer.py:270-273`).
  - It measured +5-15% decode on L4 (E042 P8).
- **[#156](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/156)** (10-22), stacked on #154. It makes two changes:
  1. A grid-stride context skip, for all GPU targets, bit-identical.
  2. FP4 tensor-core scoring, SM120 only.
  - See §4 item 0: this is the cheapest throughput lever we found.

**Sibling stacks (none newer in the sense that matters)**

| repo | base | V4.1 on SM89 | distribution | notes |
|---|---|---|---|---|
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) v0.13.1 (336 stars) | upstream 93911fcf (09-10) + #56214 | yes (sm80/86/89 Docker images) | Docker only | `fp8_ds_mla` KV, Triton sparse MLA, LMCache offload; master idle since 09-26; no Humming, so a full re-port |
| [zenglanmu/vllm `dsv41-sm80`](https://github.com/zenglanmu/vllm/tree/dsv41-sm80) (2 stars) | upstream fce64bc5 (10-20) + #55184 + `dsv41-ampere` | selected for any `major == 8`, untested on 8.9 | source | BF16 Triton sparse MLA (no FP8 MMA), A100-only reports, ~60 tok/s at c=1 |
| [LeeXzhou/vllm-deepseek-pre-hopper](https://github.com/LeeXzhou/vllm-deepseek-pre-hopper) | vLLM 0.26.0 overlay | V4-Flash only, not V4.1 | overlay | whole KV pool in pinned host memory via UVA |

## 2. Upstream vLLM and SM89 (Ada)

**V4.1 support upstream:**
- V4.1-Flash arrived in v0.30.0 (09-22; #56214/#56228/#56208).
- SWA bounded replay arrived in v0.31.0 (10-05; #56227).
- v0.32.0 (10-18) is the latest release.
- The V4.1 tracking issue #56400 lists H200, GB200, GB300 and MI350X.
- No SM89 string appears in `U:vllm/models/deepseek_v4{,1}`.
- Pre-SM90 DSv4 PRs (#55184, #55177, #47629, #38476) are unmerged, and #56120 was closed unmerged.

**Upstream main does not start V4.1 on SM89:**

| component | upstream requirement | SM89 |
|---|---|---|
| attention class | SM12 → FlashInfer SM120; else MegaAttn (SM100); else FlashMLA (`U:vllm/models/deepseek_v41/nvidia/model.py:227-255`) | falls through to FlashMLA |
| FlashMLA sparse | `capability.major in [9, 10]` (`U:.../deepseek_v41/sparse_mla.py:150-151`); built for 9.0a/10.x only (`cmake/external_projects/flashmla.cmake`) | **no** |
| FlashInfer sparse DSV41 | `capability.major in [10, 12]` (`U:.../deepseek_v41/nvidia/flashinfer_sparse.py:136-138`) | **no** |
| sparse indexer | raises without DeepGEMM (`U:vllm/model_executor/layers/sparse_attn_indexer.py:1060-1064`); DeepGEMM is SM90/100/120 (`U:vllm/platforms/cuda.py:691-697`) | **no** |
| O projection | DeepGEMM FP8 einsum (`U:vllm/models/deepseek_v4/nvidia/ops/o_proj.py:28-33`) | no |
| DeepSelect top-k (#56464/#62240) | built for 10.0f only (`cmake/external_projects/deepselect.cmake:33-38`) | no |
| MXFP4 MoE / MXFP8 dense via Humming | native `humming` backends (`U:.../fused_moe/oracle/mxfp4.py:377`, `U:.../kernels/linear/__init__.py:324-331`) | yes |
| Engram CPU offload, 128-expert routing (generic path) | `U:vllm/config/engram.py:37`; `dsv4_topk` fast path only for 256/384 experts | yes |
| mHC | tilelang/torch fallbacks | probably |

**What only the fork provides on SM89:**
- native mixed FP8/FP4-KV sparse attention through its own FlashInfer module `flashinfer.mla.deepseek_v41`
  (`F:models/deepseek_v4_1/nvidia/flashinfer_sparse.py:38-57`; upstream FlashInfer has no such module);
- SM89 query quantization;
- the Triton MXFP4 indexer (`dsv41_indexer.py`, which has **no upstream file**);
- CED prefill;
- DSpark adaptive verification.

**What only upstream provides:**
- SWA bounded replay (default on);
- decoder replay (#58132, #59532, #59894);
- the Engram rewrite (1,033 → 1,814 lines);
- offload fixes since 09-02 (#57145, #57652, #56709, #56810);
- `--max-num-active-seqs` (#56758);
- the waiting-queue rework (#58947);
- the KV-connector + MTP deadlock fix (#57104);
- the Humming refactor.

The rest (MegaMoE, mega attention, DeepSelect, specialized decode) is SM100-only.

## 3. Options

| | (a) stay on vision11 + cherry-pick into the overlay | (b) newer fork release | (c) upstream main / v0.32.0 |
|---|---|---|---|
| available? | yes | **no**: none exists (§1); the sibling stacks are not newer and use other kernels and KV formats | runs nothing on SM89 for V4.1 (§2) |
| TPS features gained | fork #156 indexer skip (exact); #58678 Engram `wkv` split (+9-10% KV); #56227 bounded replay (no 640-token-hit re-prefills, no SWA LRU churn, more streams); optionally #58132+#59894 decoder replay (prefill); E051 offload simplified by #56227 | n/a; a future rebase might bring #56227, but needs the full re-validation of (c) | all of (a)'s upstream items plus the MLA-only offload path; none of the fork's SM89 kernels |
| overlay changes | #156 → `dsv41_indexer.py` (applies to our overlay with 9/10 hunks, the miss is an `import os` we already have: `patches/fork_pr156_indexer_skip_on_overlay.diff`); #58678 → `common/engram.py` (2/2 hunks with fuzz); #56227 → 22 non-test files, 73/83 hunks apply, none in the overlay; #58132 chain → hand merge into our `nvidia/model.py` | unknown | overlay ~2-3 days (Humming hooks redone; `kernels/linear/__init__.py` and `oracle/mxfp4.py` dropped, both native upstream; Engram compact-table **rewrite** 1-2 days; `model.py` differs by 1,105 lines); **plus** porting ~11k lines of fork SM89 code and the custom FlashInfer onto upstream's rewritten V4.1 |
| versions | unchanged: torch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18 fork, humming 0.1.12 | ? | v0.32.0: torch 2.13.0, FlashInfer 0.7.0.post1, humming 0.1.17; main: **torch 2.14.0**, FlashInfer 0.7.2.post1, humming 0.1.18; transformers 5.16.1-5.20; CUDA 13.0 on all |
| Kaggle packaging | Python/Triton files only: append them to `vllm_overlay_rw.tgz` + sha file from cell 0b (the E051 HANDOFF §5 step 6 pattern), so **no new wheel set and no new dataset version** | new wheel set | new wheel set: our own wheel (upstream builds generic `_C` for 8.9, `U:CMakeLists.txt:123-134`, but not FlashMLA/DeepGEMM/DeepSelect; 311 MB at v0.32) + custom FlashInfer → a new version of the `chankhavu/dsv41-sm89-fork-py312-cu130` kernel output (198 wheels, 3.82 GB today), with **mount-error risk** on fresh versions (notes §8) and a CUDA 13 shim re-check (`kaggle/dsv41_tools/build.py:157-170`) |
| validation | per item (§5): bit-identity for #156; parity for #58678; capacity + accuracy A/B and an L4 soak for #56227 | everything | everything: parity vs the current stack on fixed prompts, 120-task accuracy, capacity, a ≥ 4 h L4 soak |
| effort / risk | ~4-5 engineer-days + 2 dev boxes × 3 days + 1 L4 pod-day; risk low (#156, #58678), medium (#56227), medium-high (#58132) | — | 3-6 weeks; very high. **No-go before 11-02** |

## 4. Upstream PRs ranked for our workload

The workload: multi-turn tool use, 36-64 streams, 20-160K contexts, prefix caching, TP4 on 50 W L4s.

"Hunks apply" means `patch` accepted the hunk with fuzz ≤ 3 against the fork source plus our overlay, after renaming `deepseek_v41` to
`deepseek_v4_1`. Fuzzed hunks still need review. Work files: scratchpad `e053/prs/`, `e053/forkcopy2/`.

The blog's numbers (https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep) are all from GB200/GB300/B300 with P/D and wide-EP.

| rank | PR | what it does | expected gain for us | port onto the fork | effort |
|---|---|---|---|---|---|
| 0 (fork, not upstream) | **fork #156**, part 1, grid-stride skip | the logits kernel strides over tiles only up to each row's end | Under FULL graphs the decode indexer width is `max_model_len // compress_ratio` (`F:vllm/v1/attention/backends/mla/indexer.py:1512-1520`). For E128 that is 81,920 columns on index sources 2/8/14 and 163,840 on layer 20, against a mean context of ~40-50K, so ~60-70% of the dense columns are empty tiles. The kernel pays for them: PR #156 measured 287 µs at W=512K with 2K context vs 310 µs at 200K context. The indexer is ~15-23 ms of a 48-stream step (E042 pod B profile). **Est. +5-10% decode, exact**; SM89 not timed by the PR author | 9/10 hunks; the miss is `import os` | 0.5 d + test |
| 1 | **[#56227](https://github.com/vllm-project/vllm/pull/56227)** encoder-side SWA bounded replay (merged 09-18) | takes SWA out of prefix caching; a hit at H recomputes [H-128, H) | **Re-prefills:** removes the "hit falls to the 640-token system prefix" full re-prefills (notes §7: ~0.2% of turns, 2-4% of throughput). **LRU churn:** a paused stream costs 1 cached block per 128 tokens instead of 29 (interval 128) or ~8 (interval 512), so SWA traffic stops evicting paused prefixes (the S3b mechanism). **Capacity:** the most likely way to run 44-48 streams safely (thrash today at KV 0.82-0.89). **Cost:** ~128 replay tokens per turn, ~1-2% of GPU time. **Not bit-identical** | 73/83 hunks. Of the 10 that miss, 5 belong to an upstream-only FlashInfer index kernel; the fork builds SWA slot lists in `sparse_swa.py` (24/24 apply). The fork's FlashInfer kernel already treats `slot < 0` as invalid and bounds-checks rows (`flashinfer/data/include/flashinfer/attention/deepseek_v41_mixed_sparse.cuh:111-134`, in the fork FlashInfer wheel), so padded replay slots should not fault; **verify on GPU**. Needs MRV2 (we use it) | 1.5-2 d + validation |
| 2 | **[#58132](https://github.com/vllm-project/vllm/pull/58132)** decoder replay + #59532 + **[#59894](https://github.com/vllm-project/vllm/pull/59894)** (+ fixes #61643, #62652) | layers 20/21-39 prefill only each request's last 128 tokens | E042's CED (the same idea) gave 1.9x prefill on L4. For our mix (first prompts ~58% of uncached tokens) **est. +5-8% e2e**, plus shorter chunk stalls. Approximate (the exact variant is Maya's 2,668 tokens, next_candidate_fixes I3) | 13/17, 16/18, 16/16, 1/1, 1/1 after #56227; hand merge into **our** `nvidia/model.py` and `model_state.py` | 1.5-2 d, after #56227 |
| 3 | **[#58678](https://github.com/vllm-project/vllm/pull/58678)** Engram `wkv` split across TP ranks (merged 09-26, +7/-3) | `ReplicatedLinear` → `ColumnParallelLinear(gather_output=True)` | Each of the 2 Engram layers' 6144×25600 MXFP8 `wkv` (~157 MB) is replicated on every rank today; splitting it frees ~0.22 GiB/GPU, **+9-10% KV tokens** (Kaggle KV 2.28 GiB/GPU), i.e. +3-4 streams at the same pressure. Costs one T×25600 all-gather per Engram layer: decode ~neutral, prefill chunk ~1-3% | 2/2 hunks with fuzz, in **our** `common/engram.py` | 0.5 d |
| 4 | #61039 length-aware decode candidate selection | scores only valid candidate blocks | ~0.2% at max-model-len 163,840; the top-k half needs DeepSelect, which the wheel lacks | 1/4 | low value |
| 5 | #59908, #62698, #62702, #61724, #60269: admission and free-path CPU cost (blog: 38.8 → 6.0 ms per 855K-token admission) | | We admit ~0.1-0.2 requests/s under async scheduling: **< 0.5%** | 19/21, 12/12, Rust frontend needed, 6/9, 7/7 | skip |
| 6 | #61725: cap on async KV loads per step | | P/D scale (90-140 loads per step); only with offload refill bursts | 7/7 | only with E051 |
| 7 | #57145 (skip scratch groups), #57652 (one host copy for multi-group MLA) | | Offload only: E051 reimplements #57145; #57652 gives 4x less pinned host RAM | 0/3 (E051 has its own), 18/19 | E051 phase 2 |
| 8 | #56926, #59327, #62359 Engram host-table paging | THP / sorted lookups / 2 MB FABRIC pages for DP-shared tables | ≤ 1-3%, unmeasured; E042 found Engram table size does not change L4 speed; #62359 is DP-only | 9/18, 7/17, 7/15 (in our `engram.py`) | skip |
| 9 | #61339 `vllm bench forward` | exact-shape forward timing tool | no TPS; handy for the A/Bs on the 4090 | new files | optional |
| — | cheap correctness fixes | #58560: compressor ring kept out of the null block (NaN under FULL graphs, seen on SM120); #60767: null SWA block-table entries | insurance (fork issue #138: FULL-graph illegal access on SM89) | #58560 1/1 clean | 0.25 d |

**Irrelevant for us:**
- #62240 and #56464 (DeepSelect, SM100 DeepGEMM);
- #59197 and #58504 (P/D);
- #61767, #62009, #61395, #61121 (DP / EPLB / NIXL);
- #56935, #56893, #56266, #56962, #61868, #61869, #58872, #58634 (SM90/SM100 kernels);
- #58586 and #57643 (fused all-reduce needs P2P; Kaggle uses NCCL over shared memory);
- HiSparse (§0c).

### Throughput arithmetic

The model is the Kaggle decode fit `step = 58 ms + s × B` with s = 0.9 ms per stream at 32K. End to end, about 0.78 of decode survives
(chunked prefill and re-prefills); that factor reproduces today's 311 at B = 36.

| per-stream slope s | B 36 | B 44 | B 48 | B 56 | B 64 |
|---|---|---|---|---|---|
| 0.90 (today) | 311 | 352 | 370 | 403 | 432 |
| 0.75 | 330 | 377 | 398 | 437 | 471 |
| 0.65 | 345 | 396 | 420 | 463 | 501 |

**400 tok/s (the gold target) needs both:**
- more streams: #56227, #58678 and util 0.97 together reach ~44-52;
- a lower slope or a better end-to-end factor: #156 lowers the slope; #58132 and #56227's removal of re-prefills raise the factor
  toward 0.82-0.85.

Estimated stack (a), if every item passes: **~370-420 tok/s.**

**550 is not reachable with this model on 50 W L4s in 7 days.** It needs B ≥ 64, s ≤ 0.65 and a factor ≥ 0.85 all at once; offload
(E051) would be required for the capacity alone.

## 5. Recommendation for the 7 days to 2026-11-02

**Stay on vision11.** No newer fork exists, and upstream is weeks of porting away. Cherry-pick in this order:

1. #156 indexer skip.
2. #58678 Engram `wkv` split.
3. #56227 bounded replay, plus #58560.
4. Only if 3 passes by 10-29: #58132, #59532 and #59894.

**How each change ships and how to fall back**
- Each change is Python/Triton only and ships through cell 0b, appended to the overlay tarball and sha file.
- No new wheel set and no new dataset version.
- Each has a kill switch:
  - #156: keep the current `dsv41_indexer.py` as the fallback file;
  - #58678: an env gate around the `ColumnParallelLinear` swap;
  - #56227: `--no-swa-bounded-replay`.
- V6 stays the fallback submission.

**Coordination**
- E051 (KV offload) continues. If #56227 lands, E051 drops its SWA checkpoint policy and offloads only the MLA+indexer group, with
  upstream semantics.
- Coordinate with dsv4-infer, which owns the solver and notebook and is running E052 on both 4090 boxes until ~20:15 UTC today.

### Day plan

**10-26/27, CPU**
- Stage an overlay branch with #156 (`patches/fork_pr156_indexer_skip_on_overlay.diff`) and #58678.
- Finish the #56227 port, including #58560 and #60767; the chain already applies in scratchpad `e053/forkcopy2/`.
- Run the PR unit tests plus the fork's prefix-cache tests on CPU.

**10-27: first 4x4090 test, about 3 h on one box**
1. **#156 kernel A/B** at E128 decode shapes under CUDA-graph replay:
   - shapes: rows 36/48/64; W 81,920 and 163,840; contexts 8K/32K/64K/128K; plus the 16,384-wide candidate calls;
   - checks: logits `torch.equal`, identical top-512 sets, µs per call;
   - also run the PR's `tests/kernels/attention/test_dsv41_native_indexer.py`.
2. **30-min tool replay A/B, old vs #156:**
   - setup: 36 streams, FULL graphs (cap 64), Kaggle-sized pool (`--num-gpu-blocks-override 21490`), production flags
     (`experiments/E046_multi_test_rollout/kaggle_bench/dsv41_logs/server_cmd.json`);
   - record: decode step ms, gen tok/s.
3. **#58678 startup:**
   - weights per GPU and KV tokens from the log;
   - greedy ids on 5 fixed prompts against the current stack.

**10-27/28: #56227 on the second box**
- Bring it up, then run a capacity A/B, replay on vs `--no-swa-bounded-replay`, 3 h each.
- Setup: E128 multi-test traffic at 44 and 48 streams, ungated or gate 0.7, on the Kaggle-sized pool.
- Count:
  - full re-prefills (prompt ≥ 50K with cached ≤ 640);
  - prefix-hit windows;
  - thrash windows;
  - KV level;
  - attempts per hour.

**10-28/29: accuracy A/B**
- 120 eval tasks, 1 attempt per arm on 2 boxes (the E050/E052 pattern): stack vs control at the same harness and effort.

**10-29/30: L4 pod soak**
- At least 4 h at Kaggle KV (util 0.938 pinned) with the chosen stack and worker count. The user rents the pod; that rule does not change.

**10-30: package and dry-run**
- Notebook build with the frozen inputs plus the cell-0b additions, and a fake-server dry run.
- Then the `next_candidate_fixes.md` gates.

**10-31 (quota resets Sat 00:00 UTC)**
- Kaggle bench, then submit by about 12:00 UTC. The score lands about 19 h later.
- 11-01 is the buffer for a fallback resubmission.

### Go / no-go criteria (each item separately)

- **#156:**
  - **GO** if:
    - indexer logits are bit-identical and top-512 sets identical on every shape, including graph replay;
    - the decode step is ≥ 4% faster at B = 36 and 48 in the 30-min replay;
    - there are 0 errors.
  - Otherwise drop it.
- **#58678:**
  - **GO** if:
    - weights drop by 0.18-0.25 GiB per GPU;
    - KV tokens rise by ≥ 7%;
    - greedy ids are identical on the 5 prompts, or the first divergence comes after ≥ 200 tokens with |Δlogprob| < 1e-3;
    - decode tok/s is ≥ -2% and prefill ≥ -3%.
- **#56227:**
  - **GO** if:
    - the CPU tests pass;
    - at 44-48 streams, full re-prefills drop ≥ 5x vs control;
    - prefix hits stay ≥ 0.9 over 3 h with no thrash window;
    - attempts/h or gen tok/s rise ≥ 8% at the same age;
    - 120-task pass@1 is within -3 pp of control (the noise band);
    - the gzip-loop-guard and NaN rates are unchanged;
    - the L4 soak runs ≥ 4 h with 0 errors.
  - **NO-GO** on any illegal memory access in replay slots, or an accuracy drop beyond the band.
- **#58132 chain:**
  - GO only if #56227 is GO by 10-29, prefill tok/s rises ≥ 15%, and pass@1 stays inside the band.
- **Cut-off:** anything not GO by 10-30 18:00 UTC is dropped for this deadline.
- **Revisit (b)** only if the fork publishes a rebase containing #56227 by about 10-28. Even then it needs the overlay port and a new wheel
  set (≥ 4-5 days), so probably not before 11-02.

## Files

- `patches/fork_pr156_indexer_skip_on_overlay.diff`: fork PR #156 commit 38c14855 rebased onto our overlay `dsv41_indexer.py`.
  - It keeps the `DSV41_INDEXER_Q_PREDECODE` gate and drops one docstring hunk.
  - It is third-party code: review it before running. It is not in any kit.
- Working notes from the three research sub-agents (scratchpad, not synced):
  - `e053/offload_findings.md` (§0);
  - `e053/sm89_overlay.md` (§2, §3);
  - `e053/pr_ranking.md` (§4; PR metadata and diffs in `e053/prs/`, apply checks in `e053/forkcopy*/`).
- Inputs read:
  - `notes/2026-10-26_dsv41_kaggle_inference_findings.md`;
  - `experiments/E051_kv_offload/{HANDOFF,DESIGN}.md`;
  - `experiments/E042_dsv41_l4_synthetic/README.md` (P1, P3, P8, P12, pod B profile);
  - `kaggle/dsv41_tools/next_candidate_fixes.md` (S2, S3, S3b, I1, I3);
  - `prior_works/project_maya.md`;
  - upstream release notes v0.29.0-v0.32.0.
