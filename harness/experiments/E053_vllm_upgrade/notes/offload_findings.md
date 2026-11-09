# Does newer vLLM support CPU KV offload for DeepSeek-V4.1? (E053 fork, 2026-10-26)

Sources: upstream main shallow clone `af6665b5dd20` (2026-10-26) at `scratchpad/e053/up_main/src` (paths below relative to it);
fork vision11 wheel source `scratchpad/src_vllm/vllm`; GitHub API (read-only).

## Short answer

Partly yes on upstream main, and no on our fork.

- **Upstream main.** For V4.1 the OffloadingConnector is designed to start and run.
  - With SWA bounded replay (default on, needs Model Runner V2) it offloads **only the compressed-MLA + indexer group**. The SWA window is
    **not stored**: after a host hit, the last 128 tokens are re-run.
  - The compressor ring is skipped as non-prefix-cacheable, and TP4 gets a single host copy (#57652).
  - What exists: CI runs an offload e2e test with **DeepSeek-V4-Flash (V4, not V4.1)** nightly on 4×H100.
  - What does not exist: no V4.1 offload e2e test, and nothing on SM89, because upstream has no SM89 sparse-MLA backend for V4.1.
- **Fork vision11.** Fails at startup, as E051 found.
  - Its draft PR #118 is unmerged and CED-oriented. It would get past the ring asserts but still stores the SWA groups densely.
- **HiSparse** does not apply. Main rejects DeepSeek-V4/V4.1 explicitly, and its sparse-MLA backends are SM90/SM100/SM120 only.

## (a) Spec types and the offload path on upstream main

**Specs registered by the V4.1 model.** V4.1 tags its specs `model_version="deepseek_v4"`, the same as V4.

| cache | spec | evidence | prefix_cacheable |
|---|---|---|---|
| compressed MLA (kv sources only) | `MLAAttentionSpec` (subclass of `FullAttentionSpec`), `tokens_per_state=compress_ratio` | `vllm/models/deepseek_v41/attention.py:1206-1233` | True |
| indexer K | `MLAAttentionSpec`, `tokens_per_state=compress_ratio` | `attention.py:1274-1300` | True |
| SWA (every layer, block 32, window 128) | `SlidingWindowMLASpec(bounded_replay=...)` from `DeepseekV4SWACache` | `vllm/v1/attention/backends/mla/sparse_swa.py:120-145`; built at `attention.py:562-580` | `not bounded_replay` (`kv_cache_interface.py:1027-1033`) |
| compressor ring | `CircularBufferSpec` (8+ rows, one block per request) | `vllm/models/deepseek_v41/compressor.py:247-255` | False (`kv_cache_interface.py:998-1000`) |

**V4 is different.** Its compressor state is a `SlidingWindowMLASpec` (`vllm/models/deepseek_v4/compressor.py:184`), so V4 has no ring.

**Bounded replay** (`vllm/config/cache.py:367-372`):
- `swa_bounded_replay` defaults to True.
- It is forced off without Model Runner V2, with PCP but no prefix caching, and on ROCm (`attention.py:538-561`).
- With replay on, `prefix_replay_tokens` is 128 (`kv_cache_interface.py:1035-1037`).

**The offload connector on main.**
- **Group selection.** It selects only prefix-cacheable groups: `get_offloading_group_ids` returns `prefix_cacheable_group_ids`
  (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:38-45`; `kv_cache_interface.py:1570-1577`).
  - The `tokens_per_block % tokens_per_hash` assert therefore runs only on the selected groups (`config.py:103-110`). This is #57145
    (merged 09-17), which supersedes the closed #55037 and #55033.
  - The ring never reaches `get_sliding_window_size_in_chunks`, whose final `assert isinstance(kv_cache_spec, FullAttentionSpec)` is at
    `config.py:81`. MLA passes as a `FullAttentionSpec`, and SWA (if replay is off) takes the `SlidingWindowSpec` branch (`config.py:61-70`).
  - #54743, open, is the same scoping in another form and is no longer needed.
- **Replicated single-copy host layout** (#57652, merged 09-30).
  - The condition is `use_mla and TP>1 and kv_tp_replicas == TP`, plus mp/single-node (`config.py:194-209`). The fork required exactly one
    `MLAAttentionSpec` group (fork `config.py:109-127`).
  - The unit test `test_replicated_layout_dsv4_flash` covers it (`tests/v1/kv_connector/unit/offloading_connector/test_config.py:866`).
- **Packed DSv4 blocks.** These are offloaded as one region per block (`offloading/worker.py:104-150`). The uint8 page is viewed as int8
  (`v1/kv_offload/cpu/gpu_worker.py:832`), which addresses the layout crash in issue #53607 (open, filed against v0.27.1).
- **Scheduler.** The scheduler resolves a manager per spec, with no FullAttentionSpec-only assert (`offloading/scheduler.py:244-251`). Its
  alignment logic names "DeepSeek V4 (MLA + SWA groups)" (`scheduler.py:179-205`).

**What is left at the restore boundary.** Main's core scheduler treats any connector hit as a replay hit (`v1/core/sched/scheduler.py:291-307,
1142-1153, 1214-1236`):
- `_load_restores_replay_window` is False for a prefix-cache store ("a prefix-cache store holds no window", `scheduler.py:3383-3390`).
- Hits are rounded down to a whole block.
- Hits of 128 tokens or fewer are dropped.
- `replay_start = H - 128` (`scheduler.py:3392-3409`).

**Conclusion for (a).**
- On main with Model Runner V2, `--kv-offloading-size` should start for V4.1 on GPUs main supports, and store only the MLA + indexer
  group.
- What could still fail or hurt:
  - Nobody has tested it end to end on V4.1. The nightly test is V4-Flash (`tests/evals/gsm8k/test_gsm8k_offloading.py:166-182`,
    `.buildkite/test_areas/lm_eval.yaml:1367-1380`).
  - Open robustness bugs on DeepSeek-V4: #54914 (prepare_load "Block not found" after days, V4-Pro/V4-Flash, v0.28-0.29) and #53569
    (fs tier misses on multi-group).
  - #56871/#56953 (capacity divided by the number of groups) are open but mostly moot here: there is one offloaded group, and a packed
    block equals one host slot.
  - Restores are not bit-identical to a cold run, by design (#56227).
  - For us: no SM89 backend on main (`models/deepseek_v41/nvidia/flashinfer_sparse.py:138`, majors 10 and 12 only). The replay clamp also
    lives in the FlashInfer/FlashMLA prefill kernels (`attention.py:553-561`), so an SM89 port must add that clamp to the fork's SM89
    kernels.

**The fork today.**
- `offloading/config.py:57-64` asserts divisibility for all groups.
- `scheduler.py:125` asserts FullAttentionSpec.
- The fork has no `prefix_replay_tokens` or bounded replay outside CED (`v1/core/sched/scheduler.py:381`, CED-only `prefix_replay_window`).

## (b) Upstream evidence of offload with DeepSeek, and the SWA state on restore

**Tests and CI.**
- `offloading-deepseek-v4-flash` and `simple-deepseek-v4-flash`: GSM8K run twice with a reload from CPU; TP4, EP, fp8, block 256,
  16 GiB. These are V4, not V4.1, run nightly on H100 ×4.
- Unit fixtures `_make_dsv4_flash_kv_cache_config` and the scratch-group skip tests (`test_config.py:405-541`, `test_worker.py:559`).

**PRs** (merge date or open):

| PR | state | what it does |
|---|---|---|
| #41445 | merged 05-01 | HMA support in the OffloadingConnector |
| #44287 | merged 06-03 | HMA support in the tiering spec |
| #57145 | merged 09-17 | skip scratch groups |
| #57652 | merged 09-30 | replicated layout for multi-group MLA |
| #56709 | merged 09-16 | restore SWA retained history |
| #56810 | merged 09-20 | SimpleCPUOffload skips non-cacheable groups (fixes V4.1 issue #56396) |
| #56227 | merged 09-18 | encoder-side SWA bounded replay: "With replay on, KV connectors transfer only the MLA and indexer caches, and hits of at most 128 tokens are not adopted"; not bit-identical; GSM8K equal on GB200 |
| #58132 | merged 09-29 | decoder replay: layers 21-39 prefill only the last 128 tokens; +25-36% prefill on TP4 GB200 |
| #59894 | merged 10-09 | trim layer 20 |
| #58504 | closed unmerged | exact non-aligned external hits |
| #41352 | open, WIP | Ascend DSv4 facade |
| #54743 | open | scoping (superseded) |
| #56953 | open | per-group slot sizing |
| #63586 | open, 10-25 | compressed-MLA for the canonical, TP-agnostic layout; affects only `canonical_layout=true` |
| #63411 | open | `bounded_replay` crash with `--disable-hybrid-kv-cache-manager` |
| #57325 | open | joint GPU/CPU lookup, DSv4 on Ascend |
| #57813 | open | pin in-flight chunks |

**Issues:**
- #53607: OffloadingConnector crashes on DSv4 MLA, v0.27.1. Open, but the int8-view code is now in place.
- #45475: crashes under concurrent DSv4-Flash offload, v0.22. Closed 10-16. 79-87% hits when it survived.
- #54914: an assert after days of uptime. Open.
- #53569: fs-tier misses across restarts on multi-group MLA+DSA. Open.
- #56396: V4.1 SimpleCPUOffload block 8 vs hash 32. Fixed by #56810.
- #63718: tracking issue for layout support, opened 10-26.

**SWA on a host restore.**
- With bounded replay it is **not stored but replayed**: 128 tokens recomputed and clamped at `replay_start`, so approximate.
- With replay off it is stored like any SWA group, using window chunks (`scheduler.py:112-131`, `config.py:61-70`).

**Fork PR #118** (draft, 09-20, based on fork main 3f3bbaea):
- It is in substance non-CED: it filters offload groups by `prefix_cacheable` (config.py), so the ring is skipped and the non-CED server
  should start.
- On the non-CED path the fork's SWA groups are still prefix-cacheable, so they would be stored densely: about 100 KB/token on the host by
  E051's numbers.
- Its replay and checkpoint logic applies only with CED.
- No full-model, throughput or multi-GPU validation (its own "Validation limits").

## (c) HiSparse

- **Series:**
  - #49629 [1/N]: closed.
  - #53781 [3/N] host-resident sparse-MLA decode hot-buffering: merged 09-12.
  - #56061 [4/N] metrics: merged 09-12.
  - #56629 [5/N] shared TP host pool: merged 09-12.
  - #55398 [6/N] P/D direct landing: open.
  - **#53592 [8/N] DeepSeek V4 support: open draft**, which notes "a substantial high-concurrency regression and an allocator failure under
    saturation" on B300.
  - Follow-ups #61243 (GLM-5.3) and #61277 (FlashInfer SM90): open.
- **On main.** `vllm/v1/hisparse/layout.py:99-104` raises `"HiSparse does not support DeepSeek V4."` for any
  `MLAAttentionSpec(model_version="deepseek_v4")`, which includes V4.1.
  - It needs Model Runner V2 (`config/vllm.py:732-738`) and the compiled `csrc/hisparse_kernels.cu` (`runtime.py:731-735`).
  - It works only through sparse-MLA backends with HiSparse paths: FlashMLA sparse (majors 9/10), FlashInfer sparse (10, 12), SM120, and
    SM90 in an open PR.
  - **Nothing on SM89 for V4.1**, and the models covered are V3.2 and GLM.
