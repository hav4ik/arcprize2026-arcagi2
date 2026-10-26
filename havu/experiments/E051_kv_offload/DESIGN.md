# KV CPU offloading for DeepSeek-V4.1 on the fork: design note

Scope: make vLLM's `OffloadingConnector` (`--kv-offloading-size` / `--kv-transfer-config`) work for DeepSeek-V4.1 (E128 2-bit) in
our fork (`vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` + the 12-file kit overlay). The goal is that an idle stream's
cached prefix spills to host RAM instead of being evicted. Everything here was built and tested on CPU only.

Paths: fork files are relative to `fp4kv/fork_ov/vllm/`; patched files are in `overlay/vllm/` (this directory); tests are in `tests/`.

## 1. TL;DR

- **Why it failed before (E042).** Two checks rejected the layout, and both reproduce on CPU (`scripts/repro_base.py`):
  1. `offloading/config.py:57-64` asserts `tokens_per_block % tokens_per_hash` for *every* group. The compressor ring is an 8-row
     `CircularBufferSpec` group that is not prefix-cacheable, while the hash unit is gcd(128, 32) = 32 (`v1/core/kv_cache_utils.py:750-757`).
  2. `offloading/scheduler.py:109-126` asserts `FullAttentionSpec` for the same ring group, whose spec is a `CircularBufferSpec`
     (`kv_cache_interface.py:783-816`).
- **What the patch does.**
  - **Ring: never offloaded.** The ring group is skipped, as upstream #57145 does (merged 09-17, after the fork's 09-02 snapshot). It keeps
    its slot, so group indices stay positional. A GPU prefix hit never restores the ring either.
  - **MLA + indexer: stored densely.** The group holds 128-token blocks.
  - **SWA: stored as checkpoints, a new opt-in policy (`swa_store_policy=checkpoint`).** Only the 4-chunk window that ends at two
    boundaries is stored: the **prompt-end** resume point and a **rolling** latest point. The rolling window is freed when the next one
    supersedes it.
  - **Hits are aligned to 128 tokens.** A host restore resumes exactly where a GPU prefix-cache hit would.
  - **Backported fix.** Upstream main no longer stores a finished request's last chunk, whose final token's KV is never written. The fork
    still does when `offload_prompt_only=false`; the patch backports the fix.
- **Host cost and the remaining risk.**
  - Host cost is **3.58 KB/token** (MLA, all 4 ranks) plus **12.8 MB per checkpoint**. Storing every SWA chunk would cost **100 KB/token**,
    which is unusable.
  - A 28 GiB budget holds about 146 conversations of 50k tokens, or 8.4M tokens of MLA prefix. That is about 3× the GPU pool.
  - Restoring 50k tokens moves about 48 MB per rank: tens of milliseconds over PCIe Gen3. The alternative is a 42 s re-prefill that stalls
    all decode.
  - The 26 CPU tests pass. Every non-equivalent mutant of the new logic is caught (§6). With default options the jobs emitted are
    byte-identical to the unpatched fork.
  - What CPU tests cannot show is GPU parity, plus the throughput gain on Kaggle. Those are the open risks (§8).

## 2. The V4.1 KV layout (verified)

I rebuilt the per-layer specs from the model code and the E128 `config.json`, then ran the fork's own grouping and placement code on CPU
(`tests/dsv41_fixture.py`, `scripts/dsv41_groups.py`):

| group | spec (per layer) | layers | tokens/block | page bytes | prefix-cacheable |
|---|---|---|---|---|---|
| 0-6 | `SlidingWindowMLASpec`, FP8 528 B rows, window 128 (`attention.py:94-123`) | 6,6,6,6,6,5,5 = 40 | 32 | 101,376 / 84,480 (block padded to 113,920) | yes |
| 7 | `MLAAttentionSpec` ×8: compressed KV, 288 B rows, for kv sources 2/8/14 (ratio 2) and 20 (ratio 1) (`attention.py:880-899`), and MXFP4 indexer K, 68 B (`:936-946`) | 8 | 128 | 3×18,432 + 36,864 + 3×4,352 + 8,704 = **113,920** | yes |
| 8 | `CircularBufferSpec` compressor ring, fp32, 8 rows (`compressor.py:135-166`), on ratio-2 sources 2/8/14 only | 3 | 8 (one block per request) | 98,304 | **no** |

- **One shared pool.** All groups use the same pool of 113,920-B blocks per GPU (block-outermost BLHNC layout; KV-tensor placement from
  `get_kv_cache_config_from_groups`). Kaggle has 2.28 GiB of KV per GPU, about 21,490 blocks (`server.log:343`).
- **Indexer K.** The index sources 24/28/32/36 own no K cache (`attention.py:368-397`). The 4 indexer layers in group 7 belong to sources
  2/8/14/20.
- **Replicated across ranks.** Every group is replicated on the 4 TP ranks: 1 KV head, and the compressor runs with `disable_tp`.
- **Block sizes and alignment.** The hash unit is 32 (prefix-cacheable groups only). The scheduler block size is lcm(32, 128, 8) = 128,
  and GPU prefix hits are aligned to it (`kv_cache_coordinator.py:679-686`).
- **SWA retention.** `--prefix-cache-retention-interval 128` with a window of 128 caches every SWA block, because `need ≥ per_segment` in
  `single_type_kv_cache_manager.py:1084`.
- **The ring is skipped in GPU lookups too** (`kv_cache_coordinator.py:693-701`).
- **Scheduler view.** The scheduler-side connector gets `generate_scheduler_kv_cache_config` (`v1/engine/core.py:332`,
  `kv_cache_utils.py:2425-2444`), which unwraps `UniformTypeKVCacheSpecs` to the first layer's spec. So the second E042 failure came from
  the ring group, not from the wrapper. The patch still unwraps the wrapper defensively.

## 3. Which groups are offloaded

### MLA + indexer (group 7): stored densely

- Every completed 128-token chunk is stored as one host slot. With `offload_prompt_only=false` (the fork default is `true`,
  `kv_offload/base.py:598`) this includes the decode tokens, because the next turn's prompt contains them.
- The chunk is final when the chunk's last token has been computed. Ratio-2 groups close on the odd position, and an aligned boundary is
  even.

### SWA (groups 0-6): checkpoints, not dense

A resume at boundary `b` needs, per group, only the 4 chunks ending at `b`: the 127 tokens before `b`, which is
`get_num_skipped_tokens = n - 127`, `single_type_kv_cache_manager.py:1104-1139`.

- **Dense storage is too big.** Dense SWA is 7 × 113,920 B per 32 tokens, about 24.9 KB/token per rank, 28 times the MLA bytes.
- **The policy.** `swa_store_policy=checkpoint` stores two windows per request:
  - **prompt end**, at `b_p = round_down(num_prompt − 1, 128)`. This is the hit for the same prompt, or for a re-rendered next turn after a
    token-exact resync;
  - **rolling**, at `b_r = round_down(computed, 128)`, the latest resume point, stored in the step that first covers it. The previous
    rolling window is freed (`CPUOffloadingManager.discard`, a new method that only removes idle, ready blocks).
- **When a request finishes**, its last rolling window is the next turn's resume point.
- **Host footprint per request:** about 2 windows. The old turns' windows age out under LRU first, because only the current turn's chunks
  are touched.

### Ring (group 8): skipped

- **No ring state is needed at an even boundary.** The ring holds the open compression group's rows (ratio 2). At any even boundary that
  group is empty: a hit at 128·k starts a fresh pair {h, h+1}.
- **The GPU cache does the same.** It never restores the ring (`CircularBufferManager`, `single_type_kv_cache_manager.py:1151-1240`), and
  the restore path stays identical to it.

### Comparison with Project Maya

Maya's `MAYA_DS4_SLOT_KEEP` (`maya/repo/docs/SETTINGS.md:623-645`) sets aside, per conversation:
- the exact position;
- every layer's 128-token window ring;
- the compressors' *partial groups*;
- the compressed KV and index keys.

We differ in three ways:
- **Resume point.** We resume at the last 128-aligned boundary instead of the exact token. That costs at most 127 re-computed tokens plus
  the usual last-token recompute, but needs no ring or partial-group state.
- **Storage path.** We use the paged block path that already exists, so there is no new kernel.
- **Sharing.** Checkpoints are keyed by prefix hash, so they are shared across requests: other attempts of the same task, and the next
  turn.

Maya's "slot" equals our final rolling checkpoint plus the dense MLA prefix.

## 4. Keys, lookup across 128/32/8-token groups, and when a hit is valid

**Keys.**
- `Request.block_hashes` is a chain at 32-token granularity.
- An offload key is the chained hash at a chunk's *last* 32-token unit plus the 4-byte group index (`kv_offload/base.py:29-31`,
  `scheduler.py:356-370`).
- MLA keys are every 4th hash, and SWA keys every hash.
- The ring gets no keys: `hashes_per_chunk=0` and `skip` (`overlay scheduler.py:513`).
- Because of the chain, a key fingerprints the whole prefix up to its chunk end, so a chunk can be reused by any request that shares that
  prefix.

**Lookup** (`_lookup_complete_chunks`, fork `scheduler.py:697-868`):
- **Bound.** The upper bound is `num_tokens − 1`: the last token is recomputed for logits. The patch also rounds it down to 128.
- **MLA.** The MLA lookup is a contiguous prefix scan from the local GPU hit.
- **SWA.** Each SWA group scans backwards for 4 consecutive present chunks. The groups iterate until they agree.
- **Re-alignment (patch).** After any group lowers the bound, it is rounded down to 128 again. If that rounding moved it, all SWA groups
  are re-scanned at the new bound (`overlay scheduler.py:1009`), including the group that triggered it. Without this, a window ending at
  1120 (two adjacent checkpoints) would be accepted at 1024 unverified; test `test_realign_reverifies_triggering_group` covers it.

**A hit at `h` is valid iff:**
1. `h % 128 == 0`, `h ≤ num_tokens − 1`, and `h` is at least the local GPU hit;
2. the MLA chunks `[local/128, h/128)` are present (below `local` they are on the GPU);
3. for each of the 7 SWA groups, the chunks `[h/32 − 4, h/32)` are present;
4. nothing is loaded into the ring.

The load maps keys to the blocks the GPU KV manager allocates for the external range (`update_state_after_alloc`). For SWA groups that
range is just the window, with null blocks before it; the ring gets size 0.

## 5. Host memory, transfer path, cost

**Host memory.** One host slot is one GPU block per rank × 4 ranks = 455,680 B, padded to 458,752 B. Multi-group layouts get no
single-copy replicated layout (fork `config.py:104-127` requires one MLA group), so all 4 ranks store their own copy (`cpu/spec.py:109`).

| quantity | per rank | host, all 4 ranks |
|---|---|---|
| MLA + indexer per token | 890 B | 3,584 B |
| one SWA checkpoint (7 groups × 4 chunks) | 3.19 MB | 12.85 MB |
| dense SWA per token (policy `all`) | 24.9 KB | 100 KB |
| 50k-token conversation, 2 live checkpoints | 51 MB | 205 MB |
| 28 GiB budget | 65,536 slots: 8.4M MLA tokens, or ~146 such conversations (~3× the 21,490-block GPU pool) | |

**Kaggle budget.**
- The cgroup is 173 GiB. Host RAM in use is 105-114 GiB (`notes/2026-10-26_dsv41_kaggle_inference_findings.md:120`); Engram pins 23.6 GiB
  per rank, about 94 GiB.
- A **25-30 GiB** cache leaves about 30 GiB of headroom. It is a `/dev/shm` mmap pinned with `cudaHostRegister` (`cpu/spec.py:167-197`,
  `gpu_worker.py:193-221`).
- **`/dev/shm` must hold it.** If Kaggle's `/dev/shm` is smaller, set `shared_region=false`: a new opt-in that uses private pinned tensors
  (`overlay cpu/spec.py:162-172`). Both count against the cgroup.

**Transfer path.** It is unchanged.
- Each direction has its own CUDA stream (`gpu_worker.py:512-680`). Stores run D2H through `ops.swap_blocks_batch` (cuMemcpyBatchAsync,
  copy engine). Loads use the same op, because 113,920 B is above Triton's 28 KiB limit.
- The packed pool is one region of 113,920 B per block, so each block is one descriptor (`offloading/worker.py:123-151`, tested on CPU).
- Stores start at the next step after `wait_stream(compute)`. Blocks reallocated before the copy finishes are flushed
  (`scheduler.py:1530-1550`).

**Cost.**

| traffic | volume per rank |
|---|---|
| steady decode at ~315 tok/s total: MLA + one rolling window per 128 tokens per stream | ~8 MB/s |
| prefill at ~1,180 tok/s | ~3 MB/s |
| restoring a 50k-token prefix (391 MLA + 28 SWA blocks) | 47.7 MB |

- The steady-state traffic is negligible, even against NCCL's SHM all-reduce, which shares the same PCIe links.
- A 50k restore takes ~4-16 ms at 3-12 GB/s on Gen3 x16. Re-prefilling it on Kaggle takes ~42 s and stalls every stream (§7 of the
  findings note).

## 6. Correctness: what makes a restore bit-identical to a GPU hit, and failure modes

**Why a restore matches.**
- A restore copies back the exact bytes the GPU block held. Each rank stores and loads its own copy, with no cross-rank mixing.
- The patch resumes only at the 128-aligned boundary a GPU hit would use. The suffix is therefore prefilled with the same chunking, and
  greedy tokens and logprobs should be bit-identical at batch size 1. `gpu/parity_test.py` checks exactly this.
- Engram, hyper-connections and the indexer's candidate buffers carry no per-request state. GPU prefix hits already work without
  restoring them.

| failure mode | effect | handling |
|---|---|---|
| store reads a block after the KV manager reused it | another request's KV under our key: silent garbage | existing flush/fence logic, plus `next_stored_chunk_idx` of checkpoint groups pinned at the earliest future window, so `_update_req_states` zeroes reused ids (`overlay scheduler.py:1645`); mutant `no_progress_protection` is caught by `test_retried_checkpoint_never_stores_reused_blocks` |
| incomplete window (a slid-out block was reused) | a hit that cannot load | the boundary is abandoned and nothing partial is stored (`_plan_swa_checkpoints`) |
| last chunk of a finished request includes the never-written final token | garbage KV for one token on the next turn | `max(num_prompt, num_tokens − 1)` (`overlay scheduler.py:1700`, upstream main `scheduler.py:1823-1827`); `test_finished_request_last_chunk_not_stored` |
| unaligned hit (e.g. 1120) | correct KV but a different resume point than a GPU hit, so not bit-identical; for ratio-2, an odd hit would need ring state | `align_hits` plus the re-scan; tests `test_unaligned_*`, `test_realign_*` |
| ring loaded or stored | assert in `update_state_after_alloc` (one 8-row block against cdiv(h, 8) slots) | skipped; mutant caught by 5 tests |
| host full | `prepare_store` returns None | allocation-failure counter; the boundary is retried while its window is alive, then abandoned |
| a superseded window is discarded while in use | — | `discard` removes only ready blocks with `ref_cnt == 0`; in-flight stores are retried later |
| load or transfer failure | `assert transfer_result.success` crashes the server (fork `worker.py:277`) | unchanged upstream behaviour; the Kaggle solver restarts the server, but the soak test must show it never fires |

**Mutation check** (`scripts/mutate.py`): 7 mutants.
- Caught: dropping the finished-request clamp, the re-scan, the ring skip in the load path, the window size, and the stale-block
  protection.
- Survived, but equivalent: the initial bound rounding, which the per-group rounding makes redundant on this layout, and reversing chunk
  order in both loops consistently.

## 7. The patch

Overlay = 4 full replacement files under `overlay/vllm/`; unified diff in `patch/dsv41_kv_offload.diff`. Each target is sha256-checked
against the wheel before install (`gpu/install_overlay.sh`). None of them overlaps the 12-file kit overlay.

| file | change |
|---|---|
| `distributed/kv_transfer/kv_connector/v1/offloading/config.py` (+14/−2) | skip non-prefix-cacheable groups in the divisibility check and in the `block_size` option |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` (+415/−10, mostly comments and the checkpoint helpers) | scratch-group `skip` (keys, lookup, load, store, touch); `UniformTypeKVCacheSpecs` unwrap; `align_hits` with re-scan; `swa_store_policy=checkpoint` (`_plan/_finish/_set_swa_ckpt_*`, discard queue); finished-request clamp (backport) |
| `v1/kv_offload/cpu/manager.py` (+34) | `discard(keys)` |
| `v1/kv_offload/cpu/spec.py` (+6) | `shared_region` knob |

**Guards.**
- With no `--kv-offloading-size` / `--kv-transfer-config`, none of this code runs, so the server is unchanged.
- With offloading on and default extra config (`swa_store_policy=all`, `align_hits` off), the only behaviour changes are:
  - the scratch skip, which affects only configs that previously failed the asserts;
  - the finished-request clamp, which fires only when `offload_prompt_only=false`.
- **Differential check** (`tests/diff_default_behavior.py`): the base and patched trees emit byte-identical job logs for full-attention
  and SWA hybrids with `offload_prompt_only=true`. With `false`, the only difference is the removed store of the uncomputed last block.

**Server flags.**

```
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":30064771072,"offload_prompt_only":false,"swa_store_policy":"checkpoint","eviction_policy":"lru"}}'
```

Here `cpu_bytes_to_use` is 28 GiB in total across the 4 ranks. Equivalently, use `--kv-offloading-size 28` together with the same JSON
minus `cpu_bytes_to_use`.

## 8. Effort, risk, and when this is not worth doing

**Effort.**
- **Done (CPU):** the patch, 26 tests, mutation and differential checks.
- **GPU validation** on the 4×4090 box, about 3-4 days:
  - install and startup, parity: 0.5 day;
  - the capacity matrix with `tool_replay.py` on E128 shapes: 1 day;
  - a 3-4 h soak with the real harness (`--token-exact --multi-test`) and an accuracy-parity check: 1 day;
  - fixes: 0.5-1.5 days.
- **Kaggle-shaped check:** one L4 pod run, about 0.5 day. You rent it; it is the only way to measure Gen3 PCIe and 50-72 W prefill.
- **Total:** realistically 4-7 working days to a submit-ready kit.

**Risks, most likely first.**
1. **An untested runtime path.** No one has run this fork's OffloadingConnector on a hybrid model with async scheduling, Model Runner V2
   and FULL CUDA graphs together. Possible asserts or hangs on the load path: medium.
2. **GPU parity.** Some per-request state outside the KV groups might exist that GPU hits also need. Low, because GPU hits already work;
   the parity test decides.
3. **Python scheduler overhead.** `_touch` touches all of a request's keys on every store, about 35k dict operations for a 160k context.
   Estimated ~1-2% of scheduler time; measure `lookup_sync_delay`.
4. **Host memory on Kaggle.** `/dev/shm` size, and pinning 28 GiB next to the 94 GiB Engram (startup +10-20 s). `shared_region=false` is
   the fallback.

**Not worth doing if any of these hold.**
- **(a) No further submission.** Version 6 was submitted on 2026-10-26, so the patch pays off only if there is another submission or rerun.
- **(b) The capacity test shows less than about 5% more attempts per hour at the best safe config.** At today's gated config (36 workers,
  gate 0.6, peak KV 0.73, 0 preemptions), the GPU cache rarely loses an active conversation. The measured loss is about 2-4% of throughput
  from rare full re-prefills.
  - Offload cannot add room for *running* streams: decode needs the whole MLA/indexer cache on the GPU.
  - The real upside is making eviction and preemption of idle streams cheap enough to run 44-52 workers without the thrash seen ungated
    at 44 (findings note §6).
  - Whether the 50 W L4 decode scales enough to pay for that is unknown.
- **(c) Parity fails** for a reason that needs kernel work.

**Phase 2, after parity passes.**
- Port upstream #57652 (replicated layout for multi-group MLA, 09-30). Only rank 0 would store and all ranks would load one copy, cutting
  host memory 4×. It first needs a check that the 4 ranks' KV bytes are identical.
- Divergent hits: when the MLA prefix is still on the GPU, load only the SWA checkpoint.
- Upstream HiSparse (#53781, DSv4 draft #53592) solves a different problem, more live decode capacity, and is not production-ready.
