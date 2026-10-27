# E054: two exact vLLM cherry-picks for the DSv4.1 Kaggle stack (fork PR #156 part 1, upstream #58678)

Status: **GPU plan run on Vast 4x4090 box 2 (57817226), 2026-10-26 22:45 - 10-27 02:37 UTC. #156: GO (exact, decode
step -5.7% / -6.5% at B = 36 / 48). #58678: NO-GO by the pre-registered gate (decode -2.4% at B = 48 against a -2%
limit; not bit-identical under Humming), REVIEW at best; its memory gate passes (+9.9% Kaggle KV).** Results and caveats:
§6. (CPU prep 2026-10-26: 18/18 checks.)

These are the first two items of E053's cherry-pick list (`experiments/E053_vllm_upgrade/README.md` §4-5). Both ship as
Python/Triton overlay files, each with its own environment kill switch:

| item | file (paths relative to site-packages) | E054 sha256 | kit (base) sha256 | kill switch (server env) |
|---|---|---|---|---|
| fork PR #156 part 1: decode indexer skips the tiles past each row's context | `vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py` | `9f13485b` | `3da076e1` | `DSV41_INDEXER_SKIP=0` |
| upstream #58678: Engram `wkv` column-parallel over TP | `vllm/models/deepseek_v4_1/common/engram.py` | `de9fc991` | `5f1e7230` | `DSV41_ENGRAM_WKV_TP=0` |

Both switches default to on. Their state is logged once per rank at startup: `E054 DSv4.1 indexer: grid-stride context
skip on|off ...` and `E054 Engram wkv: ColumnParallelLinear, 6400 of 25600 output columns per rank` (or
`ReplicatedLinear`).

Base for both: the community fork's vision11 wheel (`vllm-0.28.1rc1.dev517+glm53.dsv41.vision11`, commit 86d35745) plus our
12-file overlay (`kaggle/dsv41_tools/kit/vllm_overlay_rw.tgz`, sha256 `377d83e0`).

## 1. Fork PR #156 part 1: the grid-stride context skip

### What it changes

Source: PR #156 (yhfgyyf/vllm-deepseek-v4-sm89, open, head 38c14855, 10-22). That one commit contains two changes:

1. the grid-stride context skip, which runs on all GPU targets;
2. FP4 tensor-core scoring, which runs on SM120 only.

E054 takes part 1 only. Part 2 is SM120-only (on SM89 it would be dead code), and on SM120 it is *not* bit-identical:
only the top-k sets match. `patches/e054_vs_pr156_head.diff` shows that the E054 stride kernel is the PR head's kernel
with `FP4_DOT=False` folded in.

**The problem.** Under FULL CUDA graphs, the dense decode indexer launch is sized by the capture width,
`max_model_len // compress_ratio`, whatever the context (`F:vllm/v1/attention/backends/mla/indexer.py:1512-1520`). For E128
at `--max-model-len 163840` that means:
- 81,920 columns on index sources 2/8/14 (ratio 2, 64-token pages);
- 163,840 columns on source 20 (ratio 1, 128-token pages; its candidate selection feeds layers 24/28/32/36, which make
  16,384-wide candidate calls).

The old kernel runs one program per 16-column tile, so 40 rows x 5,120 or 10,240 tiles on a 36-stream step (36 streams
replay the 40-graph). Every tile past the row's context computes nothing useful and stores `-inf`. At a typical 40K-token
context, about 76% of the tiles are such empty tiles.

**The new kernel `_mxfp4_indexer_logits_stride_kernel`:**
- **Grid.** It runs on a grid `(rows, tile_programs)` with `tile_programs = min(col_tiles, max(8, 16384 // rows), 65535)`,
  for example 409 per row at 40 rows. Program `(row, p)` loops over tiles `p, p + P, p + 2P, ...`.
- **Rows read their own bounds.** Each row reads its bounds from device tensors:
  - `end = tl.load(row_ends + row)`;
  - for prefill, also `start = tl.load(row_starts + row)`.
- **What it scores.** It scores only the tiles in `[first_col, min(end, width))`:
  - decode: `first_col = 0`;
  - dense prefill: `first_col = min(start, width)`;
  - candidate calls keep the full 16,384 width.
- **Fill loops.** 256-wide fill loops write `-inf` over `[last_tile * 16, width)`, plus, for prefill, over
  `[0, first_tile * 16)`.
- **Hoisting.** `q` and the head weights are now loaded (or decoded) once per program instead of once per tile.
- **No axis swap.** `SWAP_GRID_AXES` is gone, because grid y is bounded by construction.

**What the file keeps:**
- The old kernel `_mxfp4_indexer_logits_kernel` stays byte for byte, with its own grid and its grid-limit `ValueError`.
  It is the `DSV41_INDEXER_SKIP=0` path, so the kill switch runs exactly the old compiled kernel.
- `DSV41_INDEXER_SKIP_CANDIDATES=0` keeps only the candidate calls on the old kernel. This matters if the GPU A/B shows
  that the hoisted-q loop is slower there; candidate calls skip nothing.
- `DSV41_INDEXER_TARGET_PROGRAMS=N` overrides the 16,384 target. The PR tuned it on an RTX PRO 6000 with 188 SMs; the L4
  has 58. Any value gives the same bits.
- Our `DSV41_INDEXER_Q_PREDECODE` gate (#154, E042) is unchanged.

**Hunk resolution.** Against our overlay, 9 of the PR's 10 hunks apply. The one that fails is the module docstring
(FP4-only text) together with an `import os` we already have, so it was dropped. The `_use_q_predecode` docstring
hunk (also FP4 text) was dropped too.

### Why it is exact

The claim: the whole `[rows, width]` logits buffer is identical bit for bit, not just the positions below the context.
- **Same tiles.** The tiles are the same 16-column tiles at the same offsets, and every computed tile runs the old
  kernel's per-tile code verbatim. That code:
  - builds the mask;
  - looks up the block table;
  - decodes K;
  - computes `tl.dot(q_bf16, k_bf16)` with fp32 accumulation;
  - applies ReLU;
  - takes the head-weighted sum;
  - stores `where(valid, score, -inf)`.

  Hoisting q out of the loop computes the same values. No floating-point operation is reordered, so for positions below
  the context the scores are identical.
- **Skipped tiles were `-inf` anyway.** A skipped decode tile has every column at or above `min(end, width)`. In the old
  kernel every such column has `valid = False` (the mask requires `token < end`), so it stored `-inf`. The fill loop
  stores the same `-inf` without loading K.
- **The partial tile.** The tile that contains `end` is computed in both kernels with the same mask.
- **Dense prefill.** Rows also skip the columns below `start`; those were `-inf` in the old kernel (`packed_col >= start`).
- **Consumers.** `persistent_topk` (decode), `top_k_per_row_prefill`, and layer 20's `_select_candidate_blocks` read
  identical buffers. So the top-512 sets and the candidate blocks are identical too, including how ties break.
- **Coverage.** For every column, either a tile or a fill chunk writes it.
  - Every trip count `(last - first + P - 1) // P` has a non-negative numerator, because `stop` is clamped to at least
    `first_tile * 16`.
  - The CPU tests fill the output with NaN first and check that none survives.
- **The padded tail.** Columns from `min(end, width)` to `width` read `-inf` in both kernels. The new kernel writes them
  without touching the cache: the tile loop's mask covers the partial last tile, and the fill loop covers the rest.
  Graph-padding rows (end 0, for example the 4 padded rows of a 36-stream batch) do no tile work and are all `-inf`.
- **Caveat.** Bit-identity on the GPU also assumes that Triton emits the same per-tile arithmetic inside the loop (the
  same MMA shape and the same reduction tree for `tl.sum` over the 32 heads; pipelining moves loads, not math).
  `gpu/indexer_ab.py` checks exactly this on SM89. The PR author reports bit-identity on SM120.

### CUDA-graph capture safety

- The launch grid is a function of `(rows, width)` only, both fixed for a captured batch size. It never depends on the
  contexts.
- Each row's context is read inside the kernel from `row_ends`, the persistent sequence-length buffer that vLLM updates
  in place before every replay. So the loop trip counts are decided on the device at replay time, and there is no host
  sync.
- The work is bounded by `width`, because `stop <= width`. A garbage or oversized `end` can never index past the capture
  width, and in the worst case (full context) the kernel does the old kernel's work plus a loop.
- The only allocation is the q predecode buffer, which is unchanged.
- Tests:
  - `tests/test_indexer_interp.py::test_grid_ignores_row_ends` records the grid for three different `row_ends` and finds
    one grid;
  - `gpu/indexer_ab.py` captures each graph with `row_ends = 1` and replays it with the real contexts.

### Review notes on the third-party diff (`patches/pr156_part1_on_overlay.diff`, 387 changed lines)

- **Indexing width.** `cols` stay int32, which is safe up to a width of 2^31. The row offsets are int64, and so is
  `row * logits_stride_row`.
- **Triton's `width == 1` specialization.** It still works: `tl.minimum(end, 1)` is valid. The GPU A/B covers widths 1,
  17, 255, 4,097 and 20,001 eagerly, because mixed steps run eager with the batch's real maximum context.
- **Grid behaviour change.** The old grid-limit `ValueError` (rows and tiles both above 65,535) cannot happen on the
  stride path. That is a behaviour change, but only in a case the old code rejected.
- **Pipelining.** `num_warps 4` and `num_stages 2` now apply to a loop, so Triton may software-pipeline the K loads. That
  affects speed only.
- **Register pressure.** q (32 x 128) stays live across the loop. Any spills would affect speed only, and the GPU A/B
  measures it.
- **Prefill.** The stride kernel also changes dense prefill: rows score `[start, end)` instead of the whole packed width.
  It is exact (tested) and should make long prefill chunks faster too.
- **Correctness concerns.** None found beyond the assumption that per-tile code generation is identical, which the GPU
  A/B tests.

## 2. Upstream #58678: Engram `wkv` split across TP ranks

### What it changes

Source: vllm-project/vllm #58678, merged 09-26, +7/-3. Our `engram.py` is a fork file with our E043C compact-table
changes, so I applied the upstream diff to it by hand.
- **The swap.** `ReplicatedLinear(6144 -> 25600)` becomes `ColumnParallelLinear(..., gather_output=True,
  disable_tp=use_sequence_parallel)`.
- **SP.** It stays replicated under sequence parallelism, where each rank holds different tokens. SP needs expert
  parallelism (`nvidia/model.py:178-186`); we run without it (TP4).
- **Kill switch.** `DSV41_ENGRAM_WKV_TP=0` restores the old `ReplicatedLinear` with the old arguments.
- **Untouched.** Nothing else changes: the table lookup, the row gather and the post-wkv kernel.

### Exact or not

On TP4, rank r computes output columns `[6400 r, 6400 (r + 1))` from its weight shard, and `all_gather(dim=-1)`
concatenates the four results in rank order.
- **Same inputs and data movement.** Every rank has the same input, because the Engram rows are all-gathered before
  `wkv`. The gather and the concatenation are pure data movement.
- **The question that remains.** The output is bit-identical to the replicated one exactly when the MXFP8 GEMM gives the
  same bits for column c with N = 6400 as with N = 25600.
  - That holds when the kernel's K-reduction order does not depend on N.
  - It can fail if the kernel picks a different split-K or tile config for the smaller N.
  - Our Ada MXFP8 kernel's `_split_k(rows, output_size)` does depend on N, but the server uses Humming (`--linear-backend
    humming`), whose configuration selection I cannot see on CPU.
- **If it fails.** The differences are fp32 accumulation-order rounding: at most one bf16 ulp in a small fraction of
  `wkv` outputs. That is noise of the same kind as the stack's existing run-to-run noise. In that case the item becomes
  "REVIEW", not "GO".
- **How it is tested.** `gpu/engram_wkv_ab.py` measures this on the real E128 weights, with the real classes and the
  server's kernel choice. It checks:
  - each GEMM against itself;
  - the full GEMM against the concatenated shards;
  - the Engram output itself (the post-wkv kernel on fixed hidden states).
- **What CPU already proved** (`tests/test_engram_wkv_tp.py`, with the real vLLM `ReplicatedLinear` /
  `ColumnParallelLinear`, the checkpoint's `DeepseekV4FP8Config`, and the real MXFP8 loaders including the [32, 32]
  checkpoint-scale repeat):
  - the four shards partition the replicated weight and per-row scales exactly, for TP 2, 4 and 8;
  - the communicator's `dim=-1` gather puts each rank's columns back in order;
  - with an exact GEMM, the output matches bit for bit.

### Memory and KV for E128 (TP4)

There are two Engram layers (1 and 14). Each layer's `wkv` holds 6144 x 25600 FP8 weights (157,286,400 B) plus per-row,
per-32 E8M0 scales (4,915,200 B), 162,201,600 B in all. These match E042's synthetic-init record: 314,572,800 + 9,830,400
B per GPU for both layers.
- **Saved per GPU.** TP4 keeps 1/4, so it saves 2 x 3/4 x 162,201,600 = **243,302,400 B = 0.227 GiB (232 MiB) per GPU**.
  Humming pads N to a multiple of 256, and 6400 = 25 x 256, so there is no padding loss.
- **Kaggle effect.** At Kaggle's 2.28 GiB of KV per GPU:
  - **+9.9% KV**;
  - about 21,490 -> about 23,625 blocks;
  - 1.554M -> about 1.709M tokens;
  - about +3-4 streams at the same KV pressure.

  This holds if peak activation does not grow; the gate checks it.

**#58678 does not change the KV page size.** It touches no attention, indexer, compressor or SWA layer, so the packed pool
keeps its 113,920 B per 128-token block. Only the number of blocks changes (`num_gpu_blocks`). The startup gate checks
it: the extra KV bytes, divided by the extra profiled blocks, must equal 113,920 B. Under `--num-gpu-blocks-override
21490` nothing about KV changes at all. E051's offload layout is unaffected.

**Costs:**
- **Decode.** One extra all-gather per Engram layer of [B, 6400] bf16 per rank (B ≤ 64: ≤ 0.8 MB per rank). The
  estimate on Kaggle's PCIe shared memory (~4.9 GB/s) is about 0.3 ms per layer, against about 0.4 ms per layer of
  weight reads saved: roughly neutral. To be measured.
- **Prefill.** For a 2048-token chunk the gather is 105 MB per layer. My estimate is about +1-2% prefill time on L4,
  against a GEMM that is a quarter of the size.
- **Transient activation.** Up to about +131 MB inside the Engram layer at T = 2048 (the shard output plus the gather
  copy). It must stay below the profiled 0.6 GiB peak; the gate checks it.
- **Compile cache.** vLLM's compile-cache key hashes the traced source files, not our environment variables, and
  `DSV41_ENGRAM_WKV_TP` changes the traced Engram graph. So never share a `VLLM_CACHE_ROOT` across switch settings.
  `gpu/serve_e054.sh` uses one per arm, and Kaggle runs start with an empty cache.

## 3. Packaging (no new wheel set, no new dataset version)

### Kit files

Kit files are in `kit/`, built by `scripts/build_kit.py`. The build is deterministic (gzip mtime 0, fixed member
metadata) and refuses to run unless the kit pair is `377d83e0`.

| file | sha256 | content |
|---|---|---|
| `kit/vllm_overlay_e054_add.tgz` | `7e5822cd5aad5c3bc272104cca24c9138582bab2853de096581fb06abc7f3baf` | the 2 E054 files (the overlay addition) |
| `kit/vllm_overlay_e054_add.sha256` | `b45a7daf…` | their 2 sha lines (same format as the kit's) |
| `kit/vllm_overlay_rw.tgz` | `3b69b51bfbaacbe365bf66be2185a2d1375f663db4198f8904a2843c43883092` | drop-in for the kit's tarball: the kit's 12 members in order, 2 replaced |
| `kit/vllm_overlay_rw.sha256` | `1e5cca29…` | the kit's 12 lines with those 2 replaced |

### Box install

`gpu/install_e054.sh [VENV] [--only indexer|engram] [--revert] [--status] [--force]`:
1. It checks the addition tarball against its sha file.
2. It refuses unless each target in the venv is the kit version, or already the E054 version.
3. It backs up the originals to `$VENV/e054_backup/` (once).
4. It installs and sha-checks the files.
5. It drops the two modules' stale `.pyc`.

`--revert` restores from the backups. I tested it here on a fake site-packages: install, partial install, idempotent
rerun, revert, and refusal of an unknown version.

### Kaggle (described only; `kaggle/dsv41_tools/build_frozen_mt.py` is not edited)

The notebook's cells run in this order:
1. 00, the settings cell;
2. 0, which unpacks and checks the kit against `MANIFEST.json`;
3. **0b**;
4. 1;
5. 2, which extracts `KIT/vllm_overlay_rw.tgz` into site-packages and asserts every line of `KIT/vllm_overlay_rw.sha256`.

So cell 0b can swap the kit's overlay pair after the MANIFEST check and before cell 2 installs it. Cell 2 itself needs no
change. There are two ways to do it; `tests/test_cell0b_sim.py` runs both, plus the fallback, against cell 2's code copied
verbatim.

- **A, recommended: pin the pair.**
  - **Option.** Add `--vllm-overlay DIR` to the builder. In `main()`, when it is set:
    `files["vllm_overlay_rw.tgz"] = DIR/"vllm_overlay_rw.tgz"` and `files["vllm_overlay_rw.sha256"] =
    DIR/"vllm_overlay_rw.sha256"`, with `DIR = experiments/E054_exact_cherrypicks/kit`.
  - **What `code_cell` already does.** Its existing logic then:
    - backs up the kit's pair as `*.v3orig`;
    - extracts the embedded pair over it;
    - checks both against `_MT_SHA`;
    - on a competition-rerun failure, restores the `*.v3orig` pair, i.e. the V6 overlay.
  - **One guard to add** inside cell 0b's `try`, before the backup: `assert sha256(KIT/vllm_overlay_rw.tgz) ==
    "377d83e0…"`. The pinned pair is a full replacement built from that version; any other kit means rebuilding with
    `scripts/build_kit.py`.
  - **Size.** It adds about 97 KB of base64 to the notebook.
- **B, append-style.** Embed only the 17.5 KB addition (about 24 KB of base64) and call
  `scripts/cell0b_overlay.py:e054_overlay_into_kit(KIT, add_tgz, add_sha)` inside cell 0b's `try`. In the
  rerun-failure path, call `restore_kit_overlay(KIT)`.
  - `e054_overlay_into_kit` checks the `377d83e0` base and each member's sha, and writes the `*.v3orig` backups.
  - A gzip tar cannot be appended to in place. So it rebuilds the tarball with the two members replaced, keeping the
    other ten byte for byte and in order, and rewrites the sha list the same way.

**Switch values.** In the settings cell, `os.environ.setdefault("DSV41_INDEXER_SKIP", "1")` and
`os.environ.setdefault("DSV41_ENGRAM_WKV_TP", "1")` record the switches; set them to `"0"` to disable. They reach the
server because `solver_dsv41.server_env()` starts from `dict(os.environ)`.

## 4. CPU checks (done here)

Run with `tests/run_cpu_checks.sh <python> <fork tree>`. Here I used the E051 CPU venv (torch 2.14.1+cpu, Triton 3.8.0)
and the read-only `fork_ov` tree. The two E054 files are injected over the tree by an import finder in
`tests/cpu_harness.py`, so the tree is never modified.

The harness keeps everything on the CPU:
- it hides CUDA;
- it stubs NVML, so vLLM resolves to UnspecifiedPlatform;
- it answers the vLLM version lookup;
- it keeps the real Triton in interpreter mode (`TRITON_INTERPRET=1`).

**Result: 18/18 tests pass** (about 80 s), plus the kit-consistency and syntax checks.

- **Imports.** Both patched modules import against the fork tree with a CPU torch.
- **#156: OLD and NEW kernels run side by side through the Triton interpreter** (`tests/test_indexer_interp.py`, 10
  tests). OLD is loaded from the kit file (sha `3da076e1`), NEW from the E054 file. The output buffers start as NaN.
  - **Coverage:**
    - dense decode at edge contexts 0, 1, 15, 16, 17, 640, 999, `width`, `> width` and negative;
    - widths 1, 2, 16, 17, 255, 256, 1,000 and 4,103;
    - tile-program targets 1, 5, 64, 7,000 and 1e9, so a program runs several tiles and the fill loops run several
      times;
    - `row_repeat` 2;
    - -1 and out-of-range block-table entries;
    - a strided page pool;
    - packed two-request dense prefill, including empty rows and rows past the width;
    - decode and prefill candidate calls.
  - **Bit-identity.** NEW equals OLD bit for bit everywhere, and so do the `DSV41_INDEXER_SKIP=0` and
    `DSV41_INDEXER_SKIP_CANDIDATES=0` variants.
  - **Reference.** An independent float64 PyTorch reference of the scoring matches to about 1e-7 relative, with
    identical `-inf` masks and identical top-k sets.
  - **Grid.** The grid ignores `row_ends`. The PR's own grid expectations hold, and E128's grids stay ≤ 65,535 (36 rows
    -> (36, 455); 64 -> (64, 256)). The old path keeps its axis swap.
  - **Mutation check.** Three plausible off-by-one bugs (a missed partial tail chunk, a dropped last column, a missing
    prefill head fill) each fail the suite.
  - **Limit.** The interpreter's `tl.dot` is numpy, not the GPU's MMA, so these tests prove the indexing, masking and
    fill logic, not GPU bits.
  - **Interpreter bug found.** Triton 3.8's interpreter multiplies bf16 bit patterns as integers in `tl.dot`. The
    harness patches it to upcast to fp32; OLD and NEW were bitwise equal even before the patch.
- **#58678** (`tests/test_engram_wkv_tp.py`, 5 tests):
  - the constructor switch at E128 shapes: column-parallel by default, SP keeps the full projection, and
    `DSV41_ENGRAM_WKV_TP=0` restores the old call;
  - the real-class shard and gather test described in §2.
- **Packaging** (`tests/test_cell0b_sim.py`, 3 tests): routes A and B, the rerun fallback, and refusal of a corrupt member
  or a second run.
- **GPU-side helpers:**
  - `startup_report.py` parses the real Kaggle log (`E046 .../server.log`: consumed 17.94-18.0 GiB, peak activation
    0.6 GiB, KV 2.28 GiB, 1,554,183 tokens);
  - its gate passes and fails as intended on synthetic logs;
  - `logprob_probe.py --compare` passes a noisy-but-same arm and fails a broken one;
  - `report_e054.py` reads a synthetic runs directory.
- **What CPU cannot check:**
  - GPU bit-identity of the stride kernel;
  - the Humming GEMM for N = 6400 against N = 25600;
  - speed;
  - real memory numbers.

  The GPU plan covers these.

## 5. GPU test plan (one 4x RTX 4090 box, about 3 h 20 min; nothing here was run)

**Before you start:**
- The box needs our stack at `/root/dsv41/venv` (fork wheel + kit overlay), the model at `/root/dsv41/model-e128t`, and
  the kit at `/root/kit`.
- Ask dsv4-infer which box is free; E052 had both until about 20:15 UTC.
- Never stop anything by process name. `run_plan.sh` stops its own server by process group.
- The environment is E051's `box_serve.sh` environment (CUDA 13 toolkit, shared JIT caches under
  `/root/dsv41/cache`). Compare it with `/root/smoke_start.sh` on the box before the first start.

**Copy (the user runs this from the repo root):**

```bash
rsync -aR experiments/./E054_exact_cherrypicks experiments/./E042_dsv41_l4_synthetic/scripts/decode_window.py \
  experiments/./E042_dsv41_l4_synthetic/scripts/tool_replay.py \
  experiments/./E043_dsv41_real_pruning/results/tool_shapes_e128t_w10_u2_gptq_eng64_sub80.json \
  experiments/./E051_kv_offload/gpu/metrics_logger.py root@<box>:/root/e054/src/
# on the box: G=/root/e054/src/E054_exact_cherrypicks/gpu
```

### Exactness design

The E051 session found that the stack is **not run-to-run deterministic at the server level**: two runs of the same
prompt differ at token 0 (logprob -4.436 vs -4.495), even with `VLLM_BATCH_INVARIANT=1`. So exact parity is tested only
on identical saved inputs, and each test first confirms that the OLD code is deterministic against itself. Server-level
checks are statistical, against OLD's own run-to-run spread.

### Steps

| step | command | time | what it measures |
|---|---|---|---|
| prep | `$G/run_plan.sh prep` | 5 min | Installs the 2 files (sha-checked, with backups) and reruns the CPU checks against the venv. |
| (a) #156 kernel | `$G/run_plan.sh kernel` (GPU 0) | 8 min | `indexer_ab.py`. **Inputs:** fixed-seed GPU inputs in Kaggle's pool geometry (21,490 pages of 113,920 B, block tables [reqs, 1280]). **Shapes:** rows 36 / 40 (36 + 4 padded) / 48 / 64; widths 81,920 (page 64) and 163,840 (page 128); contexts 8K / 32K / 64K / 128K, a 4K-160K mix and full; candidate calls (16,384); 2 dense prefill chunks; 5 small eager widths. **Checks:** OLD is self-deterministic; NEW equals OLD bitwise on the full buffer, eager and under CUDA-graph replay (graphs captured with `row_ends = 1`, replayed with the real contexts); identical top-512 sets (`torch.topk` on [0, ctx) and the production `persistent_topk`); the kill switch equals OLD. **Timing:** µs per call under graph replay with q predecode on, as in production. **Also:** a program-count sweep (4,096-32,768), and the indexer time per step for the mix (3 x 81,920 + 1 x 163,840 + 4 x candidates). |
| (c1) #58678 layer | `$G/run_plan.sh engram` (GPU 0) | 6 min | `engram_wkv_ab.py` on the real `layers.{1,14}.engram.wkv` tensors, using `ReplicatedLinear` against 4 x `ColumnParallelLinear(tp_rank=r)` with the Humming kernel the server picks. **For T** = 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 512 and 2048, it checks: each GEMM against itself; full against the concatenated shards, bitwise; the Engram output (post-wkv kernel, fixed hidden states), bitwise; the maximum difference where they differ. **Also:** per-rank bytes after the Humming repack (the real saving), and the Ada kernel as information. |
| (b, c2, c3) arm old | `$G/run_plan.sh arm old` | 80 min | The server with `--num-gpu-blocks-override 21490` (Kaggle's pool), FULL graphs (cap 64), the Kaggle flags, util 0.92, and its own `VLLM_CACHE_ROOT`. **Startup numbers:** `startup_report.py`. **Probe:** `logprob_probe.py`, 5 fixed prompts x 3 cold repeats (`/reset_prefix_cache`). **Decode step:** `decode_window.py` at B = 48 and B = 36 (seeded contexts of 8-40K, mean about 24K; B = 36 reuses the first 36 prompts, so its prefill is a prefix hit). This gives the decode step (`itl_median_ms`), decode tok/s and cold prefill tok/s. **Replays:** E128 `tool_replay.py` at 36 and at 48 streams, 30 min wall each (setup + 5 min warm-up + window), giving gen tok/s, hit rate and preemptions. `metrics_logger.py` runs throughout. |
| arm idx | `$G/run_plan.sh arm idx` | 22 min | The same startup, probe and decode with #156 on and #58678 off. It isolates #156. |
| arm both | `$G/run_plan.sh arm both` | 80 min | The same as old, with both on. Against idx it isolates #58678; against old it gives the combined replay. |
| report | `$G/run_plan.sh report` | 1 min | `report_e054.py`: go / no-go JSON. |

With `REPLAY_WALL=1500` (25-min replays) the total is about 3 h.

### Go / no-go (E053 §5, with exactness moved to saved inputs)

**#156: GO** when all of these hold:
- the kernel A/B is all ok (self-deterministic; bitwise eager and graph replay on every shape; identical top-512 sets);
- the decode step (idx vs old) is **≥ 4% faster at B = 36 and at B = 48**;
- the idx server log has 0 tracebacks;
- the probe for idx sits inside old's noise.

Otherwise drop it, or keep only the dense path with `DSV41_INDEXER_SKIP_CANDIDATES=0` if the candidate calls are what is
slow.

**#58678: GO** when all of these hold:
- `wkv` and the Engram output are bit-identical for every T;
- the startup gate (old -> both) holds:
  - weights **-0.18..-0.25 GiB** per GPU;
  - KV budget gain divided by Kaggle's 2.28 GiB **≥ +7%**;
  - peak activation up by no more than 0.02 GiB;
  - the same 113,920 B per KV block;
- decode tok/s (both vs idx) is no worse than **-2%** at B = 36 and 48, and cold prefill is no worse than **-3%**;
- the probe passes.

If everything else passes but the layer test shows differences, the verdict is **REVIEW**: look at `layer_diffs`. Only a
1-ulp kind of rounding difference may proceed, and then only through the 120-task accuracy A/B.

**Caveats:**
- The 4090 has much more memory bandwidth than the L4 and uses PCIe Gen4 rather than Kaggle's Gen3. Both the indexer's
  share of the step and the all-gather cost differ there.
- A GO on the box is necessary, not sufficient. The L4 pod soak and the 120-task accuracy A/B (E053 day plan,
  10-28..10-30; OLD run twice for the noise band) decide whether the change ships.

## 6. GPU results (Vast 4x RTX 4090 box 2, 57817226; 2026-10-26 22:45 - 10-27 02:37 UTC)

Everything is in `results/` (copied from `/root/e054/` on the box; scanned for `hf_` tokens, none found).
`results/report.json` is `gpu/report_e054.py`'s output. Its `go` flags are **false for both items, and two of the
reasons are test artifacts** (see "Gate artifacts" below). The verdicts here apply the pre-registered criteria to the
corrected readings.

### How it was run (deviations from §5)

- **Venv.** The plan ran in a copy, `/root/dsv41/venv_e054` (`cp -a`, all 53 `bin/` shebangs rewritten to the copy,
  `activate` fixed). `import vllm` and the `vllm` entry script both resolve to the copy. The E054 backups are in
  `venv_e054/e054_backup/`. The original `/root/dsv41/venv` was checked before and after: its 12 overlay files match
  the kit sha list (`377d83e0`), and `install_e054.sh --status` reports `kit` for both targets. The copy is left on the
  box for later runs (7.9 GB).
- **Step driver.** `/root/e054/step.sh NAME run_plan-args`, with `VENV=/root/dsv41/venv_e054` and the CUDA 13 env of
  `serve_e054.sh`. Logs and exit codes are in `results/logs/`.
- **Replay shapes.** The first `arm old` crashed in `tool_replay.py` within seconds. `SHAPES` points at E043's raw
  `tool_shapes_*.json` (`{summary, attempts}`), which the replay cannot read; E051 hit the same bug. The rerun used the
  converted file `E042 metadata/trace_shapes/tools_e043_e128t.json`, passed as `SHAPES=` with no script edits. That file
  matches all 334 raw attempts (`p0` and per-turn completions). The failed attempt's startup, probe and decode data are
  kept as `results/old_try1/`, a second OLD server sample.
- **Replay windows are 600 s, not 1,500 s.** Setup takes 860-900 s on the 4090 (36-48 cold histories at about 1.2K
  tok/s), so the 1,800-s wall cap leaves 300 s of warm-up and a 600-s window. At C = 48 the setup itself hit the
  replay's 900-s cap with 42 of 48 slots started, so that window is still partly setup.
- **Decode B = 36 was cold in every arm**, not a prefix hit (prefill 517-536 s). It is still the same work in each arm.

### (a) #156 kernel A/B (`results/kernel/indexer_ab.{json,log}`, GPU 0, 179 s)

- **Exact.** On all 74 cases (dense decode rows 36/40/48/64 x widths 81,920/163,840 x contexts 8K-full and mix;
  candidates; 5 eager widths; 2 prefill chunks; the program-count sweep):
  - OLD is self-deterministic.
  - NEW == OLD **bitwise on the whole logits buffer**, eager and under CUDA-graph replay (graphs captured with
    `row_ends = 1`, replayed with the real contexts).
  - The `torch.topk` top-512 sets are identical, and the kill switch (`DSV41_INDEXER_SKIP=0`) == OLD.
- **The script's `all_ok` is false** because of 3 cases (`dense_r36_w81920_full`, `dense_r48_w81920_65536`,
  `dense_r48_w163840_32768`). In each, only `persistent_topk_identical` fails, and their logits are bit-identical.
  `results/diag/topk_selfcheck.py` shows that the production `torch.ops._C.persistent_topk` is **not deterministic
  against itself on one fixed OLD buffer**:
  - over 8 reps, OLD-vs-OLD sets differ;
  - every differing index is a tie at the 512th value (511 values above it, 2 candidates for the last slot);
  - the value multisets are identical.

  Identical inputs cannot make a deterministic op disagree, so these differences belong to the top-k op, not the
  kernel. It is also one plausible source of the stack's server-level nondeterminism.
- **Speed** (µs per call under graph replay, q predecode on):

  | context | 8K | 32K | 64K | 128K | full | mix | candidates (16,384) |
  |---|---|---|---|---|---|---|---|
  | speedup, dense 81,920 / 163,840 | 11-15x / 13-16x | 4.5-4.8x / 4.3-4.6x | 2.5-2.7x / 2.4-2.7x | 1.4-1.6x | 1.24-1.36x | 1.8-2.6x | 1.07-1.16x |

  - Indexer time per step for the mix (3 x 81,920 + 1 x 163,840 + 4 x candidates): 36 rows 4.96 -> 2.92 ms, 48 rows
    7.24 -> 3.93 ms, 64 rows 9.66 -> 5.25 ms.
  - Prefill: the 1 x 30K chunk is 1.12x faster; 2 x 60K is 2.16x.
  - Tiny eager widths (1-4,097) are 0-4% slower: about 3 µs of launch overhead on calls of about 75 µs.
  - Program-count sweep (4,096-32,768): bit-identical everywhere. 8,192 is at most 5% better than the default 16,384 on
    the 4090; tune it on L4.

### (c1) #58678 layer A/B (`results/kernel/engram_wkv_ab*.json`; `results/diag/wkv_ulp.*`)

- **Not bit-identical under Humming**, the production `--linear-backend humming`:
  - full (N = 25,600) vs 4 concatenated shards (N = 6,400) differ in about 41% of `wkv` outputs at T <= 512 and 3.8% at
    T = 2,048;
  - the Engram output differs in 40-50% (T <= 512) and 8.5% (T = 2,048).
- **Size of the difference.** `wkv` |diff| is at most 4-8 on values up to about 1,264, i.e. one bf16 ulp at the top of
  the range. Near-zero outputs that cross sign give large ulp distances but tiny absolute errors.
- **Accuracy against an fp64 reference** (`x @ dequant(W)^T`) is the same class for both:
  - mean error at T = 1-64: OLD 1.10-1.27 ulp, NEW 0.97-1.21;
  - correctly rounded: OLD 61-62%, NEW 62-65%.
  - At T = 512, OLD's full GEMM happens to use a more accurate config (97% correctly rounded vs NEW's 61%). At T = 2,048
    both are about 97-98%.

  This is accumulation-order rounding, not a wrong-column bug.
- **Humming is not self-deterministic.** The full GEMM at T = 512 changes 2.3% of its outputs between two calls, and the
  shards at T = 2,048 do too. So OLD `wkv` already carries run-to-run noise of this kind.
- **The Ada kernel** (`--backend auto`, informational) is bit-identical for every T tested (1, 36, 48, 64, 2,048). It
  is 1.2-4x slower than Humming.
- **Measured saving: 243,302,400 B per GPU**, exactly as predicted.
- **GEMM time:** full 176-193 µs vs shard 51-59 µs (T = 1-64); 4.14 ms vs 1.10 ms (T = 2,048).

### (b, c2, c3) server arms (`results/{old,old_try1,idx,both}/`)

All four arms ran with the pool pinned to 21,490 blocks (1,591,736 KV tokens), FULL graphs, util 0.92, and a separate
`VLLM_CACHE_ROOT`. The arm lines in the logs confirm the switches (`old`/`old_try1`: skip off, `ReplicatedLinear`;
`idx`: skip on, `ReplicatedLinear`; `both`: skip on, `ColumnParallelLinear 6400 of 25600`). No arm logged a traceback
while serving. The one traceback in `both/server.log` is the `EngineDeadError` raised after `stop_server`'s SIGTERM,
with 48 requests in flight.

**Decode windows** (`decode_window.py`, contexts 8-40K, 60-s window):

| arm | B=48 tok/s | B=48 step ms | B=48 cold prefill tok/s | B=36 tok/s | B=36 step ms | B=36 cold prefill tok/s |
|---|---|---|---|---|---|---|
| old_try1 | 904.8 | 53.0 | 1,754 | 752.4 | 47.8 | 1,781 |
| old | 910.4 | 52.7 | 1,778 | 757.8 | 47.5 | 1,793 |
| idx (#156) | 974.4 | 49.3 | 1,767 | 803.4 | 44.8 | 1,781 |
| both (#156 + #58678) | 951.2 | 50.4 | 1,716 | 789.0 | 45.6 | 1,729 |

- OLD's server-to-server spread is 0.6% on decode and 1.4% on prefill.
- **#156 (idx vs old):** the step is 6.5% faster at B = 48 and 5.7% faster at B = 36 (7.0% / 6.3% against `old_try1`).
  Decode is +7.0% / +6.0%, and cold prefill is unchanged (-0.6% / -0.7%, inside the spread). The step saves 3.4 / 2.7
  ms, against 3.3 / 2.0 ms predicted from the kernel A/B.
- **#58678 (both vs idx):** decode is -2.38% at B = 48 (the gate is >= -2%: **fail by 0.4 pp**) and -1.79% at B = 36
  (pass). Cold prefill is -2.87% (the gate is >= -3%: pass, narrowly). The step costs +0.8-1.1 ms: the two extra
  all-gathers cost more than the 4x smaller GEMMs save, even on the 4090's PCIe Gen4. On Kaggle's Gen3 shared-memory
  all-reduce (about 4.9 GB/s) the cost is likely larger.
- **Startup gate old -> both** (`startup_report.py --gate`):
  - weights -0.23 GiB per GPU;
  - KV budget +0.227 GiB = **+9.9% of Kaggle's KV** (projected 23,625 blocks, 1,708,621 tokens);
  - peak activation +0.01 GiB;
  - 113,157 B per extra profiled block (113,920 expected), so the page geometry is unchanged.

  Every check passes except `no_tracebacks`, which counts the shutdown traceback above.
- idx's memory and KV are identical to old's.

**Logprob probe** (5 prompts x 3 cold repeats per arm; `results/diag/probe_noise.{py,json}`):
- **The pre-registered `d_lp0 <= 0.5` check cannot pass on this stack.** It scores a token present in one run's top-5
  but not the other's as -30, so any change in top-5 membership reads as about 27 nats. Even pairs from the same OLD
  server have a median of 26.7, and old vs old_try1 fails it.
- `probe_noise.py` compares token 0 on the top-5 intersection instead, with every arm judged against the OLD-vs-OLD
  (two servers) floor:

  | pair | first-divergence median | mean \|d lp\| median | token-0 \|d lp\| median / max | argmax0 same |
  |---|---|---|---|---|
  | floor: old vs old_try1 | 7 | 0.042 | 0.366 / 1.22 | 4/5 |
  | old vs idx | 6 | 0.056 | 0.369 / 2.01 | 5/5 |
  | old vs both | 6 | 0.072 | 0.370 / 1.46 | 5/5 |

  - Both is inside the noise on every check.
  - idx's medians match the floor. Its token-0 max (2.01) is the only check outside 1.5 x the floor + 0.05 (1.87). Both
    runs the same indexer and passes, and single-server maxima already reach 1.03-1.41, so this is sampling noise in a
    max over 45 pairs.

**Tool replay, combined (both vs old)**, E128 shapes, seed 1, 600-s windows:

| C | old gen tok/s | both gen tok/s | change | hit rate old / both | turns old / both | window KV mean (max) old / both | preemptions |
|---|---|---|---|---|---|---|---|
| 36 | 573.0 | 617.4 | **+7.7%** | 0.968 / 0.968 | 160 / 172 | 0.60 (0.66) / 0.60 (0.66) | 0 / 0 |
| 48 | 273.3 | 157.6 | -42% (not attributable, below) | 0.747 / 0.525 | 69 / 38 | 0.81 (0.86) / 0.80 (0.88) | 0 / 0 |

- **C = 36** is clean: same hits, no KV pressure, and #156's step gain shows through.
- **C = 48 is not a valid comparison:**
  - Both windows are still partly in setup: 42 of 48 slots started when the setup cap hit, 3-4 requests waiting.
  - KV is 0.80-0.88 in both, the documented E128 thrash band.
  - Most steps carry a 2,048-token prefill chunk and take over a second, so gen tok/s mostly measures how much uncached
    prefill landed in the window (both: 852K uncached tokens vs old's 707K, hits 0.53 vs 0.75).
  - There is one sample per arm.
  - #58678 touches no attention or KV code, and with the pool pinned its extra KV is never used.

  Treat it as open, not as a regression.

### Verdicts

- **#156 (indexer grid-stride skip): GO.**
  - **Exactness:** bit-exact logits and top-k sets on every shape, eager and graph replay; the 3 flagged cases are the
    downstream `persistent_topk`'s own tie nondeterminism, reproduced OLD vs OLD.
  - **Speed:** the decode step is 5.7% / 6.5% faster (gate >= 4%), with 0 tracebacks.
  - **Probe:** inside OLD's noise on medians.
  - **Next:** the L4 soak and the 120-task A/B (a confirmation only, since the change is exact). The L4 has less
    bandwidth, so the indexer is a larger share of its step; expect at least the 4090's gain.
- **#58678 (Engram `wkv` column-parallel): NO-GO by the pre-registered gate (REVIEW at best).**
  - **Memory:** passes (+9.9% KV, i.e. +3-4 streams at the same pressure).
  - **Speed:** decode -2.4% at B = 48 against a -2% limit.
  - **Exactness:** the layer is not bit-identical under Humming. The rounding is 1-ulp class, as accurate as OLD, and
    of the same kind as Humming's own run-to-run noise.

  To go further it needs:
  1. a capacity test where the extra KV is used: pool 23,625 blocks, or unpinned at Kaggle util, at C = 44-48 on L4,
     against old at 21,490;
  2. the decode cost measured on L4 (Gen3 shared memory);
  3. the 120-task accuracy A/B, because it is not exact.

  It ships only if the extra streams buy more than the -2 to -3% per-step cost.

### Gate artifacts to fix before the next run (not edited here)

- `gpu/indexer_ab.py`: compare `persistent_topk` tie-aware (check OLD against itself first, or compare value
  multisets).
- `gpu/logprob_probe.py`: compute `d_lp0` on the top-5 intersection (as `results/diag/probe_noise.py` does), and gate
  against a second OLD server.
- `gpu/startup_report.py`: count tracebacks only before the shutdown line.
- `gpu/run_plan.sh`: default `SHAPES` to the converted `E042 metadata/trace_shapes/tools_e043_e128t.json`, and raise
  `REPLAY_WALL` (or cap setup) so C = 48 gets a steady-state window.

## Files

- `overlay/vllm/...`: the two patched files.
- `patches/`:
  - `pr156_part1_on_overlay.diff` and `pr58678_on_overlay.diff` (against the kit versions; they reproduce the E054
    files);
  - `e054_vs_pr156_head.diff` (the E054 indexer against the PR head file).
- `kit/`: the addition pair and the drop-in pair (§3).
- `scripts/build_kit.py` and `scripts/cell0b_overlay.py` (the route B snippet).
- `tests/`:
  - `cpu_harness.py`;
  - `test_indexer_interp.py`;
  - `test_engram_wkv_tp.py`;
  - `test_cell0b_sim.py`;
  - `run_cpu_checks.sh`.
- `gpu/`:
  - `install_e054.sh`;
  - `indexer_ab.py`;
  - `engram_wkv_ab.py`;
  - `serve_e054.sh`;
  - `startup_report.py`;
  - `logprob_probe.py`;
  - `run_plan.sh`;
  - `report_e054.py`.
- `results/` (GPU run, §6):
  - `kernel/` (indexer and Engram A/Bs);
  - `old/`, `old_try1/`, `idx/`, `both/` (server logs, env, startup/probe/decode/replay JSON, `metrics.jsonl`);
  - `ctx36.json` / `ctx48.json`, `report.json`, `cpu_checks.log`, `logs/` (step logs and exit codes);
  - `diag/`: `topk_selfcheck.py` (persistent_topk self-determinism), `wkv_ulp.py` (ulp and fp64-reference analysis of
    the `wkv` difference), `probe_noise.py` (the probe against OLD's own server-to-server spread), `replay_series.py` (KV, hit rate and queue
    over the replay windows, from `metrics.jsonl`), and their outputs.
- Not in the repo: the PR head files fetched read-only (scratchpad `e054/dl_pr156_tests/`). The PR's own test file was
  not used. It targets the combined commit (it monkeypatches `_use_fp4_dot`) and needs vLLM's `tests/conftest.py`
  `workspace_init` fixture. Its part-1 checks are covered by the CPU and GPU A/Bs above: grid shape, program-count
  invariance, `-inf` fill for decode and prefill, and graph replay reading live lengths.
