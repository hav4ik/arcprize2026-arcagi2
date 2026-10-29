# E062: Manh's vLLM fork (github.com/nguyen599/vllm) with E128 + the B32 harness

User request 2026-10-28 ~07:30 UTC: plug Manh's fork into our best harness (B32), run overnight, report TPS and what
to expect on Kaggle; also check which of our B32 overlay improvements (Humming kernels, etc.) can go on top of it.
Manh's claims (via the user): ~550 tok/s, later ~700 tok/s at batch 80 with a ~4M-token KV budget (maybe GPU + host RAM).
Code review (read only, before this run): `notes/2026-10-28_manh_vllm_fork_review.md`.

Fork head at start: b9ad670a67 (2026-10-28 06:52 UTC; upstream main 08a236fa43 + 52 commits, incl. csrc changes,
so it needs a source build). DeepSeek commits: cded323df2 (Triton sparse MLA + DCP, E128 GPTQ experts),
c5d9fd8cc6 (L2 prefetch, split-k prenorm, sparse indexer), b9ad670a67 (GPU driver reserve).

## Boxes
- Vast 4x RTX 4090, instance 58063361 (rented by the user 2026-10-28): build, correctness, same-box A/B vs B32's stack.
- Runpod 4x L4 (being rented by the user): Kaggle-like replica (`l4/`).

## Layout
- `b4090/`: 4090 build, smoke, correctness and A/B logs.
- `wheels/`: the SM89 wheel built on the 4090 box + manifest (for the L4 box).
- `port/`: which B32 overlay pieces apply on top of the fork, patches.
- `l4/`: L4 replica.

## Port analysis (CPU only, 2026-10-28; `port/PORT_ANALYSIS.md`, `port/READY.md`)

**Most of B32's overlay is already in the fork or obsolete there.** Manh ported our E128 formats, and upstream absorbed
the rest.

**Already in the fork:**
- Humming dense: native, plus Humming for o_proj.
- 2-bit GPTQ experts: `expert_gptq.py` reads our `gptq/` files and needs no env vars.
- Engram compact table + `row_map`, and the (128, 6) router fast path.
- An equivalent of the #156 indexer skip: its Triton MQA logits stride over each row's own length.
- #56227 bounded replay, #58678, #58947.
- An upstream KV-offload connector that supersedes our four E051 files.

**Missing:**
1. **Engram row-chunk loading** for the Kaggle Model copy of E128 (the HF copy loads as is):
   `port/01_engram_row_chunks.diff`. Required before any Kaggle build on this fork.
2. **Previous-turn `reasoning_content`** is dropped on input: harness fix `port/02b_harness_send_reasoning.diff`
   (or the server patch `02`). This affects only re-rendered turns.

**Biggest lever found:**
- Humming pads each expert sublayer to multiples of 256 / 128. E128's TP4 shards are 1152 / 576, so that pads to
  1280 / 640: an estimated ~1.3 GiB per GPU of zeros, against B32's 2.29 GiB KV per GPU.
- Manh's fork has the knob: `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64` (default 256,128).
- `port/R1_b32_overlay_humming_moe_pad.diff` puts the same knob on B32's own overlay: est. Kaggle KV 1.55M -> ~2.3M
  tokens, if humming 0.1.12 accepts it. Untested.

**Config traps:**
- Upstream's offload default is `offload_prompt_only: true`: set false.
- Retention interval 512 is a no-op here.
- DCP4 makes prefix hits land on 512-token blocks at block size 128.

**What the claims most likely mean:**
- **4M KV tokens:** plausible from DCP4 on the GPU pool alone, because sharded groups count `block_size x dcp`. It
  doesn't need host RAM.
- **~700 tok/s at batch 80:** about what B32's own stack already does on 4x4090 (600-870 gen tok/s at 56 workers in
  E059's B16 twin). On 72 W L4s it would beat B32's 389. Our decode fit gives ~480 at B = 80, and DCP's fp8 query and
  partial traffic is ~30 ms per step over Kaggle's PCIe.
- **Expected on Kaggle:** ~430-550 tok/s until measured on the L4 pod. The claim doesn't state hardware, workload,
  settings or accuracy validation.

## 4090 (Vast 58063361)

**Verdict (10:35 UTC): the fork is correct to within about B32's own run-to-run noise** (with patches 01-03; 03 is required on SM89): its new-kernel unit tests pass on SM89 (1,877 passed); it loads the Kaggle copy (row-chunked Engram, patch 01) and serves; on 24 real tool prompts (3.4k-40.8k tokens) first-token argmax 24/24 equal, greedy divergence like B32-vs-B32; teacher-forced on B32's own 256-token continuations (4,912 tokens) per-token |dlogprob| mean 0.042 vs B32-repeat 0.037 (median 0.0030 vs 0.0026, p99 0.38 vs 0.33), mean NLL +0.0039 nats/token vs B32 (repeat +0.0002): a small numeric shift (fp8 DCP transport / new kernels), not a bug; only the 120-task accuracy A/B can say whether it costs points. **L4 options (pad 64,64, embed UVA, interleave 64; 11:04 UTC): clean.** vs the fork's defaults: |dlogprob| mean 0.036, median 0.0024, p99 0.29, NLL +0.0001 (= B32's repeat noise); vs B32: NLL +0.0040 (the fork's own shift). They raise the 4090 pool at util 0.92 from 2.45M to 3.98M tokens (+62%; B32 2.12M).

- Box: 4x RTX 4090 (300 W cap, 24 GiB), 256 cores, cgroup 342 GiB (RAM use kept <= ~250 GB), 296 GB disk, CUDA 13.0
  toolkit, driver 590. Vast's proxy SSH (ssh7.vast.ai:25442) never answered; direct SSH works:
  `ssh -p 41451 root@66.114.157.224`. The image's own Qwen3.5 vLLM service (supervisor `vllm`, `model-ui`, `ray`) was
  stopped to free the GPUs.
- Model: the HF repo `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` returns 404 to the workspace's HF token
  (fine-grained; it reads the livobereshka datasets but not this private model). The box uses the **Kaggle Model copy**
  (`chankhavu/dsv41-flash-1017-arc-tool-e128-2bit` V1, 136 files, kagglehub per file, sha256-checked against
  `E043/results/kaggle_e128_manifest.json`) -> `/root/dsv41/model-e128k` (B32 loads it as Kaggle does). Manh's fork
  has no loader for row-chunked Engram tables, so `b4090/merge_engram.py` writes an HF-layout view
  `/root/dsv41/model-e128m` (symlinks + the chunks concatenated back into whole tensors; the chunks are exact row
  slices of the HF tensors, E043 verify_reshard_equiv).
- B32's stack: E042 `setup_box.sh` / `box_install.sh` + podprep + E056's 16-file overlay (all shas OK), kit_b = kit +
  pinned_candb (solver 77fe80a3, harness 882c817f, train_check 8fe36175, auto_repair 2292369c).
- Manh's fork: b9ad670a67 (private repo; source copied from the workspace clone), `b4090/build_manh.sh` (SM89 only,
  torch 2.14.0+cu130, uv venv `/root/manh/venv`).

### 4090 log (UTC, 2026-10-28)
- 08:21 E128 Kaggle copy downloaded (136/136 sha256 OK); fork wheel built (20 min, `wheels/`); B32 stack installed.
- 08:31 B32 server (util 0.92, max-num-seqs 64, no offload): weights 16.9 GiB/GPU, KV 3.09 GiB = **2,118,735 tokens**.
- B32 repeat noise (same server, 24 real tool prompts of 3.4k-40.8k tokens, `b4090/parity.py`): greedy decoding is not
  repeatable (first-token argmax 24/24 equal, but the greedy continuations diverge after a median of 8 tokens);
  teacher-forced on a fixed 256-token continuation (4,912 tokens): per-token |dlogprob| mean 0.037, median 0.0026,
  p90 0.116, p99 0.33, max 1.31; mean NLL 0.2909 vs 0.2910. This is the noise floor the fork is judged against.
- 09:32 fork unit tests on GPU (4090, SM89) with patches 01-03 installed: `test_triton_sparse_mla_dsv4`,
  `test_triton_mqa_logits`, `test_dcp_indexer_topk`, `test_deepseek_v41_expert_gptq`, `test_humming_mxfp8_bmm`:
  **59 passed**; `test_engram`, `test_topk_softplus_sqrt`, `test_mla_short_prefill_indexer`: **1818 passed, 33 skipped**.
- 09:36 the SM89-only wheel cannot start E128 without patch 03 (indexer requires DeepGEMM; same as the L4 agent saw).
- 09:53 fork server up on the **Kaggle copy** (row-chunked Engram via patch 01; TP4 + DCP4, interleave 128, humming
  linear, util 0.92): weights 16.36 GiB/GPU, KV 2.8 GiB = **2,454,526 tokens (+16% vs B32 at the same util)**;
  `cudagraph_mode` FULL falls back to FULL_DECODE_ONLY (sparse-SWA FlashInfer backend); first start 17 min (Triton /
  TileLang JIT; graph capture 604 s on one rank), later starts use the caches.
- 10:06 fork (defaults above) vs B32 on the same 24 prompts: first-token argmax 24/24; greedy continuations diverge
  like B32's own repeats (median 9.5 tokens); teacher-forced on B32's continuations: |dlogprob| mean 0.042 / median
  0.0030 / p90 0.127 / p99 0.38 (B32 repeat 0.037 / 0.0026 / 0.116 / 0.33), mean NLL 0.2948 vs 0.2909 (+0.0039; B32
  repeat +0.0002). Render: with the harness's tools + `chat_template_kwargs` (thinking, effort 50) the fork's
  `/tokenize` is prefix-stable on 24/24; without kwargs its default effort header is 75 (B32: 50), so a client that
  sends no kwargs gets a different system prompt (the B32 solver always sends them).
- 10:36 fork with the L4 replica's capacity options (`VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64`, `--cpu-offload-gb 1
  --cpu-offload-params embed_tokens` (0.31 GiB offloaded), interleave 64; util 0.92): weights **14.8 GiB/GPU** (defaults
  16.36), KV 4.5 GiB = **3,979,602 tokens** (defaults 2,454,526: +62%; B32 2,118,735: +88%); start 2.5 min with caches.
  Teacher-forced (24 prompts, 4,912 tokens): vs the fork's defaults |dlogprob| mean 0.036 / median 0.0024 / p90
  0.116 / p99 0.29, NLL +0.0001 (identical to B32's repeat noise); vs B32 mean 0.042 / 0.0029 / 0.133 / 0.40,
  NLL +0.0040 (same as the defaults' +0.0039). **The options do not change outputs.**
- 11:04 throughput A/B started (`b4090/ab_run.sh`): fork x56 with the L4 replica's exact server config (offload 32 GiB,
  util 0.938), 50-min window (`ab_manh_w56`); then B32 x56 (`ab_b32_w56`); then the fork x80 if time allows.
- 11:30 fork x56 (`ab_manh_w56`) server up after **925 s** (CUDA-graph capture 543 s, not cached across launches here;
  the L4 agent saw 440 s vs B32's 235 s), KV 4,336,285 tokens at 0.938. In the 34 min from server-up to stop-new:
  **179.5 gen tok/s** mean (10-min bins 44 / 213 / 244 / 264), 50.6 running, KV 8.9% (never KV-bound), 5 attempts
  finished. Pure-decode 10-s windows (no prompt tokens, >= 50 running): **308 tok/s** median (5.5 tok/s per stream,
  ~180 ms per decode step); windows with prefill 196 tok/s. All-reduce ran on PYNCCL (`VLLM_HOST_ALLREDUCE` not set);
  `cudagraph_mode` FULL_DECODE_ONLY, so steps with any prefill run without graphs.
- 12:16 render parity with the harness's tools + kwargs, B32's server (8100) vs the fork's: **full renders equal 24/24,
  prefix renders equal 24/24**, prefix-stable 24/24 on both (patch 02 works; `--token-exact` sees identical prompts).
- 12:16 B32 x56 (`ab_b32_w56`) server up after 355 s, KV 2,414,269 tokens at 0.938.
- 13:05 **x56 A/B result (same box, workload, solver, settings, util 0.938, offload 32 GiB; first 34 min after
  server-up, `b4090/tps_ab.py`):**

  | stack | server start | gen tok/s (34 min) | 10-min bins | pure-decode 10-s windows | windows with prefill | KV mean | attempts done | local pairs at window end |
  |---|---|---|---|---|---|---|---|---|
  | B32 (our fork + 16-file overlay) | 355 s | **351.2** | 111 / 420 / 444 / 462 | 556 tok/s (n 11) | 372 tok/s | 19.1% | 14 | 26/344 (44 min) |
  | Manh's fork, L4 replica config | 925 s | **179.5** | 44 / 213 / 244 / 264 | 308 tok/s (n 48) | 195 tok/s | 8.9% | 5 | 6/344 (34 min) |
  | ratio fork / B32 | 2.6x slower | **0.51** | 0.40-0.57 | 0.55 | 0.52 | | | |

  The fork is **~2x slower than B32 on the 4090 at x56** in every phase (pure decode too), while far from KV-bound
  (9% used); the per-step cost, not KV, limits it here. Its larger pool cannot pay off at x56. This config uses NCCL
  all-reduce (no `VLLM_HOST_ALLREDUCE`) and FULL_DECODE_ONLY graphs; Manh's 700 tok/s setup may differ (host
  all-reduce + L2 prefetch, `VLLM_TRITON_SPARSE_INDEXER_LOGITS`), not tested here.
- 13:05 fork x80 (`ab_manh_w80`, max-num-seqs 96, 45-min window) launched.
- 13:22 fork x80 (`ab_manh_w80`, max-num-seqs 96): server up after 1,015 s (graph capture 643 s), KV 4,192,997 tokens;
  only 28 min to stop-new, mostly the initial prefill of 80 prompts: 98.7 gen tok/s over the 28 min (bins 10 / 108 /
  195), pure-decode 10-s windows at >= 70 running **264 tok/s** (x56: 308): on this box the fork does not gain from a
  larger batch.
- **Why the 4090 and the L4 disagree (hypothesis from these runs):** on the L4 pod the fork x80 runs ~600 tok/s vs
  B32's ~430-470 (L4 agent); here it is 0.51x B32. The fork on the much weaker L4 does a decode step at x80 in ~133 ms,
  but needs ~180 ms at x55 on the 4090, so this box penalizes something other than compute. Both stacks draw only
  ~115-160 W of the 4090s' 300 W at full clocks (mem.log): neither is compute-bound here. This box has **no GPU P2P
  ("CNS") and its GPUs sit on two sockets** (GPU0-1 NUMA 0, GPU2-3 NUMA 1, SYS between them), so every NCCL transfer
  goes through host memory and half of them cross sockets. DCP4 adds collectives to every compressed layer (query
  all-gather, partial-output send/recv, on top of TP's all-reduces), so the fork pays that latency per step several
  more times than B32 does, and the cost grows with the batch (x80 pure decode 264 < x56 308). On the L4 pods (one
  NUMA node, PHB, faster NCCL) that overhead is smaller, while B32's own step is held down by the L4's 72 W
  compute / bandwidth, where the fork's 4x-sharded compressed-KV reads and its larger pool (no preemption) pay off.
  The prefill share does not explain it (the fork is ~0.52x both in pure-decode and in prefill windows). The kernel
  difference (B32's FlashInfer sparse MLA vs the fork's Triton sparse MLA) cannot be separated from the
  communication cost with these runs. **The 4090 ratio does not transfer to Kaggle; use the L4 numbers.**
- 13:56 logs copied off (`b4090/runs/`, `b4090/parity/`); box idle (GPUs empty, no processes of ours). Stopping
  instance 58063361 through the Vast API was blocked by the session's permission classifier: **the user needs to stop
  it** (or allow it).

## L4 (Runpod ddj7xv53e5rgre)

**Status 14:20 UTC:** replica `manh_x2t_w80` (Manh's fork, 80 workers) running healthy at pod 4.4 h; morning summary below. B32 same-pod window `b32pod_x2t` done (08:25-09:31).

- Pod and fidelity: `l4/fidelity.md`.
  - **GPUs:** 4x L4 at 72 W, 2,040 MHz. Kaggle's scored rerun runs at a 50 W cap (user), so pod numbers are
    optimistic vs Kaggle. The Kaggle-clock projection corrects with factors from a real Kaggle run (prefill x1.45,
    decode x0.94).
  - **NCCL all-reduce is ~2x faster than Kaggle:** 3.3 ms per 2048-token all-reduce, against Kaggle's ~6.4 ms (E027).
  - **CPU is the weak point:** a 22.95-CPU cgroup quota.
- Setup (`l4/stage_pod.sh`):
  - E128 from HF at revision 9bce98c5, plain HTTP with the full-access token: 121 files, 131,090,911,045 B, the HF
    listing.
  - B32's stack as on the 4090: box_install + podprep + the 16-file overlay, all shas OK, kit_b
    (77fe80a3 / 882c817f / 8fe36175 / 2292369c).
- Manh's fork: the 4090 agent's wheel (`wheels/`, b9ad670a67, sha256 cf111141), installed with `freeze.txt` into
  `/root/manh/venv` (torch 2.14.0+cu130, flashinfer 0.7.2.post1, humming 0.1.18).
  - Patches 01 (Engram row chunks) and 02 (reasoning_content input) are applied in site-packages.
  - The 3 patched files are byte-identical to `nguyen599/vllm` branch `arc/e128-kaggle` (e4988d15), so the install is
    that branch's code.
  - **First start failed** on every rank in `SparseAttnIndexer.__init__`: "Sparse Attention Indexer CUDA op requires
    DeepGEMM support". An SM89-only build (`TORCH_CUDA_ARCH_LIST=8.9`) ships no vendored DeepGEMM; DeepGEMM is
    built only for 9.0a+. On SM89 the logits dispatch already uses the Triton kernels, because
    `is_deep_gemm_mqa_logits_supported()` is False there, so the `__init__` guard was the only blocker.
  - **Patch 03** (`port/03_indexer_deepgemm_guard_sm89.diff`) requires DeepGEMM only on SM90+.
  - **The L4 replica therefore runs b9ad670a67 + patches 01 / 02 / 03.** The 4090 agent commits 03 to
    `arc/e128-kaggle` once it has run there.
- Standalone starts before the replica (`l4/serve_manh.sh`: the solver's fixed server flags plus the replica's extra
  args; logs `/root/e062/serve_manh_8011_try*.log`):
  1. **Start 1:** DeepGEMM guard (above) -> patch 03.
  2. **Start 2:** "v4.1 candidate blocks must not straddle DCP interleave shards".
     - The default `cp_kv_cache_interleave_size` is 1. It must be a multiple of E128's `candidate_block_size` (8) and
       divide `--block-size` 128.
     - Set to **64**, the value in Manh's `test_dcp_indexer_topk.py`.
  3. **Start 3:** loaded and profiled.
     - Weights 16.36 GiB/GPU; CUDA-graph capture took 226 s; 1.78 GiB/GPU left for KV.
     - **KV pool 1,545,435 tokens at util 0.938**, the same as B32's 1,556,850. DCP4 alone does not give a large pool
       at these settings.
     - It then failed: "prefix_cache_retention_interval (128) must be ... a multiple of scheduler_block_size (512)".
       The solver hard-codes 128; under DCP4 a scheduler block is 128 x 4 = 512 tokens. Fixed by passing
       `--prefix-cache-retention-interval 512` (B32's value) in the extra args (the later flag wins).
  4. **Start 4:** + Manh's capacity options (coordinator OK, closer to his 700 tok/s setup):
     `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64` and `--cpu-offload-gb 1 --cpu-offload-params embed_tokens`.
     - **Correctness of the options: cleared by the 4090 agent (11:04 UTC).** Pad 64,64, embed_tokens UVA and
       interleave 64 don't change outputs: options on vs off is within B32's own run-to-run noise (|dlogprob| mean
       0.036 vs 0.037, NLL +0.0001).
     - **Remaining caveat for any score from this replica:** the fork itself shifts numerics slightly vs B32 (NLL
       +0.004/token; first token identical on 24/24 prompts). Only the 120-task accuracy A/B on the 48 GB box can
       show whether that costs points.
     - Result: up in ~2 min (CUDA graphs cached from start 3). Weights 14.8 GiB/GPU (from 16.36); 3.47 GiB/GPU left
       for KV.
     - **KV pool 3,058,602 tokens** at util 0.938: 1.96x B32's 1,556,850, max concurrency 18.7x at 163,840.
     - Host memory after start: cgroup shmem 124.9 GiB + rss 11.6 GiB = **136.5 GiB unreclaimable** (32 GiB of it
       the offload pool in /dev/shm). Kaggle's limit is 173 GiB; B32 peaked at 141 GiB.
     - **API checks pass** (`l4/api_check.py`):
       - `/tokenize` keeps prior thinking under both `reasoning_content` and `reasoning` (patch 02);
       - a greedy chat call returns a parsed `run_python` tool call with separate reasoning;
       - token-id `/v1/completions` returns canonical DSML.
- **Replica `manh_x2t_w80`, launched 09:49:33 UTC** (`l4/launch_manh.sh manh_x2t_w80 80 96`).
  - B32's kit_b and settings (multi-test 0.5, context limit, gate 0.85 + queue hold, auto-repair), workload
    evalx2t_s2026, toolkit deadlines.
  - Server: `/root/manh/venv`, util 0.938 (pods' equivalent of Kaggle's KV pool settings), `--max-num-seqs 96`, and
    extra args:
    - `--linear-backend humming`
    - `--decode-context-parallel-size 4 --cp-kv-cache-interleave-size 64`
    - `--prefix-cache-retention-interval 512`
    - `--cpu-offload-gb 1 --cpu-offload-params embed_tokens`
    - `--compilation-config {"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 96}`
    - `--kv-transfer-config` OffloadingConnector, 32 GiB, `offload_prompt_only: false`, lru
  - Server env: `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64`, `VLLM_GPU_DRIVER_RESERVE_MB=512`,
    `VLLM_GPU_MEMORY_LOG_INTERVAL=300`.
  - **80 workers:** Manh's batch size for his 700 tok/s claim on 4xL4 (coordinator), and the pool is 3.06M tokens,
    above the 2.5M threshold.
  - Server up after 440 s; B32 needs 235 s on this pod, and the difference is mostly the fresh CUDA-graph capture for
    96 sizes.
  - **Smoke (first 9 attempts, pod 0.40 h): pass.**
    - All 9 submitted, train_pass 9/9, 7/9 first attempts correct.
    - token_exact: 40 turns, 3 breaks, 0 rerenders / resyncs / parse fallbacks; 0 server errors.
    - 80 running, 0 waiting, 0 preemptions; KV 0.24.
    - Gen 499 tok/s mean since server up, against B32's 423 on this pod at the same hour (+18%); last window
      624 tok/s.
    - CPU: 5.5 of the 22.95 quota, no throttling.
    - Host memory not reclaimable: AnonPages 25.5 + Shmem 125.1 = 150.6 GiB, against B32's ~143 GiB on the old pod
      and Kaggle's 173 GiB limit.
  - Hourly interim tables: `l4/reports/interim_<UTC>.txt` and `l4/reports/manh_x2t_w80_<UTC>.md` (`l4/interim.sh`).
  - **Interim, pod 1.43 h (11:16 UTC):**
    - **Throughput:** 2.59M generated tokens, 587 gen tok/s mean since server up; hour 1 607 tok/s.
      - Same pod, B32 (`b32pod_x2t`): hour 1 474 tok/s, so **+28%**.
      - Old pod, B32 (`candBf_x2t`): 448 tok/s cumulative at 1.43 h (+31%).
    - **Load:** 80 running, KV 0.46, prefix hits 0.95-0.97, 0 preemptions, the host offload not used yet.
    - **Attempts:** 58 first attempts done (B32 old pod: 46), 37 correct (63.8%; B32 old pod 54.3% on its first 46,
      a different set of tasks).
    - **Token-exact:** 510 turns, 102 breaks, 2 resyncs, 2 parse fallbacks; B32 had 0 / 0 at the same point. Watched.
    - **Memory:** the container's unreclaimable memory (cgroup memory.stat rss + shmem) is 141.8 GiB, B32's peak
      level; Kaggle's limit is 173 GiB.
      - `/proc/meminfo` on Runpod is host-wide (it moved +7 GiB with no change in our processes), so from 11:17
        `cgmem.log` records the cgroup figure.
    - Token-exact detail: the 2 resyncs are 1 parse fallback each, in `010a9cf8` and `aa44c2e7`, both submitted.
      That is 2 of 510 turns (B32's 11 h run: 5 of 6,197).
  - **Interim, pod 2.39 h (12:16 UTC):**
    - **Throughput:** 4.84M tokens, 594 gen tok/s mean since server up; hours 0 / 1 / 2: 580 / 613 / 582.
      - B32, old pod: 3.53M, 434 tok/s cumulative; hours 0 / 1 / 2: 441 / 451 / 371.
      - B32's hour 2 is where its KV reached 0.89 and host restores started (1,269 tok/s).
      - Manh: KV 0.56, 0 host restores, 0 preemptions, 80 running.
    - **Attempts:** 94 first attempts (B32 old pod: 79, so +19%), 52 correct (55.3%; B32 53.8%); train_pass 85% vs
      86%; mean 23.6k out tokens and 11.5 turns per record vs 25.9k / 12.0.
    - **Memory:** cgroup unreclaimable 142.2 GiB, flat.
    - **Partial raw score:** 34.4 (113/344 pairs). The run is far from the deadline, so this is only a progress
      number.
  - **Interim, pod 3.34 h (13:10 UTC):**
    - **Throughput:** 6.67M tokens, 591 tok/s cumulative; hours 0-3: 580 / 613 / 586 / 569 tok/s.
      - Manh: KV 0.61, still 0 host restores and 0 preemptions.
      - B32, old pod, hour 3: 395 tok/s at KV 0.93, prefix hits 0.52, 3,661 tok/s restored from host.
    - **Pace:** 140 first-attempt tasks done; B32 reached 140 at pod 4.46 h, so a 1.33x time ratio. At equal task
      counts the ratio was 1.16-1.18 up to 80 tasks and grows once B32's KV fills.
    - **Paired accuracy** (`l4/paired_acc.py`, 94 tasks with every pair finished in both runs): first attempts 76.6
      vs 77.7, -1.1 [95% -8.3, +5.4], 6 better / 7 worse. A tie so far.
    - Stops: 131 submitted, 5 output_budget, 2 max_turns (B32 at 107 attempts: 104 / 3 / 0).

### L4 morning summary (pod 4.42 h, 14:15 UTC; the replica keeps running, ~21:20 UTC to the end)

**Throughput (generated tokens/s, all streams, vLLM metrics; this pod runs at 72 W, Kaggle at a 50 W cap, so raw pod
numbers are optimistic):**

| run | where | workers | KV pool | raw pod gen tok/s | per hour 0 / 1 / 2 / 3 / 4 | Kaggle-equivalent* |
|---|---|---|---|---|---|---|
| **Manh's fork** (`manh_x2t_w80`) | pod 1 | 80 | 3.06M | **588** (0-4.4 h) | 580 / 613 / 586 / 570 / 596 | **~516** |
| B32 (`b32pod_x2t`) | pod 1, **same pod** | 56 | 1.56M | 474 (0-1.1 h) | 473 / 474 | ~416 |
| B32 (`candBf_x2t`, the B32 replica) | old pod, cross-pod | 56 | 1.56M | 410 (0-4.4 h); 389 over 11 h | 441 / 451 / 385 / 359 / 430 | ~360 / ~342 |
| B32 + R1 (`b32r1_x2t`) | pod 2, cross-pod | 56 | 2.49M | ~470-485 by hour (pod 2's table) | 480 / 485 / 477 / 469 | ~420 |
| Manh's claim | ? | 80 | ~4M | 700 | | |

\*Kaggle-equivalent = raw x 0.94 / (1 + 0.45 x prefill share). The factors are the replica report's central Kaggle
clock (decode x0.94, prefill x1.45, measured with B32 on a real Kaggle run); the prefill share is 15.5% for the fork
and 15.7% for B32.

- **Same-pod (pod 1, hours 0-1): the fork is +22% over B32** (576 vs 474 tok/s cumulative at 1.0 h; hour 1 613 vs
  474, +29%). This is the cleanest comparison.
  - B32 itself ran ~7.5% faster on pod 1 than on the old pod (`l4/fidelity.md`), so cross-pod gaps vs `candBf_x2t`
    overstate the fork by about that much.
- **Cross-pod vs the B32 replica at the same pod hour (4.4 h):** +43% raw tok/s (588 vs 410), about +33% after the
  pod adjustment.
  - The gap grows with time: B32's KV hits 0.97 in hour 4 (156 preemptions, prefix hits 0.23, 6.2k tok/s restored
    from host). The fork sits at KV 0.68 with 0 preemptions and only starts using the host offload (63 tok/s).
- **Versus Manh's 700:** ~588 raw on real multi-turn tool traffic at batch 80 (84% of his number, hardware and
  workload of his measurement unknown), ~516 Kaggle-equivalent.

**Progress and score (paired by task, pod clock, `l4/paired_cut.sh`):**
- At pod 4.38 h the fork has 184 first-attempt tasks done; the B32 replica had 132 at that hour. The time ratio vs B32
  is 1.43 and rising: B32 reached 190 tasks at 6.30 h.
- Same-hour score: fork 59.2 vs B32 43.3 (**+15.9, sd 3.8**) vs B32+R1 46.9 (+12.2, sd 3.3), all cut at 4.38 h.
  - This is coverage: the fork finishes more attempts in the same time. It is not final accuracy.
- **Paired first-attempt accuracy on the 124 tasks both finished: fork 74.2 vs B32 77.4, -3.2 [95% -9.6, +2.5]**
  (6 better / 10 worse).
  - Within noise, but leaning negative; at 94 tasks it was -1.1.
  - This fits the fork's small numeric shift vs B32 (4090 agent: NLL +0.004/token, first token identical on 24/24
    prompts).
  - Only the 120-task accuracy A/B on the 48 GB box can settle it.
- Health: 0 server errors, 0 restarts, 0 preemptions; token-exact 4 resyncs / 4 parse fallbacks in 2,894 turns.
  Host memory (cgroup rss + shmem) is flat at 142.7 GiB, against Kaggle's 173 GiB limit. CPU 6 of the 23-CPU quota.

**Projection to Kaggle (`l4/project.py`; partial, pod 4.42 h of ~11):**
- **Coverage:** at the current time ratio (1.25-1.43 over the last hour of samples), all 240 first attempts are done
  by pod ~6.1-6.9 h (B32: 8.63 h).
- **Second attempts before the Kaggle-clock force deadline (pod 9.68 h):** ~125-180 at the fork's last-hour rate of
  45-50 tasks/h, against B32's 32 kept. Each adds B32's mean +0.020 points.
- **If first-attempt accuracy equals B32's** (77.01 attempt_1-only): Kaggle-clock **~79.5-80.6**, LB by the
  pre-registered rule (-1.2) **~78.3-79.4**, against B32's 77.43 / 76.2.
- **If the current paired accuracy gap (-3.2) holds:** Kaggle-clock ~76.3-77.4, LB **~75-76**, i.e. a tie with B32.
- One replica draw has an sd of ~4 task-weighted points; the projection holds the time ratio constant and uses B32's
  per-second-attempt gain.
- The run's own Kaggle-clock score at the end (~21:20 UTC) replaces this.

**Risks the replica does not cover:**
1. **Power and PCIe.**
   - Kaggle is 50 W and PCIe Gen3; this pod is 72 W and Gen4 x8.
   - The decode factor 0.94 was measured for B32. The fork adds DCP traffic every decode step (fp8 query all-gather
     plus partial exchange; the port analysis estimates ~30 ms/step at batch 80 on Gen3), so its Kaggle decode
     factor may be lower.
   - Only a Kaggle throughput window can measure it (the user's call; no Kaggle run was made).
2. **Accuracy** (above).
3. **Packaging:** a new offline wheel set (torch 2.14.0+cu130, flashinfer 0.7.2.post1, humming 0.1.18, the SM89
   wheel), patches 01-03, and new launch flags (DCP4, interleave 64, retention 512, embed UVA, pad 64,64).
4. **Startup:** the server came up in 440 s vs B32's 235 s (fresh CUDA-graph capture for 96 sizes), already inside
   the replica's clock.

## R1 (Humming pad on B32)

**Accuracy A/B (17:00 UTC; same 48 GB box, same B32 kit, settings and stack, only R1 differs): R1 does not cost
accuracy.** This supersedes the 14:05 warning below.

| arm (48 GB 4090, 120 eval tasks x 1 attempt) | task pass@1 | pairs | out tok/task | gen tok/s | no answer |
|---|---|---|---|---|---|
| `acc_r1`: B32 stack + R1 (humming_utils 4dd6392b, pad 64,64) | 67.8% | 112/172 | 41.9k | 694 | 8 |
| `acc_b32`: B32 stack (9ae7acb8) | 67.9% | 117/172 | 44.3k | 666 | 8 |
| `acc_manh3`: Manh's fork, same kit, same box | 77.1% | 133/172 | 39.2k | 551 | 4 |
| E052 A (B32-era code, B32 stack, 24 GB 4090, x36 gate 0.6) | 76.5% | 132/172 | 42.9k | 530 | 5 |

- **R1 vs B32, paired by task:** -0.1 pp [-8.3, +8.3], tasks won 16 / 16. Per pair 18 R1-only vs 23 B32-only right
  (172 pairs). Tokens x0.95 [0.87, 1.03]; gen tok/s +4%. Both arms: 0 errored records, token-exact resyncs 1-2,
  rerenders 0. KV 18.5M vs 17.6M tokens; this box is not KV-bound.
- **So the replica's 71.1-vs-76.0 gap on common pairs was draw / cross-pod noise, not R1.** The throughput gain
  stands, and B32 + R1 stays a B32.2 candidate. The replica's projection without an accuracy haircut (LB ~77.5 by the
  rule) is the one to use.
- **New anomaly: B32's stack scores ~68% on this box in both arms.** That is vs 77.1% for the fork on the same box
  and kit (B32 -9.2 pp [-17.4, -0.8], 10 / 22 tasks, p 0.05), and vs 76.5% for E052 A (-8.6 pp [-16.7, -0.4]).
  - Per pair vs the fork: 13 B32-only vs 29 fork-only right. Two independent B32-stack runs on this box both land
    at ~68%, so it looks systematic, not a bad draw.
  - Candidates:
    - the modded 48 GB cards with B32's FlashInfer sparse-MLA kernels;
    - a size-dependent bug at a 17-18M-token KV pool (an int32 offset overflow; Kaggle runs 1.55M, R1 2.49M);
    - x56 / gate 0.85 / offload at this size.
  - B32 on the L4 replicas is normal (first-attempt accuracy 78.9%).
  - So this box is a valid R1-vs-B32 pair but **not a reference for B32's absolute accuracy**. Before using it to
    rank stacks: the same B32 arm at util ~0.454 (L4-sized pool) on this box.

**Verdict at 14:05 UTC (replica at 5.0 h): +60% KV and +20-30% throughput in the KV-bound hours, but a possible
accuracy cost.** Not ready to ship without a same-pod accuracy A/B. On the pairs both runs finished by 5 h, R1
scores 71.1% vs B32's 76.0% (12 vs 23 discordant, sign p ~0.09; cross-pod, one draw each); details under L4 pod #2.

Server-level gate (09:00 UTC): passes the practical gate, so B32 + R1 went to a full replica on L4 pod #2 (`b32r1_x2t`).
It does not pass READY's strict gate (identical ids, or a first divergence of 200 tokens or more). No change to the
reduction order can pass that gate: the stack's own repeats diverge after 75-121 tokens on the long prompt.

**What R1 is:** B32's 16-file overlay with `r1/humming_utils_r1.py` (sha `4dd6392b`; B32's file is `9ae7acb8`). It
reads `DSV41_HUMMING_MOE_PAD="N,K"`; unset keeps B32's 256,128. With `64,64`, E128's TP4 expert shards (w13 N 1152,
w2 K 576) are no longer padded to 1280 / 640. Humming 0.1.12 accepts it, so no fallback pad was needed.

**Setup:**
- Measured on L4 pod #2 (Runpod 4x L4 at a 72 W limit; optimistic vs Kaggle, where the user reports a 50 W cap).
  The 48 GB 4090's model download was too slow.
- B32's exact server flags (`r1/scripts/serve_r1.sh`), util 0.938 (Kaggle's pool setting on pods), no block override.
- Script `r1/scripts/run_r1_arm.sh`, logs `r1/runs/`.

| per GPU / server | B32 padding (256,128) | R1 64,64 | change |
|---|---|---|---|
| Model loading took | 16.90 GiB | 15.64 GiB | **-1.26 GiB** |
| Available KV cache memory | 2.29 GiB | 3.62 GiB | +1.33 GiB |
| GPU KV cache size (tokens) | 1,556,850 (= the earlier pods / Kaggle 1.554M) | **2,486,783** | **+59.7%** |
| decode tok/s, B = 48 / 56 / 64 (4,096-token prompts, 60 s window) | 728.8 / 785.8 / 857.6 | 755.2 / 821.3 / 883.2 | **+3.6 / +4.5 / +3.0%** |
| inter-token median, B = 48 / 56 / 64 | 65.8 / 71.2 / 74.6 ms | 63.5 / 68.1 / 72.4 ms | |
| cold prefill (48 x 4,096 tokens) | 1,415 tok/s | 1,759 tok/s | +24% (one sample) |
| startup (server up) | 275 s (cold compile cache) | 135 s (warm) | not comparable |

**Numerics** (greedy, cold, 3 repeats per prompt):
- **E054's 5 short prompts:** the across-arm stats sit inside the noise floor. First divergence median 7 tokens both
  across arms and within B32; mean |d logprob| 0.076 across vs 0.095 within B32; token 0's argmax agrees on 4 of 5
  prompts. E054's "|d lp(token 0)| <= 0.5" check fails, but it fails within B32 too (27.7), because the top-5 sets
  differ.
- **One real 48,888-token tool transcript** (a B32 replica trace, harness tools):
  - token 0's argmax agrees 9/9;
  - mean |d logprob| over the shared prefix is 0.043 across, vs 0.019 within B32 and 0.011 within R1 (inside E054's
    2x + 0.01 limit);
  - first divergence: 15 tokens across, vs 121 within B32 and 75 within R1;
  - token-0 |d logprob| max 0.59 across vs 0.26 within.
- **Reading:** a small, systematic change in rounding (w2's K reduction is no longer padded), not a bug. Both arms
  write the same coherent analysis of the task, and a broken kernel fails every check at once.
- **Accuracy decides it.** The replica's commonly finished pairs lean against R1 (see L4 pod #2). That fits a small
  real numeric cost as well as draw noise. Next: a same-box 120-task x 1 attempt A/B, R1 vs B32 padding, as the
  acc48 arms did for the fork.

**What it would give Kaggle:** B32's settings at 1.55M KV tokens ran at KV mean 0.88 / max 0.99 with 487 preemptions.
The same util gives ~2.49M tokens with R1. Decode is also ~3-4% faster and nothing else changes. The Kaggle notebook
needs the patched `humming_utils.py` in the overlay and `DSV41_HUMMING_MOE_PAD=64,64` in the server env (env or
`ARC_SERVER_ENV_JSON`).

## L4 pod #2 (Runpod 5z3srm3108wcwl)

**Status 09:01 UTC:** replica `b32r1_x2t` started.
- Setup: same layout as pod #1.
  - `box_install.sh` + podprep, B32's 16-file overlay (16/16 sha), kit_b (77fe80a3 / 882c817f / 8fe36175 / auto-repair
    2292369c / 571f47eb).
  - E128 from HF at revision 9bce98c5: 131,090,911,045 B, in 9 min.
  - The R1 `humming_utils.py` swapped in. B32's original is kept at `/root/e062/humming_utils_b32.py`.
- Pod: 4x L4 23,034 MiB, 72 W, 2,040 MHz, PCIe Gen4 x16 (pod #1: x8). Topology PHB pairs, one NUMA node. EPYC 9254,
  nproc 48 (Runpod: 24 vCPU), Runpod RAM 250 GB (the earlier pods: 286 GB and nproc 128; candBp's pod: 250 GB / 48).
- Replica: `l4pod2/launch_replica.sh` (NAME=b32r1_x2t PAD=64,64 WORKERS=56).
  - Settings: candBf_x2t's exactly (x56, gate 0.85 + queue hold, multi-test 0.5, context limit, auto-repair,
    retention 512, offload 32 GiB, util 0.938), plus `DSV41_HUMMING_MOE_PAD=64,64`.
  - Workload `evalx2t_s2026`, toolkit deadlines (-600 s).
  - Workers stay at 56, not 64: one change at a time, so the replica pairs cleanly with candBf_x2t. 64 is the next arm
    if R1 holds.
- The server picked up R1: env `DSV41_HUMMING_MOE_PAD=64,64` (server_cmd.json), weights 15.64 GiB, KV 2,486,783
  tokens, up at +0.07 h (265 s).

**Interim reports:** `l4pod2/interim.sh`.
- It mirrors the run and writes `runs/b32r1_x2t/report_live.md`.
- It also cuts candBf_x2t (B32, same workload, earlier pod hegn) at the same solver hour with `l4pod2/cut_run.py`.
- Paired report: `runs/b32r1_x2t/report_vs_candBf_<h>h.md`. Score = the pod-clock submission rebuilt from finished
  records.

| solver hour | B32 + R1 score | B32 (candBf_x2t) at the same hour | paired diff (sd) | R1 gen tok/s that hour | B32 gen tok/s | R1 KV mean / max | B32 KV mean / max |
|---|---|---|---|---|---|---|---|
| 1.00 | 11.46 | 10.42 | +1.04 (1.3) | 480 | 441 | 0.30 / 0.46 | 0.44 / 0.72 |
| 2.01 | 26.32 | 25.42 | +0.90 (1.8) | 485 | 451 | 0.54 / 0.64 | 0.85 / 0.95 |
| 3.01 | 36.32 | 32.08 | **+4.24 (2.3)** | 477 | 385 | 0.69 / 0.75 | 0.88 / 0.96 |
| 4.00 | 44.86 | 41.39 | **+3.47 (2.7)** | 469 | 359 | 0.76 / 0.85 | 0.90 / 0.98 |
| 4.99 | 53.96 | 51.18 | +2.78 (2.9) | 468 | 384 | 0.84 / 0.90 | 0.93 / 1.00 |

- **Hour 4-5:** R1 ran 468 gen tok/s with 56 running, 0 preemptions and prefix hits 0.955. B32 ran 384 with 46
  running, 126 preemptions and hits 0.46 (KV 1.00).
  - Over the 5 h: generated tokens 8.34M vs 7.03M (+19%); attempts 180 vs 159; per-pair records 253 vs 229.
  - All numbers are raw pod tok/s at 72 W. The replica has not reached Kaggle's force deadline, so the report has no
    Kaggle-clock score yet. On the measured Kaggle twin factors (prefill x1.45 slower, decode x0.94) both runs slow
    alike.
- **Projection (pod #1's rule, `l4/project.py b32r1_x2t candBf_x2t 9.73`; force deadline at pod 9.73 h from the
  measured Kaggle twin clock):**
  - Inputs: 184 tasks' first attempts done at 5.06 h; time ratio vs B32 1.185; first attempts all done by 7.29 h
    (B32: 8.63 h); 88 second attempts kept (B32: 32).
  - Result: Kaggle-clock 78.7, **LB by the rule 77.5** (B32: 77.4 / 76.2).
  - That assumes B32's first-attempt accuracy. With the gap below (x 71.1/76.0), the attempt_1 score is 72.0 and the
    LB ~72.5.
  - So the expected LB is 72.5-77.5, depending on whether the accuracy gap is real.
- **Accuracy warning (5 h, pairs both runs finished, multi-test answers split per pair):**
  - 225 common pairs: R1 71.1% vs B32 76.0%; R1-only right 12, B32-only right 23 (sign p ~0.09).
  - At 3 h: 76.9 vs 80.8 (4 vs 9). At 4 h: 74.9 vs 78.4 (11 vs 17).
  - Pairs only R1 reached: 28, 71.4% right.
  - So the score lead comes from throughput, partly offset by lower accuracy on the same work. This is one draw
    per arm, cross-pod and cross-day; E052's same-config repeat differed by 2.5 pp [-11, +5.8]. It fits noise, but
    it is also what a small numeric cost would look like (the long-prompt probe sat at the edge of the noise
    floor).
- **Hour 2-3:** R1 ran 477 gen tok/s with 56 running and 0 preemptions. B32 ran 385 with 41 running (the gate
  held it at KV 0.88-0.96). That is +24% throughput; cumulative attempts 110 vs 95. Per-attempt accuracy of R1's
  finished attempts was 78.6% then (B32's full replica: 78.9% on first attempts); the paired view above supersedes
  this.
- **Hours 1-2:** B32 hit its 0.85 gate (52 running, max KV 0.95). R1 still had headroom (56 running, KV 0.64 max):
  +7.5% gen tok/s, 0 preemptions. Cumulative attempts 78 vs 69.
- **Cross-pod comparison:** pod #2 is not hegn (48 vs 128 visible CPUs, Gen4 x16). B32 itself ran ~10% faster on
  pod #1 (same CPU type as pod #2) than on hegn, so about 10 points of R1's +7-31% gen tok/s can be the pod. The
  KV-bound hours (+22-31%) still exceed that.
- **Memory:** cgroup max 232.8 GiB is Runpod's cgroup-v1 usage including page cache. RSS sum 147 GiB (double counts
  shared pages); R1 does not change host memory vs B32 (141 GiB PSS on hegn).

**Final (2026-10-28 20:02 UTC, pod 11.01 h; E064b agent): B32 + R1 `b32r1_x2t` finished normally.** The final
submission was written at the force point; the local score is 270/344 pairs; 0 preemptions; KV max 0.903 at 4.38 h.
- **Raw (pod clock):** 79.51 task-weighted (sd 3.2), against candBf_x2t (B32, old pod) 79.10 at the same hour.
- **Kaggle clock (central k 1.45, d 0.94): 77.92** (optimistic 78.19, pessimistic 77.29), against B32's 77.43:
  **+0.49** paired (sd 2.4, 240 tasks). That's a tie in score, though R1 finished 209/240 second attempts against
  B32's 106 (`runs/b32r1_x2t/report_live.md`, `report_vs_candBf_11.01h.md`).
- **LB projection:**
  - by the pre-registered rule (mean gap of V5 +0.6 and V6 −3.0): 76.7;
  - with B32's measured gap included (B32 LB 68.89 vs replica 77.43: −8.5; mean of the three: −3.6): **~74.3**;
  - if R1 has B32's own gap (−8.5): ~69.4.
- **Cross-pod caveat as before:** B32 ran ~7.5% faster on pod #1 than on the old pod.

## Accuracy of Manh's fork (48 GB 4090, Vast 58066982)

**Result (12:27 UTC): the fork's accuracy is the same as B32's stack within noise.**
- Task pass@1: 77.1% (pairs 133/172 = 77.3%), vs E052 arm A's 76.5% (132/172).
- Paired: +0.6 pp [-7.5, +8.5], task wins 16 / 15.
- Output tokens per task: 39.2k vs 42.9k (ratio 0.91 [0.84, 0.99]); turns 17.3 vs 19.3; no answer 4 vs 5; 0 errored
  records.
- Run: `acc_manh3`, 2.62 h, 551 gen tok/s at x56 on this 48 GB box.
- Harness side matches B32-era runs:
  - token-exact 2,076 turns, 624 retokenization breaks (30%, as E052 A 32% and candBf_x2t 31%), 2 resyncs, 0
    rerenders;
  - auto-repair 114 proposed / 114 applied / 92 retries ok.
- Analysis: `experiments/E050_effort_ab/analyze.py --a acc48/runs/acc_manh3/out --b ../E052_next_code_gate/runs/new_ar/out`.
  Arm "eff25" in its output is Manh's fork, "eff50" is E052 A. JSON: `acc48/acc_manh_vs_e052a.json`.
- **Caveats:**
  - E052 A ran on a 24 GB 4090 box with V6 solver settings (x36, gate 0.6) and the E052-era code (harness 13ebdb70,
    solver 24aaa376). B32's kit is that code plus off-by-default levers and solver fixes.
  - A same-box B32 arm (`acc_b32`, same kit and settings, B32's stack) started at 12:28 for a clean pair; it needs
    ~2.6 h.

**Same box, B32's stack (`acc_b32`, started 12:28; 70/120 tasks done at 14:05):**
- On those 70 tasks: fork 79.3% vs B32 77.9% (6 vs 5 tasks better): a tie.
- It finishes ~15:10 UTC. Re-score with the same `analyze.py` call (`--b acc48/runs/acc_b32/out`).
- A watcher on the box (`/root/e062/acc_b32_cleanup.sh`) stops the run's memlog (PID 30035) when it ends. The box
  is then idle; I don't stop it.

**Throughput, same box, same settings, KV not binding (48 GB):** the fork is slower per stream.
- Median gen tok/s over the first 10 status windows with >= 55 running: B32 762.6 vs fork 611.6 (-20%).
- Attempts done at ~0.72 h after start: 31 vs 23.
- So on 4090s the fork's gain can only come from KV capacity (DCP4: 23.6M vs 17.6M tokens here), not from decode
  speed. On Kaggle-sized L4 pools, capacity is what binds. Pod #1 measures that.

**Status 10:05 UTC:** run `acc_manh3` started (server up 10:02).

**Box:** Vast 58066982, 4x modded RTX 4090 48 GB, 450 W, kept running for the user's debugging.
- I stopped the image's own Qwen3.5 vLLM: `supervisorctl stop vllm model-ui`, then its orphaned process group by PGID.
- The box is not Kaggle-sized: util 0.92 of 48 GB, for accuracy only.

**Install:**
- E128 from HF at revision 9bce98c5: 131,090,911,045 B, in 65 min at ~35 MiB/s.
- Fork venv `/root/manh/venv` from the 4090 wheel (sha256 cf111141) + `freeze.txt` (torch 2.14.0+cu130,
  humming 0.1.18).
- Patches 01 + 02 of the branch (installed shas bbf92e10 / def7ad87 / 39016a73), plus
  `port/03_indexer_deepgemm_guard_sm89.diff` (installed `sparse_attn_indexer.py` 3113fa17).
- The B32 arm uses `/root/dsv41/venv`: box_install + podprep + the 16-file overlay (16/16 sha), B32's own
  `humming_utils.py`.

**Arms:** `acc48/launch_acc.sh`, the notebook pipeline (podrun + local_run).
- Kit and settings: B32's kit_b and solver settings (x56, max-num-seqs 64, gate 0.85 + queue hold, multi-test 0.5,
  context limit, auto-repair), ATTEMPTS=1, the 120 Kaggle eval tasks, deadlines at 30-33 h. The same pipeline as E052
  arm A (76.5 task pass@1).
- Fork server flags: pod #1's (`l4/launch_manh.sh`, 09:50 version):
  - DCP4, interleave 64, retention 512, embed_tokens UVA (`--cpu-offload-gb 1`), upstream offload 32 GiB with
    `offload_prompt_only:false`, FULL graphs (cap 64);
  - env `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64`, `VLLM_GPU_DRIVER_RESERVE_MB=512`.
  - Startup 740 s (JIT warmup). KV 23,623,307 tokens. Weights 14.8 GiB per GPU.

**Startup failures before this run:**
- `acc_manh`: every rank raised "Sparse Attention Indexer CUDA op requires DeepGEMM support". Fixed by patch 03.
- `acc_manh2` (pod #1's earlier flags, without interleave / retention 512 / UVA / pad): one rank (TP1) failed in
  profile_run inside Humming's dense MXFP8 kernel init (`wq_b`): `humming/config/config.py gemm_type_id`
  `AttributeError: 'str' object has no attribute 'value'`. The other 3 ranks hung, so I stopped the groups by PGID.
  - With the current flags the server started cleanly.
  - Not understood: one rank failing points to a race in Humming's JIT config path, not a deterministic bug. Watch
    for it on Kaggle-like starts.

## Manh's flags (E062b)

Manh's recommended server env (via the user, 2026-10-28 ~16:30 UTC), on top of the fork config the L4 replica used:
`VLLM_HOST_ALLREDUCE=1 VLLM_HOST_ALLREDUCE_MODE=60 VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS=16 VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB=36
VLLM_HOST_ALLREDUCE_LARGE_SHAPE=32x32x256 VLLM_HOST_ALLREDUCE_SHAPE=12x12x256 VLLM_HOST_ALLREDUCE_UNIT_KB=16
VLLM_DCP_ASYNC_QUERY_GATHER=1 VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64 VLLM_TRITON_SPARSE_INDEXER_LOGITS=1
VLLM_GPU_DRIVER_RESERVE_MB=512 VLLM_GPU_MEMORY_LOG_INTERVAL=60`.
- New vs pod #1's replica: the host all-reduce with its tuning, and `VLLM_TRITON_SPARSE_INDEXER_LOGITS`. The pad,
  driver reserve and async query gather were already on.
- **MODE 60 = 32 + 16 + 8 + 4 = PF_STOP | Q8 | SIMD_HOST | PACK.** MODE_Q8 is lossy: TP all-reduce inputs and sums
  cross the host link as int8 with an fp16 scale per 16 values (`host_all_reduce.py` lines 38-50). The indexer-logits
  flag is documented as the same selection as masking the dense logits; the prefetch blocks are loads only.
- Fork build in every E062b run: wheel b9ad670a67 (sha256 cf111141) + patches 01/02/03 over site-packages (installed
  shas model.py bbf92e10, chat_message.py def7ad87, types.py 39016a73, sparse_attn_indexer.py 3113fa17), the same
  as pod #1. That is branch `arc/e128-kaggle` 5f61bdc; the comm autotune (33dcc71, 786468e) is not installed.

**Plan (launched 16:18 UTC):**
- L4 pod #2 (`e062b/chain_pod2.sh`): after the B32+R1 replica ends, a same-pod A/B at x80 (max-num-seqs 96) on the
  240-task workload, 55-min windows (`e062b/ab_run_pod2.sh`): arm A `e062b_A_base` = pod #1's replica config, arm B
  `e062b_B_flags` = A + Manh's flags. Compared over the same minutes after server-up.
- **Changed 17:55 UTC (coordinator):** Manh pushed 5 commits to main (16:16-16:18, head 5e1bccd07c: expandable
  segments after the allocator cap (his OOM fix), host all-reduce mode bit 512 (warm AVX-512 reducers, csrc), humming
  MoE tuning config, page-cache drop, DSA decode indexer row shard). Arm B is now the local merge
  `arc/e128-kaggle-main5e1b` (95376a1 = his main + our `arc/e128-kaggle` 786468e; no conflicts, both the SM89 guard and
  his row shard kept; not pushed) with his flags at MODE 572 + `VLLM_DSA_DECODE_INDEXER_ROW_SHARD=1`
  (`ab_run_pod2.sh flags2`, `e062b/chain_pod2b.sh`). A new SM89 wheel is built on the 48 GB box's CPU
  (`e062b/build_manh2.sh`, MAX_JOBS 32, nice 15). Fallback if it isn't installed on pod #2 in time: the old build
  with MODE 60 (`e062b_B_flags`). Arm A is unchanged.
- **Changed 19:56 UTC (coordinator): pod #2 was reassigned to a B32-M replica.** `chain_pod2b.sh` was cancelled before
  arm A started (pod #2 keeps two unused fork venvs, /root/manh and /root/manh2, ~20 GB). The A/B moved to **pod #1**
  (`e062b/chain_pod1.sh`): after manh_x2t_w80 ends normally (final submission + report, mirrored to `l4/runs/`), arm A
  `e062b_A_base` then arm B `e062b_B_main5e1b`, the same 55-min windows at x80 (`ab_run_pod2.sh` with E062_PORT /
  E062_IP); /root/manh2/venv installed on pod #1 the same way (FREEZE=/root/manh/freeze_nov.txt). Pod #1 is stopped
  after the A/B.
- **Changed 20:00 UTC (coordinator, after acc_manhflags' -7.5 pp):** pod #1 runs `e062b/chain_pod1b.sh` with arms in the
  order A `e062b_A_base`, **C `e062b_C_lossless556`** (the new build 95376a1 + Manh's flags with the lossless MODE 556
  = 572 - 16, no Q8, + row shard; `ab_run_pod2.sh lossless`), then B `e062b_B_main5e1b` (MODE 572, lossy) only if it
  can end by ~00:45 UTC. A vs C is the main comparison.
- 48 GB box (`e062b/chain_48g.sh`), after acc_r1: teacher-forced numerics on E062's 24 prompts with the flags, then
  `acc_manhflags` (acc_manh3's exact env + the flags; 120 tasks x 1), then the same probe without the flags.
**Numerics with Manh's flags (48 GB box, 17:13-17:18 UTC):** the flags server started cleanly (host all-reduce logged
"enabled for bf16 messages up to 24.0 MiB ... mode 60, prefetch blocks 16", shapes 12x12x256 / 32x32x256 from 2 MiB;
KV 23,645,396 tokens vs acc_manh3's 23,623,307). Teacher-forced on E062's 24 prompts (B32's 256-token continuations,
4,912 tokens; `e062b/48g/tf_manhflags.json`):

| A vs B | mean abs dlogprob | p99 | mean NLL B - A |
|---|---|---|---|
| fork, pod #1 options (4090 box) vs **fork + Manh's flags** (48 GB box) | 0.0373 | 0.330 | **-0.0010** |
| B32 (4090 box) vs fork + Manh's flags | 0.0429 | 0.357 | +0.0031 |
| reference: fork defaults vs pod #1 options (same 4090 box) | 0.0364 | 0.287 | +0.0001 |
| reference: B32 vs B32 repeat (same 4090 box) | 0.0371 | 0.332 | +0.0002 |

So the q8 host all-reduce (MODE 60) does not shift teacher-forced numerics beyond run-to-run noise. **Same-box check
(20:03 UTC, `e062b/48g/tf_base.json`, fork without the flags on the 48 GB box): fork vs fork + flags 0.0375 mean abs
dlogprob, p99 0.323, NLL +0.0010**; and the fork across the two boxes, 4090 vs 48 GB, 0.0359, NLL -0.0020. All rows
sit at the B32 repeat's noise level. The 48 GB box chain is done; the box is idle (never stopped).

**Accuracy with Manh's flags (48 GB box, `acc_manhflags`, 17:18-19:55 UTC, 2.61 h):** acc_manh3's exact env, server
env + Manh's flags (MODE 60). Paired with acc_manh3 (`E050_effort_ab/analyze.py --a acc48/runs/acc_manhflags/out --b
acc48/runs/acc_manh3/out`; JSON `e062b/48g/acc_manhflags_vs_manh3.json`; the tool labels a "eff25", b "eff50"):

| | acc_manhflags (fork + Manh's flags) | acc_manh3 (fork) |
|---|---|---|
| task pass@1 | **69.6%** | 77.1% |
| pairs | 119/172 = 69.2% | 133/172 = 77.3% |
| output tokens per task (mean / median) | 43.9k / 37.0k | 39.2k / 33.0k |
| no answer | 7 | 4 |
| train pass | 84 | 89 |

- Paired: **-7.5 pp [95% -14.9, +0.0], task wins 9 / 21 (sign p = 0.04)**, tokens x1.12 [1.04, 1.21].
- Same-config noise for scale: run7 a0 vs a1 -2.5 pp [-11.0, +5.8], 15 / 16.
- One draw per arm, so not conclusive. But it is the direction expected from the lossy q8 all-reduce, and it is
  significant at p = 0.04. The teacher-forced probe shows no shift: it scores single 256-token continuations,
  whereas accuracy compounds over 17-18 turns and ~40k generated tokens.
- **Don't ship MODE 60 / 572 (they include Q8) without a second draw.** A lossless mode (e.g. 556 = 572 - 16) keeps
  the host path without int8.
- Throughput on this box (not Kaggle-like, reported only for completeness): 616 vs 551 gen tok/s.

- acc_manhflags is compared only with `acc_manh3` (fork vs fork, same box, identical memory settings: util 0.92,
  max-num-seqs 64, the same server args; only the server env differs). Not with acc_b32 / acc_r1: B32's stack scored
  ~68% on this box in both, against 76.5 (E052 A) on a 24 GB 4090, so the box or its huge pool affects B32 there.

### L4 A/B result (pod #1, 2026-10-28 20:20-23:10 UTC, 72 W pod; Kaggle is 50 W per the user, so all optimistic)

Same pod (Runpod ddj7xv53e5rgre), same 240-task workload start, B32 kit and settings, x80 (max-num-seqs 96), util 0.938,
KV 3,058,602 tokens in all three arms. 55-min windows; compared over the first 46 min after server-up
(`b4090/tps_ab.py`). Runs: `e062b/runs/e062b_{A_base,C_lossless556,B_main5e1b}/pod/`.

| arm | build | host all-reduce | server start | gen tok/s (46 min) | per 10 min | attempts done | pure-decode tok/s per stream (>= 70 running) | with prefill per stream (>= 60 running) |
|---|---|---|---|---|---|---|---|---|
| **A** base | b9ad670a67 + 01-03 | off (NCCL) | 515 s | **555** | 430 / 637 / 601 / 559 / 541 | 28 | 8.97 (~718 at 80 streams) | 8.24 |
| **C** lossless | merge 95376a1 | MODE 556 (no Q8) + row shard + Manh's tuning | 505 s | **590 (+6.2%)** | 473 / 664 / 636 / 597 / 571 | 31 | **9.85 (~788, +9.8%)** | **9.00 (+9.2%)** |
| B lossy | merge 95376a1 | MODE 572 (Q8) + row shard | 525 s | 590 (+6.3%) | 486 / 666 / 649 / 575 / 422* | 32 | 9.82 | 8.94 |

\* B was stopped by SIGTERM at ~40 min after server-up (coordinator: skip B), so its last bin is partial; over the
first 40 min it ties C.
- **Manh's flags (lossless variant) give +6% end-to-end and ~+10% decode per stream on L4.** The lossy q8 bit adds
  nothing measurable on top (B = C), and costs accuracy on the 48 GB box (-7.5 pp, p = 0.04). So the shippable choice
  is C's MODE 556.
- Pure-decode windows at >= 78 running are rare in a 55-min window (the early phase is dominated by new tasks'
  prefill), so decode is compared per stream at >= 70 running. Pod #1's 11 h replica (config A) had pure-decode
  windows at >= 78 running at median 688 / p90 704 tok/s, i.e. 8.6-8.8 per stream; C's 9.85 per stream would be
  ~790 at 80 streams. **Manh's 700 is exceeded for pure decode; end-to-end (with prefill and tool gaps) C runs at ~590
  in its first 46 min, against A's 555** (the replica's whole-run mean for A was ~596, so the early-window numbers
  are lower than full-run means for every arm).
- Startup is the same (505-525 s). No OOM, no restart, 0 preemptions in any arm. Host memory: the mem.log here only
  has host-wide numbers (Runpod containers see the host); the replica of config A peaked at 142.8 GiB unreclaimable
  in its cgroup (Kaggle 173 GiB), and C adds only the host all-reduce's pinned buffers (24 MiB per rank by default).

### Server-crash / restart test of the fork with B32's solver (pod #1, `e062b_crash`, 2026-10-28 23:14-23:5x UTC)

Setup: config A (b9ad670a67 + 01-03, B32-M's server flags: DCP4, interleave 64, retention 512, embed UVA, offload
32 GiB with offload_prompt_only false, graphs 96, max-num-seqs 96, pad 64,64, driver reserve 512, util 0.938), B32's
kit solver at x80, gate 0.85 + queue hold, the 240-task workload, stop-new at 40 min. For the test only
`ARC_RESTART_MIN_LEFT_S=300`: the default 2700 s makes `restart_allowed()` refuse a restart in a 40-min window. A first
launch without it was stopped before load (`e062b_crash_aborted`). Kill: `e062b/crash_killer.sh` on the pod, SIGKILL
to the solver's own `VLLM::EngineCore` (found by PID ancestry, never a name pattern) after 6 min of load with 80 running.

| event | time (UTC) | |
|---|---|---|
| server up (first, cold) | 23:22:21 | 510 s |
| SIGKILL EngineCore 272715, 80 running | 23:28:23 | |
| 80 in-flight attempts: APIError 500 (EngineDeadError) -> `retry` (1 of JOB_TRIES 2 used) | 23:28:24 | +1 s |
| watchdog: "server died (code 0)" (the API server exited) | 23:28:34 | +11 s |
| restart 1 started | 23:28:53 | +30 s |
| server up again (warm compile caches), KV 3,058,602 | 23:30:48 | +145 s (115 s start) |
| running 79-80, waiting 0; gen 76 -> 533 -> 630-663 tok/s | 23:31-23:34 | ~4 min to full speed (re-prefill: the prefix cache is empty after a restart) |

- **Attempts:** all 480 jobs accounted for (queued 397 + running 80 + done 3 at 23:34). **1 attempt lost:**
  aa8caa8d.0#0 launched at the kill instant, failed twice (APIError) and was recorded done with stop=error; its
  attempt #1 is still queued.
- **Weak spot (S21):** during the 2.3-min outage the launch loop kept starting attempts into the dead server: 232
  "Connection refused" re-queues, several attempts at 4/4 CONN_TRIES. A Kaggle restart of ~8-10 min would exhaust
  CONN_TRIES and then JOB_TRIES, and attempts would be lost in bulk. B32.1's S21 (`ARC_CONN_HOLD`: hold launches until
  `/v1/models` answers) fixes exactly this; B32's kit does not have it. B32.1's S20 (stall watchdog) is not needed for
  a crash, since the dead server is noticed through the process exit.
- No leftover GPU memory blocked the restart. One zombie (defunct python3, a child of local_run.py from before the
  kill) is harmless. The submission was written at 23:28:26 on the error path, then again at 23:35:43 when the first
  post-restart attempts finished (7b5033c1.0#0, one of the attempts re-queued 3 times, finished correct), and at the
  final SIGTERM write (23:36:37, valid). Logs: `e062b/runs/e062b_crash/pod/` (killer log beside the run).
- Pod #1 (ddj7xv53e5rgre) was stopped at ~23:40 UTC after all runs were copied off (`l4/runs/`, `e062b/runs/`,
  `e062b/pod1_extra/`; secret scan: 0 hits).

## Results (morning summary, 2026-10-28 ~14:30 UTC; the replicas are still running)
Details in the sections above. "Pod" numbers are raw Runpod 4xL4 at 72 W. Kaggle runs at a 50 W cap (user) and PCIe
Gen3, so the pods are optimistic. Kaggle-equivalent tok/s and the projections use the decode x0.94 / prefill x1.45
factors measured on a real Kaggle run with B32's server.

| | Manh's fork (x80) | B32 + R1 (x56) | B32 (x56) |
|---|---|---|---|
| KV pool, L4 at util 0.938 | 3,058,602 | 2,486,783 | 1,556,850 |
| gen tok/s, raw pod (Kaggle-equiv.) | 588 (~516), pod #1 | ~470-485 (~420), pod #2 | 474 same pod #1, hour 1; 389 over the 11 h replica |
| same-pod comparison | +22% vs B32 (pod #1, hour 1) | none (cross-pod) | - |
| first-attempt accuracy | 77.1 task pass@1 on 120 tasks (48 GB box) vs E052 A 76.5: +0.6 [-7.5, +8.5]; replica-paired -3.2 [-9.6, +2.5] (124 tasks, cross-pod) | 71.1 vs 76.0 per pair on 225 pairs (12 vs 23, p~0.09, cross-pod): same-box A/B `acc_r1` running, ~17:50 UTC | reference |
| projected LB (pre-registered rule) | ~78.3-79.4 if accuracy ties; ~75-76 at the replica's -3.2 | ~77.5 if accuracy ties; ~72.5 at the observed gap | 76.2 (submitted B32) |
| server start on L4 | 440 s | ~135-275 s | 235 s |

Findings:
- The fork's speed-up on L4 comes from KV capacity (DCP4 + pad 64,64 + embed UVA: 2x B32's pool, 0 preemptions at
  KV 0.68 when B32 is at 0.97 with preemptions), not per-step speed. With KV not binding (48 GB box), B32 decodes 20%
  faster (763 vs 612 tok/s). On the 4090 with no P2P across 2 sockets the fork ran at 0.51x B32 (DCP traffic).
- So Kaggle's slower PCIe (Gen3) and 50 W cap are the main risk for the fork: its DCP collectives on every decode step
  may make Kaggle's decode factor worse than B32's 0.94. Only a Kaggle throughput window (user approval) measures it.
- The fork's accuracy ties B32 on the controlled 120-task test; its numerics shift slightly (NLL +0.004/token).
- R1 is a one-file change to the stack already on Kaggle (+60% KV), but its accuracy is in doubt until `acc_r1`.
- Kaggle packaging for the fork: a new offline wheel set (torch 2.14, FlashInfer 0.7.2.post1, humming 0.1.18, the SM89
  fork wheel) + patches 01-03 + new flags (DCP4, interleave 64, retention 512 for DCP).

Open (patch 03 pushed to `arc/e128-kaggle` 5f61bdc and box 58063361 stopped, 2026-10-28 ~16:10 UTC; earlier both were blocked for the agents, user
action); the two L4 replicas' final scores (~20:00 and ~21:20 UTC); `acc_r1` (~17:50 UTC).

## Dynamic-count MoE: per-layer routed-expert counts on Manh's fork (2026-10-29)

For E066's per-layer pruned candidates (floor 72, cap 160, average 120 or 124; multiples of 8).
- **Branch `dynamic-count-moe`** in the clone `scratchpad/e062_port/manh_vllm`:
  - commit b3a3f642a9 on Manh's main 26c5906214, kept in parallel with his main (rebase as it moves);
  - local only until the user says to push;
  - patch copy: `port/04_per_layer_experts.patch`;
  - the model hunk alone (`port/04b_per_layer_model_only.diff`) also applies to the shipped wheel's 5e1bccd07c, so it
    can be a Kaggle runtime patch.
- **Why it's needed:**
  - V4.1 routes through the Triton `dsv4_topk` (adapted from SGLang) only for 128/256/384 experts. That is a Python
    whitelist; the kernel itself masks lanes.
  - Every other count falls to the CUDA `topk_softplus_sqrt`. Its switch accepts 1-512 powers of two plus 192, 320,
    384, 448 and 576, and errors ("Unsupported expert number") on 72-224.
- **The change:**
  - Both whitelists become 6..512.
  - The config gets `n_routed_experts_per_layer` (40 ints). `DeepseekV4MoE` sizes each layer's gate, biases,
    bias_vl and FusedMoE from it.
  - The expert-name mapping uses max(per-layer). EPLB is refused with mixed counts.
  - The GPTQ loader already yields whatever experts each `gptq/layer_LL.safetensors` holds.
- **Router cost** (`router_bench/bench_dsv4_topk.py`: standalone copy of the kernel, RTX 3090, CUDA graph of 40 calls
  = one step):
  - Decode (80 rows): 94 us/step at 128, 108-123 at 72/96, 100-115 at 136-224.
  - Prefill (2,048 rows): 190-277 us/step.
  - That is <0.1% of an L4 step, and per-layer counts cost at most ~+30 us/step.
  - Expert ids equal a torch reference at 72-224, image rows included.
- **Tests** (fork, not yet run on GPU):
  - `tests/kernels/moe/test_topk_softplus_sqrt.py::test_dsv4_topk_per_layer_expert_counts` and `::test_dsv4_topk_expert_count_limits`;
  - `tests/models/deepseek_v41/test_per_layer_experts.py`.
- **Pool check without the export:** `per_layer/make_dummy_ckpt.py` builds a shape-only checkpoint for
  `--load-format dummy`. It contains config.json and tokenizer files, plus per layer a sparse gptq file with e0's
  w1/w2/w3 q/s/z. The fork picks Humming GPTQ per layer from those files and reads the schema from e0's shapes.
  - Template: the shipped E128 config (`data/e066/e128_shipped/config.json`, with vision_config).
  - Shapes are confirmed against the C1 export: w1/w3 q [2304,320] s BF16 [2304,40] z [144,40]; w2 q [5120,144]
    s [5120,36] z [320,36].
  - Planned on a 4xL4 pod: C1 (uniform 128 + vision) vs floor72/avg120 vs floor72/avg124 at util 0.938 with
    B32-M's flags, measuring KV pool, peak, encoder profiling and a short decode run. Is 124 >= 3.1M?
- **Combined runtime patch for the shipped wheel (5e1bccd07c), for the real-export check and Kaggle:**
  `port/04c_rowchunk_plus_per_layer_model.diff`, which is the Engram row-chunk fix (a6c3ee8c2b) plus the per-layer
  hunks on `deepseek_v41/nvidia/model.py`. sha256 base ead1e087... -> row-chunk only b2a04fc2... -> combined
  75f3725adc5d131e.... `dsv4_topk.py` 7e1e101a -> ab32ea7e (measured on pod 2). Hash-guard both in the notebook
  cell.
- **Pool check result (dummy weights, util 0.938, vision on, B32-M's DCP4 flags; Kaggle-eq = pod x 1.0377):**

  | Layout | Pool (Kaggle-eq) |
  |---|---|
  | uniform 128 text-only (control; = B32-M exactly) | 3.17M |
  | C1 128 + vision | 2.93M |
  | f72 avg120, step 8 | 3.27M |
  | f72 avg124 | 2.88M (fails the 3.1M bar) |
  | **f72c152s16 avg120 (chosen, E72-152)** | **3.38M** |

  - Per-layer layouts carry 0.2-0.36 GiB/GPU of PyTorch caching-allocator waste (allocated - requested: 320 MiB vs
    116 MiB for uniform; the requested bytes scale exactly). Fewer distinct counts recover about a third.
- **Real E72-152 on pod 2:** sha256 of all 137 files verified; the image request is coherent.
  - KV pools: 3,259,129 with B32-M's DCP4; 1,336,475 with Manh's older cell (no DCP, util 0.92); **2,264,661 with his
    production cell on the v10 wheel** (no DCP, util 0.95, cpu-offload 0.35, f506's peak-memory trims).
  - His real launcher (arc2-2026-src v22) with the notebook's args: launch.json matches, contract passes. The chat path
    keeps prior reasoning and the prefix is stable (render check with tools).
  - --vision-max-images is capped at 32 by his launcher (64 is rejected).
  - No-DCP load (older cell, 120 workers): GPU prefix hits fell to ~3% after 16 min, but ~90% of prompt tokens came back
    from his CPU KV offload (12 GiB/rank); recompute stayed at 3.5-8%. The cost is running streams 120 -> 60.
  - Greedy parity v10 vs v8: within noise.
  - Runs: `per_layer/phase_h/` (README, scripts, tables).
- **Audits (3 Opus read-only reviewers, 2026-10-29):** notebook build, image chat path, per-layer vLLM patch. The fixes
  are in build_img_manh.py and pinned_img/; findings are listed in kaggle/dsv41_tools/CANDIDATE_E066_IMG.md.
- **Rebased:** branch `dynamic-count-moe` = 66cf71a0b7 on Manh's main 71dca8bc24 (old base kept as
  `dynamic-count-moe-26c5`). The patch for the v10 wheel (gf506afa210) is
  `kaggle/dsv41_tools/b32m_patches/dynamic_count_model_f506.diff` (029aa62e -> ad4ac0fd).
