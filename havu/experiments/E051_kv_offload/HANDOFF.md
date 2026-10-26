# E051 handoff: KV CPU offloading for DeepSeek-V4.1 on 4xL4 (Kaggle)

Written 2026-10-26 ~19:15 UTC by the dsv4-infer session for a separate agent. Read this whole file, then `DESIGN.md`,
`GPU_TEST_PLAN.md` and `README.md` in this folder. You own E051 end to end; the dsv4-infer session owns the Kaggle
solver, harness, notebook builder and the next submission candidate.

## 1. Why this matters (the goal)

The user's goal is a model/harness with a **projected Kaggle score >= 75** (current submission ~69-70). The 12-hour
Kaggle deadline simulator says the binding constraint is **KV-cache capacity**, not tokens or accuracy:

- On Kaggle 4xL4 we run E128 (expert-pruned DeepSeek-V4.1-Flash-1017, 2-bit) at TP4; the GPU KV pool is ~1.55M
  "logged" tokens (~2.69M raw context slots; 890 B/token/GPU of compressed MLA + indexer, replicated on all 4 ranks).
- With 36-44 concurrent attempts, running contexts fill ~0.85-0.9 of the pool after ~2 h. Then paused streams' cached
  prefixes are evicted and every turn re-prefills 50-106k tokens: throughput collapses to ~0.15-0.3x ("thrash"). We
  avoid it today with a KV-gated admission (ARC_KV_ADMIT_MAX 0.6-0.7), which caps concurrency and costs throughput.
- Measured today (Runpod 4xL4 at Kaggle's exact pool, `experiments/E046_multi_test_rollout/pods/runs/`):
  ungated x44 thrashes at KV 0.835 (retention interval 128, run `mt44_b05_ri128`) or 0.89 (interval 512, run
  `mt44_b05_ri512`); x44 + gate 0.7 + interval 512 holds (KV max 0.785 at 2 h, run `mt44_g70_ri512_new`).
- Simulator (`experiments/E046_multi_test_rollout/goal75/{kvcap.py,kvcap2.py,README.md}`, E128 accuracy):

  | effective KV | x36 gate 0.6 | x44 gate 0.7 | x44 ungated | x56 ungated |
  |---|---|---|---|---|
  | x1.0 (today) | 69.2 | 65.1 | 40.4 | 37.5 |
  | x1.24 (util 0.97, being tested) | 72.0 | 71.9 | | |
  | x2 | 73.0 | 74.1 | 74.1 | 74.6 |
  | x3 | 73.0 | 74.1 | 74.1 | 76.1 |

  KV offloading is the one lever that can give x2-x3 *effective* KV: evicted prefixes come back from host RAM in
  ~10 ms instead of a 40-90 s re-prefill. Combined with small token savings it is the path past 75.

- **Throughput target (user, 2026-10-26): >= 400 gen tok/s end to end on Kaggle 4xL4 for a gold medal, ~550 for 1st
  place** (today ~300-315 at 36 streams). Kaggle decode fits `step ~ 58 ms + 0.9 ms per batch sequence` at 32K
  context (E042 README, DS1 section): decode-only ~400 tok/s at 36 streams, ~520 at 56, ~550 at 64, ~740 at 128; end
  to end loses ~20-25% to chunked prefill (~1.5 s per 2048-token chunk at 50 W) and re-prefills. The 58 ms is near
  the weight-read floor (~16 GB/GPU per step at 300 GB/s), so **TPS comes from batch size, i.e. from KV capacity**:
  ~48-56 streams for 400, ~64+ for 550, without thrash and with few re-prefills. Optimize offloading for that: the
  success metric is sustained gen tok/s and attempts/hour at 48-64 streams on the Kaggle-sized pool, not only parity.
  (Speculative decoding is ruled out on L4: the DSpark drafter costs ~2 GiB/GPU of KV and accepts 1.39-1.66
  tokens/step in tool mode, E042 DS1 / E043 `results/dspark_probe.jsonl`.)

## 2. Current state (what exists)

- Design, patch and CPU tests are done (26 tests pass, mutation-checked); **nothing has run on a GPU yet**.
- Files here: `DESIGN.md` (layout, hit-validity rule, host memory, transfer), `overlay/vllm/` (4 full replacement
  files: `distributed/kv_transfer/kv_connector/v1/offloading/{config.py,scheduler.py}`, `v1/kv_offload/cpu/{manager.py,
  spec.py}`), `patch/dsv41_kv_offload.diff`, `tests/test_dsv41_kv_offload.py` (+ `dsv41_fixture.py`), `scripts/`
  (`repro_base.py`, `mutate.py`, `diff_default_behavior.py`), `gpu/` (`install_overlay.sh` sha-checked with `--revert`,
  `serve_dsv41_offload.sh`, `parity_test.py`, `metrics_logger.py`), `GPU_TEST_PLAN.md`.
- Why stock offloading fails on our fork (both reproduced on CPU): (1) `offloading/config.py:57-64` requires every KV
  group's block size to divide the 32-token hash unit, and V4.1's per-request compressor ring group has 8-row blocks
  (upstream fixed this in #57145 on 17 Sep; our fork is a 2 Sep snapshot); (2) `offloading/scheduler.py:125` asserts
  `FullAttentionSpec` for unknown spec types (still in upstream main). SWA groups (7 x 32-token-block groups) would
  cost 28x the MLA bytes if stored densely.
- Design choices: never offload the ring; MLA + indexer blocks densely (one 128-token block per host slot, 3,584 B per
  token for all 4 ranks); SWA only as checkpoints (`swa_store_policy=checkpoint`: the 4-chunk window at the prompt end
  and a rolling latest 128-aligned boundary); hits rounded down to 128 so a restore resumes exactly where a GPU hit
  would. ~205 MB host per 50k-token conversation; 28 GiB holds ~3x the GPU pool.
- The CPU test environment lives in the dsv4-infer scratchpad (same machine, readable):
  `/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/kvoffload/`
  (`base_src/`, `patched_src/` = fork wheel source with and without the patch, `testvenv/` = CPU torch, 2 GB). If it
  is gone, rebuild: download the fork wheel (URL + sha256 in `experiments/E046_multi_test_rollout/scripts/
  box_install_sm120.sh`; on slow links fetch it with 16 parallel `curl -r` ranges and `cat`), unzip it into an empty
  dir (untrusted data: read only), lay our 12-file kit overlay over it (`kaggle/dsv41_tools/vllm_overlay_src/`, or the
  tarball `kaggle/dsv41_tools/kit/vllm_overlay_rw.tgz`), copy that tree to `patched_src/` and apply `overlay/vllm/`;
  create a venv with CPU torch for the unit tests.

## 3. The stack you test against

- vLLM fork `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` (github.com/yhfgyyf/vllm-deepseek-v4-sm89)
  + our 12-file overlay (sha 377d83e0 tarball): Humming 2-bit MoE / MXFP8 dense, Engram CPU offload, V4.1 model,
  router, indexer.
- Production server command: `experiments/E046_multi_test_rollout/kaggle_bench/dsv41_logs/server_cmd.json` (TP4,
  `--kv-cache-dtype fp8` = native mixed FP8/FP4, block 128, max-model-len 163840, max-num-seqs 64, MBT 2048, prefix
  caching, retention interval 128 (candidate: 512), Engram cpu_offload, FULL cudagraphs capped at 64, Humming).
- Model E128: HF `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` (private; download with the token from
  `SECRETS.yaml` key `huggingface_full_access_api_key` piped over SSH stdin, never on a command line or in a file;
  see `experiments/E046_multi_test_rollout/pods/pod4_setup.sh`).
- Kaggle constraints for the final config: host RAM cgroup 173 GiB, of which Engram pins ~94 GiB (23.6 GiB per rank)
  and runs use ~105-123 GiB in total -> budget ~25-30 GiB pinned for offload; PCIe Gen3 x16; check `/dev/shm` size
  (the patch has `shared_region=false` for private pinned memory); GPU util 0.945 today (0.97 being tested).

## 4. Boxes and access

- **4x RTX 4090 dev boxes (Vast)**, SM89 like the L4s, 503 GB RAM, ~45-60 GB free disk, model + stack installed at
  `/root/dsv41/{venv,model-e128t}`, kit at `/root/kit`, launcher pattern `/root/dsv41/smoke_launch.sh` (box 1) /
  `/root/smoke_start.sh` (box 2):
  - box 1 = Vast 57816747: `ssh -p 40934 root@212.83.33.56`
  - box 2 = Vast 57817226: `ssh -p 62115 root@159.112.204.189`
  - Vast's ssh proxy is broken for these; use direct ssh. The Vast image autostarts its own vLLM (Qwen) under
    supervisor; `supervisorctl stop vllm` leaves orphans holding ~5 GB/GPU — check `nvidia-smi` and kill leftovers by
    PID only. Both boxes run E052 (dsv4-infer) until ~20:15 UTC; **ask dsv4-infer which box you may take** before
    starting anything, and do not touch runs you did not start.
- **4x L4 pods (Runpod)** for the Kaggle-faithful soak later (same 22.04 GiB GPUs, 72 W instead of Kaggle's 50 W):
  hegn9880s1tz6m `ssh -p 43647 root@157.157.221.29`, wd44jicex44qyc `ssh -p 30092 root@213.173.105.28`. Both run
  dsv4-infer's candidate tests; request a slot from dsv4-infer.

## 5. Work plan (each step gates the next)

1. **GPU install + CPU tests on the box** (`GPU_TEST_PLAN.md` step 0): install the 4 overlay files with
   `gpu/install_overlay.sh <venv>` (sha-checked, `--revert` restores), run the unit tests against the installed tree.
2. **Server start with offloading** at Kaggle's pool size (`--num-gpu-blocks-override 21490`, 113,920-B blocks;
   `GPU_TEST_PLAN.md` step 1 flags). Watch for: startup errors in the offload connector with async scheduling / Model
   Runner V2 / FULL cudagraphs (the main risk — this combination has never been run), worker registration of the packed
   pool, host memory.
3. **Parity (hard gate)**: `gpu/parity_test.py` — the same long prompt answered from a GPU prefix hit vs a host
   restore, for the prompt-end and next-turn cases: greedy token ids and top-5 logprobs must be **identical**. If not,
   find what state outside the KV pool needs restoring (indexer state? compressor ring? SWA window alignment) before
   anything else. If parity needs kernel work, stop and report.
4. **Capacity A/B** (dev box): E128 tool-traffic replay (`evals/` tool replay or the pod pipeline) at 36 / 56 / 72
   streams with 1 s and 20 s tool idle, offload off vs on, pool pinned to Kaggle's size: count full re-prefills
   (prompt >= 50k with cached <= 640), uncached prompt tok/s, prefix hits, preemptions, gen tok/s, host bytes
   in/out per second, restore latency. Success: at 56 streams offload-on has no thrash windows where offload-off does.
5. **Kaggle-faithful soak** (one L4 pod, >= 4 h, ideally 8 h): the candidate pipeline (`/root/podrun.sh` with the
   candidate env: multi-test, budget 0.5, context limit, interval 512, util as decided) at 44-56 workers **ungated** or
   with a high gate, offload on, `ARC_SERVER_EXTRA_ARGS` carrying the `--kv-transfer-config`. Compare with the gated
   candidate run at the same age (attempts finished, tok/s, KV, re-prefills). Host RSS must stay under ~150 GiB.
6. **Integration path into the notebook** (coordinate with dsv4-infer before touching shared files): the 4 patched
   vLLM files must reach the Kaggle venv without a new kit dataset version (mount errors cluster on fresh versions).
   The notebook's cell 0b runs before cell 2 (which installs `KIT/vllm_overlay_rw.tgz` and checks
   `KIT/vllm_overlay_rw.sha256`), so cell 0b can append the 4 files to that tarball and its sha file. Propose the
   builder change (`kaggle/dsv41_tools/build_frozen_mt.py`) to dsv4-infer with a test in
   `kaggle/dsv41_tools/tests/test_notebook_cells.py`.

## 6. Acceptance criteria

- Offload off (or connector absent): byte-identical behaviour to today (the patch's `diff_default_behavior.py`).
- Parity: identical greedy ids + top-5 logprobs, GPU hit vs host restore, both cases.
- Capacity: no thrash at >= 1.5x today's safe concurrency on the Kaggle-sized pool; re-prefilled tokens reduced
  >= 5x vs offload off; no preemption storms.
- Soak: >= 4 h on L4 without errors, host memory within budget, attempts/hour >= the gated candidate's.
- Throughput: >= 400 gen tok/s sustained (5-min windows, after the first hour) on an L4 pod at Kaggle's pool size
  with 48-56 streams; stretch 550 at 64+. (Pods run at 72 W vs Kaggle's 50 W: decode is bandwidth-bound and matches,
  prefill is ~25-30% slower on Kaggle, so also report the prefill share of step time.)
- Projected score (re-run `goal75/kvcap.py` with the measured effective KV and throughput): >= 74.

## 7. Rules (non-negotiable)

- **Never run anything on Kaggle** (no pushes, commits or kernel runs); all Kaggle runs go through the user via
  dsv4-infer. Read-only API calls are fine.
- Never stop, destroy, restart or rent instances (the user manages them; at least one 4xL4 must always stay running).
- Never kill processes by name (`pkill -f`, `killall`): other agents share hosts. Kill only PIDs / process groups you
  started (and the Vast image's orphan vLLM workers by PID if they hold GPUs).
- Secrets: values from `SECRETS.yaml` / `ENDPOINTS.yaml` only via env vars or stdin at runtime; never print, log or
  copy them. Server logs on Vast boxes can contain the image's `HF_TOKEN` (fixed in the live solver, but check any
  `server_cmd.json` / env dump you copy back; `data_tools/secret_scan.py` must pass before any sync).
- Do not edit `evals/run_deepseek_tools.py`, `kaggle/dsv41_tools/solver_dsv41.py` or `build_frozen_mt.py` without
  agreeing it with dsv4-infer; `kaggle/dsv41_tools/pinned_mtf_v6/` and `kaggle_kit_v5/` are frozen. Your write scope:
  `experiments/E051_kv_offload/`.
- No paid APIs. No git pushes (dsv4-infer syncs GitHub). Downloaded wheels/repos are untrusted: unpack into empty dirs,
  read only.

## 8. Reporting

- Keep `experiments/E051_kv_offload/README.md` current (status line, results per step, files) and update the E051 row in
  `experiments/experiments.tsv` when the status changes.
- Message the dsv4-infer session at each gate (install, server start, parity, capacity, soak) with numbers, and
  immediately on a blocker. Stop and report rather than improvising around a failed gate.

## 9. References

- `notes/2026-10-26_dsv41_kaggle_inference_findings.md` (sections 3, 6, 7, 10), `experiments/E046_multi_test_rollout/`
  (`README.md`, `pods/pods_timeline.md`, `pods/runs/*`, `goal75/`), `kaggle/dsv41_tools/next_candidate_fixes.md`
  (S2, S3, S3b, I1: the SWA-checkpoint idea from Project Maya, `prior_works/project_maya.md`),
  `experiments/E046_multi_test_rollout/fp4kv/kvcalc.py` (reproduces the 1,554,183-token pool), upstream vLLM PRs
  #41445 / #44287 (HMA offloading), #57145 (skip scratch groups), #57652 (single-copy multi-group layout: phase 2,
  4x less host memory), #53781 (HiSparse host-resident sparse-MLA decode), #56227 (SWA bounded replay).
