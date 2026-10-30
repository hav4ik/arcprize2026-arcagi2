# Pod 3: C1 (E128 + vision) on Manh's production server + DCP4, image-harness load run (2026-10-29)

Part of the E066 model choice for tonight's Kaggle candidate.
- **DCP4 model A/B:** C1 here vs E72-152 on a 4th pod.
- **No-DCP vs DCP4:** E72-152 on pod 2.

Scripts: `../pod3_scripts/`. Raw tables and metrics: this folder.

## Pod and hardware
- **Pod:** Runpod 4xL4, pod 3 = o5r3oa6tgfpc8i, location IS, deployed by the user 18:40 UTC; SSH up 19:04.
- **GPUs:** L4 23,034 MiB, power limit 72 W.
- **Topology differs from pod 2:** only GPU1-GPU2 are a PHB pair; GPU0 and GPU3 are NODE to every other GPU. Pod 2 has
  pairs 0-1 and 2-3. Manh's `--gpu-ids 0,2,1,3` therefore maps logical ranks to different physical pairs on the two
  pods, so throughput is not strictly comparable across pods.
- **Host:** 128 CPUs (pod 2: 48), 503 GB RAM, /dev/shm 104 GB, 200 GB disk, CUDA 13.0.
- **Incident 19:05-19:10:** my bundle tar reset /root to uid 1000 / mode 775, which broke sshd StrictModes. The user
  fixed it from the web terminal (`chown root:root /root && chmod 700 /root`). Staging now uses
  `tar --no-same-owner --no-overwrite-dir` from inside root/.

## Stack (identical to the other pods)
- **Environment:** Manh's offline install (`install_b32m.sh`, all steps offline OK), Python 3.13.18.
  - vllm-wheels **v10**: 293 wheels identical to v8 plus vllm-0.32.1rc1.dev0+**gf506afa210** (sha256 f3c297dc...).
  - torch 2.14.0+cu130, numpy 2.4.6, scipy 1.18.1.
- **Our runtime patch** (`apply_patch_f506.sh`), base hashes checked first:
  - `model.py` 029aa62e -> **ad4ac0fd** (`b32m_patches/dynamic_count_model_f506.diff`; per-layer counts, row chunks
    native in his wheel);
  - `dsv4_topk.py` 7e1e101a -> **ab32ea7e** (router whitelist).
- **Manh's source:** arc2-2026-src **v22** (kagglehub path .../versions/22), copied to the work dir as his cell 2 does.
- **Model:** C1 = chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit, read-only HF key. **Full sha256 of all 137
  files OK** against `E066/results/c1_gptq_sha256_manifest.json`.
- **Server:** Manh's **v5 production cell 2m**. Run 1: verbatim through `run_prod_cell.py`. Run 2: as cell 14
  of the notebook itself, TP4, no DCP.
  - Cell settings: util 0.94, VLLM_GPU_DRIVER_RESERVE_MB 256, --context 163840, --concurrency 80 (max-num-seqs and
    graphs), MODE 556 lossless, retention 128, tool parser.
  - Env: MANH_VISION=1, MANH_VISION_MAX_IMAGES=32.
  - **Run 1 only, plus DCP4:** `--vllm-arg=--decode-context-parallel-size=4 --vllm-arg=--cp-kv-cache-interleave-size=64
    --vllm-arg=--prefix-cache-retention-interval=512`, inserted after his last flag, so 512 overrides his 128.
- **Solver:** kit = kaggle_kit_v5 + `pinned_img/`.
  - Files: solver **e96f7cb7** (external-server probe tolerates timeouts), harness 1d1a281e, grid_image 18ed78fb,
    train_check 8fe36175, auto-repair 2292369c / 571f47eb.
  - Launcher: pod 2's `launch_ext_e72c152.sh` with C1's dir and **80 workers**.
  - Settings: ARC_GRID_FORMAT=text+image, multi-test 0.5, context limit 163840, gate 0.85 + queue hold, auto-repair,
    workload evalx2t_s2026, replica deadlines.

## Run 1: C1 + DCP4, stopped before serving (19:35-19:57): slow Engram THP load on this host
- **Timeline:** the v5 cell 2m + DCP4 flags started at 19:35:56.
  - TP0 loaded in 357 s and TP1 in 860 s ("Model loading took 14.56 GiB").
  - TP2/TP3 (physical GPUs 1 and 3) were still loading after 21 min: 6 GB on the GPU, ~136% CPU each, no log after
    19:36:52 "Engram table offloaded to pinned host memory" (10.36 / 16.31 GiB per rank).
  - TP1 finished right after "Engram MADV_COLLAPSE failed; keeping existing pages".
- **Evidence** (`dcp4_slow_load/`, read-only, 19:56):
  - `/proc/vmstat` over 10 s: compact_migrate_scanned **+143.6M**, compact_stall +260, compact_fail +253,
    compact_success +7, thp_fault_fallback +63.
  - The workers were R (running), VmRSS 30.0 / 23.9 GB, wchan 0.
  - py-spy was refused (the container lacks SYS_PTRACE).
  - So the Engram THP collapse (`--engram-config.use_thp=true`) drives synchronous memory compaction that mostly
    fails.
  - THP settings are identical to pod 2 (enabled madvise, defrag madvise, khugepaged defrag 1).
  - Suspected cause: this host's memory state (fragmentation; 128 CPUs), possibly made worse by the C1 sha256 verify
    that ran during the first 3 min. **A Kaggle startup risk to watch:** Kaggle runs the same use_thp=true.
- **Stopped** per the coordinator at 19:57: TERM to run_variant and the cell runner, TERM then KILL to the server's
  process group 3475. No DCP4 throughput data.

## Run 2 (main job): the exact submission notebook, C1 copy, RERUN mode, 240-task x2t workload
- **Notebook:** `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-img-manh/dsv41-tools-submit-img-manh.ipynb` (sha
  8d1fac9e).
  **Build actually run: 8d1fac9e** (the C1 copy c31c5d5d).
  - The later rebuild c23eb5fb (fourth audit, GitHub main dd38b0555) changes only commit mode: new cell 3w, the 4b
    light-smoke checks, the 1b size pass. Its 2r also writes {WORK}/e066_vllm_patch.json.
  - So rerun behaviour is the same, and this run was not restarted. In the copy, ONLY cell 1b's model manifest is swapped to C1's 137 entries (sha c31c5d5d; diffed: only
  cell 6 differs). Cell 1b runs only in commit runs, so the rerun never uses it.
- **Runner** (`../pod3_scripts/nb_setup.sh`, `nb_restore_vllm.sh`, `nb_run.sh`):
  - papermill with --log-output, a Python 3.13.18 kernel (the solver venv + ipykernel), cwd /kaggle/working,
    KAGGLE_IS_COMPETITION_RERUN=1, VIRTUAL_ENV = the venv (so his `uv pip install` cells target it).
- **/kaggle/input mapped to local copies:**
  - datasets/nguyennguyen599/arc2-2026-src = v22;
  - datasets/nguyennguyen599/vllm-wheels = the downloaded v10 set (wheels/wheels/, 294 files);
  - datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz = kaggle_kit_v5's;
  - models/chankhavu/dsv41-flash-1017-arc-tool-img-e128-2bit/pytorch/default/1 = the C1 copy;
  - competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json = evalx2t_s2026 (240 tasks).
  - **Not mapped:** cu13-install, so his CUDA cell fails harmlessly in the rerun ("CUDA installation failed (rerun:
    continuing; the fallback stands)") and the pod keeps its CUDA 13.0; and unsloth-main (training only).
- **vLLM files restored** to the wheel's originals before the run (029aa62e / 7e1e101a), so cell 2r patches them as on
  Kaggle. The 21 GB uv cache was removed from /dev/shm so it doesn't inflate host-RAM numbers.
- **Cell results (rerun mode):**
  - cell 0: fallback written for 240 tasks, kit sha256 OK;
  - 0b: E046 CHECK OK, pins solver e96f7cb7 / harness 1d1a281e / grid_image 18ed78fb / train_check 8fe36175;
  - Manh's cells 1-5 ran (wheels copied to /dev/shm; uv audited the venv; the unsloth installs fail, harmless);
  - **2r: E066 VLLM PATCH PATCHED both files** (029aa62e -> ad4ac0fd, 7e1e101a -> ab32ea7e);
  - cell 2: import check OK on vllm 0.32.1rc1.dev0+gf506afa210;
  - **2m: model staging "ok after 0s wait"** (the files were page-cached), Manh's server launched (PID 7403).
- **Fidelity gaps vs Kaggle:**
  - 72 W pod GPUs (the user says Kaggle is 50 W);
  - 128 CPUs, and no 173 GiB cgroup (host RAM peak is logged by mem.log);
  - this pod's topology (only 1-2 PHB);
  - Kaggle-clock deadlines unshifted (the notebook's own: stop-new 10.83 h, force 11.17 h, kill 11.75 h).

## Run 2 startup timeline (per rank)
| UTC | Event |
|---|---|
| 19:58:0x | notebook start (papermill); cells 0-2 done within ~20 s |
| ~19:59:00 | cell 2m launches Manh's server (PID 7403) |
| 19:59:31-33 | first Engram table offloaded to pinned host memory: TP0 1.41 GiB, TP1 5.39, TP2 10.36, TP3 16.31 GiB per rank (TP0 also its second table, 1.41 GiB) |
| 20:00:01 | TP0 model loaded: 14.56 GiB, **36.6 s** |
| 20:04:00 | TP1 "Engram MADV_COLLAPSE failed; keeping existing pages" |
| 20:04:04 | TP1 model loaded, **279.9 s** |
| 20:07:51 | **FLAG: TP2 and TP3 still loading after 8.3 min.** All workers at 125-173% CPU. Host compaction counters since boot: compact_stall 13.1M, compact_fail 12.3M, compact_success 0.74M |
| 20:03:37 | TP1 second Engram table (5.39 GiB) |
| 20:13:10 | TP2 second Engram table (10.35 GiB) |
| 20:13:35 / 20:13:39 | TP2 MADV_COLLAPSE failed / TP2 model loaded, **854.6 s** |
| 20:19:31 | TP3 second Engram table (13.38 GiB) |
| 20:19:56 / 20:19:59 | TP3 MADV_COLLAPSE failed / TP3 model loaded, **1235.4 s** (20.6 min) |
| 20:20:23 | host all-reduce enabled; TileLang kernel compile, then shm_broadcast "no block in 60 s" while the engine profiles (normal during warm-up) |
| 20:23:04 | KV profile: **GPU KV cache size 1,878,692 tokens**, max concurrency 11.47x at 163,840 |
| 20:24:00 | **MANH SERVER UP after 1525 s** (+1544 s since notebook start; pod 2: 125 s) |
| 20:24:00 | cell 3 (solver) starts: 240 tasks, 344 pairs; deadlines stop-new 10.83 h, force 11.17 h, hard stop 11.50 h |

Each rank holds **two** Engram tables, so the pinned Engram totals per rank are TP0 2.82, TP1 10.78, TP2 20.71 and TP3 29.69 GiB.
Load time scales with that size, and the second table is the slow step: it is logged 4-20 min after the first, and on TP1-TP3 the collapse then fails.
This matches run 1; pod 2 had the server up in 125 s.

## Run 2 host RAM with `--engram-config.use_thp=true` (read-only)
Probe: `pod3_scripts/memprobe.py`, samples `/proc/meminfo`, `/proc/<pid>/status` and `/proc/<pid>/smaps_rollup`; raw output in `/root/e066/nbrun/memprobe.jsonl` on the pod.
Samples: one after all ranks loaded, one at server up, then every 10 min for an hour.

Per rank after load (20:22:52, before the KV pool and CPU offload buffers are allocated). Units are GiB.

| rank (pid) | Engram total | AnonHugePages | huge share of Engram | RssAnon | RssShmem | VmRSS | VmPTE MiB |
|---|---|---|---|---|---|---|---|
| TP0 (8550) | 2.82 | 2.82 | 100% | 5.24 | 1.97 | 8.14 | 20 |
| TP1 (8553) | 10.78 | 8.81 | 82% | 13.20 | 1.97 | 16.10 | 36 |
| TP2 (8552) | 20.71 | 13.73 | 66% | 23.13 | 1.97 | 26.03 | 56 |
| TP3 (8551) | 29.69 | 18.73 | 63% | 32.11 | 1.97 | 35.01 | 74 |
| API server (7403) | | 0 | | 0.88 | 0 | 1.34 | 6 |
| EngineCore (8246) | | 0 | | 0.68 | 0 | 1.13 | 6 |

Host at 20:22:52 (`/proc/meminfo`): AnonPages 88.5 GiB, AnonHugePages 44.2 GiB, Cached 129.2 GiB (mostly model files), Shmem 6.9 GiB, PageTables 0.27 GiB, Mlocked 0.03 GiB, MemAvailable 387.5 of 503.5 GiB.
The server tree (7 processes) has 87.8 GiB RSS.

RssAnon minus Engram is about 2.4 GiB on every rank, so the gap between ranks is the Engram tables.
TP0's tables are fully huge-page backed. On TP1-TP3, where MADV_COLLAPSE failed, the 4 KiB remainder is 2.0, 7.0 and 11.0 GiB.
The bytes are the same either way. Failed collapses cost load time and page-table size (TP3 VmPTE 74 MiB), not host RAM.

At server up every rank's RssAnon has grown by 12.6 GiB (the 12 GiB/rank CPU KV offload plus about 0.6 GiB). AnonHugePages has not changed, so the offload buffers are 4 KiB pages on every rank.
The server's anonymous memory is about 140 GiB. Cell 5's report checks against Kaggle's 173 GiB cgroup limit.

| UTC | tag | AnonPages | AnonHugePages | MemAvailable | tree RSS | TP0 / TP1 / TP2 / TP3 RssAnon | TP0 / TP1 / TP2 / TP3 AnonHuge |
|---|---|---|---|---|---|---|---|
| 20:22:52 | loaded, pre-up | 88.5 | 44.2 | 387.5 | 87.8 | 5.24 / 13.20 / 23.13 / 32.11 | 2.82 / 8.81 / 13.73 / 18.73 |
| 20:24:03 | server up (KV pool + CPU offload allocated) | 142.0 | 44.2 | 334.1 | 139.7 | 17.89 / 25.80 / 35.73 / 44.73 | 2.82 / 8.81 / 13.73 / 18.73 |
| 20:34:06 | +10 min | 147.4 | 44.2 | 328.8 | 143.5 | 17.95 / 25.93 / 35.81 / 44.81 | 2.82 / 8.81 / 13.73 / 18.73 |
| 20:44:09 | +20 min | 145.7 | 44.2 | 323.1 | 143.5 | same | same |
| 20:54:11 | +30 min | 147.6 | 44.2 | 328.5 | 143.8 | same | same |
| 21:04:14 | +40 min | 147.5 | 44.2 | 328.5 | 144.1 | same | same |
| 21:14:16 | +50 min | 147.2 | 44.2 | 329.0 | 144.1 | same | same |
| 21:24:19 | +60 min | 146.7 | 44.2 | 329.5 | 144.1 | same | same |

Columns: AnonPages, AnonHugePages, MemAvailable and tree RSS are host-wide or tree totals in GiB; the last two columns are per rank, GiB.
Under load, the API server's RssAnon is 2.0 GiB and EngineCore's grows from 0.9 to 4.0 GiB over the first 40 min, then holds.
Each worker's RssShmem is 2.07 GiB (a shared segment, counted once per process in the tree RSS). VmPTE is 46 / 62 / 82 / 100 MiB.
The page cache grows from 128 to 159 GiB, which is model files and is reclaimable.

**Steady host RAM of the server (from +40 min): 144.1 GiB tree RSS, 130.6 GiB of it anonymous** (workers 124.5, API server 2.0, EngineCore 4.0).
That is close to Manh's ~132 GiB with THP, so pod 3 keeps the saving although MADV_COLLAPSE failed on TP1-TP3.
The failed collapses cost load time (1525 s to server up) but no RAM.
The CPU KV offload buffers (12 GiB/rank) are not huge-page backed on any rank, so AnonHugePages stays at 44.2 GiB, all of it Engram.
Host AnonPages with the solver and sandboxes is ~147 GiB. The peak RSS of all pod processes (`mem.log`) is 149.2 GiB, against Kaggle's 173 GiB cgroup.

## Run 2 KV pool and contract
- GPU KV pool: 1,878,692 tokens (util 0.94, reserve 256 MiB, no DCP), plus 12 GiB/rank CPU offload.
- Contract: cells 2b (sandbox at full load) and 2c (API contract) are commit-only, so in RERUN mode papermill runs them as no-ops. This is the Kaggle rerun path; there is no contract result for run 2.
- Startup cost on Kaggle's clock: the solver started 26 min after notebook start, all of it in the server load. The cell-2m deadline is 7200 s and stop-new is 10.83 h.

## Run 2 load windows (5 min, then hourly)
`pod3_scripts/pull_nb.sh` copies the solver logs (no traces) and `metrics_scrape.txt`, then runs `windows.py`.
Minutes count from the first scrape sample (20:24, solver start). "1st done/corr" are first attempts (pair 0 records) finished by the window's end.

| win (min) | gen/s | prompt/s | GPU-hit % | CPU-rest % | recomp % | KV max | run/wait max | holds | preempt | 1st done | 1st corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25-30 | 673 | 6110 | 91.5 | 0.0 | 8.5 | 0.77 | 80/0 | 0 | 0 | 21 | 14 |
| 30-35 | 638 | 7712 | 92.3 | 0.0 | 7.7 | 0.83 | 80/0 | 0 | 0 | 28 | 20 |
| 35-40 | 661 | 7619 | 92.6 | 0.0 | 7.4 | 0.88 | 80/1 | 1 | 0 | 32 | 22 |
| 40-45 | 665 | 8342 | 94.4 | 0.0 | 5.6 | 0.89 | 78/2 | 1 | 0 | 40 | 30 |
| 45-50 | 640 | 8762 | 95.2 | 0.0 | 4.8 | 0.90 | 70/3 | 1 | 0 | 45 | 35 |
| 50-55 | 660 | 6352 | 95.4 | 0.0 | 4.6 | 0.91 | 64/2 | 1 | 0 | 50 | 40 |
| 55-60 | 599 | 7831 | 93.9 | 0.0 | 6.1 | 0.90 | 57/5 | 1 | 0 | 54 | 44 |

Checkpoint at 1 h (21:29):
- Server alive. server.log has 0 tracebacks, OOMs, engine deaths, HTTP 400s or image-count errors. The solver has 0 errors and 0 restarts.
- 56 first attempts done, 73 of 344 pairs answered. 6 KV-gate holds.
- GPUs at 21.6 of 23.0 GiB, 72 W, SM 1470-1545 MHz.
- CPU offload restored nothing so far: the GPU pool has held every reused prefix.

## Run 2 paired first-attempt accuracy vs the C1 eval on the E043 stack
Why: Manh's decode-only paths (DSA indexer row shard, Triton sparse indexer logits, host all-gather) engage only at >= 8 concurrent decodes, and no parity test exercised them.
Run 2 runs at 55-80 concurrent attempts, so it is the first accuracy evidence for them at scale.

- Script: `pod3_scripts/paired_c1.py`.
- Comparison: run 2's attempt 0 against `evals/results/tools_persistent_grid_lean2sub_dsv41_low_e066_c1x_img.jsonl` attempt 0, per pair. Multi-test records of the run are split per pair.
- Id mapping: x2t ids map to eval ids by `evalx2t_s2026_idmap.json` (`orig`). Copies are d8 + colour-permuted versions of the same task.
- Context: the eval served max-model-len 262,144 and run 2 serves 163,840; both are above C1's ~130k max context.

| snapshot | pairs | n (tasks) | run 2 | eval a0 | run-only | eval-only | McNemar exact p |
|---|---|---|---|---|---|---|---|
| 21:26 | originals | 36 (28) | 80.6% | 88.9% | 0 | 3 | 0.25 |
| 21:26 | copies -> original | 36 (27) | 80.6% | 83.3% | 2 | 3 | 1.0 |
| 21:26 | all | 72 (55) | 80.6% | 86.1% | 2 | 6 | 0.29 |

Noise control, the eval's own attempt 0 vs attempt 1:
- They disagree on 21 of 133 pairs (16%). On the eval tasks run 2 covered, it is a0-only 0, a1-only 4.
- Run 2 against the eval's two attempts: 56/62 on pairs the eval solved 2/2, 2/6 on 1/2, 0/4 on 0/2.

Run 2's misses where the eval was right:
- 6e453dd6 (orig and copy), 1ae2feb7 pair 2, 4c3d4a41 copy (2 pairs), 8f3a5a89.
- All are right-shape near misses with 1-36 cells wrong; no garbled output.
- 5 of the 6 pairs passed the train check; 8f3a5a89 did not (train_frac 0, after 4 auto-repairs).

Decode health of run 2:
- All 56 attempts submitted; 0 rescues.
- Tool errors per call 0.093 (eval 0.12).
- Auto-repair proposals in 8 of 56 attempts; the eval's harness records none.

Verdict at 21:26: no significant difference, and the gap is within the eval's own run-to-run noise. The common set is the easy early finishers, and n = 72 detects only large drops.

Final snapshot at 21:45, when run 2 was stopped for run 3: 71 first attempts, 98 pairs scored, 97 in common with the eval.

| snapshot | pairs | n (tasks) | run 2 | eval a0 | run-only | eval-only | McNemar exact p |
|---|---|---|---|---|---|---|---|
| 21:45 | originals | 48 (35) | 77.1% | 87.5% | 2 | 7 | 0.18 |
| 21:45 | copies -> original | 49 (35) | 73.5% | 79.6% | 2 | 5 | 0.45 |
| 21:45 | all | 97 (70) | 75.3% | 83.5% | 4 | 12 | 0.077 |
| 21:45 | tasks, all pairs right | 70 | 51 | 56 | 3 | 8 | 0.23 |

- Distinct eval tasks behind the discordant pairs: 7 lost (6e453dd6, 1ae2feb7, 4c3d4a41, 8f3a5a89, 16de56c4, d35bdbdc, 71e489b6) vs 3 won (b0039139, 3dc255db, 291dc1e1). Sign test p = 0.34.
  - 4c3d4a41 alone is 4 discordant pairs (orig + copy, 2 tests each), so the pair-level p overstates the evidence.
- Eval-defined subsets, which avoid run 2's early-finisher selection: tasks whose eval a0 and a1 both used < 20k / < 30k output tokens.
  - Run 2 finished 54% / 51% of them (the queue had not reached the rest).
  - Run-only 0 / 1 vs eval-only 6 / 9, p = 0.031 / 0.021, with the same correlation caveat.
- Noise: the eval's own a1 beats its a0 on these 97 pairs (89 vs 81; a0-only 2, a1-only 10).
- Lengths of run 2's wrong answers against the eval's right ones: some are much shorter (4cccce55 4k vs 22k / 14k, 6e453dd6 10k vs 22k / 14k), one is longer (4c3d4a41 40k vs 13k / 16k), the rest similar.

Verdict at stop: a consistent ~8 pp negative trend at the pair level. It is borderline and not significant at the task level, and a real accuracy cost of the new decode paths (or of the 163,840 context, or concurrency) is not ruled out.
Run 3 adds more pairs at the submitted configuration.

### Run 2 stop (21:46)
- `stop_run2.sh`: notebook and kernel killed 21:46:30, solver + 38 sandboxes 21:46:31, server group 7403 (SIGTERM) gone 21:46:33.
- At 21:46:51 nvidia-smi still listed pid 8551 (TP3, "[No data]", 21.6 GiB); the GPUs were empty at 21:47:00.
- So the rank with the largest pinned Engram tables takes ~20-25 s after its process exits to release the GPU.
- Run 2 data: `/root/e066/run2_working` and `/root/e066/nbrun/run2_attempts_final.jsonl` on the pod; local copy of the logs taken at 21:45.

## Run 3: new build with the notebook-side restart supervisor (cell 2s), TP-worker kill test, then the C1 replica
Plan from the coordinator, ~22:30 UTC on the go:
1. Stop run 2.
2. Run the new build (Manh's `--supervise 3`; status file passed to the solver as `ARC_EXTERNAL_SERVER_STATUS`; pinned solver S26, sha 832d3367). Rerun mode, C1, x2t workload, fresh `/kaggle/working`.
3. ~10 min after server up, SIGKILL one TP worker (TP2 or TP3) and time the supervisor restart.
4. Continue as the CV replica.

Scripts (`pod3_scripts/`, on the pod in `/root/e066/r3/`):

| Script | What it does |
|---|---|
| `stop_run2.sh [--dry]` | Discovers papermill, the kernel, the solver + its sandbox workers and Manh's server (launch.json pid), checking each by cmdline. Kills by process group / PID only: notebook first, then solver, then server (SIGTERM, 30 s, SIGKILL on the group), scraper, memory logger. Prints the GPU process list. Dry run at 21:33: papermill 6900 (pgid 6897), kernel 6902, solver 15554, 50 sandboxes, server 7403. |
| `dl_src.sh <version>` | Pinned kagglehub download of `nguyennguyen599/arc2-2026-src` (token over stdin). |
| `setup_run3.sh <nb> <kernel-metadata.json> <src dir> [kit]` | Archives run 2's `/kaggle/working`, `/tmp/dsv41` and `/tmp/arc_vision` to `/root/e066/run2_*` (not deleted). Maps the build's `model_sources` path to the local C1 copy and the src dir. Checks cell 1b's model manifest against C1's. Restores the vLLM files so cell 2r patches them as on Kaggle. |
| `nb_run3.sh` | papermill in rerun mode, as run 2. Log: `/root/e066/r3/run.log`. |
| `find_tp.sh` | Live workers by exact cmdline `VLLM::Worker_TP<r>` (padded with NULs), pgid, RSS, parent chain, GPU process list. |
| `kill_tp.sh <2\|3>` | SIGKILL exactly one worker, after checking that exactly one process has that exact cmdline. Logs the time. |
| `restart_sampler.py <out> 5 <status glob>` | Every 5 s: meminfo, RSS of all processes and of every vLLM process (pids), GPU compute apps, the status file(s). |
| `grep_restart.sh` | Expected lines: solver "server died", "(wait 1): started by the notebook", "server up after ... (external, start 1)", re-queue/retry; supervisor lines in the server dir; status files; status.json; attempt errors. |

Differences between the current build (out/nb_dsv41-tools-submit-img-manh, sha af16851c) and run 2's notebook:
- `model_sources` is `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit`, and cell 1b's manifest already equals C1's 137 entries, so no manifest swap is needed.
- ARC_WORKERS and Manh's `--max-num-seqs` are 64 (Manh's Kaggle value); run 2 used 80. Run 3 is therefore the replica of the submitted concurrency.
- The server root is `{WORK}/dsv41_logs` (server.log under `dsv41_logs/server/`).

GO from the coordinator (21:45): our notebook-side supervisor, not Manh's `--supervise`.
- Build `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-img-manh/` sha 55992cf482fd. The kit is unchanged (f24e624b, the same as kaggle_kit_v5's) and src is v22.
- Cell 2s: a daemon thread watches MANH_SERVER_PID, triggering on exit or on 25 min of no generated tokens while requests are running. It then:
  1. writes `{WORK}/dsv41_logs/e066_supervisor.json` with state "restarting";
  2. kills the server group and waits until the group and all GPU compute apps are gone;
  3. reruns the launcher with `--output-root {WORK}/dsv41_logs/r<N>`, waits for /v1/models, and writes "up".
  It restarts at most twice, and only with >= 60 min left before stop-new.
- Cell 3 passes ARC_EXTERNAL_SERVER_STATUS and ARC_MAX_SERVER_RESTARTS=3. The pinned solver is 324ea6ee (S26 + waits while the status is not "up"). Cell 3s stops the supervisor before cell 4.
- Server: util 0.90, `--kv-offload-gib-per-rank 10`, MANH_SEQS 64, MANH_CONTEXT 131072, `--mm-processor-cache-gb=2`, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900. ARC_WORKERS 64, ARC_CONTEXT_LIMIT 131072.

Run 3 timeline:

| UTC | Event |
|---|---|
| 21:47:24-37 | `setup_run3.sh`: run 2's working dirs archived. Model source `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pytorch/default/1` -> C1 copy; cell 1b manifest == C1's (137). vLLM files restored to 029aa62e / 7e1e101a. Run 2's `chankhavu/...` model mapping removed, so the model glob finds one dir as on Kaggle. |
| 21:47:58 | papermill start (RERUN). Samplers: `restart_sampler.py` every 5 s, `mem.log` every 60 s, `scrape_loop.sh` (re-attaches after restarts) |
| ~21:48:30 | cell 0b E046 CHECK OK (solver 324ea6ee); dpkg fails as in run 2 ("CUDA installation failed (rerun: continuing...)"); cell 2r PATCHED both files; cell 2m launches the server |
| 21:49:09-12 | first Engram tables offloaded (TP0 1.41, TP1 5.39, TP2 10.36, TP3 16.31 GiB) |
| 21:49:46 | TP0 and TP1 loaded in 43.5 s (both second tables in at once) |
| 21:50:26 | TP2 loaded in 84.0 s (MADV_COLLAPSE failed at 21:50:23) |
| 21:52:32 | TP3 loaded in 209.7 s (second table at 21:52:03, MADV_COLLAPSE failed at 21:52:29) |
| 21:55:41 | Available KV cache memory 3.76 GiB per GPU; **GPU KV cache size 1,453,240 tokens** (11.09x at 131,072). util 0.90 is equivalent to 0.8951 without CUDA graphs |
| 21:56:22 | CUDA graph pool 0.15 GiB actual vs 0.11 estimated |
| 21:56:31 | **MANH SERVER UP after 495 s** (+513 s since notebook start), max_model_len 131,072, API server pid 19687, EngineCore 20496, workers TP0-TP3 20827 / 20829 / 20830 / 20828. Supervisor: "watching pid 19687 (max 2 restarts, stall 1500s ...)" |
| 21:56:40 | GPUs 20.48-20.53 of 23.03 GiB used (**2.04-2.09 GiB free**); host 162 GiB used (free -g), 340 GiB available |

Run 3's load is 3x faster than run 2's (495 s vs 1525 s): TP3 took 210 s vs 1235 s. The collapse still fails on TP2 and TP3. The cause of the speed-up is not established; one possibility is the host's memory state (fragmentation) right after run 2 freed ~140 GiB.
The load time under THP compaction varies by a factor of 3 on the same pod; pod 2 took 125 s.

### Run 3 kill test (TP3 SIGKILL, 22:06:37 UTC, ~10 min after server up, 63 requests running)
Tools: `kill_tp.sh 3` (exactly one process, cmdline `VLLM::Worker_TP3`), `restart_sampler.py` every 5 s, `analyze_restart.py`, and the solver, supervisor and server logs.

| UTC | +s | Event |
|---|---|---|
| 22:06:37.4 | 0 | SIGKILL TP3 worker pid 20828 (pre-kill: API server 19687, EngineCore 20496, TP0-2 20827 / 20829 / 20830; supervisor state "up", start 0) |
| 22:06:49 | 12 | EngineCore notices the dead worker; the API server answers every in-flight request with HTTP 500 (server.log: `EngineDeadError: EngineCore encountered an issue`) |
| 22:06:50 | 13 | **The API server exits by itself**: "[shutdown] Process manager: force killing remaining process EngineCore pid 20496", force kills TP0-2, then "Finished server process [19687]" |
| 22:06:50 | 13 | supervisor: `[+1140s] E066 SUPERVISOR: server pid 19687 exited; restart 1/2 (631 min to stop-new)`; state file "restarting" |
| 22:06:51 | 14 | solver: `server died (code 1); tail: ...` and `external server http://127.0.0.1:18084 (wait 1): started by the notebook, not by the solver` |
| 22:06:57 | 20 | every old pid is gone and the nvidia-smi compute-app list is empty |
| 22:07:02 | 25 | relaunch: new vLLM pids (API server 27380, EngineCore 27811, workers 28147-28150); GPU contexts by 22:07:33 |
| 22:08:13-48 | | load: TP0 / TP1 40 s, TP2 48 s, TP3 75 s (MADV_COLLAPSE failed on TP3); KV pool 1,453,240 tokens, unchanged |
| 22:09:31 | 174 | supervisor: `[+1292s] E066 SUPERVISOR: start 1 up after 150s: pid 27380, log .../r1/server/server.log`; solver: `server up after 160s (external, start 1); GPU KV cache size: 1,453,240 tokens` |
| 22:09:31 | 174 | 64 x `re-queue <task>.0#0 after a server-side error (1): TurnError: APIConnectionError: Connection error.` |
| 22:10:21 | 224 | new server: 19 running + 45 waiting = 64, so every worker is back; generation resumed |

Host RAM (5 s samples):
- AnonPages went 136.7 GiB before the kill, 14.4 GiB at 22:06:57 (old engine gone), and peaked at 135.7 GiB at 22:09:52 (new engine).
- **The old and new engines never coexisted.** Minimum MemAvailable was 340 GiB.
- GPU after the restart: 20.48-20.53 GiB used, 2.04-2.09 GiB free per GPU, the same as at first start.

**Problems found (solver S26, pinned 324ea6ee; identical in the text build):**
1. The dying server's HTTP 500 is not classed as a server loss.
   - At 22:06:49-50 there are 64 lines `retry <task>.0#0: TurnError: APIError: The server had an error while processing your request.`
   - The HTTP body carries no "EngineDeadError" / "EngineCore encountered an issue"; those strings are only in server.log. So `CONN_ERRORS` misses it, and each in-flight attempt used one of its two `ARC_JOB_TRIES`. The retries were sent straight away, to the dying server.
   - They then failed again at 22:09:31 with `APIConnectionError: Connection error.` and were re-queued free.
   - The 64 jobs are now at tries = 1. A second death would end every attempt that was in flight at both deaths as an error.
2. **One attempt is lost**: f4c952f2.0#0.
   - Its 615 s first turn got the 500. The retry, 0.3 s later on the dying server, got `TurnError: InternalServerError: Error code: 500 - The server had an error while processing your request.`
   - With no try left it was finalized: `[5] f4c952f2.0#0 pred=no stop=error turns=1 out=0 train_pass=None cuts={} 0.5s`.
3. Cosmetic: the first window after the restart has negative rates (`server: -246.2 gen tok/s, ... uncached -1396.9 tok/s`) because the server's counters reset.

Suggested fix:
- Count the vLLM 500 text / InternalServerError as a server loss.
- Or more robustly: any error while the supervisor state is not "up", or right after a burst of 500s / "server died", is a free re-queue.
- And do not resubmit until the state is "up" again.

The supervisor itself (cell 2s) behaved as designed: death detected in < 1 s of the API server's exit, the group and GPU apps confirmed gone, a clean relaunch, and "up" in 150 s.

### Run 3 hourly checks (stability run after the kill test)
`pod3_scripts/pull_r3.sh` prints the supervisor events, server deaths, error records, host RAM, GPU memory and 10-min windows. `vram_series.py` gives the per-process GPU memory series from the 5 s sampler.

**22:57 (1 h after first server up, 48 min after the restart):**
- Stability: supervisor state "up", start 1 (pid 27380). No deaths beyond the deliberate kill. The r1 server.log has 0 tracebacks, 400s or 500s, and there are no new error records.
- Windows after the restart: 607-665 gen/s, 91-95% GPU prefix hits, 0% CPU-restored, KV 0.89-0.90, running 60-64, 2 KV-gate holds, 0 preemptions.
- Progress: 36 first attempts done, 28 correct; 47 of 344 pairs answered.
- Host: AnonPages 134.3 GiB; RSS of all processes 136.7 GiB; MemAvailable 342.5 GiB. Run peak AnonPages 139.1 GiB at 21:59.
- GPU memory per worker process (nvidia-smi compute apps, the same on all 4 GPUs): 20,438 MiB from server up to 22:20, 20,478 MiB at 22:30, 20,580 MiB at 22:50.
  - That is +142 MiB in 45 min, in steps. Board used went from 20,477-20,531 to 20,619-20,673 MiB, and free from 2,036-2,090 to 1,894-1,948 MiB.
  - The rate is ~0.09 GiB per 30 min; pod 2 sees ~0.13.

**23:57 (2.15 h elapsed):**
- Stability: supervisor "up", start 1. No deaths beyond the kill; the r1 server.log has 0 tracebacks, 400s or 500s.
- **Second lost attempt: f95b36dd.0#0.**
  - It is one of the 64 that used a try at the kill (`retry ... APIError ...` at 22:06:49, then `re-queue ... Connection error` at 22:09:31).
  - At 23:03:04 it ended as `[49] f95b36dd.0#0 pred=no stop=error turns=8 out=20811 ... 3212.1s`, error `TurnError: APITimeoutError: Request timed out.`
  - The harness streams, and the solver's client uses `httpx.Timeout(7200, connect=30, read=900)`, so turn 8 got no chunk for 900 s.
  - Turn 7 had completed; the trace (`traces/f95b36dd_p0_a0_r2.json.gz`) ends with the session-answer fallbacks, none accepted.
  - The likely cause is a > 15 min wait in vLLM's waiting queue: KV 0.89-0.93, 3-6 waiting, 0 preemptions.
  - `APITimeoutError` ("Request timed out.") is not in `CONN_ERRORS` (only "ReadTimeout" is), so a timeout costs a try even without a server death.
- Throughput: KV-bound from ~60 min, at 395-536 gen/s with running 30-48 and KV 0.88-0.93. CPU-restored was 37-42% of prompt tokens in the 23:18-23:38 windows. 0 preemptions.
- Progress: 79 first attempts done, 60 correct; 109 of 344 pairs answered; 48 running; 351 queued.
- Host: AnonPages 137.6 GiB; RSS of all processes 136.7 GiB; MemAvailable 339 GiB. The peak is still 139.1 GiB.
- GPU memory per worker process:

  | UTC | MiB |
  |---|---|
  | 22:10 | 20,438 |
  | 23:00 | 20,620 |
  | 23:10 | 20,700 |
  | 23:20 | 20,740 |
  | 23:40 | 20,760 |
  | 23:50 | 20,780 |
  | 23:56 | 20,820 |

  - That is +382 MiB in 1 h 46 min (~0.11 GiB per 30 min, in 20-100 MiB steps). Free per GPU is 1,654-1,708 MiB.
  - Extrapolated linearly, the headroom runs out near 9.6 h elapsed, before the 11.5 h hard stop.
