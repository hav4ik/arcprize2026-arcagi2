# Phase H: the E066 image candidate (E72-152 avg120) on Manh's fork, pod 2 (2026-10-29)

Model: `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E72-152-avg120-2bit` @ c61b70728b.
- Downloaded with the read-only HF key over ssh stdin.
- All 137 files match `e72c152_gptq_sha256_manifest.json` (size and full sha256).
- Config: n_routed_experts 152, n_routed_experts_per_layer (sum 4,800; counts 72/88/104/120/136/152), vision tower.

## Image path checks
- **Real-weights sanity**, on a B32-M-flags server: 0934a4d8 (9 images) through the harness's own text_image_content.
  - Image tokens = 5,681, exactly the harness's count.
  - The reasoning is coherent about the grids ("light blue (8) rectangle ... hidden region by symmetry") and it calls run_python.
  - Scripts: `make_img_request.py`, `send_img_request.py`.
- **Pools** (pod):
  - B32-M flags (DCP4): 3,259,129 tokens.
  - Manh's config as published (no DCP, util 0.92, seqs 120): 1,336,475 tokens.
  - Manh's production cell (v10 wheel, util 0.95, seqs 80, cpu-offload 0.35): 2,264,661 tokens.

## Manh's real launcher (arc2-2026-src v22)
- **Blocker found:** `--vision-max-images 64` is rejected ("vision-max-images must be between 2 and 32", qwen38_vision.py:76). The notebook must use <= 32.
- Runner scripts:
  - `run_manh_cell.py`: the notebook's "Manh's server cell" verbatim.
  - `run_prod_cell.py`: his production cell 14 verbatim (`MANH_VISION=1`, `MANH_VISION_MAX_IMAGES=32`). It can append extra launcher args, e.g. DCP4.
- The `launch.json` receipt matches the intended command.
- **Contract** (`run_contract.py` = cells/b32m_contract.py unchanged): all pass on every Manh server.
  - /tokenize of the tool transcript = 377 tokens, as on B32-M's Kaggle smoke.
  - The fork defaults to tokenizer_mode deepseek_v41, so `--tokenizer-mode` is not needed.
- **Render check** (dsv4-pruning's render_check_chat.py, with `--tools` = tools_for(args)): every turn prefix-stable, prior reasoning kept in the history. That includes a multi-test 9-turn trace.

## Load run 1: v8 wheel, Manh's config (no DCP), 120 workers (`runs/ext_nodcp_w120`)

| min | gen/s | prompt/s | GPU-hit/s | CPU-restored/s | recompute/s | KV | run | wait | done |
|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 18 | 3,177 | 219 | 0 | 2,958 | 0.18 | 24 | 96 | 0 |
| 1-6 | 128 | 5,248 | 2,461 | 0 | 2,787 | 0.75 | 120 | 0 | 0 |
| 6-11 | 893 | 6,767 | 6,151 | 0 | 616 | 0.89 | 108 | 8 | 5 |
| 11-16 | 912 | 5,263 | 2,300 | 2,629 | 334 | 0.90 | 90 | 21 | 11 |
| 16-22 | 844 | 5,536 | 205 | 4,998 | 333 | 0.89 | 77 | 26 | 18 |
| 22-27 | 742 | 5,744 | 178 | 5,082 | 484 | 0.89 | 68 | 33 | 22 |
| 27-32 | 764 | 5,663 | 159 | 5,307 | 197 | 0.89 | 60 | 33 | 29 |

- **KV binds at the gate**, so GPU prefix hits collapse after ~16 min. SimpleCPUOffload then restores ~90% of prompt tokens from host; recompute stays 3.5-8%.
- 0 preemptions; running streams fall 120 → 60.
- **Harness checks** (35 finished attempts, 176 turns >= 2):
  - image span cached on turns >= 2: 176/176;
  - chat-path re-render break 6.8%;
  - image-preprocessing cache hits 85%;
  - max 11 images per prompt (mean 6.7).
- **Baseline** b32m_x2t (DCP4, E128 text, 80 workers): 575-636 gen/s, hits 0.89-0.96, KV 0.21-0.38, 0 waiting.

## v10 switch and parity
- vllm-wheels **v10** (gf506afa210, wheel sha f3c297dc), vllm reinstalled offline, no deps (`switch_v10.sh`).
- Patches: model.py 029aa62e → ad4ac0fd (dynamic_count_model_f506.diff); dsv4_topk 7e1e101a → ab32ea7e.
- **Greedy parity v8 vs v10** (`parity.py`):
  - e8686506 text, 2.9k tokens: identical 88 tokens.
  - 0934a4d8 image, 20.6k tokens: flips at token 7, the same near-tie flip as v10 vs v10.
  - 0934a4d8 text, 14.9k tokens: v8 vs v10 diverge at token 2 (" analyze" / " inspect"), while v10 vs v10 agree for 125 tokens. That fits reordered prefill sums on the long-prefill path; no large logprob gap.

## Load run 2: v10 + Manh's production cell (no DCP), 80 workers (`runs/prod_nodcp_w80`, from 19:35 UTC, ~90 min)
Stopped at ~25 min (19:35-20:02 UTC) when the priority moved to the exact-notebook stress test.
- KV pool: 2,264,661 tokens. Kaggle-equivalent ≈2.35M.
- Contract: all pass (/tokenize 377, tool call parsed, image +242, token ids, metrics).

| min | gen/s | prompt/s | GPU-hit/s | CPU-restored/s | recompute/s | GPU % | KV | run / wait | done |
|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 24 | 3,287 | 279 | 0 | 3,008 | 8.5 | 0.11 | 30 / 50 | 0 |
| 1-6 | 466 | 5,610 | 3,904 | 0 | 1,705 | 69.6 | 0.33 | 80 / 0 | 3 |
| 6-11 | 816 | 5,498 | 4,954 | 0 | 544 | 90.1 | 0.43 | 80 / 0 | 9 |
| 11-16 | 852 | 6,187 | 5,809 | 0 | 377 | 93.9 | 0.53 | 80 / 0 | 11 |
| 16-22 | 786 | 7,240 | 6,610 | 0 | 630 | 91.3 | 0.60 | 79 / 0 | 19 |

- 0 gate holds; KV never above 0.85 (still climbing, ~17k/stream); 0 preemptions; 0 server errors.
- First attempts: 23 done, 19/23 correct (82.6%), all submitted.

## Exact notebook end-to-end (from 20:05 UTC)
- **Notebook:** `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-img-manh/dsv41-tools-submit-img-manh.ipynb`, sha256 c23eb5fbb61c6c77 (GitHub main dd38b0555). That is Manh's v5 cells plus our cells settings, 0b, 1b, 2r, 2c, 3w and 4b.
- **Runner:** `scripts/run_nb.py` executes it with nbclient in a fresh Kaggle-like Python 3.13 venv (`/root/kenv`: ipykernel, nbclient, numpy, scipy, pandas, psutil, requests, pillow, pyyaml). It runs the cells in order, stops at the first error, logs a cell timeline, and saves the executed notebook with outputs.
- **/kaggle mapping** (`scripts/make_kaggle_tree.sh`): read-only inputs are symlinks to the verified pod copies.
  - Datasets: arc2-2026-src v22; cu13-install v2 (107 files, fetched per file, since the whole-dataset download returns 404); dsv41-tools-kit v5; unsloth-main v15.
  - vllm-wheels v10: v8's 294 top-level wheels plus v10's vllm-0.32.1rc1.dev0+gf506afa210 wheel. Per the API listing, the vllm wheel is the only file that differs.
  - Model: the verified E72-152 copy at `models/chankhavu/dsv41-flash-1017-arc-tool-img-e72-152-avg120-2bit/pytorch/default/1`.
  - Competition: `competitions/arc-prize-2026-arc-agi-2` holds the local data/ files. In the rerun phase, arc-agi_test_challenges.json is replaced by the 240-task evalx2t_s2026 workload.
  - A pass-through `sudo` shim (the pod is root without sudo). Manh's cell 1 really installs the CUDA 13 debs.
- **Phase 1:** commit mode, the light smoke. **Phase 2:** KAGGLE_IS_COMPETITION_RERUN=1, the full ~12 h CV replica.
- **cu13-install v2, 5 of 107 files:** the Kaggle API returns 404 for every file whose name holds an epoch colon (`%3a`): default-jre(-headless), libxcomposite1, libxtst6, libxxf86dga1. Encoded variants return 404 or 403.
  - Those 5 came from the Ubuntu 24.04 archive instead, via `apt-get download <pkg>=<exact version>` (signature-checked by apt).
  - Their names and byte sizes match the dataset listing. The other 102 files, including the empty `lock`, came from Kaggle.
- **Launcher** (`scripts/launch_nb.sh <commit|rerun> <name>`):
  - Fresh /tmp/dsv41, /tmp/arc_vision, /dev/shm/wheels and /kaggle/working.
  - `env -i`, with HOME=/root, LANG and VIRTUAL_ENV=/root/kenv; PATH = kenv, /usr/local/cuda/bin, then the system dirs. So Manh's `uv pip install` cells install into the kernel venv, as into the kernel's Python on Kaggle. `uv` is /usr/bin/uv 0.13.1.
  - Host memory is logged every 60 s (`mem.log`).
  - Outputs go to `/root/nbruns/<name>/`: `nb.timeline`, `nb.ipynb` (the executed notebook with every cell output) and `run_nb.log`.
- **First commit attempt (p1_commit, 20:10 UTC): the pod's disk filled during the install cells.** This is a pod problem, not a notebook problem.
  - Manh's cell 4 `uv pip install` wrote ~20 GB into the uv cache (cell 0 puts it in /tmp/dsv41/cache/uv) and copied another ~20 GB into the venv. uv copies rather than hardlinks here: link count 1.
  - At the same time, cell 1's `dpkg -i ./*.deb` installed new packages: Nsight Compute, Nsight Systems, OpenJDK.
  - With 46 GB free, the disk reached 0 B. dpkg failed with "No space left on device", and cell 4 raised "CUDA installation failed" (commit mode), so nothing else ran.
  - Fix: removed the run's /tmp/dsv41 and the regenerable uv cache (/root/b32m/uvcache, 21 GB). Also removed /root/b32m/venv (20 GB): the earlier non-notebook server env, rebuildable from /root/b32m/dl with `freeze.txt` and `install_b32m.sh`; the notebook installs the same stack into /root/kenv.
  - Recreated /root/kenv fresh (`kenv_fresh_freeze.txt`, 48 packages). That left 59 GB free.
  - Relaunched as p1b_commit at 20:13 UTC. Kaggle has its own large /tmp, and Manh's notebook installs fine there.
- **Second attempt (p1b_commit, 20:13 UTC): the pod image lacks system packages that Kaggle's image has.**
  - The install cells took only 42 s, because the wheels were already in page cache.
  - Then `dpkg -i ./*.deb` exited non-zero, and cell 4 raised "CUDA installation failed" again. The cause was dependency problems: openjdk-21 needs ca-certificates-java, java-common, libnss3, libgif7 and libpcsclite1; nsight-systems needs libegl1 and libopengl0; x11-utils needs libxaw7, libxft2, libxkbfile1, libxmu6, libxmuu1 and libxt6t64.
  - Kaggle's image must have these: Manh's own cell raises on that error, and his notebook passes on Kaggle.
  - Fix: `apt-get install -f --no-install-recommends` from the Ubuntu 24.04 sources only. That installed 16 packages, including libnspr4, libegl-mesa0 and libxpm4, and removed nothing. `dpkg --audit` is clean; nvcc reports 13.0. Log: `/root/e062pl/apt_deps.log`.
  - A private sources dir was needed: Manh's cuda-keyring deb adds `cuda-ubuntu2404-x86_64.list` (signed-by), which conflicts with the pod image's own `cuda.list`, so plain `apt-get` now fails with "Conflicting values set for option Signed-By".
- **The launcher now resets /root/kenv** from a pristine copy (`/root/kenv_fresh`, built at /root/kenv so its shebangs are right; freeze identical to `kenv_fresh_freeze.txt`).
- **Third attempt (p1c_commit, 20:16:35 UTC)** on c23eb5fb. Per the coordinator, phase 1 is then repeated on the v6-based rebuild, sha256 1e773bbc42e85f27, before phase 2. v6 changes one line of cell 2m: the server output root is `{WORK}/dsv41_logs`, so the server log is `dsv41_logs/server/server.log`.

### Phase 1 (commit mode, light smoke) results

| run | notebook | install cells (Manh 1-5) | server up | solver | notebook end | verdict |
|---|---|---|---|---|---|---|
| p1c_commit | c23eb5fb (v5-based) | 45 s | +325 s (cell 2m: 4.5 min) | 40 s, 4 attempts | +399 s | B32-M SMOKE OK |
| p1d_commit | 1e773bbc (v6-based) | 48 s | +328 s | 35 s, 4 attempts | +397 s | B32-M SMOKE OK |
| p1e_commit | **1707e4da** (v6-based, light-smoke deadlines 4200/4320/4380/4470 s) | 51 s | +341 s | 47 s, 4 attempts | +423 s | B32-M SMOKE OK |

Install times are short because the wheels and the uv cache are on local disk in page cache; Kaggle's FUSE inputs are slower.

Both runs:
- **2r:** `e066_vllm_patch.json`.
  - model.py: patched, 029aa62e → ad4ac0fd (base check: ead1e087 / 029aa62e).
  - dsv4_topk.py: patched, 7e1e101a → ab32ea7e. No .pyc to remove.
  - The vLLM is the wheel cell 4 installed into the kernel venv: 0.32.1rc1.dev0+gf506afa210, torch 2.14.0+cu130, humming 0.1.18.
- **Server:** Manh's cell 2m with util 0.94, TP4, no DCP, GPU order 0,2,1,3.
  - KV 2,171,851 tokens (×1.0377 Kaggle-equivalent ≈ 2.25M).
  - Model loading 47-48 s per rank, 13.88 GiB.
  - EngramConfig(cpu_offload=True, use_thp=True).
- **Contract (2c):** all pass, `fail: []`.
  - models dsv41 / 163840; reasoning_content and reasoning both carry thinking; /tokenize 377.
  - completion token ids 64; all 6 metrics present; chat_tool_calls 1; image prompt +242 tokens.
- **3w:** "E066: contract thread finished (waited 0 s)".
- **Solver:** 2 tasks / 3 pairs (25ff71a9 x2 pairs, 00576224), 4 attempts. All `stop=submitted`, all train_pass.
  - Both tasks are also public training tasks with identical test inputs. Scored against `arc-agi_training_solutions.json`, the p1c answers are correct on every pair (attempt_1).
  - 4b says "0 correct" because the commit-mode test file has no solutions.
- **The run ends right after the 2 tasks** (solver "done" at +0.11 h), not at the stop-new deadline (0.75 h). Then "submission.json valid" and "Manh's server stopped".
- **4b lines:**
  - `E066 vLLM patch: [('model.py', 'patched'), ('dsv4_topk.py', 'patched')]`
  - `E066 model files (names, sizes): {'files': 137, 'missing': [], 'size_mismatch': []}`
  - `E066 light smoke: 4 attempt records, 4 with an answer, 0 correct`
  - Also from 4b: INPUT INTEGRITY OK (every model file matches its sha256).
  - Minor: the 4b timeline reports `server_start_min: None`. The server is started by cell 2m, not the solver; `server_up_min` is 6.0.
- **Cell 5:** host MemAvailable min 343 GiB of 503 (no cgroup limit on the pod); GPU peak 21.4 of 23.0 GiB per card; power limit 72 W; tool sandboxes peak 16 processes, 1 GiB RSS.
- **v6 log layout (1e773bbc):** the server log is `dsv41_logs/server/server.log` next to the solver's files. `dsv41_logs/server` did not exist before the run (fresh /kaggle/working). The final verdict cell passes, with no missing-log raise. Cell 4's archive (`dsv41_traces.tar.gz`) holds only the traces; server.log stays in `dsv41_logs/server/`, which is part of /kaggle/working and so of the Kaggle output.
- **1707e4da:** the solver printed "deadlines: stop-new 1.17h, force 1.20h, hard stop 1.22h" and still ended at +0.12 h, right after the 2 tasks. Its answers are again correct on every pair.
- **Full cell outputs:** `runs/nb_p1*_commit_*/nbruns/<run>/nb.ipynb` (executed notebook) and `nb_outputs.txt` (plain text, as Kaggle's log shows them), plus `nb.timeline`.

### Host memory with use_thp=true (pod 2, p1d_commit; for the pod 3 comparison)

`scripts/mem_probe.py` is read-only: /proc/meminfo, per-process /proc/<pid>/status + smaps_rollup, /proc/vmstat. Samples are every 30 s here, because the light smoke's server lives only ~1.5 min. Phase 2 samples every 10 min for the first hour.

- **THP setting:** `transparent_hugepage/enabled` = madvise, defrag = madvise.
- **Engram tables per rank:** two tables, pinned host memory.

  | rank | Engram tables | total |
  |---|---|---|
  | TP0 | 1.41 + 1.41 GiB | 2.82 GiB |
  | TP1 | 5.39 + 5.39 GiB | 10.78 GiB |
  | TP2 | 10.36 + 10.35 GiB | 20.71 GiB |
  | TP3 | 16.31 + 13.38 GiB | 29.69 GiB |

  All ranks logged the offload within 3 s (20:26:21-24). Model loading took 47.4-47.7 s per rank.
- **No MADV_COLLAPSE lines in server.log, and compaction never ran:** compact_stall / fail / success were 25,755,912 / 24,715,573 / 1,040,339 both before and after the load (Δ 0). The huge pages came straight from free memory: this host had ~390 GiB available and mostly page cache.
- **Every byte of the Engram tables is THP-backed.** AnonHugePages per worker equals that rank's Engram size: 2.8 / 10.8 / 20.7 / 29.7 GiB (host total 64.2 GiB).

| min after ready | MemAvailable | AnonPages | AnonHugePages | Mlocked | Unevictable | Cached | Shmem | PageTables | server tree RSS |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 343.4 | 136.1 | 64.2 | 0.0 | 0.0 | 231.0 | 6.8 | 355 MiB | 139.6 |
| 0.5 | 343.8 | 135.9 | 64.2 | 0.0 | 0.0 | 231.1 | 6.9 | 352 MiB | 139.6 |
| 1 | 343.5 | 136.2 | 64.2 | 0.0 | 0.0 | 295.4 | 7.0 | 353 MiB | 140.2 |

GiB unless noted. "Cached" includes the 120 GiB model read by the warm stage and by the load.

| process | VmRSS | RssAnon | RssShmem | VmPTE | AnonHugePages |
|---|---|---|---|---|---|
| API server | 1.4 | 0.9 | 0.0 | 5 MiB | 0.0 |
| EngineCore | 1.3 | 0.9 | 0.0 | 5 MiB | 0.0 |
| Worker TP0 | 21.1 | 18.1 | 1.9 | 46 MiB | 2.8 |
| Worker TP1 | 29.0 | 26.0 | 1.9 | 62 MiB | 10.8 |
| Worker TP2 | 38.9 | 35.9 | 1.9 | 82 MiB | 20.7 |
| Worker TP3 | 47.9 | 44.9 | 1.9 | 100 MiB | 29.7 |

Per worker, RssAnon minus Engram is ~15.2 GiB on every rank. That is the 12 GiB/rank KV offload plus the rest. Mlocked stays 0: pinning through cudaHostRegister does not count as mlock here. The server tree totals ~140 GiB, of which 136 GiB is anon, at the start before any KV offload traffic.

## Phase 2: rerun mode, 240-task x2t CV replica (from 20:40:45 UTC)
- **Notebook:** 1707e4da22f2c6b2, KAGGLE_IS_COMPETITION_RERUN=1 (checked in the runner's environment).
- **Workload:** /kaggle/input/.../arc-agi_test_challenges.json, byte-identical to evalx2t_s2026_challenges.json (240 tasks, 344 pairs).
- **Clock:** the full ~12 h Kaggle clock from the solver's own deadlines.
- **Run dir:** `/root/nbruns/p2_rerun`; /kaggle/working = `/root/kwork_p2_rerun`.
- **Start:** server up at 20:46:19 (cell 2m: 4.6 min; Engram offload 20:42:30-32; model loading 48.1-48.2 s per rank). The commit-only cells (1b, 2b, 2c, 3w, 4b, 5, verdict) do nothing in rerun mode.
  - Solver: "attempting 240 tasks, 344 pairs; deadlines: stop-new 10.83h, force 11.17h, hard stop 11.50h".
- **Monitoring:**
  - A live grep of server.log, solver.log and the timeline for OOM, engine death, Traceback, ERROR, HTTP 4xx/5xx, context and image-count errors, watchdog lines and the notebook end.
  - `scripts/p2_report.sh` at each checkpoint: split table (`windows2.py`, pool 2,171,851), attempts scored against evalx2t_s2026_solutions (multi-test pred lists as in `first_correct.py`), gate holds, server errors, host RAM, GPU memory lines, disk.

### Checkpoint ~25 min (21:06 UTC)

| min | gen/s | prompt/s | GPU-hit/s | CPU-restored/s | recompute/s | GPU % | KV | run / wait | done |
|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 25 | 3,388 | 288 | 0 | 3,100 | 8.5 | 0.12 | 31 / 49 | 0 |
| 1-6 | 487 | 5,540 | 3,900 | 0 | 1,640 | 70.4 | 0.35 | 80 / 0 | 3 |
| 6-11 | 842 | 5,307 | 4,856 | 0 | 451 | 91.5 | 0.47 | 80 / 0 | 7 |
| 11-17 | 813 | 6,718 | 6,187 | 0 | 531 | 92.1 | 0.56 | 80 / 0 | 11 |

- **Attempts:** 21 first attempts, 15 correct (71%), all submitted, 0 errors.
- **Health:** 0 gate holds, 0 preemptions. Server: 0 HTTP 4xx/5xx, 0 ERROR, 0 Traceback.
- **GPU memory per rank:** 19.54 GiB allocated (peak 19.88), 20.28 reserved, 1.05-1.11 GiB free, 0 allocator retries, 0 OOMs.
- **Host:** used 186 GiB (peak 188), MemAvailable 317 GiB. Disk 16 GB free.
- **use_thp, phase 2's server:** compaction counters unchanged before vs after the load (Δ 0); 0 MADV_COLLAPSE lines.

  | UTC | AnonPages | AnonHugePages | server tree RSS | PageTables |
  |---|---|---|---|---|
  | 20:50 | 161.0 | 64.3 | 144.2 | 423 MiB |
  | 21:00 | 161.3 | 64.3 | 144.1 | 426 MiB |

  Per worker, VmRSS / AnonHugePages: TP0 21.3 / 2.8, TP1 29.3 / 10.8, TP2 39.2 / 20.7, TP3 48.2 / 29.7 GiB. EngineCore 3.8, API server 2.4 GiB.
- **Probe caveat:** the probe started before the launcher re-pointed /kaggle/working. Its "after_load" record and log lines came from phase 1's server.log; the values above were re-read from phase 2's own server.log (`mem_probe_after.jsonl`). The probe now ignores a server.log older than its own start.

### Checkpoint ~1 h (21:41 UTC; the solver has run 55 min)

| min | gen/s | prompt/s | GPU-hit/s | CPU-restored/s | recompute/s | GPU % | KV | run / wait | done |
|---|---|---|---|---|---|---|---|---|---|
| 17-22 | 775 | 7,299 | 6,642 | 0 | 657 | 91.0 | 0.62 | 79 / 0 | 22 |
| 22-27 | 802 | 7,636 | 7,086 | 0 | 550 | 92.8 | 0.70 | 80 / 0 | 28 |
| 27-33 | 731 | 10,002 | 9,172 | 0 | 830 | 91.7 | 0.75 | 80 / 0 | 38 |
| 33-38 | 776 | 8,547 | 7,889 | 0 | 658 | 92.3 | 0.81 | 80 / 0 | 47 |
| 38-43 | 768 | 10,592 | 9,935 | 0 | 657 | 93.8 | 0.86 | 80 / 0 | 55 |
| 44-49 | 787 | 10,689 | 10,112 | 0 | 577 | 94.6 | 0.90 | 76 / 0 | 62 |
| 49-54 | 800 | 10,626 | 10,265 | 0 | 361 | 96.6 | 0.88 | 68 / 0 | 71 |

- **Attempts:** 71 first attempts, 57 correct (80.3%), all submitted, 0 errors.
- **KV gate:** KV above 0.85 in 3 of 11 windows (from ~40 min, peak 0.90). 4 gate-hold lines, the last "KV gate: usage 0.881 (limit 0.85); holding new attempts at 72 running".
- **No CPU restores yet:** GPU prefix hits are 92-97%, recompute 3-9%.
- **Server health:** 0 preemptions, HTTP 4xx/5xx, ERROR or Traceback.
- **GPU memory per rank:** 19.54 GiB allocated (peak 19.96), 20.41 reserved, 0.92-0.97 GiB free, 0 OOMs.
- **Host:** used 170 GiB now (peak 188 at load). Disk 16 GB free.
- **use_thp memory, first hour:** compaction Δ 0 and no MADV_COLLAPSE throughout.

| UTC | MemAvailable | AnonPages | AnonHugePages | Cached | Shmem | PageTables | server tree RSS |
|---|---|---|---|---|---|---|---|
| 20:50 | 317.8 | 161.0 | 64.3 | 261.3 | 7.2 | 423 MiB | 144.2 |
| 21:00 | 317.5 | 161.3 | 64.3 | 261.2 | 7.2 | 426 MiB | 144.1 |
| 21:10 | 317.1 | 161.8 | 64.3 | 186.1 | 7.2 | 426 MiB | 144.6 |
| 21:20 | 316.6 | 162.3 | 64.3 | 186.1 | 7.2 | 428 MiB | 145.0 |
| 21:30 | 316.0 | 163.0 | 64.3 | 186.1 | 7.2 | 430 MiB | 145.6 |

| process (21:30) | VmRSS | RssAnon | RssShmem | VmPTE | AnonHugePages |
|---|---|---|---|---|---|
| API server | 2.4 | 2.0 | 0.0 | 8 MiB | 0.0 |
| EngineCore | 5.2 | 4.7 | 0.1 | 13 MiB | 0.0 |
| Worker TP0 | 21.3 | 18.1 | 2.1 | 46 MiB | 2.8 |
| Worker TP1 | 29.3 | 26.1 | 2.1 | 62 MiB | 10.8 |
| Worker TP2 | 39.2 | 36.0 | 2.1 | 82 MiB | 20.7 |
| Worker TP3 | 48.2 | 45.0 | 2.1 | 100 MiB | 29.7 |

### Phase 2 stopped at 21:53:30 UTC (coordinator: the decode-numerics A/B replaced it)
- **Run time:** 1.21 h of the notebook, 1.12 h of the solver. Stopped by process group: kernel, solver, server and runner.
- **First attempts:** 82 done, 65 correct (79.3%), all submitted. Errors: 0 attempt errors; 0 HTTP 4xx/5xx; 0 ERROR or Traceback lines; 0 preemptions; no context or image-count errors.
- **KV gate:** 7 gate-hold lines. KV stayed at 0.86-0.90 from minute ~40. Running fell to 56-63 with 3-5 waiting (gate plus ARC_ADMIT_MAX_WAITING). GPU prefix hits 97-98%; still no CPU restores.
- **Trends:**
  - GPU memory per rank: 0.88 GiB free at the end (1.11 at 25 min); peak allocated 20.03 GiB; 0 OOMs.
  - Host: used 170 GiB, flat after load; peak 188 at load. Process RSS 150 GiB.
  - Server tree RSS 144.2 → 145.6 GiB in the first hour.
- **Archive:** `runs/nb_p2_rerun_1707e4da/` (executed-notebook state, timeline, solver/server logs, attempts, traces, mem probe).

## Decode-vs-prefill logprob A/B (coordinator, 21:52 UTC; E72-152, because E128 was not on the pod)
- **Why:** Manh's decode-only paths engage only at >= 8 concurrent decodes and had never been checked numerically. They are VLLM_DSA_DECODE_INDEXER_ROW_SHARD, VLLM_TRITON_SPARSE_INDEXER_LOGITS and VLLM_TP_HOST_GATHER.
- **Server:** the b32m-fix notebook's (sha 3cf579f62484) cell 2m, run as-is by `scripts/run_srv.py`, with Manh's dsv41_vllm.py from arc2-2026-src v22 and his MANH_SERVER_ENV, including VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900.
  - Settings: util 0.90, --kv-offload-gib-per-rank 10, --concurrency 64, --context 163840, --max-num-batched-tokens 2048, MANH_VISION=0 (--language-model-only), TP4, GPU order 0,2,1,3.
  - vLLM: gf506afa210 + 2r, as needed for E72-152's per-layer expert counts.
  - KV 1,909,056 tokens; GPU 2.47-2.54 GiB free per rank.
- **Prompts:** 32 b32m_x2t (E128 text harness) transcripts, each cut after a tool turn and rendered with /tokenize (messages + tools, add_generation_prompt). 32.3k-44.9k tokens, 1.35M total (`ab_decode/prompts_index.json`).
- **Passes** (`scripts/ab_logprob.py`):
  - Decode: all 32 at once, temperature 0, max_tokens 256, logprobs 1, token ids. The server ran 13-14 requests (about 13 decoding plus 1 prefill chunk) for most of the pass, so >= 8 concurrent decodes was in effect.
  - Prefill: prompt + generated[:k], max_tokens 1, logprobs 20, every 4th position, concurrency 4 (< 8 decodes possible). The decode token was always in the prefill top-20.

| comparison | n | mean \|d\| | p99 | max | \|d\| > 0.5 | mean d | prefill argmax != decode token |
|---|---|---|---|---|---|---|---|
| A: flags ON (Manh's production env), decode (conc 32) vs prefill | 2,040 | 0.0362 | 0.277 | 1.68 | 4 | +0.009 | 56 |
| B: ROW_SHARD=0, SPARSE_INDEXER_LOGITS=0, TP_HOST_GATHER=0, decode vs prefill | 2,038 | 0.0347 | 0.278 | 1.86 | 5 | +0.007 | 62 |
| decode A vs decode B, common greedy prefixes | 1,554 | 0.0216 | 0.158 | 0.30 | 0 | +0.002 | n/a |
| control on A: decode conc 1 vs prefill (8 prompts, common prefixes) | 97 | 0.0156 | 0.170 | 0.17 | 0 | -0.001 | n/a |
| control on A: decode conc 32 vs decode conc 1 | 374 | 0.0230 | 0.273 | 0.46 | 0 | -0.002 | n/a |

- **Verdict:** the decode paths are numerically clean. Flags on and off are indistinguishable, at fp-noise level, with no systematic offset; per prompt, mean |d| is 0.02-0.06 on both servers.
- **B's flags were really off:** the workers' /proc/<pid>/environ shows ROW_SHARD=0, SPARSE_INDEXER_LOGITS=0, TP_HOST_GATHER=0 and AR_OVERLAP_CHUNKS=2.
- **(f) AR_OVERLAP_CHUNKS=0 was skipped:** it is a prefill-only path, so a wrong overlap would have shown as a decode-vs-prefill gap on both servers.
- **Files:** `ab_decode/` (dec*/pre* JSON, server logs and launch.json). Scripts: `scripts/ab_logprob.py`, `ab_control.py`, `run_srv.py`.

## B32-M fix candidate on E128 (text): commit smoke + rerun replica (from ~23:45 UTC)
- **Candidate notebook:** `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/dsv41-tools-submit-b32m-fix.ipynb`, sha256 **bec71fb1e3f1**.
  - Built on Manh's v6 server cell (no --supervise) with our cell 2s supervisor.
  - Solver 1ea2b25d, ARC_CONN_TRIES 12. Settings: util 0.90, offload 10 GiB/rank, 64 seqs/workers, context 163840, exec timeout 900 s, MANH_VISION=0.
  - kernel-metadata pins `nguyennguyen599/vllm-wheels/11`.
- **Pod inputs:**
  - **Model:** E128, the Kaggle model chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1 (136 files, 131.09 GB), downloaded sequentially per file (`dl_kaggle_files.py`) to /root/models/e128. E72-152 was deleted first to make room (it is on HF).
    - Kaggle serves ~30-40 MB/s per 1.15 GB file, so ~25-29 MB/s overall: about 80 min.
  - **E128 download:** DL_DONE ok=136 failed=0 in 4,128 s (~32 MB/s).
    - sha256 of all 136 files matches the manifest embedded in the notebook's cell 1b (`verify_manifest.py`: VERIFY_OK).
    - The dataset's `e043_manifest.json` is the pruning manifest, not a hash list.
  - **vllm-wheels v11** (listing through every page: 1,669 entries, 9.17 GB):
    - 293 of its 294 top-level wheels equal v10's by name and size. The vllm wheel changes from gf506afa210 to **g8cf0c0664b** (295,627,100 B); torch is unchanged.
    - The tree also holds 1,374 files (122 MB) in 4 nested source dirs: dion-0.1.0, liger_kernel-0.8.5, quack-kernels-0.6.5, sonic-moe-0.1.2.post1. These were also missing from the earlier v10 mapping. Manh's cell 3 copies them to /dev/shm, but no cell installs from them: cell 4 installs `/kaggle/working/quack-kernels`, which does not exist on Kaggle either. 1,374 sequential per-file downloads would take ~45 min, so they stay out.
  - **arc2-2026-src:** v22. The coordinator says any version works, because v6's cell 2m does not use --supervise.
  - **Unchanged:** cu13-install v2, dsv41-tools-kit v5 (same as the latest), unsloth-main v15.
- **Tree:** `scripts/make_kaggle_tree.sh` now takes KT_MODEL_SLUG, KT_MODEL_DIR, KT_SRC_DIR and KT_WHEELS_DIR.
- **v11 tree on the pod:** kagglehub resolved v11 for the new vllm wheel (sha256 5039fd7680a4) and .gitattributes. The other 293 wheels are hardlinks of the v10 set, and the nested source dirs are left out (see above).

### Commit smoke f1_commit (23:36 UTC, bec71fb1e3f1): every check passes except the host RAM check, which fails on the pod's cgroup v1

| cells | end |
|---|---|
| 0-9 | +3 s |
| Manh's cell 4 | +44 s |
| 2 (env) | +54 s |
| 2m | +294 s |
| 2s | +294 s |
| 2b | +328 s |
| 3 | +378 s |
| 3s | +380 s |
| 4 | +381 s |
| verdict | +381 s, **raised** |

- **Verdict:** `B32-M SMOKE FAILED: ['host RAM check: ValueError: max() iterable argument is empty']`.
  - The pod runs cgroup v1: no /sys/fs/cgroup/memory.max or memory.stat. So every smoke_env sample has `stat: {}` and `memory.max: null`, and 4b's projection gets an empty list.
  - Kaggle uses cgroup v2 (173 GiB), so the check computes there. But it fails closed wherever cgroup v2 is missing.
  - The same formula from /proc/meminfo: peak anon 129.1 GiB (2b's sample) + Shmem 8.7 + 4 + 0.2 x 62 = ~154 GiB, ~19 GiB under 173.
- **Settings:** "E046 settings: ARC_WORKERS 64, ARC_MULTI_TEST 1, ARC_MULTI_BUDGET 0.5, ARC_CONTEXT_LIMIT 163840, ARC_KV_ADMIT_MAX 0.85, …".
- **0b:** "E046 CHECK OK", with solver_dsv41.py 1ea2b25d, run_deepseek_tools 882c817f, train_check 8fe36175, auto_repair 2292369c / 571f47eb. The kit unpacked with sha256 ok. **No cell 2r**; vLLM is the v11 wheel, unpatched.
- **2m:** server up after 240 s. **KV 1,617,253 tokens at util 0.90** (E128).
  - GPU free 2.46-2.51 GiB per rank after load; 2.01-2.07 GiB under the smoke (reserved 19.38, peak allocated 18.94 GiB). 0 OOMs.
  - Model loading 14.33 GiB, 44.7 s per rank.
- **2s:** "E066 SUPERVISOR: watching pid 156480 (max 2 restarts, stall 1500s, status …/e066_supervisor.json)".
- **2b:** SANDBOX CHECK OK.
- **2c:** contract `fail: []`.
- **Cell 3:** 4 attempts on 2 tasks / 3 pairs, all submitted with train_pass, correct on every pair against the public training solutions. It ended at +0.10 h, right after the 2 tasks.
- **3s:** "E066 supervisor stopped: {'restarts': 0, …}".
- **3w:** contract thread finished.
- **Cell 4:** submission.json valid; "Manh's server stopped".
- **4b:** supervisor status `{"state": "up", "pid": 156480, "up_s": 240}`; model files 136, none missing; "E066 light smoke: 4 attempt records, 4 with an answer, 0 correct"; INPUT INTEGRITY OK (sha256 of every model file).
- **Files:** `runs/nb_f1_commit_bec71fb1/`.

### CV replica r1_rerun (bec71fb1e3f1, rerun mode, x2t; notebook from 23:44:11 UTC, solver from 23:49:07)
- **Solver:** "attempting 240 tasks, 344 pairs; deadlines: stop-new 10.83h, force 11.17h, hard stop 11.50h".
  - Env has ARC_EXTERNAL_SERVER, ARC_EXTERNAL_SERVER_LOG, **ARC_EXTERNAL_SERVER_STATUS** (= dsv41_logs/e066_supervisor.json) and ARC_CONN_TRIES=12.
- **Server:** KV 1,617,253 tokens. Server up 240 s after cell 2m started.
- **Pod-only accommodation:** after the install cells (+46 s) the 20 GB uv cache in /tmp/dsv41/cache/uv was deleted to keep disk headroom (34 GB free). Nothing reads it afterwards.
- **Watchers:**
  - `scripts/r1_watch.py` on the pod: read-only, 60 s. It writes a summary every 10 min to `watch.log`, and FLAG lines to `watch.flags` for VRAM free < 1 GiB, allocator retries or OOMs, server deaths, stop=error, Tracebacks, a supervisor not up for > 5 min, and the notebook's end.
  - `scripts/p2_report.sh <run> <pool>` for the hourly report.

#### Checkpoint 00:08 (19 min of solving)
- gen 751-798 tok/s; KV 0.59; 64 running / 0 waiting.
- 14 first attempts, 9 correct. 0 errors, 0 gate holds.
- Host used 154 GiB.
- VRAM per rank: 18.62 GiB allocated (peak 18.94), 19.39 reserved, 2.01-2.06 free, 0 retries, 0 OOMs.

#### TP-worker kill test (00:19:38, 30 min into solving, 63 running): PASS (`runs/r1_rerun_bec71fb1/killtest.log`)

| t after the kill | event |
|---|---|
| 0 | SIGKILL of exactly `VLLM::Worker_TP2` (pid 165301) |
| +2 s | its GPU (GPU 1: --gpu-ids 0,2,1,3) freed; host anon 129.8 → 96.2 GiB |
| +11 s (00:19:49) | API server pid 164585 exited. Supervisor status `{"start": 1, "state": "restarting", "error": "server pid 164585 exited"}`. Solver: `server died (code 1)`, then 65 x `re-queue <task> after a server-side error (1): TurnError: APIError: The server had an error while processing your request.` (64 of them; 1 x `InternalServerError: Error code: 500 …`; 2 with "its session answer is kept"), then `external server … (wait 1)` |
| +17 s | old process group gone, all GPUs at 0 MiB; host anon 11.7 GiB |
| +42 s | relaunch (new output root `dsv41_logs/r1/server/`) |
| +116 s (00:21:34) | solver: `server up after 105s (external, start 1); GPU KV cache size: 1,617,253 tokens`. Supervisor `{"start": 1, "state": "up", "pid": 170790, "up_s": 95}` |
| +8 min | 64 running, 0 waiting, 857 gen tok/s |

- **Host RAM never held two engines:** anon 12 GiB in the gap, 126.6 GiB once the new engine was up. Peak `free` used in the window: 153 GiB, vs 154 before the kill.
- **No attempt finalized as an error:** 0 `stop=error` in solver.log.
  - attempts.jsonl holds 65 per-try records with stop_reason "error", one per re-queued job. They are re-run; their final records are checked in the hourly reports.
- **VRAM after the restart:** identical to before (2.01-2.06 GiB free, reserved 19.39, 0 retries / OOMs).
- The supervisor's own "E066 SUPERVISOR: … restart 1/2" lines are in the kernel cell output and appear in nb.ipynb when the run ends.

#### Checkpoint ~1 h of solving (00:50 UTC; `runs/r1_rerun_bec71fb1/report_0050.txt`)
- **Attempts:** 41 first attempts finalized, 34 correct (82.9%), all submitted, 0 finalized as error.
  - Of the kill test's 65 re-queued jobs, 14 had already finished normally.
  - No other errored attempts, and no APITimeoutError.
- **Throughput:** gen 720-806 tok/s. KV 0.80 at 54-59 min; 64 running, 0 waiting. 1 KV gate hold (0.798).
- **VRAM, per rank (allocated / peak / reserved / free, GiB):** 18.62 / 18.94 / 19.39 / 2.01-2.06 at 00:08, then 18.62 / 19.00 / 19.43 / 1.97-2.02 at 00:48. 0 retries, 0 OOMs.
  - Allocated is flat; reserved grows with the peak, which is cache growth, not a leak.
- **Host:** used 151-153 GiB (peak 154). Disk 33 GB free.
