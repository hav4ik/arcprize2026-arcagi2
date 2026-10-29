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
