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
  8d1fac9e). In the copy, ONLY cell 1b's model manifest is swapped to C1's 137 entries (sha c31c5d5d; diffed: only
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
pending

## Run 2 KV pool and contract
pending

## Run 2 load windows (5 min, then hourly)
pending
