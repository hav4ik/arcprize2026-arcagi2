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
- **Server:** Manh's **v5 production cell 2m**, run verbatim through `run_prod_cell.py`.
  - Cell settings: util 0.94, VLLM_GPU_DRIVER_RESERVE_MB 256, --context 163840, --concurrency 80 (max-num-seqs and
    graphs), MODE 556 lossless, retention 128, tool parser.
  - Env: MANH_VISION=1, MANH_VISION_MAX_IMAGES=32.
  - **Plus DCP4:** `--vllm-arg=--decode-context-parallel-size=4 --vllm-arg=--cp-kv-cache-interleave-size=64
    --vllm-arg=--prefix-cache-retention-interval=512`, inserted after his last flag, so 512 overrides his 128.
- **Solver:** kit = kaggle_kit_v5 + `pinned_img/`.
  - Files: solver **e96f7cb7** (external-server probe tolerates timeouts), harness 1d1a281e, grid_image 18ed78fb,
    train_check 8fe36175, auto-repair 2292369c / 571f47eb.
  - Launcher: pod 2's `launch_ext_e72c152.sh` with C1's dir and **80 workers**.
  - Settings: ARC_GRID_FORMAT=text+image, multi-test 0.5, context limit 163840, gate 0.85 + queue hold, auto-repair,
    workload evalx2t_s2026, replica deadlines.

## Startup
| Step | UTC | Notes |
|---|---|---|
| cell 2m start | 19:35:56 | concurrent with the C1 sha256 verify |
| model loaded | 19:42:43 | 14.56 GiB/GPU, 357 s (disk shared with the verify) |
| server up | pending | |

## KV pool and contract
pending

## Load run windows (5 min)
pending
