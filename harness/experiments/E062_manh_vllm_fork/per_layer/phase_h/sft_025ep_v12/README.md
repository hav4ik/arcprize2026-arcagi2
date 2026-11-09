# "SFT 025ep on V12": Manh's V12 notebook with the SFT 025ep checkpoint and the g22df163d2d wheels (pod 2)

Coordinator request (2026-10-30): rerun the V12 replica (`../manh_v12/`, Kaggle-style 80.83) with two changes and
nothing else, so the score compares directly with V12's:
1. the model: the fine-tuned E128 checkpoint (SFT + QAT of the tool E128 2-bit model);
2. Manh's updated vLLM wheels (vllm 0.32.1rc1.dev0+g22df163d2d, with his record_stream fix for the illegal-memory-access race).

The first choice, `sadalsuud/ds-qat-e0128-arazdSFT-v01`, was dropped: its only working key was unusable. On the user's
instruction the run uses the earlier checkpoint `chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch`.

## Verdict: stopped by the user at 4.76 h

**The fine-tune ties V12 (the routed experts are byte-identical to E128; only the shared experts differ). The g22df163d2d
wheel ran clean for 4.76 h: 0 allocator retries or OOMs, and cap − reserved flat at 1.02 GiB from 2.3 h.**

- **Stop:** at 17:31:28 UTC, Kaggle clock 4.76 h, the user stopped the run to free pod 2 for the "V12 + images" replica.
  - All 39 process groups of the run were killed by PGID (`scripts/stop_run.py`). The supervisor got SIGTERM first, and the server exited in 16 s. GPUs read 0 MiB afterwards.
  - The 1 "server died" line and the 1 errored record in the logs are the stop itself. Before it: 0 deaths, restarts, allocator retries, OOMs, errored attempts or timeouts.
- **Score at the stop.** The final scores are not comparable: V12's 80.83 came from a run that went to the end.
  - The latest `submission.json` (snapshot taken just before the stop) scores **66.32** Kaggle-style (222/344 pairs). No second attempt had run yet; V12's second attempts started at ~5 h.
  - Score so far at 4.76 h (best finished attempt per task; unanswered = 0): **66.3 vs V12 67.5**. Tasks answered 205 vs 204.
  - First-attempt accuracy so far: 152/205 = 74.1% vs 154/204 = 75.5%.
  - Paired per task at 4.76 h, on the 194 tasks both runs answered (`scripts/paired_at_hour.py`): 19 gained, 21 lost, mean −1.9 pp (bootstrap 95% CI −6.9 to +3.5). A tie.
    - Gained: 19c612ab 1bfd59d6 2250ae3e 2d0172a1 35ab12c3 38d6b6bc 3a25b0d8 4ac1ab1c 66a29f68 7b3084d4 7b5033c1 85ab8b0b 88bcf3b4 aa8caa8d b5328f0d b5ca7ac4 dd6b8c4b dfadab01 f560132c.
    - Lost: 06d13ca6 247ef758 25105d21 291dc1e1 296b7a44 2fba34fb 367a372d 36a08778 4c3d4a41 4c7dc4dd 4cccce55 5cbc143c 62593bfd 8698868d 89565ca0 9385bd28 9aaea919 cda7f206 d35bdbdc de809cff fca9e286.
  - `replica_report.md` scores the partial submission. Its paired section compares a 4.76 h run with V12's 10.17 h run, so read the clock-matched numbers above instead.
- **Score so far by hour** (`scripts/hourly_vs_v12.py`):

| Kaggle h | score (SFT / V12) | first-attempt acc so far (SFT / V12) | gen tok/s in the hour (SFT / V12) | running (SFT / V12) | worst cap − reserved GiB (SFT / V12) | Kaggle-eq GiB (SFT / V12) |
|---|---|---|---|---|---|---|
| 1 | 17.4 / 18.1 | 40/50 = 80.0% / 43/50 = 86.0% | 726 / 729 | 58.6 / 58.7 | 1.30 / 1.37 | 132.3 / - |
| 2 | 31.2 / 29.8 | 72/94 = 76.6% / 70/87 = 80.5% | 580 / 559 | 37.5 / 36.3 | 1.08 / 1.19 | 132.8 / - |
| 3 | 40.3 / 40.4 | 93/128 = 72.7% / 92/126 = 73.0% | 537 / 536 | 35.3 / 33.0 | 1.02 / 1.08 | 132.9 / 134.2 |
| 4 | 57.8 / 53.3 | 133/178 = 74.7% / 121/164 = 73.8% | 580 / 554 | 40.0 / 36.4 | 1.02 / 1.08 | 132.4 / 134.0 |
| 4.76 | 66.3 / 67.5 | 152/205 = 74.1% / 154/204 = 75.5% | 501 / 524 | 31.0 / 35.9 | 1.02 / 1.08 | 132.4 / 134.4 |

  (V12's sampler had no cgroup reading before 03:57 UTC, its 2.1 h.)
- **Serving with the g22df163d2d wheels:**
  - KV pool 1,593,757 tokens (V12: 1,608,578; −0.9%). Same model load (14.56 GiB per rank) and the same allocator caps.
  - cap − reserved fell in steps to 1.02 GiB by 2.32 h and stayed flat to the stop (V12: 1.08 from 2.83 h).
  - Peak allocated was a little lower than V12's (19.51 vs 19.57 GiB); reserved was higher (20.13 vs 20.07).
  - Throughput matched V12 (726 / 580 / 537 / 580 tok/s per hour vs 729 / 559 / 536 / 554). Host RAM was flat at ~132.4 GiB Kaggle-eq.
- **Where the checkpoint differs from E128:** the routed experts are byte-identical; only the shared experts differ. This comes from the other session's check of the checkpoint. Here: no 025ep file is byte-identical to E128's, and the `gptq/layer_*.safetensors` files are 48 bytes shorter (header).
- **Raw data:** see `data/RAW_README.md`. The files over 1 MB are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab` under `replicas/sft_025ep_v12/`.

## Setup

### Notebook
- **File:** V12's notebook (`kaggle/dsv41_tools/out/manh_v12_363638159/`, sha256 c47be5a819ad).
- **The one change:** cell 1b's embedded INTEGRITY manifest now lists the staged model dir: 99 files, 131.09 GB, sha256 per file.
  - The patched notebook's sha256 is 8398b01e8997 (`notebook/`, with `cell1b_integrity.patch` and `staged_model_manifest.json`).
  - It was patched by `scripts/patch_nb.py`, a raw-text replacement of that one JSON string. Applied with the original manifest, it reproduces V12's bytes exactly.
  - Cell 1b runs in commit mode only, so the change has no effect in this rerun.
- **Runner:** the same as V12's: `scripts/launch_nb.sh rerun` via `scripts/launch_s025.sh`, `run_nb.py` (nbclient), KAGGLE_IS_COMPETITION_RERUN=1, a fresh `kenv` and `/kaggle/working`.
  - **Workload:** the same evalx2t_s2026 file (240 tasks, 344 pairs; `data/workload.txt`).
  - **Inputs:** arc2-2026-src v23, cu13-install v2, dsv41-tools-kit v5 and unsloth-main v15, all the same as V12.
  - **Launched** 12:45:42 UTC.

### Model: SFT 025ep + E128's Engram tables, zero-copy (`/root/sft_025ep_model`)
- **Checkpoint:** `chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch` @3ca689afc229: 52 files, 59.30 GB, an E043 export.
  - Weights: `gptq/layer_00..39.safetensors` and `model-0000{1,2}-of-00002.safetensors`.
  - Index, config, tokenizer, `e043_manifest.json`, `engram_top64gib.json`, the keep-set npy, and `EXPORT_COMPLETED.json` (epoch 0.2456, step 28, val_loss 0.6655, run `v12_fixed_oom_moe_only_full_sft`).
  - Downloaded in 257 s. All 43 LFS files were checked against the Hub's sha256, and the 9 small files against the Hub copies.
- **Compatibility with the E128 Kaggle model**, checked before download from the Hub metadata and safetensors headers (`scripts/inspect_sft.py`):
  - **Byte-identical to E128's:** `config.json` (128 experts, Humming GPTQ 2-bit schema, `engram_compact_kept`), `e043_manifest.json` (keep ids, Engram row counts and mask sha256s), `tokenizer.json`, `tokenizer_config.json`, `README.md`.
  - **Keep set:** the npy is bool [40, 384]. Its array-bytes sha256, f4c946c3, equals E128's `expert_keep_sha256`, and its rows equal the manifest's keep ids.
  - **Engram table:** `engram_top64gib.json` rows_kept [136114884, 124186164] and rows_total [384006168, 384016682] equal E128's Engram counts and the shapes of its row chunks and scales.
  - **Tensors:**
    - All 47,334 of E128's non-Engram-table tensors are present, with the same dtypes and shapes and no extras. They include the 40 GPTQ layers and `layers.{1,14}.engram.{k_weight,q_weight,wkv.weight,wkv.scale}`.
    - The 025ep's own weights are used for these tensors; no file is byte-identical to E128's.
  - **Index:** the 025ep index already names E128's Kaggle Engram shards. Its 18 Engram-table entries (`layers.{1,14}.engram.embed.weight.rows_*`, `.embed.scale`, `.embed_tokens.row_map` → `model-engram-000NN-of-00016`) equal E128's exactly.
- **Staging** (`scripts/stage_sft.py`): real directories holding file symlinks.
  - The 51 SFT files, with the 025ep index used unchanged.
  - E128's 16 Engram shards (70 GB).
  - E128's small non-weight files that the 025ep lacks: LICENSE, `encoding/`, `inference/`, `serving/`. E128's `gptq/*.json` quantization reports are left out; the loader globs only `gptq/layer_*.safetensors`.
  - Checks: each indexed file's header holds exactly the tensors the index maps to it, because vLLM loads whole files; index paths are relative with no `..`; and nothing of E128's backbone is linked.
  - The kit's `stage_model.py` preflight passes.
- **Input mapping:** the model input `models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1` → `/root/sft_025ep_model`. The notebook finds it by its `e043_manifest.json`, as on Kaggle.

### vLLM wheels: the new full set with g22df163d2d (`/root/kin/vllm_wheels_sft`, 319 files)
- **vllm-wheels-util:** `vllm-0.32.1rc1.dev0+g22df163d2d-cp313-cp313-linux_x86_64.whl`, sha256 1a6def82.
- **tool-wheel:** `.gitattributes` plus `wheels.tar.gz` (9.57 GB). Despite the name it is a plain, uncompressed tar of `wheels/` (324 files). It was stream-extracted and its sha256 checked.
  - 293 wheels are byte-identical to v11's; they are hard-linked to them, using no extra disk.
  - 25 are newer wheels.
  - 4 are source zips: dion, liger_kernel, quack-kernels, sonic-moe.
  - 2 are vllm wheels: g22df163d2d, and g70867e279f, which is an ancestor of g22df163d2d without the fix.
- **Mapped set:** the full set without g70867e279f and without the zips. V12 also left the 4 source dirs out, and no cell installs from them.
  - g70867e279f had to go: its local version tag sorts higher, so `uv pip install vllm -U` would have installed it.
- **Check before launch:** a uv dry-run of cell 4's `uv pip install --no-build-isolation vllm torch torchvision -U` (offline, find-links) resolves `vllm==0.32.1rc1.dev0+g22df163d2d`.
  - It differs from v11 only by patch bumps: transformers 5.22.1 (cell 4 then pins 5.20.2), pydantic 2.14.2, openai 3.41.0, huggingface-hub 2.4.0, hf-xet 1.8.0, tokenspeed-triton 3.9.10.post20261028, opentelemetry 1.45.2, setuptools 80.10.2, mistral-common 1.13.0, cachetools, cuda-pathfinder, dnspython, nvidia-cudnn-frontend, oss-harmony.
  - Run record (`b32m_env.json`): vllm 0.32.1rc1.dev0+g22df163d2d, torch 2.14.0+cu130, transformers 5.20.2, openai 3.41.0.

### Pod changes (disk; coordinator decision "option b")
- **Deleted from E128:** 42 non-Engram weight files from `/root/models/e128` (`gptq/layer_*.safetensors`, `model-0000{1,2}-of-00002`, 59.29 GB).
  - Before deleting, all 42 were checked against the Kaggle model listing (same sizes), with no symlinks into them.
  - Restore: download them from Kaggle `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1` (`/root/models/E128_PARTIAL.txt`).
- **Also removed:** `/root/kenv` (launch_nb.sh resets it from `kenv_fresh`) and `/root/.cache/uv`.
- **After the stop:** the 025ep weights (`/root/models/sft025ep`) and the staged dir were deleted from the pod to free disk for the image replica. They can be downloaded again from the Hub at 3ca689afc229.
- **As in V12:** the run's 20 GB uv cache was deleted after the install cells (`uvcache_cleanup.sh`, 12:46:37 UTC; 14 → 34 GB free).

### Monitoring (same as V12)
- **Sampler:** `scripts/v12_sampler.py`, read-only, every 60 s, with the same alerts. It writes `samp.jsonl`, `samp.alerts` and `samp.tps.tsv`.
- **Dashboard sync:** every 5 min (`scripts/sft025_dash_sync.sh`, `sft025_dash_loop.sh`) into `data/`. Same files and formats as `manh_v12/data`: gpumem.log, ram.tsv, tps.tsv, status.json, `dsv41_logs/{attempts.jsonl, solver.log}`, `nbruns/s025_rerun/samp.jsonl`, workload.txt.
- **Hourly report:** `scripts/sft025_wait.sh`, comparing with V12 at the same Kaggle hour (`scripts/hourly_vs_v12.py`).

## Results (as the run went)

### Start
- Install cells done at +50 s. Server up at +280 s (V12: +297 s): his supervisor, start 0.
- Model loading 14.56 GiB per rank (V12: 14.56).
- **KV pool 1,593,757 tokens** (V12: 1,608,578; −0.9%).
- Allocator caps 21.15-21.20 GiB with 19.14 reserved (V12: 21.15-21.20 / 19.15), so cap − reserved is 2.01-2.06 GiB at start.
- Solver from 12:50:23: "attempting 240 tasks, 344 pairs; deadlines: stop-new 10.83h, force 11.17h, hard stop 11.50h".
- Kaggle-eq host RAM (cgroup) 131.5 GiB at start.

### Report 1 (13:46 UTC; Kaggle 1.0 h)
**Verdict: stable, no events.** 0 deaths, restarts, allocator retries, OOMs, errored attempts or timeouts.

| measure at 1 h | SFT 025ep | V12 |
|---|---|---|
| score so far over 240 | 17.4 | 18.1 |
| first-attempt accuracy so far | 40/50 = 80.0% | 43/50 = 86.0% |
| gen tok/s, running (0-1 h mean) | 726, 58.6 | 729, 58.7 |
| worst cap − reserved | 1.30 (TP0-3 now 1.33 / 1.31 / 1.30 / 1.28) | 1.37 |
| Kaggle-eq RAM (cgroup) | 132.3 GiB, flat | ~134 |

- cap − reserved fell 0.35 GiB/h over the first 0.6 h. That is the load ramp: V12 fell 0.24/h in hours 1-2, then stayed flat at 1.08 from ~3 h.
- Alerts were startup artifacts only:
  - the supervisor's starting → up;
  - "server waiting 48" from the initial burst;
  - the load ramp of cap − peak;
  - a "RAM rising 250 GiB/h", because its 1 h window included the pre-server samples.

### Report 2 (14:46 UTC; Kaggle 2.0 h)
**Verdict: stable, no events.**

| measure at 2 h | SFT 025ep | V12 |
|---|---|---|
| score so far over 240 (tasks answered) | 31.2 (94) | 29.8 (87) |
| first-attempt accuracy so far | 72/94 = 76.6% | 70/87 = 80.5% |
| gen tok/s, running (1-2 h mean) | 580, 37.5 | 559, 36.3 |
| worst cap − reserved | 1.08 (TP0-3 1.13 / 1.11 / 1.10 / 1.08), flat since ~1.6 h | 1.19 |
| Kaggle-eq RAM (cgroup) | 132.8 GiB, flat | ~134 |

- **VRAM:** the run reaches V12's final plateau earlier.
  - Peak allocated tracks V12's: 19.45 vs 19.43 GiB at 1.75 h.
  - Reserved runs ~0.13 GiB higher: 20.07 vs 19.94 at 1.75 h. This run reached 20.07, V12's final plateau, at ~1.6 h; V12 reached it at ~3 h.
- **One new EARLY alert** at 14:23: "cap − peak fell 0.19 GiB in 29 min". It is the load ramp; there were no retries.

### Report 3 (15:47 UTC; Kaggle 3.0 h)
**Verdict: stable, no events; tied with V12.**

| measure at 3 h | SFT 025ep | V12 |
|---|---|---|
| score so far over 240 (tasks answered) | 40.3 (128) | 40.4 (126) |
| first-attempt accuracy so far | 93/128 = 72.7% | 92/126 = 73.0% |
| gen tok/s, running (2-3 h mean) | 537, 35.3 | 536, 33.0 |
| worst cap − reserved | 1.02 (TP0-3 1.07 / 1.05 / 1.04 / 1.02) | 1.08 |
| Kaggle-eq RAM (cgroup) | 132.9 GiB | 134.2 GiB |

- **cap − reserved moves in steps, as in V12.**
  - This run's steps: 1.16 → 1.08 at 1.63 h, then → 1.02 at 2.32 h. It has been flat since.
  - V12's last step was to 1.08 at 2.83 h, after which it stayed flat to the end.
- **Peak vs reserved:** peak allocated is lower than V12's (19.51 vs 19.57 GiB), but reserved is higher (20.13 vs 20.07). The reserved − peak gap is ~0.12 GiB wider: more cached or fragmented blocks.

### Report 4 (16:46 UTC; Kaggle 4.0 h)
**Verdict: stable, no events; ahead of V12.**

| measure at 4 h | SFT 025ep | V12 |
|---|---|---|
| score so far over 240 (tasks answered) | 57.8 (178) | 53.3 (164) |
| first-attempt accuracy so far | 133/178 = 74.7% | 121/164 = 73.8% |
| gen tok/s, running (3-4 h mean) | 580, 40.0 | 554, 36.4 |
| worst cap − reserved | 1.02 (TP0-3 1.07 / 1.05 / 1.04 / 1.02), unchanged since 2.32 h; 2 h slope −0.024/h | 1.08 |
| Kaggle-eq RAM (cgroup) | 132.4 GiB, flat | 134.0 GiB |
