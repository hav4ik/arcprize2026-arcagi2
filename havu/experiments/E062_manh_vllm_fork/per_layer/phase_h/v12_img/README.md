# "V12 + images FINAL": Manh's V12 notebook with the image switch, his --watermark-reload-only and the new solver recovery, served the causal image E128 model (pod 2)

User decision (2026-10-30, via the coordinator): use pod 2 for a 12 h "V12 + images" replica.
- **Notebook:** Manh's V12, the gold standard (replica 80.83), with only a surgical image switch.
- **Model:** the HF causal re-prune `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`, in place of E066 C1, which was dropped mid-download on the user's instruction.
- **Baseline:** pod 2's own V12 text replica (`../manh_v12/`).

## Setup

### Notebook
- `kaggle/dsv41_tools/out/nb_v12_img/dsv41-tools-v12-img.ipynb`, sha256 790b1bd9d7b1, built by another agent.
- Its README holds the cell-by-cell diff vs V12; tests passed (23 + 29 + 20). Unchanged here.
- Cell 1b's integrity manifest is C1's (137 files). Rerun mode skips cell 1b, so it is not checked in this run.
- **Runner:** the same as V12's: `launch_nb.sh rerun` via `scripts/launch_v12img.sh` → `launch_nb_owner.sh`, `run_nb.py` (nbclient), KAGGLE_IS_COMPETITION_RERUN=1, a fresh `kenv` and `/kaggle/working`.
  - `launch_nb_owner.sh` / `make_kaggle_tree_owner.sh` are V12's `launch_nb.sh` / `make_kaggle_tree.sh` with one change: the model owner in the mount path is a variable (`KT_MODEL_OWNER`).
- **Workload:** evalx2t_s2026, 240 tasks, 344 pairs (`data/workload.txt`).
- **Launched** 17:55:56 UTC.
- **Inputs:**
  - arc2-2026-src v23, cu13-install v2, dsv41-tools-kit v5, unsloth-main v15;
  - vllm-wheels = the new full set with vllm 0.32.1rc1.dev0+g22df163d2d (`/root/kin/vllm_wheels_sft`; see `../sft_025ep_v12/README.md`, the wheels section).
  - V12's own replica ran g8cf0c0664b; the SFT 025ep run showed g22df163d2d serving text at V12's level.
- **Env:** nothing extra. Cell 00 sets ARC_GRID_FORMAT=text+image and MANH_VISION_MAX_IMAGES=32.

### Model: the HF causal model + E128's Engram tables, zero-copy (`/root/img_causal_model`)
- **Checkpoint:** `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal` @**209c810a747c043c2aca474843de8e357f920ef6**, the head on 2026-10-30 (created 14:14:56 UTC). 122 files, 60.27 GB.
  - Downloaded in 257 s with the readonly key (over stdin).
  - All 44 LFS files were checked against the Hub's sha256; the 77 small files against copies fetched at the same revision.
- **Checks before staging** (Hub metadata and headers, then `scripts/check_c1.py` on the staged dir):
  - **config.json:** every `engram_*` field equals E128's (`engram_compact_kept`, `engram_num_embeddings`, ...). It differs only in `e043_lowbit.name`, `e043_pruning.expert_keep_sha256` (the image re-prune's own expert keep set), `text_only` false and `no_mtp` true.
  - **e043_manifest.json:** the Engram section equals E128's: rows 384006168 / 384016682, kept 136114884 / 124186164, mask sha256 2f4ab8e2… / e8a4ebab…, so it is the same Engram table.
    - Only the provenance string `engram_from` differs: `/workspace/dsv4/export/c1_gptq` vs `export/e72_w30_eng64`.
    - The expert keep ids differ from the tool E128's, by design.
  - **Index:** its 18 Engram-table entries (`layers.{1,14}.engram.embed.weight.rows_*`, `.embed.scale`, `.embed_tokens.row_map` → `model-engram-000NN-of-00016`) equal E128's index exactly, so the index is used unchanged.
    - Its metadata has no `e043_reshard`. The fork maps `.rows_<start>` chunks by tensor name (`deepseek_v41/nvidia/model.py` `_ENGRAM_ROW_CHUNK`), not by index metadata.
  - **Headers:** each of the 19 indexed files' safetensors header holds exactly the tensors the index maps to it (1,538 tensors), and no tensor is in two files.
  - **Vision tower:** `vision_config`, plus 266 `vision.*` / `aligner.*` / `image_*` tensors in `model-00001.safetensors`.
  - **Experts:** 40 `gptq/layer_*.safetensors` (= num_hidden_layers).
  - **E128 tensors:** all 47,334 of E128's non-Engram-table tensors are present, with no dtype or shape mismatch.
- **Staging** (`scripts/stage_sft.py`): 137 file symlinks in real directories: the 121 files of the causal repo and E128's 16 Engram shards. E128's 16 shards were sha-checked against the C1 manifest the same day. Kit preflight ok.
- **Mount:** `/kaggle/input/models/chankhavu/deepseek-v41-flash-1017-arc-tool-img-e128-2bit-causal/pytorch/default/1` → `/root/img_causal_model`, the only model mounted.
  - The notebook's bounded lookup (`models/*/*/*/*/*`) found it; the server's model path is that dir.
  - The arazzd C1 path in INPUT_ROOTS does not exist on the pod, as intended.
- **Comparison with C1:** the staged dir has C1's exact file set (137 paths). 52 files are byte-identical to C1's: the 16 Engram shards, `model-00003`, the index, tokenizer, code and docs. `model-00001/2`, the 80 gptq files, `config.json`, `e043_manifest.json` and `README.md` differ.

### Pod changes
- **Deleted:** the SFT 025ep weights (re-downloadable from the Hub at 3ca689afc229).
- **C1:** the partial download was stopped (2 processes, by PGID) and deleted. It is re-downloadable from Kaggle `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/PyTorch/default/1`.
- **As in V12:** the uv cache was deleted after the install cells (17:56:46 UTC; 13 → 33 GB free).

### Monitoring
Same sampler and alerts as V12. Every 5 min the data are synced into `data/` (`scripts/replica_dash_sync.sh` / `replica_dash_loop.sh`): gpumem.log, ram.tsv, tps.tsv, status.json, `dsv41_logs/{attempts.jsonl, solver.log, metrics.jsonl}`, `nbruns/v12img_rerun/samp.jsonl`, workload.txt.

Hourly (`scripts/replica_wait.sh`):
- `hourly_vs_v12.py`: score, first- and second-attempt accuracy, throughput, cap − reserved, RAM.
- `hourly_extra_vs_v12.py`, the serving metrics vs V12 text:
  - prefix-cache hit rate;
  - uncached prompt tokens per turn;
  - preemptions;
  - server running and solver in-flight under the KV gate;
  - host-KV restores;
  - MM processor-cache hit rate;
  - API-server + EngineCore RSS (where the MM caches live);
  - cap − reserved;
  - score and tasks answered.

## Result: "V12 + images FINAL" (d5e545d4e05c), stopped as planned at Kaggle 7.5 h
**Verdict: images tie V12 text on accuracy at 7.3 h, cost ~9% throughput, and run stably. The image candidate was not submitted today (coordinator decision).**
- The run was stopped by PGID at 03:31 UTC (Kaggle 7.55 h; 46 groups, GPUs to 0 MiB). The 1 "server died" line is the stop itself.
- **Stability over 7.5 h:** 0 deaths, restarts, allocator retries, OOMs or errored attempts.
- **Alerts:** the only critical alert was the cap − reserved slope rule at 1.5 h (−0.35 GiB/h during the ramp).
  - No absolute threshold fired: cap − reserved stayed ≥ 0.75 GiB, nvidia-smi free ≥ 0.5 GiB and Kaggle-eq RAM ≤ 150 GiB.
- **Score at 7.3 h:**
  - Score so far 74.7, V12 76.0.
  - Tasks answered 236 vs 234; tasks with both attempts finished 57 vs 73.
  - First-attempt accuracy 162/233 = 69.5% vs 166/222 = 74.8%.
  - `submission.json` at the stop (7.55 h): Kaggle-style 75.69 (attempt_1 only 71.67).
- **Paired per task at 7.3 h** (`../scripts/paired_at_hour.py`), on the 232 tasks both runs answered: images 76.8% vs text 78.0%. 24 gained, 23 lost; paired −1.2 pp, 95% CI −6.0 to +3.6. **A tie.**
- **Throughput:** mean gen tok/s over 0.17-7.3 h is 525 vs V12's 577, a **ratio of 0.909**.
  - Uncached prompt tokens per turn ran at 2.0-2.4k vs 0.8-1.3k. Image re-prefills cost more, and image turns cannot use --token-exact.
  - Preemptions were 11-24 per hour from 4 h (V12: 0).
  - The GPU prefix-hit rate fell to 0.03-0.19 at 6-7.3 h, while host-KV restores ran at 7.2-9.5k tok/s (V12: 0.54-0.71 and 2.9-4.3k).
- **cap − reserved trajectory:** it fell in steps and flattened.
  - Steps: 2.01 at load → 1.17 (1.0 h) → 1.03 (1.5 h) → 0.95 (1.77 h) → **0.93 at 2.32 h**.
  - Flat at **0.93-0.98 GiB per rank** (TP3 worst) from 2.32 h to the stop, 5.2 h with no further step.
  - V12 text flattened at 1.08 from 2.83 h. The ~0.15 lower plateau matches the replay graphs (+0.14 GiB graph memory) and the slightly lower cap.
  - cap − peak 1.66 at the end. Retries 0.
- **Host RAM:** Kaggle-eq peaked at ~141.7 GiB at ~6 h (V12 ~134.4), 31 GiB under Kaggle's 173.
  - API-server + EngineCore RSS grew 6.1 → 8.7 GiB as the MM processor caches filled (2 × 4 GiB default). It leveled off at 8.6-8.7 GiB from 6 h.
  - MM-cache hit rate 93-100% per hour.
- **KV pool:** 1,531,221 tokens (−4.2% vs the image run without replay graphs; V12 text 1,608,578).

### Hour by hour vs V12 text (`../scripts/hourly_vs_v12.py`)
| Kaggle h | score (new / V12) | 1st-att acc so far (new / V12) | 2nd-att acc so far (new / V12) | gen tok/s in hour (new / V12) | running (new / V12) | worst cap-reserved GiB (new / V12) | Kaggle-eq GiB (new / V12) |
|---|---|---|---|---|---|---|---|
| 1 | 16.0 / 18.1 | 38/47 = 80.9% / 43/50 = 86.0% | - / - | 647 / 729 | 55.4 / 58.7 | 1.21 / 1.37 | 136.6 / - |
| 2 | 28.5 / 29.8 | 67/90 = 74.4% / 70/87 = 80.5% | - / - | 503 / 559 | 32.9 / 36.3 | 0.95 / 1.19 | 137.7 / - |
| 3 | 39.6 / 40.4 | 92/125 = 73.6% / 92/126 = 73.0% | - / - | 517 / 536 | 34.7 / 33.0 | 0.93 / 1.08 | 138.4 / 134.2 |
| 4 | 52.5 / 53.3 | 120/172 = 69.8% / 121/164 = 73.8% | - / - | 491 / 554 | 35.0 / 36.4 | 0.93 / 1.08 | 139.0 / 134.0 |
| 5 | 65.8 / 68.3 | 151/214 = 70.6% / 156/206 = 75.7% | - / 3/4 = 75.0% | 477 / 555 | 32.9 / 38.0 | 0.93 / 1.08 | 139.7 / 134.3 |
| 6 | 71.2 / 74.0 | 161/230 = 70.0% / 166/222 = 74.8% | 7/10 = 70.0% / 9/18 = 50.0% | 544 / 552 | 35.8 / 33.2 | 0.93 / 1.08 | 141.7 / 134.1 |
| 7 | 72.5 / 75.6 | 162/233 = 69.5% / 166/222 = 74.8% | 14/35 = 40.0% / 23/51 = 45.1% | 491 / 549 | 32.0 / 34.4 | 0.93 / - | 139.8 / 134.7 |
| 7.3 | 74.7 / 76.0 | 162/233 = 69.5% / 166/222 = 74.8% | 27/52 = 51.9% / 34/68 = 50.0% | 515 / 568 | 34.9 / 38.4 | 0.93 / 1.08 | 139.4 / 134.4 |

### Serving metrics vs V12 text (`../scripts/hourly_extra_vs_v12.py`)
| Kaggle h | score (new / V12) | tasks answered | gen tok/s | prefix hit | uncached tok/s | uncached prompt tok/turn | prompt tok/turn | preemptions | server running | solver in-flight | host-KV restore tok/s | MM cache hit (new) | API+Engine RSS GiB | worst cap-reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16.0 / 18.1 | 47 / 50 | 671 / 748 | 0.890 / 0.889 | 551 / 254 | 1792 / 1021 | 21098 / 18572 | 0 / 0 | 55.7 / 58.5 | 56.3 / 60.1 | 247 / 572 | 0.980 (6808 q) | 6.13 / 2.83 | 1.21 / 1.37 |
| 2 | 28.5 / 29.8 | 90 / 87 | 501 / 559 | 0.944 / 0.558 | 499 / 276 | 2264 / 831 | 36300 / 35800 | 0 / 0 | 33.3 / 37.6 | 33.3 / 43.0 | 0 / 3872 | 0.942 (5344 q) | 7.30 / 2.88 | 0.95 / 1.19 |
| 3 | 39.6 / 40.4 | 125 / 126 | 519 / 537 | 0.832 / 0.499 | 441 / 246 | 2015 / 1084 | 46235 / 44837 | 0 / 0 | 35.0 / 33.9 | 36.1 / 39.3 | 1469 / 5170 | 0.977 (6400 q) | 7.93 / 2.90 | 0.93 / 1.08 |
| 4 | 52.5 / 53.3 | 172 / 164 | 484 / 566 | 0.821 / 0.973 | 629 / 260 | 2210 / 1185 | 48160 / 42307 | 11 / 0 | 35.5 / 36.3 | 36.7 / 37.6 | 1163 / 0 | 0.933 (6105 q) | 8.18 / 2.90 | 0.93 / 1.08 |
| 5 | 65.8 / 68.3 | 214 / 206 | 483 / 564 | 0.950 / 0.937 | 570 / 331 | 2322 / 1297 | 45646 / 42273 | 24 / 0 | 32.5 / 38.7 | 33.3 / 40.6 | 7 / 320 | 0.950 (6632 q) | 8.59 / 2.90 | 0.93 / 1.08 |
| 6 | 71.2 / 74.0 | 231 / 224 | 553 / 553 | 0.026 / 0.705 | 357 / 145 | 2364 / 1050 | 49948 / 49416 | 22 / 0 | 36.3 / 32.7 | 47.0 / 36.2 | 9504 / 2894 | 1.000 (5567 q) | 8.61 / 2.90 | 0.93 / 1.08 |
| 7 | 72.5 / 75.6 | 236 / 233 | 486 / 545 | 0.166 / 0.601 | 461 / 191 | 2117 / 853 | 48264 / 47586 | 24 / 0 | 32.4 / 34.3 | 43.6 / 37.3 | 7165 / 3733 | 0.965 (5511 q) | 8.71 / 2.92 | 0.93 / - |
| 7.3 | 74.7 / 76.0 | 236 / 234 | 514 / 579 | 0.186 / 0.543 | 465 / 198 | 2396 / 899 | 46558 / 45056 | 22 / 0 | 35.4 / 38.0 | 47.2 / 41.7 | 7495 / 4348 | 0.962 (5885 q) | 8.71 / 2.92 | 0.93 / 1.08 |

### Runs on the way to this one
- **r0** (790b1bd9d7b1, no watermark flag): `../v12_img_r0/`.
- **r1** (5f4cfe5d1694: watermark-reload-only and solver 0c2f81e8, no replay graphs; stopped at 0.98 h): `../v12_img_ab_r1/`.
- **e5385b5994fe** (replay graphs via a dotted flag, which replaced the compilation config; stopped at load): `../v12_img_rg_bad/`.

### Raw data
The files over 1 MB are on HF `livobereshka/arc-agi2-vllm-serving-lab` under `replicas/v12_img_final/` (commit bf6018d9c25d; private check and secret scan, 374 files, 0 hits):
- `dsv41_traces.tar.gz` (690 traces);
- `server.log`;
- `gpumem.log`;
- `samp.jsonl`.

Everything else is in `data/`.

## The final run: setup and load

- **Notebook (final, d5e545d4e05c):** `kaggle/dsv41_tools/out/nb_v12_img_final/dsv41-tools-v12-img.ipynb`. It is 5f4cfe5d1694 plus one full-JSON `--compilation-config` arg (FULL, max capture 64, decoder replay graphs 32-256). Launched 19:58:24 UTC; server up 20:02:59.
- **Earlier notebook in this line:** `kaggle/dsv41_tools/out/nb_v12_img_ab/dsv41-tools-v12-img.ipynb`, sha256 **5f4cfe5d1694**. Relative to 790b1bd9d7b1 (r0) it adds two things:
  - Manh's `--vllm-arg=--watermark-reload-only` in cell 2m;
  - solver **0c2f81e8**: held free re-queues and 500/timeout classification, active only on failures.
  - The harness is unchanged (ed668f98).
- **Same as r0:** the staged causal model and mount, the wheels (g22df163d2d) and the runner (`launch_v12img.sh v12img2_rerun <notebook>`).
- **Launched** 18:49:43 UTC.

### Load of the final run (20:03 UTC)
- Non-default and engine config agree: cudagraph_mode FULL, max_cudagraph_capture_size 64, decoder replay sizes 32-256.
- Replay graphs were captured; graph memory 0.24 GiB.
- KV pool 1,531,221 tokens. Caps 21.08-21.13 GiB with 19.12 reserved, so cap − reserved started at 1.96-2.01.
- watermark_reload_only True, watermark 0.1, disable_chunked_mm_input True, image limit 32.
- Vision loaded, 0 bidirectional lines, solver 0c2f81e8.

### Load of r1 (5f4cfe5d1694, 18:54 UTC)
- Server up at +262 s; solver from 18:54:06 (240 tasks, 344 pairs, the usual deadlines).
- **Non-default args:** `watermark_reload_only: True`, `watermark: 0.1`, `disable_chunked_mm_input: True`, `limit_mm_per_prompt: {image: 32}`, `max_num_seqs: 64`, util 0.91, max_model_len 163840.
- **Vision tower loaded:** MMEncoderAttention initialised; the encoder cache "profiled with 2 image items". 0 "bidirectional" lines.
- **Solver:** sha 0c2f81e8 in the run (not the v3 fallback), with env ARC_GRID_FORMAT=text+image and MANH_VISION_MAX_IMAGES=32. 476 MM-cache queries in the first minute.
- **KV pool 1,597,824 tokens** (V12: 1,608,578, −0.7%). Model loading 14.56 GiB per rank. Caps 21.15-21.20 GiB with 19.15 reserved.
- As in V12, the uv cache was deleted after the install cells (18:50:33 UTC).

## Earlier run r0
The first run (r0, 17:55-18:45 UTC, notebook 790b1bd9d7b1, without `--watermark-reload-only`): stopped by the user at 0.82 h,
see `../v12_img_r0/README.md`.

## Staged model: sha256 of every file (for the Kaggle re-upload check)
A teammate will re-upload this model to Kaggle. Before any submission, the Kaggle copy must match this list byte for byte (cell 1b's format: `staged_model_manifest.json`).
137 files, 132,061,521,361 bytes. Source: `causal` = HF @209c810a747c; `E128` = the E128 Kaggle copy's Engram shards, which are byte-identical to C1's.

<details><summary>137 files</summary>

| path | bytes | sha256 | source |
|---|---|---|---|
| `LICENSE` | 1,084 | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` | causal |
| `README.md` | 3,553 | `256759449fa043b57817a89edc8084987f1c5dbc7c90caaf5246fe4d54cd5dde` | causal |
| `config.json` | 5,293 | `49919628ddf71c844cf2defd1a5ee4a4320ce8adee79f5a84dd850520fdad4d3` | causal |
| `e043_manifest.json` | 40,578 | `d2a47999f1a7d1b9b53bfe1d3eec4aded4c074a0e11362dc6a8598b1890918b8` | causal |
| `encoding/README.md` | 12,164 | `a34a5ca98c65e5b5b2bedd693cc2e18e732fe50192723fb6937ae07219c23739` | causal |
| `encoding/encoding.py` | 37,962 | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` | causal |
| `encoding/test_encoding.py` | 22,449 | `288b79c9ea7d018ae03b0dd3e54a8220cc3a2df1ff97f59907e75ed8112765d6` | causal |
| `encoding/tests/test_input_1.json` | 2,761 | `6a5a86694e47169c8f0c9243a7e24cc5db05447be0d3d3a9891034e913c518de` | causal |
| `encoding/tests/test_input_2.json` | 527 | `fd985cda88e8640b8e2404d8e659f7006cad204eb41f73c9e6662b1f218c01aa` | causal |
| `encoding/tests/test_input_3.json` | 2,554 | `4207613f15c79b95ad92074bdb8b2b81c5ce9cfdf15b7a67f58ded6bb3ec4fa5` | causal |
| `encoding/tests/test_input_4.json` | 712 | `4c3890a7bc042ec6396d98d024a39a73cdcffbe1d0253e852da0c35e65296f3e` | causal |
| `encoding/tests/test_input_5.json` | 1,143 | `88289b0aac8d650aa55c4ffeb392818fff3a0ac83c2293ddfd90546804d5e5c4` | causal |
| `encoding/tests/test_output_1.txt` | 2,476 | `91f79aa955717b414a82aabfb38bcd83ab5a998bd47db89d2a26338e8d3de550` | causal |
| `encoding/tests/test_output_2.txt` | 294 | `9fa162587b73ab9b315be5c88d433f9e0e681b4b36926952f289099ed9bc1dab` | causal |
| `encoding/tests/test_output_3.txt` | 2,481 | `cf2b801d0738c88abb691bf4a096000c0e0737faab8a54a4ba5243c5513a3711` | causal |
| `encoding/tests/test_output_4.txt` | 574 | `94433b68370dcc69fa61efc5011b5d0e28685e4716fee1ee77fbbfc0c9935b54` | causal |
| `encoding/tests/test_output_5.txt` | 408 | `4d673091746553d1cb2e40a3fc01dc09a7c53c40627096a3e49c97a269a81f64` | causal |
| `gptq/layer_00.json` | 34,340 | `070aac9c2d3e5efb5d67c791a42c48f7f0851380b5bedbfc933abfc99a4f4c85` | causal |
| `gptq/layer_00.safetensors` | 1,238,726,280 | `317a443d76d1e38a3c217c9ea0aad4c90f22411b8cb1b5a01f101e804d7740ff` | causal |
| `gptq/layer_01.json` | 34,241 | `1ffdc01259997f226f7cf4a8d6585a4e0037db567a3aa8ff37aac38780ec1624` | causal |
| `gptq/layer_01.safetensors` | 1,238,726,280 | `7538fdc99e7c0a9426a0907b80cc5bc19abb2b9828a39fcc737f8ca684e021cb` | causal |
| `gptq/layer_02.json` | 34,261 | `a160f0dbacb1cd41856a01f9a22d61bdd2c8049c3bc890de98b98dc042e29bf7` | causal |
| `gptq/layer_02.safetensors` | 1,238,726,280 | `f74972458567cc90cc455fafdab9d3bc645c3d56bb58982e8020a539268c9336` | causal |
| `gptq/layer_03.json` | 34,299 | `fc2a80ea284929ff92dea4281fd7344d703f8262a5fffad91348ce8b9369028d` | causal |
| `gptq/layer_03.safetensors` | 1,238,726,280 | `269b976a4c69fcdc1b03d0ce85e4cc9e8e98b4a0798bea24ef1f5314fea49e8a` | causal |
| `gptq/layer_04.json` | 34,264 | `c26b2c42eb759e867f658453659ba5d10923d406580d45dfee9faa9289b402e8` | causal |
| `gptq/layer_04.safetensors` | 1,238,726,280 | `a7d171edfff175cf6b10f32a6f38205b714729027a3d64901910be7755c57492` | causal |
| `gptq/layer_05.json` | 34,219 | `d02f762fa77cd61c34daa592d6939741ee207d305ef80e84fd918b6e5d757881` | causal |
| `gptq/layer_05.safetensors` | 1,238,726,280 | `c0f5bc7523e2c4dd209b2b14feba523e1d890af945ef8eef0aadbb7cbbd3f7cc` | causal |
| `gptq/layer_06.json` | 34,243 | `7f38e6e92138983d416f53778ef368539005993b1fa390b8bc378218b255cd1b` | causal |
| `gptq/layer_06.safetensors` | 1,238,726,280 | `b11c3032524aa6b15a0925d91d4883f3c5efc227eada802c4edcaed8cd850c24` | causal |
| `gptq/layer_07.json` | 34,220 | `9268b12d990ed8d0d34dca3f7098bc80bc2e1fec718dd21c96a17c7346303884` | causal |
| `gptq/layer_07.safetensors` | 1,238,726,280 | `2202eebfa49da9a84f80f762c6abfb8f37ccf43d5c1b1faf817cbdfd6c392228` | causal |
| `gptq/layer_08.json` | 34,226 | `ec36434d5ab963d6e1d80eb69f02e1e2718e07faf911852cc387b1fd00d66f0d` | causal |
| `gptq/layer_08.safetensors` | 1,238,726,280 | `a6833ed77ed13a0d1b56424c3aaca08edc975bb4a6f8de5cdb6f10ca099755b6` | causal |
| `gptq/layer_09.json` | 34,221 | `71fc390bfcc6e3b8a2a45c2997e5a4eb6d92009d8bc39c5af76f693ab57bb066` | causal |
| `gptq/layer_09.safetensors` | 1,238,726,280 | `892e97272cbcd62db53e3078f69c4ba9342ce4411306e680fa3562394bcac364` | causal |
| `gptq/layer_10.json` | 34,179 | `f3ecd2f3e6f31932d2eac48b8a0c90b73cf2139f5b883c89576bc381b7cd5f43` | causal |
| `gptq/layer_10.safetensors` | 1,238,726,280 | `3b867f63df53a588720f091344a562a5281325c5ca419faa64ae355142a27c0f` | causal |
| `gptq/layer_11.json` | 34,189 | `1dc89b8c6e9d1ae883eb5841167cbd135776924f00d4d366fb8cdd0f3228dde9` | causal |
| `gptq/layer_11.safetensors` | 1,238,726,280 | `c082b95236552f2c78b8ea0a58d5eda00e1d5da8612d1917678227ae4bcdaca9` | causal |
| `gptq/layer_12.json` | 34,149 | `5c28479d3ff99fb7636a757add096a7031b094ff54148120c3b6d86f7a0f8fe9` | causal |
| `gptq/layer_12.safetensors` | 1,238,726,280 | `2f11d2342449defa3d134de479c4d06fce0c1e13f7df93ea23532c58c9ef67a8` | causal |
| `gptq/layer_13.json` | 34,179 | `cb280279b9df00834ff530cf3ea51068f5fff29f1f9bd4b32ee87e22a5c56ee8` | causal |
| `gptq/layer_13.safetensors` | 1,238,726,280 | `f65a2852858b873166e12784e5766f799517c6697c748848a7eafdc1dd8347b7` | causal |
| `gptq/layer_14.json` | 34,170 | `862e66a8891e09c8ea08a52e21c07ebe74216008475bf887a9f050082dbea4ab` | causal |
| `gptq/layer_14.safetensors` | 1,238,726,280 | `5c44952aff5e7603c762505ed69d4548c350b0836693afde9e29cb1ba9f09237` | causal |
| `gptq/layer_15.json` | 34,180 | `3a6eae3e1559fdaf83a00429486c3d2c541fe5a4b068f4413ed42803484fc1e2` | causal |
| `gptq/layer_15.safetensors` | 1,238,726,280 | `74802900fc10f00af1d9c382f9ad8129a0660c465ba27442a1d910990d44cfd9` | causal |
| `gptq/layer_16.json` | 34,132 | `2bd79ef99d0afa9323a9f6fcdc4516883ea56e1cdb6398728d7f84cf17ff15ab` | causal |
| `gptq/layer_16.safetensors` | 1,238,726,280 | `b5e2b015b533a00098b5584413f82aff1429249953ed85aaa7b2ad89e42aefc1` | causal |
| `gptq/layer_17.json` | 34,155 | `5bb4eb7bd2be50ebc7b1a0378a7f7147d7e596e4c88d5477122521d318c6892c` | causal |
| `gptq/layer_17.safetensors` | 1,238,726,280 | `6274e79575a09ceb8b54057d53228f1d3ab528500d0193e5ef254000fe34b90a` | causal |
| `gptq/layer_18.json` | 34,132 | `2fa05fbe530ccdea385d7c41be04789816e5d20d07bb770bc48cec420e6b0434` | causal |
| `gptq/layer_18.safetensors` | 1,238,726,280 | `7f7fe9b6b4bb0b5eb619cb1058b854419e4e56990e6ef7d41b426890ed3edccd` | causal |
| `gptq/layer_19.json` | 34,186 | `74b9fe2f2966fdce4f8d1658ba8bdda1bf0cf92bc91c2504e80e6f0e524b9194` | causal |
| `gptq/layer_19.safetensors` | 1,238,726,280 | `a4c9e8a81e55c044b63bfbf6051e0a259aa2fb399c683251eaa0d67f5592b785` | causal |
| `gptq/layer_20.json` | 34,162 | `88c7bef916092e3e25c38c9105cfef16377cba7f4ebc3105e0baa3fc13bf5b93` | causal |
| `gptq/layer_20.safetensors` | 1,238,726,280 | `92b35a1c7467d4a0d6c2f6b8cc0f19dc31216eee79034cb9c07eb59508335956` | causal |
| `gptq/layer_21.json` | 34,153 | `8041215746ce85f4800ab2da886f79e0967b3f02d5531405b9d7a9462ff1e494` | causal |
| `gptq/layer_21.safetensors` | 1,238,726,280 | `14127a8e9a611d83e78a5d6390e63f02c71e6dae861eab3ffd1266fda9d51db2` | causal |
| `gptq/layer_22.json` | 34,151 | `47f3c810a7ef6e025092a19eea338b06bd8751db4b72ed1a76dcdc934e6d88c5` | causal |
| `gptq/layer_22.safetensors` | 1,238,726,280 | `3b2bbfae7c41606de33e65fb48b7a9e5e978e266a4e49e7c75e794be130417e5` | causal |
| `gptq/layer_23.json` | 34,164 | `1783709d308527c2625abaddc42292a807cc6f24891c524c510c5569d5aadf3e` | causal |
| `gptq/layer_23.safetensors` | 1,238,726,280 | `2ecfa9b6087195c5da3a00ca84f6b201d4d0954b7ba05667d72e143bfdec6284` | causal |
| `gptq/layer_24.json` | 34,142 | `47485f070621facadd7825034e1820054f01668bfba6f97b099767137ba5c413` | causal |
| `gptq/layer_24.safetensors` | 1,238,726,280 | `1c8dcb06ab6d892c771e785f4263bf801330c161cc2d24ebd8874490a00872f7` | causal |
| `gptq/layer_25.json` | 34,176 | `ec4a04eb0e3a057aeaa1d744c1bb2fea1fc54a3ece3fed7f1d0674bc64aa91d4` | causal |
| `gptq/layer_25.safetensors` | 1,238,726,280 | `510da18ab1ea9346d8b9fa08cb10d50a1447738de77b503f452d095acaf6708d` | causal |
| `gptq/layer_26.json` | 34,118 | `0810c688fcdd1803ac79eae9f1c5d40f7c1c4ed49f7fe882f6876f2685b5bfa3` | causal |
| `gptq/layer_26.safetensors` | 1,238,726,280 | `f53b5ea0e70c53be08ea6b4a8af14e32b7883bc2b2fc12060328240ff6f1e933` | causal |
| `gptq/layer_27.json` | 34,140 | `384551797a7105b448185de6e18c93b05d005bcae2a444be4b797b31789eb824` | causal |
| `gptq/layer_27.safetensors` | 1,238,726,280 | `60ccd8cad901c39a57e44219e9bdcaeabec7af85721cd51cf734af958f87a735` | causal |
| `gptq/layer_28.json` | 34,139 | `5c1e700286018aabf3af2dfc82fc71074e28176abe1bb5996955f6f99548a3c2` | causal |
| `gptq/layer_28.safetensors` | 1,238,726,280 | `bed03e8132b5a5827df0def5aea73888d03120a808c4fb83a75cf6c6d58cd21a` | causal |
| `gptq/layer_29.json` | 34,132 | `e422ca8265c161270e63177a5e80810770f3a805330428e7b87ea2ef012935de` | causal |
| `gptq/layer_29.safetensors` | 1,238,726,280 | `709383155071bbce83202f6a9c2797f99bf44925699e6de61c418775655a97c0` | causal |
| `gptq/layer_30.json` | 34,129 | `777288cf3b860e496ee81c0da4b4c5bda9e4e6b85459c4f2385d4e9b452404a4` | causal |
| `gptq/layer_30.safetensors` | 1,238,726,280 | `b96b900026702ab97dc64539aeac836b563e423d0c939ff49c6ab3c57ad1cce3` | causal |
| `gptq/layer_31.json` | 34,144 | `234d4b39f58327963dfff8b724fabfbc21db3e2ae818731b10c8130ae03f402f` | causal |
| `gptq/layer_31.safetensors` | 1,238,726,280 | `4c1e6b1b030a7bb954bb7da515c5ccef4d143eaaf266b750d378f08edb708240` | causal |
| `gptq/layer_32.json` | 34,133 | `70c2c36cb82b0af47d133627bb33de31d2e2f650dcce3787747195e5e29f361c` | causal |
| `gptq/layer_32.safetensors` | 1,238,726,280 | `85e8066b6ccd720d256b0b423541074e2eea0852536dbf3f7a1a70271b709943` | causal |
| `gptq/layer_33.json` | 34,128 | `68bda52492a29bb85dca0887c48339ce79f5337a95020e7a78ce97d3f6c81311` | causal |
| `gptq/layer_33.safetensors` | 1,238,726,280 | `0ce9a73fc3c63ac5b689cc094f7d7b959cbeec77285075e671e860788910588b` | causal |
| `gptq/layer_34.json` | 34,160 | `c4b0531b9e3554c4bdba9c7f5b589e467619a0365eb53945beae1beb01396eb7` | causal |
| `gptq/layer_34.safetensors` | 1,238,726,280 | `d229b97c1fa761557a288912dc920a7a422c9cf4cfe7beea9a515b5dbde69224` | causal |
| `gptq/layer_35.json` | 34,198 | `f0e06c93951436311a9528aa33f4a8f19381db58a95487e8407101060115b1bb` | causal |
| `gptq/layer_35.safetensors` | 1,238,726,280 | `d74a45d3c5316c6853d3dd887ab6af8331ab571cdff39b43c06512b9da4d09de` | causal |
| `gptq/layer_36.json` | 34,168 | `9011da348df1c159087db0c2abe1ab457e41ed9f7c03d8b96c8e1c8d94da25e1` | causal |
| `gptq/layer_36.safetensors` | 1,238,726,280 | `4115669409d498dcac16c8e3c62de00f29d391a9559da7e87dae13336cbaa45e` | causal |
| `gptq/layer_37.json` | 34,175 | `7a317a064246b4abbbf96edecf8ae83eda502a3153b04f2147337d186f579b5d` | causal |
| `gptq/layer_37.safetensors` | 1,238,726,280 | `0ea91d3238a944c6c2656d4bdb1b6885ab8df8a30840536800b8a446929fea35` | causal |
| `gptq/layer_38.json` | 34,224 | `db1f954fd4f2125e2957344e0e89ce841b33e965c06fb2edf1ea58daf248ec33` | causal |
| `gptq/layer_38.safetensors` | 1,238,726,280 | `2e1da89783fa20ddce7beb70e79c1463f9917e83cafa02335d933f8a2eaff0db` | causal |
| `gptq/layer_39.json` | 34,308 | `ad9912d1dd56de652925a89e3d90cd1b3e5f4873f96855832a562b2275328518` | causal |
| `gptq/layer_39.safetensors` | 1,238,726,280 | `81d772a75e0887f6ef8b9ae976e8d4bff184c3cac66d74d03f4cf5daeaf674e1` | causal |
| `inference/README.md` | 2,029 | `2834402823199ee24e9a42bdf36a0fc6daf94448f444cb062c042a057a798f1c` | causal |
| `inference/config.json` | 1,982 | `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809` | causal |
| `inference/convert.py` | 9,458 | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` | causal |
| `inference/engram.py` | 8,138 | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` | causal |
| `inference/examples/example.txt` | 332 | `8456a2f1d5a15b7aa1f697567e34f79af0522fb694df7ac789dae9853fff165d` | causal |
| `inference/examples/example_harmony.json` | 2,156 | `44ae84a3cbd360b1b8d303ed4006756afb77e5d014c4abb7c61ff9b9a094f6ea` | causal |
| `inference/examples/images/carrots.jpeg` | 212,495 | `5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a` | causal |
| `inference/examples/images/corn.jpeg` | 56,122 | `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1` | causal |
| `inference/generate.py` | 8,722 | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` | causal |
| `inference/image_processor.py` | 7,699 | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` | causal |
| `inference/kernel.py` | 23,790 | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` | causal |
| `inference/model.py` | 61,669 | `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d` | causal |
| `inference/requirements.txt` | 98 | `be82bcd5775ccf8361ddd22cac53ef7b3b83434697fc3c61c75870450fb9437e` | causal |
| `inference/run.sh` | 1,759 | `91eddd7895f9fcd9bace53c21128c506e741e770f3c1a02464eff51f60401dfb` | causal |
| `inference/vision.py` | 4,457 | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` | causal |
| `model-00001.safetensors` | 4,890,753,920 | `307cf222a9ecf9dff1fbe1ac3ec3d8fb711800f320ce001c568aeb1cd74731c2` | causal |
| `model-00002.safetensors` | 4,182,866,440 | `a96c7d6bbc090746e6df11e37f8029aa5765a1de3002072d4213210e1b511d63` | causal |
| `model-00003.safetensors` | 1,638,882,320 | `331c2723b1c6ef214d99eba344d4fd1dde45969d5a42bdbdca1d23c249be6b12` | causal |
| `model-engram-00001-of-00016.safetensors` | 1,088,919,216 | `55c00ea48a06a67028a7c5eeb3897ae8e7f4119b0de0d1c2c8088b2674ded583` | E128 |
| `model-engram-00002-of-00016.safetensors` | 4,998,951,576 | `fd43f6cc023ff267c37085ccecb074305d235fb3a23747d3bc99d22db6a81ef3` | E128 |
| `model-engram-00003-of-00016.safetensors` | 4,998,951,584 | `3fa7839ef761a09e7937b78aa917c706570bd2309fb0a410104880a46cb3f05c` | E128 |
| `model-engram-00004-of-00016.safetensors` | 4,998,951,584 | `1c897280d993b03794cd6d14712e2f995c5a683935e686d1f0d6c43920e1e13a` | E128 |
| `model-engram-00005-of-00016.safetensors` | 4,998,951,584 | `d214812b29e538caaa0831ece23c636690699587763a9458ad44ce04bce513ef` | E128 |
| `model-engram-00006-of-00016.safetensors` | 4,998,951,584 | `261a5fef67de77a7877395737f783b4adaadb7c9c0098019bd7cfda95afa4804` | E128 |
| `model-engram-00007-of-00016.safetensors` | 4,998,951,584 | `f781ae4a959389211dbf92f4ad248c52c4de675d59d71f32a87aa59bb200809b` | E128 |
| `model-engram-00008-of-00016.safetensors` | 4,851,701,920 | `c1d6d4993c9316a331464f9d07f97cbb45d6b68ff58dcb370e574a8a08bfecaa` | E128 |
| `model-engram-00009-of-00016.safetensors` | 2,529,514,248 | `5de637f1aaecb973114866a6af06a67ffcafed5c7ab7f21a1746b93c86e5a66a` | E128 |
| `model-engram-00010-of-00016.safetensors` | 4,998,951,576 | `5a5d3e532bf817a2753685f133ab1144b07c0771f0a5389d44033e62c656e13e` | E128 |
| `model-engram-00011-of-00016.safetensors` | 4,998,951,584 | `3560c30b277c7cfcf826ba3d9ff38212fcdc7703f087a858f0ccdd914e3e427f` | E128 |
| `model-engram-00012-of-00016.safetensors` | 4,998,951,584 | `041646d1e4c7ae4e4c8473504ed0777dda00278b3210532b449693964add675b` | E128 |
| `model-engram-00013-of-00016.safetensors` | 4,998,951,584 | `5e9f091824203c027f2cdeba6d294de700dd2587b452a9cbfcdc97f81125f6f9` | E128 |
| `model-engram-00014-of-00016.safetensors` | 4,998,951,584 | `32c53926732fa89c634147a5048c44221ed6851f7915f6db85045a1cebad3360` | E128 |
| `model-engram-00015-of-00016.safetensors` | 4,998,951,584 | `e6c9a6bed470838bc472519603f53646e2b727c64b937a12b3ca2f2e67850851` | E128 |
| `model-engram-00016-of-00016.safetensors` | 3,334,016,448 | `69b910b6c8e4c1c98d4a117b0cf844218f10e6b07b72346386e0e7752dd548b8` | E128 |
| `model.safetensors.index.json` | 93,569 | `cfd5f921f9b42f9bcf0f129c7a4160e1d5163d07e264e16c483d7ef18867670f` | causal |
| `serving/apply_compact_engram.py` | 14,059 | `fd3ef2f9d51eb2e44b572db783b0b384952b0e3319fed74c9e147cac5aa92f17` | causal |
| `serving/apply_e043_humming.py` | 10,843 | `991074c8f2a47834d98b369eee757b0e86a777802c811fc47e05f7bb522f787b` | causal |
| `serving/e043_humming.py` | 4,780 | `74dcbb141ff0ad29b56123f1ae24eef8e0249dea48a36dbe76d983a7fa2be8d8` | causal |
| `tokenizer.json` | 6,367,257 | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` | causal |
| `tokenizer_config.json` | 801 | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` | causal |

</details>
