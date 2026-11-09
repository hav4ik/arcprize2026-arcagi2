# DSv4.1 Kaggle track: state, findings and next steps (handoff, 2026-10-30 ~23:45 UTC)

Read this first to continue the DeepSeek-V4.1 Kaggle work (ARC-AGI-2, Kaggle 4xL4, 12 h scored rerun). It summarizes
2026-10-29/30 and points to the detailed notes and data. Author: the "dsv4-infer" session.

## 1. Where things stand

**Leaderboard (best first):**

| Submission | LB | Pod replica | Notes |
|---|---|---|---|
| **Manh V12** (`nguyennguyen599/dsv41-tools-submit-b32m-external-server`, v12, 2026-10-29) | **76.25** | 80.83 (0 events) | the gold standard; B32-M + Manh's vLLM server |
| B32 (`candb` V2, 10-27) | 68.89 | 77.4-79.1 | our own vLLM 0.28 stack |
| B32-M V3 (10-28) | 46.5 | 50.4 Kaggle-matched / 77.9 pod | repeated GPU OOMs (sec. 3.1) |

- **Smoke-tested today:** `chankhavu/dsv41-tools-v12-img` v1, "V12 + images". Kaggle commit run COMPLETE: 8/8 correct, 0 retries. It was the candidate for the 2026-10-30 slot. The user was choosing between it and a V12 resubmit; check the LB for which went in.

**Two pods were running when this was written** (Runpod 4xL4, $2.36/h each; the user rents; never stop pods without the user):

| Pod | Run | Data |
|---|---|---|
| `5z3srm3108wcwl` future_tan_falcon ("pod 2", V12's own pod) | "V12 + images FINAL" (notebook `d5e545d4`), launched 19:58 UTC; stop at Kaggle-clock 7.5 h, then write up | `experiments/E062_manh_vllm_fork/per_layer/phase_h/v12_img/` |
| `ccn5ns0b8emb3o` yawning_salmon_cobra | "V12-fast" (notebook `9f7e7074`), launched 22:39 UTC; run to >= 8 h | `phase_h/v12_fast/` |

- `9u4jdy2rczh9fu` (pod 4) is stopped (EXITED). Its host is ~6% slower (sec. 3.4).
- Runpod balance at 22:00 UTC: ~$47.

## 2. Notebooks (all under `kaggle/dsv41_tools/out/`)

| Folder | sha256 (prefix) | What | Status |
|---|---|---|---|
| `manh_v12_363638159/` | c47be5a8 | Manh's V12 as submitted | LB 76.25 |
| `nb_v12_img_submit/` | **5bea5b7f** | V12 + images on the causal model (Kaggle `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-causal/pyTorch/default/1`) + Manh's flags + replay graphs + recovery A+B + no early server give-up + latest docker + 4-task commit smoke (stop-new 45 min, kill 51 min) | pushed as `chankhavu/dsv41-tools-v12-img` v1; smoke COMPLETE |
| `nb_v12_img_final/` | d5e545d4 | the same server/solver as the submit notebook, model input C1 placeholder (the pod mounts the causal model at the same path) | running on pod 2 |
| `nb_v12_img_ab/` | 5f4cfe5d | the above without replay graphs | replica r1 (0.98 h) |
| `nb_v12_img/` | 45544302 | + turn replay (resume an attempt's turn after a server death; `ARC_REPLAY`, `ARC_REPLAY_CONCURRENCY`) | fake-server tests only, no GPU run |
| `nb_v12_img_te/` | 361a6998 | + PR #3 token-exact image turns (`/inference/v1/generate`) | correctness passed on GPU; A/B not run |
| `nb_v12_fast/` | **9f7e7074** | V12 TEXT (old tool E128 `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1`, image limit 16) + `--watermark-reload-only` + replay graphs + recovery A+B + startup-timeout fix | running on the new pod |
| `manh_review_v12_img/` | b488ee1e | review package sent to Manh (README_FOR_MANH.md + diffs) | — |

- **Build scripts:** `kaggle/dsv41_tools/build_v12_img.py`, `build_v12_img_te.py`. `nb_v12_img_final`, `nb_v12_img_submit` and `nb_v12_fast` were made by byte-minimal string patches of `nb_v12_img_ab`; each patch is recorded in this note and in the session transcript.
- **Tests:** `kaggle/dsv41_tools/tests/test_v12_img.py`, `test_v12_img_te.py`, `test_img.py`, `test_input_scans.py`, plus the fake-server recovery/replay suites.
- **Run tests** with `.venv/bin/python -m unittest tests.<name>` from `kaggle/dsv41_tools`; pytest is not installed.

**Recovery A+B** (solver `0c2f81e8`, active only on failures):
- An attempt that hits a server-side failure (connection drop or refusal, HTTP 500, client timeout) is held until the supervisor reports a newer start as up, or `/health` answers.
- It gets at most one free re-queue per server start.
- The normal path's request log is byte-identical to V12's (tested).

**Replay graphs** go in as ONE full JSON: `'--vllm-arg=--compilation-config={"cudagraph_mode":"FULL","max_cudagraph_capture_size":64,"decoder_replay_cudagraph_capture_sizes":[32,64,96,128,160,192,224,256]}'`.

## 3. Findings (each with its evidence)

### 3.1 GPU memory and OOM (`notes/2026-10-30_b32m_oom_modes.md`)

- **The margin that matters is the allocator cap − reserved**, not cap − peak.
  - B32-M OOMed at cap − reserved ≈ 0.03 GiB while cap − peak still read 0.40.
  - Alarm thresholds: the first allocator retry, or cap − reserved < 0.15 GiB on Kaggle.
- **Pod margins:** V12 flat at 1.08 GiB from 2.8 h. The image candidate's slides faster (1.46 → 1.17 → 1.03 → 0.93 at 0.25/1/1.5/2.9 h) and is flattening.
  - The extra is reserved, not peak: fragmentation from image prefills, replay graphs, and the record_stream wheel (+0.12 reserved in the SFT run).
- **Kaggle vs pod memory (the image smoke, 2026-10-30):** reserved + free on Kaggle is only ~0.05-0.1 GiB below the pod's; KV pool 1,527,155 vs 1,531,221; graph memory 0.25 vs 0.24 GiB.
  - Older B32-M commits showed CUDA graphs at 0.65 vs 0.34 GiB, but that was a different stack.
- **Don't raise util to 0.95** (Manh's prod value): ~0.9 GiB less margin per GPU, which is B32-M territory.

### 3.2 Recovery cost (pod 4, V1 fault tests; `per_layer/pod4_runs/v1_qual/README.md`)

- **Kill of one TP worker:** back in 176 s, 0 attempts lost, but ~40 min of serving lost (31 attempts restarted from scratch, 1.16M tokens).
- **Engine SIGSTOP (hang):** 33 client timeouts burned tries at +900 s; the stall restart at +25 min; ~78 min lost.
- **Fixes:** R3, R4, R9 in `kaggle/dsv41_tools/next_candidate_fixes.md`. A+B covers R3/R4; turn replay (R9) is built in `nb_v12_img`, not GPU-tested.

### 3.3 V12 recovery as submitted

- Manh's supervisor restarts the server on exit, up to 3 times.
- The solver re-queues cut attempts (CONN_TRIES 4, JOB_TRIES 2).
- vLLM's 300 s step timeout turns a stuck GPU step into a death.
- **Gaps:** a dead port burns re-queues in seconds; plain 500s and client timeouts spend real tries; attempts restart from scratch.
- **Startup give-up:** V12 gave up a server start after 2 h (`ARC_SERVER_UP_TIMEOUT_S` 7200), after which the rerun scores the fallback, ~0. Our notebooks set it to 39000 (the whole window), the launcher call timeout to 3 h, and the import check to 1 h. Other teams report very slow input loading on Kaggle.

### 3.4 Pod speed differs; compare notebooks on the same pod

- V1 trailed V12 by ~10 tasks at 3 h. The cause was pod 4's host (`pod4_runs/v1_qual/SLOWDOWN.md`):
  - decode 0.937x, then 0.70x after its restart;
  - the TP3 rank sat on the other NUMA node;
  - the same notebook on pod 2 matched V12 (1.003x).
- **Rule:** A/B on the same pod, or calibrate decode speed at matched running/KV.

### 3.5 LB vs replica

- V12: replica 80.83 at 10.2 h, LB 76.25. Two explanations, not yet separated:
  - the hidden test set is harder (other teams report a 3-5 point gap);
  - Kaggle runs ~1.5x slower (the user's rule "8 h pod ~ 12 h Kaggle"). Kaggle's ~11 h of solving ~ 7.3 pod hours, where V12 stood at ~76.5.
- **To separate them:** Kaggle's own gen tok/s under full load against the pod's. The 4-task smoke was too light to measure it.
- **V12's replica by hour:** score 18.1 / 29.8 / 40.4 / 53.3 / 68.3 / 74.0 / 75.6 / 79.2 / 80.2 / 80.8. Tasks with both attempts: 0 / 0 / 0 / 0 / 4 / 16 / 40 / 101 / 157 / 215.

### 3.6 Images (E066/E058; `experiments/E066_dsv41_image_pruning/README.md`)

- **Causal image model ("c1f"):** recalibrated after fixing the bidirectional-image-attention bug (E068). Manh's wheels v10/v11/g22df163d2d contain upstream #57152 (causal), so no `--hf-overrides` is needed.
  - Risk: a model `config.json` with `"is_mm_prefix_lm": true` would force bidirectional. c1f's config has no such key.
- **Node 5 (V12 harness + images + auto-repair, partial 77 tasks):** pass@1 +3.7 [-1.1, +8.9], pass@2 -1.3 vs the same-box old-E128 text.
- **Pod 2 replica (same pod as V12):**
  - at 1.5 h: paired -4.0 on 63 tasks;
  - at 2.9 h: score 38.8 vs 40.4, 121 vs 126 tasks;
  - first-attempt accuracy 75.4% vs ~73%; gen tok/s ~-10%.
- **Text + image second attempt (offline, node-5 rollouts, 115 shared tasks):** pass@2 text+text 80.9, text+image 83.3 (+2.4 [-0.6, +5.8]), image+image 84.8. The text baseline there is a weak run.
- **The chat path re-tokenizes history.** V12's token-exact records show a BPE re-tokenization mismatch on **29% of turns** (2,532 / 8,699). The image path loses its cached prefix there: +64% uncached prompt tokens per turn, prefix hits 90% vs 94%.
  - PR #3 (`hav4ik/arcprize2026-arcagi2#3`, Manh) fixes this via `/inference/v1/generate`. Correct on GPU; see R12.

### 3.7 Manh's server changes (2026-10-30)

- **Wheel:** `vllm-0.32.1rc1.dev0+g22df163d2d`, Kaggle dataset `nguyennguyen599/vllm-wheels` v16+, unpinned.
  - The HF copy `livobereshka/tool-wheel` also holds the stale `g70867e279f`, which uv prefers. Exclude it.
- **`--watermark-reload-only`:** the 0.1 KV watermark then gates only preempted/offload-reload requests. Exists only in g22df163d2d.
- **Replay graphs:** his launcher `--dsv41-vllm-replay-graph-max 256` = sizes `range(32, 257, 32)`. Claimed +9-11% decode; **not confirmed on our workload yet** (R11).
- **New launcher options (arc2-2026-src latest, all default off):**
  - `--dsv41-vllm-engram-model`: Engram tables from a second model input, for weights-only uploads;
  - `--dsv41-vllm-vision-weights`;
  - `--dsv41-vllm-replay-graph-max`.
- **Concurrency:** Manh suggests 80 (he stress-tested 120). We kept 64: the run is KV-bound after ~35 min, and decode CUDA graphs are captured only up to 64.

### 3.8 Fine-tune (Anastasia's QAT SFT)

- **`chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch` ties the base model:**
  - pod 2 at 4.76 h: paired -1.9 [-6.9, +3.5];
  - 2x RTX PRO 6000 (E069): pass@1 71.3 vs 69.9, pass@2 80.0 vs 81.4.
- **Its 2-bit routed experts are byte-identical to the base E128.** Only the 240 shared-expert tensors differ.
- **Open question for Anastasia:** were routed experts trained and exported? Check tensor checksums of any new checkpoint before spending GPU time.

### 3.9 Answer keys: GitHub vs Kaggle evaluation versions (memo: https://claude.ai/artifact/8pkidsHURYVSsAR8Y1n6kp)

- **6 tasks differ:**
  - 4a21e3da, abc82100, b6f77b65, f560132c, faa9f03d: Kaggle adds test inputs; faa9f03d's training pairs also differ.
  - d8e07eb2: training pairs only.
  - 114 tasks are identical. Test pairs: Kaggle 172, GitHub 167.
- **Team rule: score on the Kaggle version.**
  - The harness default `--challenges` is the GitHub dir.
  - The pod workload `evalx2t_s2026` (120 Kaggle originals + 120 rotated/flipped, colour-permuted copies) is built from the Kaggle files.
- **Effect:** ~1 point on 120 tasks. Kaggle's versions of the 6 score ~53-56% pass@2 vs ~83% elsewhere. The transformed copies score ~2 points below the originals.

### 3.10 The pass@2 ceiling (why ~81-83 and not 85)

- **V12's setup on the 115 shared tasks:** 83.0 (pod replica originals) and 82.9 (node 5).
- **Higher runs on the same 115:**
  - E046 run 7, old E128, E046-best flags: 85.7;
  - C1 text, E066 settings: 87.0;
  - API: 89.9.
- **Ruled out (local evidence):**
  - **Context limit:** 2 of 480 V12 attempts hit 163,840.
  - **Hardware/kernels:** V12 setup on L4 (83.0) = on RTX PRO 6000 (82.9).
  - **Auto-repair over-correction:** repaired attempts are as or more accurate at matched turn counts, 84.9% vs 78.9% among train-passing attempts.
- **Remaining suspects:** Docker sandbox vs the solver's local executor, the harness build, noise (one run each, ~3 points).
- **Next:** a same-box A/B (V12 vs V12 + Docker sandbox vs E046-run-7 settings, optionally auto-repair off), old E128, Kaggle versions, 4 attempts/task, ~2-3 h on 4x RTX PRO 6000.

### 3.11 Where a 3rd rollout would go (V12 replica)

- First attempt alone 73.1, best of two 80.8.
- 53 tasks are not fully solved after two attempts: 26 where neither attempt passed the train check, 27 where an attempt passed it but was wrong.
- A targeted third attempt for the first group needs time that Kaggle may not give.

### 3.12 The rerun fallback is a crash path (R10)

- If cell 0b's check fails in the scored rerun, the notebook restores the kit's v3 solver. That solver needs a vLLM venv that Manh's environment lacks, so the run scores ~0.
- **Fix before the next submission:** keep the pinned files on failure.
- Any edit to a notebook must update `_MT_EXPECT` and `_MT_NEED` in cell 0b to match cell 00.

## 4. Tools and infrastructure

- **Dashboard (shared, link-shared by the user):** https://claude.ai/artifact/RvbHR1L9axaNcDi73nSzbM
  - It is a Dashboard-type artifact; the page lives in its db `files/index.html`; datasets point at uploaded JSON blobs.
  - Source of truth in git: `kaggle/dashboard/submissions_tracker/`.
  - `tools/auto_dash.sh` rebuilds the JSONs every 5 min from `phase_h/*/data/` (new runs are auto-discovered; finished runs are frozen in `frozen_*.json`).
  - **Local live page:** `python3 -m http.server 8765` in that folder → `/local.html`. It was served on the LAN at the time.
  - **Publishing the shared page needs the Artifact tools:** `tools/publish_list.py` lists changed files; upload each as an asset, update `datasets/<id>` with `if_version`, then `publish_list.py --mark`.
- **Pod scripts:**
  - `experiments/E062_manh_vllm_fork/per_layer/phase_h/scripts/` (launch_v12img.sh, stage_sft.py, hf_snap.py, the hourly comparisons);
  - `per_layer/pod4_scripts/` (Kaggle model downloads with the token over stdin).
- **Large data:** HF private dataset `livobereshka/arc-agi2-vllm-serving-lab` (`replicas/<run>/`).
  - The readonly HF key is OK on pods.
  - Uploads go from the local machine with `huggingface_api_key`, after a private check and a secret scan.
- **Kaggle smoke of v12-img v1:** logs are in `experiments/E062_manh_vllm_fork/per_layer/kaggle_smoke_v12img_v1/`. Server up after 458 s; solving 23:22-23:27 UTC.

## 5. Working rules learned these two days (also in the user's memory)

- **Kaggle:**
  - **Runs:** only with the user's OK for each run. About one ~40 min smoke per day fits the quota; 4xL4 bills 2x quota.
  - **Before Kaggle:** test everything locally (pods) first, including a commit-mode rehearsal of the exact notebook.
  - **Visibility:** never public. Notebooks private; models and datasets shared only within the team.
- **Agents and budget:**
  - The weekly agent quota is tight: few sub-agents, low cadence, haiku/sonnet for small tasks.
  - No recurring refresher agents: the background loops cost no turns.
- **Data hygiene:**
  - Every number states its eval version (Kaggle), task count and pass@k.
  - Replica comparisons are made on the same pod.
- **Serving changes:** never trust a flag translation without checking the server's "non-default args" line at load. The dotted `--compilation-config` override silently replaced the whole config.

## 6. Next steps (priority order)

1. **Read the overnight runs.**
   - Pod 2 image run to 7.5 h: does cap − reserved stay > ~0.8 GiB? Score at 7.3 h vs V12's ~76.5.
   - V12-fast to >= 8 h: does it beat V12's curve at 7.3 h once host speed is calibrated? Its early decode was ~17% below V12 on a different pod.
   - Each pod agent writes a README and uploads large files at the end.
2. **Fix R10 in every candidate notebook** before the next Kaggle submission, with a test.
3. **Ceiling A/B (sec. 3.10).** Then decide on V12 + images vs V12-fast vs V12 for the next submissions.
4. **If images stay:** a text-first, image-second attempt mode (offline +2.4 pass@2), PR #3 (fewer re-prefills), and turn replay (R9). Each needs its own replica before shipping.
5. **SFT:** confirm the export with Anastasia (sec. 3.8) before evaluating more checkpoints.
6. **Ask Manh** how he measured +10% from replay graphs, and pass R12 to him for PR #3.

Related notes: `notes/2026-10-30_b32m_oom_modes.md`, `notes/2026-10-30_vllm_failure_paths.md`,
`notes/2026-10-30_vllm_fork_acceleration_map.md`, `notes/2026-10-30_illegal_memory_access_risk.md`,
`experiments/E067_vllm_lab/README.md`, `experiments/E062_manh_vllm_fork/per_layer/phase_h/README.md`,
`kaggle/dsv41_tools/next_candidate_fixes.md` (R1-R12), `kaggle/dsv41_tools/out/nb_v12_img/README.md`.
