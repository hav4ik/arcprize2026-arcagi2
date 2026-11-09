# V12 + images

Manh's V12 notebook (`nguyennguyen599/dsv41-tools-submit-b32m-external-server`, scriptVersionId 363638159, sha256
c47be5a819ad; replica 80.83 with 0 events) with the E058 image harness switched on. Nothing else changes its behaviour.
Built by `kaggle/dsv41_tools/build_v12_img.py`. Nothing is pushed or uploaded.

| File | sha256 |
|---|---|
| `dsv41-tools-v12-img.ipynb` (23 cells; Manh's flag + recovery R1 + R2) | 45544302b092832396437cb8361508edd65600ffbb8d8be4b946bfb09304e0b2 |
| A+B only (no replay): `../nb_v12_img_ab/` | 5f4cfe5d16946e1b5c458c6385782612e9f08186f50237fc25e3f5c4ccccf004 |
| Manh's review package (A+B + model switch): `../manh_review_v12_img/` | b488ee1e83eb9b1adf354ced569423203ccca6f63794cafc71f95fcb34890afe |
| `kernel-metadata.json` (`chankhavu/dsv41-tools-v12-img`) | 824e56d8efe9 |
| `model_manifest_c1.json`, `model_manifest_causal_predicted.json` | cell 1b's two manifests (137 files each), picked by cell 00's `V12IMG_MODEL` |
| `cells_vs_v12.diff` | every changed line, cell by cell (`v12img_diff.py`) |
| `solver_11bfaaaa_to_0c55beeb.diff`, `harness_882c817f_to_acb8a733.diff` | the two code diffs |
| `harness_882c817f_vs_1d1a281e.diff` | V12's harness vs pinned_img's image harness (why it was not used) |

## Pinned files (cell 0b overlay, `kaggle/dsv41_tools/pinned_v12img/`)

| File | V12 | V12 + images |
|---|---|---|
| `solver_dsv41.py` | 11bfaaaa | **0c55beeb** = 11bfaaaa + the image switch + recovery R1 (held free re-queues) + R2 (replay hook) |
| `repo/evals/run_deepseek_tools.py` | 882c817f | **acb8a733** = 882c817f + the minimal E058 text+image code (ed668f98) + the R2 turn-replay hook |
| `repo/evals/arc_harness/grid_image.py` | (kit 8f4e68cf) | **18ed78fb** (pinned_img; only adds aligned_cell / deepseek_tokens / qwen_tokens) |
| `repo/evals/train_check.py` | 8fe36175 | 8fe36175 (unchanged) |
| `repo/evals/auto_repair.py` | 2292369c | 2292369c (unchanged) |
| `repo/evals/auto_repair_names.json` | 571f47eb | 571f47eb (unchanged) |

## Cell-by-cell diff vs V12 (full text: `cells_vs_v12.diff`)
14 of V12's 21 cells are byte-identical.

- **Markdown:** an 8-line "V12 + images" note on top.
- **Cell 00 (settings):**
  - 2 header comment lines.
  - `("ARC_GRID_FORMAT", "text+image")` and `("MANH_VISION_MAX_IMAGES", "32")` join the setdefault list. Cell 2m reads the second one; V12 uses 16.
  - The model-input setting `V12IMG_MODEL_SRC = ""`. When set, it becomes `ARC_MODEL_SRC`; `""` means the attached Kaggle Model.
  - A settings print line.
  - The load-test lookup made bounded (see the last bullet).
- **Cell 0b (pins):**
  - A 2-line comment.
  - A new `_MT` overlay: the 6 files above, as a deterministic tar.gz.
  - `_MT_SHA` lists the 6 files.
  - `_MT_EXPECT` adds `ARC_GRID_FORMAT=text+image` and `MANH_VISION_MAX_IMAGES=32`.
  - `_MT_NEED` adds the solver's `GRID_FORMAT = E("ARC_GRID_FORMAT"` and `MM_KEYS`, the harness's `def chat_context_cap` and `def text_image_content`, and grid_image's `def aligned_cell`. These are the image feature checks; a commit run raises and the rerun falls back, as in V12.
  - `_MT_NEW` is unchanged. The kit has a grid_image.py, so the rerun fallback restores it like the other files.
- **Cell 1b (integrity):** `INTEGRITY_BY_MODEL` holds the C1 and causal manifests (137 files each, 132.06 GB; V12: 136 files, 131.09 GB), and `INTEGRITY` is picked by `V12IMG_MODEL`. The one INTEGRITY line becomes three.
- **Model switch:** set cell 00's `V12IMG_MODEL = "causal"` and change the `model_sources` line in kernel-metadata.json. The default is `"c1"`.
- **Cell 2m (his server):** two changes.
  - **Manh's ask (2026-10-30), text and image alike:** `"--vllm-arg=--watermark-reload-only"` is inserted right after `"--vllm-arg=--watermark=0.1",`, on the same line. With it, the 0.1 watermark applies only to preempted requests and KV-connector or offload reloads (fork `SchedulerConfig.watermark_reload_only`, commit 274ddccd88). The flag is in g22df163d2d but **not in V12's g8cf0c0664b**, so this notebook needs his g22df163d2d (v16+) wheels.
  - His other ask, `"VLLM_GPU_DRIVER_RESERVE_MB": "256"` in MANH_SERVER_ENV, is already in V12 and unchanged here; a test checks it.
  - Two lines are added after `_cmd`:
  `if os.environ.get("ARC_GRID_FORMAT") == "text+image": _cmd.append("--vllm-arg=--disable-chunked-mm-input")`.
  This is the one server-flag change on our side. V12's cell 2m and Manh's launcher (`dsv41_vllm.py` `command_for`, arc2-2026-src v23) do not pass it. Each image span is prefilled whole, as DeepSeek's reference assumes and as the E068 fix ran. With mnbt 2048 and the fork's `get_mm_max_tokens_per_item` = `vision_max_n_token` 1024 (a 30x30 grid is 932 tokens), it cannot trip vLLM's start-up check.
- **New cell 2i (commit runs only, after 2c; `cells/v12img_smoke.py`):** the image checks, run in a background thread.
- **New cell 4i (commit runs only, last; `cells/v12img_verdict.py`):** the verdict on cell 2i's checks.
- **Metadata:**
  - id `chankhavu/dsv41-tools-v12-img`; private; no internet.
  - `model_sources` = `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/PyTorch/default/1`, replacing the E128 tool model.
  - V12's datasets (latest versions at push time), competition, machine and `docker_image` pin are kept. That pin is Manh's private BYOD image, which our account may not be able to use; `--docker-latest` switches to Kaggle's latest image, as earlier builds did.
- **AGENTS.md rule (the only non-image change):** V12's recursive `/kaggle/input` scans become `input_paths.py`'s bounded lookups, in cells 00 (load test), 0, 1b and 2. They find the same files on Kaggle's and the pods' layouts (`test_input_scans`). The model is found at its exact mount dir first: `INPUT_ROOTS` includes `models/arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pytorch/default/1`. Build with `--keep-v12-lookups` to keep V12's scans.

## Solver diff 11bfaaaa -> 0c55beeb (`solver_11bfaaaa_to_0c55beeb.diff`; recovery: see "Recovery")
- **Ported verbatim from pinned_img's solver 64ef0a0f (15 lines):**
  - `GRID_FORMAT = E("ARC_GRID_FORMAT", "json")`; any value other than json or text+image raises SystemExit; `IMAGES`.
  - make_args: with images, `HARNESS` minus `--token-exact`, plus `--grid-format text+image --image-cell 0`. `--context-limit`, `--multi-test`, `--multi-budget` and `--auto-repair` are kept.
  - server_cmd drops `--language-model-only`. This only matters if the solver starts its own server; in V12, cell 2m serves.
- **New (coordinator's monitoring request), image mode only:**
  - `vllm:mm_cache_queries_total` and `vllm:mm_cache_hits_total` (the MM processor cache) are added to `METRIC_KEYS`.
  - `metrics.jsonl` gains `mm_queries` / `mm_hit`, and the 30-s server line gains ", MM cache hits X of N images".
  - In text mode, `METRIC_KEYS` and the log line are V12's exactly.
- **Not ported from pinned_img** (each checked absent by a test): the APITimeoutError and HTTP-500 retry markers, the supervisor "up" wait, and find_model_dir.

## Harness diff summary
882c817f vs pinned_img's 1d1a281e (`harness_882c817f_vs_1d1a281e.diff`) has these non-image differences:
- E065 `offpeak_wait` / `--offpeak-only`.
- E064 `--show-tool`, `--check-image`, `--image-in`, `--image-note` (image tool results, `append_text`, `pending_images`, `image_check` fallback, `n_errors` and rescue changes).
- The `submit_answer` refactor (`_answer_body` / `_answer_feedback`).
- New `V6_DEFAULTS` keys.
- `chat_context_cap` is also applied to final and no-think turns under `--token-exact` (`tok is None or final or not think`).

None of these fires with V12's flags: `--final-mode session` and `--think-turns -1` mean no final or no-think turns. Even so, the build uses **882c817f + the minimal image code** (ed668f98) so that the text path is V12's code:
- `aligned_cell` and `data_uri` imports;
- `CHAT_*` constants, `message_token_bound` and `chat_context_cap`, verbatim;
- `IMAGE_NOTE` and `text_image_content`, verbatim;
- the text+image prompt in `solve()`;
- `ctx` tracking;
- `chat_context_cap` gated on `tok is None` only, which is the image (chat) path;
- the `--grid-format text+image` choice and help texts;
- the `--token-exact` + images guard in `main()`.

Only 2 added lines are not 1d1a281e's (a test checks this).

## Checks
- **Image limit:** counted with the harness's own builder under the solver's multi-test.
  - The 120 eval tasks need at most **14** images per prompt (36a08778: 6 pairs + 2 tests; mean 7.4), at most 8.4k image tokens.
  - The local 240-task test file needs 17 (2 prompts exceed V12's 16).
  - The 1,000 training tasks need 22 (794b24be).
  - The limit is set to 32, the most Manh's launcher accepts (`--vision-max-images` 2-32 -> `--limit-mm-per-prompt`). No turn adds images.
- **Causal image attention:** no `is_mm_prefix_lm` override is needed.
  - #57152 (9f9e1dac26) removed `self.is_mm_prefix_lm = vision_n_layers > 0`. It is an ancestor of g22df163d2d and of V12's g8cf0c0664b, but not of g5e1bccd07c or a6c3ee8c2b.
  - At g22df163d2d, model_type `deepseek_v41` has no arch-config convertor, the fallback list does not contain it, sparse_swa widens only for `DeepseekV4Config`, and the C1 / causal configs carry no `is_mm_prefix_lm`.
  - Note: vLLM logs "multimodal-bidirectional" only when `--disable-chunked-mm-input` is off. With the flag, cell 2i's real checks are the installed `configs/deepseek_v41.py` source and vLLM's own config resolution.
- **Model (C1 Kaggle Model):**
  - The read-only Kaggle file listing matches the manifest: 137 files, 132.062 GB, 0 missing, 0 extra, 0 size mismatches.
  - Its 16 Engram files equal V12's E128 entries (sha256), its index equals the causal repo's (it names those 16 shards), and the vision file `model-00003` equals the causal repo's. The E066 README calls it uniform E128 with the vision tower kept.
  - V12's E128 is also uniform, so no per-layer expert patch is needed.
  - I could not read C1's own config.json or README: its HF repo is gone (replaced by `-causal`), and Kaggle offers no per-file download. The checks above rely on the sibling causal config, which is the same pipeline (vision_config, image_token_id 129264, max_wh_ratio null, as Manh's `read_metadata` requires).
- **Calibration caveat:** C1's GPTQ calibration and expert selection replayed image traffic on the pre-#57152 stack, which attends to images bidirectionally (E068). Kaggle's g22df163d2d wheel is causal. The model may lose some image accuracy because of that mismatch. The causal re-prune is `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal@209c810a74`; it differs from C1 only in its weights and keep set, uses the same 16 Engram files, and builds with `--model-source/--integrity-manifest` (`model_manifest_causal_predicted.json`).
- **VRAM / host RAM:**
  - V12 already builds the vision tower: MANH_VISION=1, no `--language-model-only`, 0.90 GiB BF16. So the weight VRAM should not change. Watch the KV pool line anyway.
  - The MM processor cache defaults to 4 GiB in each of the API server and the engine core, so up to 8 GiB. V12's host headroom was 38 GiB. E066's final build used `--mm-cache-gb 2`; it is not changed here.

## Pod run (rerun mode, as V12 and 025ep)
- **Notebook:** `kaggle/dsv41_tools/out/nb_v12_img/dsv41-tools-v12-img.ipynb` with its `kernel-metadata.json`.
- **Model input:**
  - `/kaggle/input/models/arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pytorch/default/1` -> the C1 dir (137 files).
  - `make_kaggle_tree.sh` hard-codes `models/chankhavu/$KT_MODEL_SLUG`. That also works: the bounded lookup's fixed depth `models/*/*/*/*/*` finds the model's `e043_manifest.json`. Mount only one model.
  - Alternatively, set env `ARC_MODEL_SRC=<dir>`.
- **Other inputs, as 025ep:**
  - `KT_WHEELS_DIR=/root/kin/vllm_wheels_sft` (g22df163d2d). This is required: `--watermark-reload-only` is not in v11's g8cf0c0664b (V12's own replica), whose server would reject the argument.
  - `KT_SRC_DIR=.../arc2-2026-src/versions/23`; cu13 v2, kit v5, unsloth v15; workload evalx2t_s2026.
- **Env:** nothing extra. Cell 00 sets `ARC_GRID_FORMAT=text+image` and `MANH_VISION_MAX_IMAGES=32`, in reruns too.
  - Do **not** override these two with env: cell 0b's `_MT_EXPECT` requires them, and a mismatch in a rerun drops to the kit's v3 fallback (per pair, 36 workers).
  - For a text twin, run V12's own notebook with the model input remapped.
- **Rerun mode skips cells 2i and 4i.** For the image checks, run the commit mode once, or E068's `perception_probe.py --n 60` against the pod's server.

## Monitoring plan (pod, against V12 text on pod 2)
Sampler at 60 s; solver `metrics.jsonl` every 30 s.
- **Prefix cache:** the hit rate over time (`prefix_hit`), and re-prefill (uncached prompt) tokens per turn (`uncached_tok_s` and per-turn usage).
- **Queue:** preemptions (`num_preemptions_total`), and running streams under the KV gate (`srv_running`, "KV gate" lines).
- **MM processor cache:** the hit rate (new `mm_hit`), and the host RAM of the API-server and EngineCore RSS.
- **Encoder cache:** GPU encoder-cache hits are not exposed by vLLM. Use a short `--enable-logging-iteration-details` window to count encoder inputs.
- **CPU KV offload:** restores (`host_tok_s`). Image-span blocks restored this way are not re-encoded: an item fully inside the computed prefix skips the encoder.
- **VRAM:** cap - reserved.
- **Progress:** tasks answered at 3 h and 6 h.

## Risks
- The chat path re-tokenizes about 1 turn in 8 (E043), which breaks the cached prefix; images make re-prefills dearer.
  - ~932 tokens per 30x30 grid, ~7 images per task, up to 22.
  - First-turn prompts reach ~41-45k tokens.
- The expected API-level gain is ~+3.4 pass@1 (E058/E064), so L4 throughput decides.

## Recovery (solver R1 + harness/solver R2; nothing runs and no request is added while nothing fails)
| event | V12 | now |
|---|---|---|
| **server death** (connection drop, then refused) | in-flight attempts error; re-queued at once (4 free re-queues) and re-sent as soon as the port answers, burning re-queues and tries in the seconds before the supervisor's status flips; each restarts from turn 1 | each failing turn waits in its attempt until the server is up again: supervisor `up` with a newer start, or `/health` 200; polled with backoff 1, 2, 4 .. 30 s. Then it is **replayed**: same history, the token-exact prompt ids again, the session kept, the partial turn dropped. At most 8 replay prefills at once (`ARC_REPLAY_CONCURRENCY`), each also behind the KV gate. After 3 replays (`ARC_REPLAY_MAX`), or with no restart left / past stop-new or force: R1 |
| **HTTP 500** (dying engine) | not a server error: each 500 spent one of the 2 JOB_TRIES | a server error, as above. `/health` is 503 while the engine is dead, so nothing is re-sent into it |
| **hang past the client timeout** (read 900 s) | APITimeoutError spent tries (33 in V1's hang test) | a server error: replayed up to 3 times, then R1 |
| **R1** (an attempt that still ends with a server error) | immediate re-queue | free re-queue at most once per attempt per server start (CONN_TRIES in all), held until the server is up again; later failures use the normal tries |

Settings: `ARC_REPLAY=0` turns replay off; `ARC_REPLAY_CONCURRENCY` (default 8), `ARC_REPLAY_MAX` (default 3). Records gain `replays` and `replay_errors` only when a turn was replayed; the log line gains ` replays=N`.

**Risks:**
- A server that hangs every request keeps each attempt up to 4 x 900 s before R1.
- `/health` 200 does not prove a hung engine is healthy.
- Replays after a restart are paced, but 64 attempts at 8 at a time with 30-100k-token prefills take minutes.

**Next:** a stall watchdog (a hung engine), and resume from the transcript after a solver restart.

## Tests (2026-10-30, all pass; no GPU, network or Kaggle)
- `tests/test_v12_img_recovery.py` (R1, the solver's real main loop against a fake supervised server): **5 OK**.
  - die + refuse, then a new start; 500s, then death; a hang past the client timeout; once per start; the normal path's requests equal V12's (no `/health`).
  - It caught a `status` name-shadowing bug on the normal path, now fixed.
- `tests/test_v12_img_replay.py` (R2): **11 OK**.
  - Death mid-stream at turn k: the same history and prompt ids, the session kept, the same record.
  - Chat path with images: the same messages again.
  - The replay cap; no replay when `wait_for_server()` says no or for other errors.
  - The normal path's calls equal the hook-less ones, in both modes.
  - The concurrency cap (peak 8 of 30), the KV gate, and `wait_for_server` against a fake supervisor (a new start: True; failed or forced: False).
  - `ARC_REPLAY=0`.
- `tests/test_v12_img.py` (new): **23 OK**.
  - Cell and line approvals.
  - Cell 0b runs on the v5 kit and its checks pass.
  - The solver's text-mode argv and Namespace equal V12's exactly; the image argv is V12's minus `--token-exact` plus `--grid-format text+image --image-cell 0`.
  - Approved solver and harness diffs; the image code is 1d1a281e's verbatim.
  - `solve()` on the text path (token-exact, multi-test, auto-repair, ContextFull) is equal to 882c817f's.
  - On the image path it is equal to 1d1a281e's: the same request bodies with images, the caps, and context-full.
  - Eval-task images decode to the grids at the aligned geometry, <= 32 per prompt.
  - Cell 2i's source check on the fork's files (9f9e1dac26^ flagged; 9f9e1dac26, g8cf0c0664b and g22df163d2d clean).
  - Probe scoring and cell 4i's verdict logic.
- `tests/test_img.py`: 29 OK. `tests/test_input_scans.py`: 20 OK. `pinned_v12img` is added to FROZEN (V12's solver lookup sits behind ARC_MODEL_DIR), and the builder joins the "Built" set.

Run: `.venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_v12_img.py'`
