# V12 + images: what changed vs your V12, and what we'd like you to check

This is your V12 notebook (`dsv41-tools-submit-b32m-external-server`, scriptVersionId 363638159) with our image harness switched on. Your server cell is the same apart from three lines. The full line-by-line diff is in `cells_vs_v12.diff`, and the code diffs are the two other `.diff` files.

## Changes by cell
- **Cell 00 (settings):**
  - `ARC_GRID_FORMAT=text+image` turns the image harness on.
  - `MANH_VISION_MAX_IMAGES=32`. Your cell 2m passes it as `--vision-max-images`, which becomes `--limit-mm-per-prompt {"image": 32}`.
  - `V12IMG_MODEL = "c1"` selects the model; see "Model" below.
  - `V12IMG_MODEL_SRC = ""` means "use the attached model". Also one print line.
- **Cell 0b (pinned code):**
  - Solver `0c2f81e8` = your `11bfaaaa` + the image switch + recovery A+B (below).
  - Harness `ed668f98` = your `882c817f` + the minimal image code. The text path is your code unchanged.
  - `arc_harness/grid_image.py` `18ed78fb` is new in the overlay. train_check and auto-repair are yours, byte for byte.
  - The settings and feature checks now also cover the image switch.
- **Cell 1b (integrity):** the model manifest is the image model's (137 files, 132.06 GB). It is picked by `V12IMG_MODEL`.
- **Cell 2m (your server):** three changes, nothing else.
  - `--vllm-arg=--watermark-reload-only` (your ask), always on.
  - `--vllm-arg=--disable-chunked-mm-input`, only with images, so each image span is prefilled in one step.
  - `MANH_VISION_MAX_IMAGES` is 32 through cell 00.
  - `VLLM_GPU_DRIVER_RESERVE_MB=256` was already in your env; it is unchanged.
- **New cell 2i (commit runs only, after 2c):** image checks in a background thread:
  1. causal image attention: the installed `configs/deepseek_v41.py` has no `is_mm_prefix_lm = ...` line, vLLM's config resolution gives False, and no log line says "multimodal-bidirectional";
  2. one grid image is accepted with the expected token count;
  3. a prompt with 32 images is accepted (33 should be rejected);
  4. the `vllm:mm_cache_*` metrics are present;
  5. a small-grid perception probe: 15 grids, warning only.
- **New cell 4i (commit runs only, last):** raises if a hard check in 2i failed.
- **Input lookups:** the `**` recursive globs over `/kaggle/input` in cells 00, 0, 1b and 2 are replaced by exact mount paths plus fixed-depth globs (your 2026-10-30 rule). They find the same files.
- **Metadata:**
  - slug `chankhavu/dsv41-tools-v12-img`, private;
  - `model_sources` = the image model;
  - your datasets, unpinned: vllm-wheels latest = v16, `g22df163d2d`;
  - your `docker_image` pin is kept.

## Model
- **Placeholder:** `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit` (E066 candidate 1). It is uniform E128 with the vision tower kept and the same 16 Engram row-chunk files as E128.
- **To switch to the causal re-prune** once a teammate uploads it, change two things:
  - cell 00: `V12IMG_MODEL = "causal"`;
  - the `model_sources` line in `kernel-metadata.json`.
  The model dir is found by its `e043_manifest.json`. C1 was calibrated on the old bidirectional stack.

## vLLM as it will run (your launcher, from V12's `launch.json`)
- **Command** (`--limit-mm-per-prompt` was `{"image": 16}` in V12; the last two flags are new):

  ```
  python -m vllm.entrypoints.cli.main serve <model> --served-model-name dsv41 --host 127.0.0.1 --port 18084 --trust-remote-code
  --tensor-parallel-size 4 --distributed-executor-backend mp --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840
  --max-num-seqs 64 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.91 --enable-prefix-caching
  --enable-batch-sharded-sampling --engram-config {"cpu_offload": true} --linear-backend humming
  --compilation-config {"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64} --reasoning-parser deepseek_v41
  --enable-scale-out --enable-prompt-tokens-details
  --kv-transfer-config {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": 10737418240, "lazy_offload": false}}
  --limit-mm-per-prompt {"image": 32} --offload-backend=uva --cpu-offload-gb=0.35 --cpu-offload-params=embed_tokens
  --engram-config.use_thp=true --watermark=0.1 --watermark-reload-only --attention-config.sparse_mla_prefill_workspace_factor=2
  --enable-auto-tool-choice --tool-call-parser=deepseek_v41 --prefix-cache-retention-interval=128 --disable-chunked-mm-input
  ```
- **Env:**
  - your `MANH_SERVER_ENV`, unchanged: host all-reduce 556, Humming pad 64,64 + MoE tuning, sparse indexer, `AR_OVERLAP_CHUNKS=2`, `DRIVER_RESERVE_MB=256`, memory log 60 s;
  - your launcher's own: `CUDA_VISIBLE_DEVICES=0,2,1,3`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_BATCH_SHARD_TOPK=64`.
- **Solver:** 64 workers, KV gate 0.85, multi-test with budget 0.5, `--context-limit 163840`, auto-repair. All as V12.

## What the image path does
- The JSON prompt is unchanged. One token-aligned PNG per grid is added: the training inputs and outputs, then the test inputs. The cell is 42·k px, so the processor never resizes; a 30x30 grid costs 932 tokens.
- It goes over the **chat path** (`/v1/chat/completions`, `--grid-format text+image --image-cell 0`). `--token-exact` is off for images, because token ids carry no pixels. `--context-limit` is applied from each turn's usage.
- Images per prompt: at most **14** on the eval set (mean 7.4), 17 on the local 240-task test file, 22 on training. No turn adds images.

## Recovery change (A+B, solver only; nothing runs when nothing fails)
- A server-side failure (connection drop or refusal, HTTP 500, client timeout) re-queues the attempt for free, at most once per server start (your supervisor's `start`), and at most `CONN_TRIES` times in all.
- The attempt is then held until the server is up again: the supervisor says `up` with a newer start, or `/health` returns 200. Polling backs off 1, 2, 4, 8, 16, 30 s.
- Before, a dying engine's 500s spent real tries, and a dead port burned re-queues within seconds.

## Please check
1. Image-path correctness on your wheel:
   - causal image attention (we believe #57152 covers it in `g22df163d2d` and `g8cf0c0664b`, so no `--hf-overrides`);
   - `--disable-chunked-mm-input` with mnbt 2048 (the largest image item is 1,024 tokens);
   - anything else the V4.1 image path needs.
2. MM caches and host RAM: processor cache default 4 GiB × 2 processes; encoder cache = mnbt 2048 tokens. Should we set `--mm-processor-cache-gb` or the encoder cache size?
3. TPS flags for image traffic (prefill-heavy re-sends), with `--watermark-reload-only`.
4. Can a notebook under **chankhavu** use your BYOD docker image (`gcr.io/kaggle-private-byod/...`)? If not, we switch to Kaggle's latest image.

## Commit run ("Save & Run All")
- It runs V12's load-test smoke: 80 eval tasks, stop-new 38 min, force 40 min, kill 42.5 min after the first cell.
- Alongside it run your contract check, our 2i image checks in the background, and the integrity hash of the model (background, short-smoke rules).
- Verdicts: your 4b and verdict cells, then 4i.
- Expected: about 45-50 min (install about 1 min, server up about 5 min). The scored rerun skips every smoke cell.

## Early pod numbers (interim run: the causal image model, without your `--watermark-reload-only`, first 0.8 h; vs V12 text on the same pod)
- Generation throughput: −10% gen tok/s.
- Prompt re-processing: +64% uncached prompt tokens per turn.
- MM processor cache: 97% hits.
- Host RAM: +3.4 GiB.

## Hashes
| File | sha256 |
|---|---|
| notebook `dsv41-tools-v12-img.ipynb` | b488ee1e83eb9b1adf354ced569423203ccca6f63794cafc71f95fcb34890afe |
| `kernel-metadata.json` | 824e56d8efe9b92d396b6caccb22189bb96dd8cb3043f8d7080df73a5ab10d2d |
| solver | 0c2f81e8c728b82afe4bedf176997b642220c92ce2b21f764fa924823a252e5e |
| harness | ed668f98fe555dc26b139ea4424b7cd6c55d2ef72cb7d0f6189145620ed2d1d0 |
| grid_image | 18ed78fbc81568ee040371e40f31d178e38fe68780c07685681e2d0f27e385df |
| your V12 | c47be5a819ad63a09c6ee831bff7f9770e070cd91fe04a27865f6ddd6e7ec42f |
