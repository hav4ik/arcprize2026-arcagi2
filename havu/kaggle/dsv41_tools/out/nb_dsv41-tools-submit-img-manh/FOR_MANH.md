# Image candidate notebook: built on your v5 (for testing our harness on your server)

`dsv41-tools-submit-img-manh.ipynb` is your notebook v5 (`dsv41-tools-submit-b32m-external-server`). Your install
cells, server cell 2m, solver cell 3 and stop cell 4 are byte-identical. Builder: `kaggle/dsv41_tools/build_img_manh.py`.
Full record: `kaggle/dsv41_tools/CANDIDATE_E066_IMG.md`.

## What we changed (6 cells)

- **Settings (00):**
  - `ARC_GRID_FORMAT=text+image` (the E058 image harness: chat path, no `--token-exact`, `--context-limit` on the chat
    path);
  - `MANH_VISION=1`, `MANH_VISION_MAX_IMAGES=32` (your launcher's maximum; ARC prompts carry up to ~22-24 images, about
    7 on average);
  - the commit run is a light smoke: 2 tasks, ending when both are done.
- **0b:** our pinned solver `e96f7cb7` and harness `1d1a281e`.
  - The solver is your S23 ARC_EXTERNAL_SERVER plus the image switch. Its liveness probe ends the run only on
    "connection refused", or after 3 failed probes in a row.
  - Your v5 marks the server dead after one failed TCP connect, and with no restarts that stops all new attempts.
- **1b:** input integrity against the model's sha256 manifest.
- **2r (inserted after your install cells):** our patch to your installed vLLM, so per-layer expert counts load
  (E72-152, `n_routed_experts_per_layer`):
  - `model.py` 029aa62e -> ad4ac0fd (wheel gf506afa210 / main 71dca8bc24);
  - `dsv4_topk.py` whitelist 6..512, 7e1e101a -> ab32ea7e.
  - It is sha256-guarded: an unknown file hash fails a commit run, and a scored rerun leaves the file as it is. With a
    uniform 128 model (C1) it is harmless.
- **2c:** the contract also checks parsed chat tool calls (max_tokens 4000) and that one image adds prompt tokens.
- **4b:** the report regex accepts `server up after Ns (external);`. Your v5 regex fails a good run on this line. It
  also reads `MANH_SERVER_LOG`; with images, contract failures are fatal.

## The model

- The image harness needs a model with the vision tower. The text E128 runs with `--language-model-only` and can't take
  images.
- The candidates are E72-152 (per-layer 72-152 experts, average 120, vision kept) and C1 (uniform E128 + vision). Only
  one will be a Kaggle Model, and the choice is pending.
- `kernel-metadata.json` points at the E72-152 handle, which is not uploaded yet.

## What we see on Runpod 4xL4 with your production cell (TP4, no DCP), E72-152 + images

- KV pool 2,264,661 tokens at util 0.95 (pod). The server is up in 125 s and the contract passes.
- Image streams hold more KV than text: ~8k at the start, ~20k+ later; the first image turn is ~8.4k prompt tokens.
- On your older cell at 120 workers:
  - GPU prefix hits fell to ~3% after ~16 min, but ~90% of prompt tokens came back from your CPU KV offload, and only
    3.5-8% were recomputed;
  - the solver's KV gate (0.85) held running streams at ~60-77 with 26-33 waiting;
  - 0 preemptions, 0 errors.
- The solver runs 80 workers (`ARC_WORKERS`) to match `--concurrency 80`. If you test 120, set both.
