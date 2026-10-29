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
Results are added below as the run progresses.
