# Tonight's candidate (2026-10-29): image harness + E066 model on Manh's server

**Manh: please vibe-check this before we submit, ideally by ~21:30 UTC.** The submission deadline is ~23:30 UTC.
Serving is your code and config, and it stays the source of truth. We changed the harness side, and added one
small patch to your vLLM so per-layer expert counts load. Everything we changed is listed and diffed below.

## Files here

| File | What |
|---|---|
| `dsv41-tools-submit-img-manh.ipynb` + `kernel-metadata.json` | the candidate notebook, built on your v5 by `kaggle/dsv41_tools/build_img_manh.py` |
| [`review/cells_vs_manh_v5.md`](review/cells_vs_manh_v5.md) | cell-by-cell status and the diff of every cell we changed (generated) |
| [`review/solver_vs_manh_v5.diff`](review/solver_vs_manh_v5.diff) | our pinned solver vs the one in your v5 cell 0b |
| [`review/harness_vs_manh_v5.diff`](review/harness_vs_manh_v5.diff) | our pinned harness vs the one in your v5 cell 0b (B32's) |

Longer record: [`kaggle/dsv41_tools/CANDIDATE_E066_IMG.md`](../../CANDIDATE_E066_IMG.md).

## What is yours and what is ours

16 of the 22 cells are yours, byte-identical and in your order. That includes your install cells 1-5, server cell 2m,
solver cell 3, stop cell 4 and the verdict cell. Your cell 2m ships exactly as in v5: TP4, no DCP, util 0.94, driver
reserve 256 MiB, `--concurrency 80`, `--context 163840`, 12 GiB/rank CPU KV offload, `--watermark=0.1`, retention 128,
MODE 556 + row shard + AR overlap.

| Cell | Change | Why |
|---|---|---|
| settings | adds `ARC_GRID_FORMAT=text+image`, `MANH_VISION=1`, `MANH_VISION_MAX_IMAGES=32`; commit run = light smoke (2 tasks, ends when both are done, capped at ~50 min) | image harness; 32 is your launcher's maximum (ARC prompts: max ~22-24 images, mean ~7); the light smoke saves Kaggle quota, since load testing runs on our pods |
| 0b | our pinned files (below) | harness side = ours |
| 1b | integrity check against the E066 model's sha256 manifest | it pointed at E128's manifest |
| 2r (inserted after your install cells) | patches your installed vLLM: `model.py` 029aa62e -> ad4ac0fd, `dsv4_topk.py` 7e1e101a -> ab32ea7e | E72-152 has per-layer expert counts (72-152, average 120); see below |
| 2c | the contract also requires a parsed chat tool call (max_tokens 4000) and an image request that adds prompt tokens | the image path is the chat path; the probe's old max_tokens cut the tool call |
| 4b | the regex accepts `server up after Ns (external);`, reads `MANH_SERVER_LOG`, and makes contract failures fatal with images | your v5 regex fails a good commit run on the external solver's line |

## Our fixes on top, in detail

**Serving side (the only change): per-layer expert counts.** Branch `dynamic-count-moe` of your fork = 66cf71a0b7 on
your main 71dca8bc24. It is not pushed; the diffs are in `kaggle/dsv41_tools/b32m_patches/`.
- `model.py` reads `n_routed_experts_per_layer` (40 ints, validated: length = layers, each >= top-k).
  `DeepseekV4MoE` builds each layer with its own count. `n_routed_experts` = the max (152) is used for the expert
  mapping. EPLB is refused with mixed counts.
- `dsv4_topk.py`: the Triton router is generic, but its whitelist allowed only 128/256/384. It is widened to
  top-k..512. Other counts used to fall through to the CUDA `topk_softplus_sqrt` kernel, which errors.
- Tests on a Runpod 4xL4: your kernel tests plus ours, 1,720 pass; per-layer model tests 5/5; Triton = CUDA at 128.
- Cell 2r is guarded by sha256. A file already at the fixed hash is left alone; a known base hash is patched and
  verified, and the pyc is removed. An unknown hash raises in commit runs and is left as it is in the scored rerun.
- With uniform 128 (the C1 model) the patch changes nothing at runtime.

**Solver (`solver_vs_manh_v5.diff`, 43 lines).**
- Image switch `ARC_GRID_FORMAT=text+image`: the harness gets `--grid-format text+image --image-cell 0` and drops
  `--token-exact`, because token ids carry no pixels.
- S25 = your S23 plus a sturdier liveness probe. Your probe marks the server dead after ONE failed TCP connect, and
  with 0 restarts that stops all new attempts for the rest of the 12 h. Now a refused connection (nothing listening)
  is dead at once; any other connect error counts only after 3 failures in a row.

**Harness (`harness_vs_manh_v5.diff`; yours is B32's 882c817f, ours is 1d1a281e).**
- The E058 image harness: one token-aligned image per grid next to the JSON prompt. On the eval set it gains +3.0-3.4
  pass@1 over text with stock Flash-1017.
- `--context-limit` on the chat path (`chat_context_cap`). Before this fix, self-hosted image runs had no context
  limit: an E066 eval lost 68 of 334 attempts to HTTP 400s at max-model-len. The bound comes from the last turn's
  usage plus the appended messages. On 3,075 real image turns it was never below the real prompt.
- `--token-exact` combined with images is an argparse error.
- The other changes in that diff are opt-in options this candidate does not use (`--show-tool`, `--check-image`,
  `--offpeak-only`, ...).
- train_check, auto_repair and auto_repair_names are byte-identical to yours.

## Settings that ship (ours)

`ARC_WORKERS=80` (= your `--concurrency 80`), `ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`, `ARC_CONTEXT_LIMIT=163840`
(= your `--context`), KV admission gate `ARC_KV_ADMIT_MAX=0.85`, auto-repair on, image harness as above.

## Please vibe-check

1. **2r vs the rest of your fork:** does anything still assume one expert count for every layer? For example Humming
   pad / repack, ROW_SHARD, MODE 556 host gather, CUDA graph sizes, Engram. On our pod, your production cell loads
   E72-152 and passes the contract, and greedy output matches the v8 wheel within noise. Still, you know where the
   uniform assumptions live.
2. **Vision under load at util 0.94:** 80 image streams with up to 32 images per prompt. Is there any encoder-cache or
   mm-cache memory peak that your 120x128k text stress test would not have hit?
3. **Our KV gate with your CPU offload:** with offload, GPU KV sits near ~0.89 by design, and our gate stops admitting
   new attempts above 0.85. You suggested 0.9-0.95. We are measuring how often it holds on the pods; if it binds, we
   test 0.9 tonight.
4. **Context 163840 with images on your server:** any limit we would hit, such as the vision token budget or
   max-num-batched-tokens?
5. **Kaggle timeline:** your install cells, then 2r (seconds), then server start. Six distinct expert counts add
   ~3-4 min of Humming compiles. Is anything else slower on Kaggle than you'd expect?

## Evidence so far (Runpod 4xL4)

- **E72-152 on your production cell, v10 wheel gf506afa210 + 2r:**
  - KV pool 2,264,661 tokens (util 0.95);
  - server up in 125 s; contract passes (tool call, image, `/tokenize`, token-id completions);
  - no Humming/MoE warnings.
- **Older cell, 120 workers, ~35 min:**
  - 0 errors, 0 preemptions;
  - GPU prefix hits fell to ~3% after ~16 min, but ~90% of prompt tokens came back from your CPU offload and only
    4-8% were recomputed;
  - 21/27 first attempts correct.
- **Harness checks:**
  - the cached prefix covered the images on 176/176 later turns;
  - the vision encoder does not re-run on prefix hits; mm cache hits 85%;
  - prior reasoning is kept and the prefix is stable on the chat path (multi-test trace, 9 turns).
- **Running now:** 90-min loads of E72-152 and C1 on your v5 cell, and a full-workload stress test of this exact
  notebook. Results go to `experiments/E062_manh_vllm_fork/per_layer/` (`phase_h/`, `pod3_runs/`).

## The model (decided tonight)

The image harness needs the vision tower, so text-only E128 (`--language-model-only`) can't run it. Kaggle's quota
fits one model.

| Model | HF (private) | Notes |
|---|---|---|
| E72-152 | `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E72-152-avg120-2bit` | per-layer 72-152 experts, average 120, needs 2r |
| C1 | `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit` | uniform 128 + vision |

The sha256 manifests are `experiments/E066_dsv41_image_pruning/results/{e72c152,c1}_gptq_sha256_manifest.json`. If
you serve E72-152 outside this notebook, apply both `b32m_patches/dynamic_count_*.diff` first; cell 2r does exactly
that (`kaggle/dsv41_tools/cells/img_patches.py`). `kernel-metadata.json` points at the E72-152 Kaggle handle, which
is not uploaded yet; it gets rebuilt for the chosen model.

## Known and accepted

- No server restart (your design): a mid-run engine death ends new attempts. The solver still writes the submission.
- No Kaggle smoke before the submission's own commit run (quota).
- Per-layer layouts hold ~0.2 GiB/GPU of PyTorch caching-allocator waste (~0.19M KV tokens).

## Rebuild

```
python3 kaggle/dsv41_tools/build_img_manh.py --manh-nb <kaggle kernels pull -m of your notebook> \
  --model-source chankhavu/<model>/PyTorch/default/1 \
  --model-manifest experiments/E066_dsv41_image_pruning/results/<e72c152|c1>_gptq_sha256_manifest.json
.venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_img.py'
```
