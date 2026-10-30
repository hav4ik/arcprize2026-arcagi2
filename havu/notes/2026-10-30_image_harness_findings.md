# 2026-10-30: the image harness, self-hosted and on the API (E068, E070)

Two questions about DeepSeek-V4.1-Flash-1017's image harness (E058: the JSON prompt plus one image per grid; on the
DeepSeek API pass@2 90.7 / pass@1 81.1 vs text 88.6 / 77.6). Details: `experiments/E068_selfhost_image_gap/README.md`,
`experiments/E070_image_layout/README.md`.

## 1. Why images lost points self-hosted (E068)
E066 measured images costing ~5 pass@1 on our vLLM fork, for the full model and for the pruned candidates, while they
gained ~3-5 on the API.

- **Cause: a serving bug.** The fork's config sets `is_mm_prefix_lm = vision_n_layers > 0`, so the language model
  attends bidirectionally inside each image span (image tokens see later image tokens). DeepSeek's reference
  `inference/model.py` is causal there (and with causal attention small grids read as well as on the API).
  Upstream vLLM fixed it in PR #57152 ("Restore causal image SWA for DeepSeek V4.1", merged 2026-09-16) for its later
  code; V4.1 had inherited the rule from DeepSeek V4 Vision-Exp.
- **Fix on the E043 fork (`...vision11`):** `--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input`.
  On this fork the flag gates every widening path, so the override equals the upstream fix. Port of the PR's config
  hunk: `experiments/E068_selfhost_image_gap/patches/apply_pr57152_port.py`. Manh's newer fork may carry the
  upstream-style code and need the real PR.
- **How to verify it took:** the server's "non-default args" line shows `'hf_overrides': {'is_mm_prefix_lm': False}`;
  or a small-grid perception probe (exact transcription ~2/3 with the fix, ~1/3 without). A missing
  "multimodal-bidirectional" warning proves nothing once `--disable-chunked-mm-input` is explicit.
- **Evidence (full model, 4x RTX PRO 6000):**
  - Image-only perception probe, exact transcription: API 25.8%, fork default 7.5%, fixed 18.3%. Small grids
    (<= 36 cells): API 63%, fixed 67%.
  - Teacher-forced NLL of the API's own image rollouts: −0.0081/token with the fix (−0.0200 on the first turn), and
    no change on grey images.
- **Ruled out:**
  - the prompt (byte-identical tokens);
  - the sampler;
  - the KV format (the fork runs only DeepSeek's native FP8/FP4 KV);
  - the indexer precision (FP4 in the reference too);
  - the Engram image mask and the router image bias;
  - vision preprocessing and math;
  - the vision tower's TP split (for size and exact transcription).
- **Open:**
  - a residual gap on larger grids (exact −7.5 vs the API);
  - the ARC accuracy with the fix: candidate 1's image re-eval is dsv4-pruning's, and the full model is not
    measured.
- **Kaggle:** today's submissions serve the text-only E128 (no vision tower), which is unaffected. The parked image
  candidate (`kaggle/dsv41_tools/CANDIDATE_E066_IMG.md`) now carries the fix and the checks.

## 2. Does Manh's image layout help on the API (E070)?
Manh (Qwen3.8): input and output of an example in one image beat separate images. New harness option
`--image-layout grid|pair|sheet` (default grid = E058). DeepSeek API, 120 tasks x 2, paired:

| layout | pass@2 | pass@1 | pass@2 vs per-grid [95% CI] |
|---|---|---|---|
| one image per grid (E058) | 90.7 | 81.1 | |
| `pair`: one image per example, input left of output | 88.8 | 78.7 | −1.9 [−6.9, +2.8] |
| `sheet`: all examples in one image | 84.2 | 75.8 | −6.5 [−12.5, −1.1] |

It doesn't transfer to DeepSeek on the API: pair ties, sheet is worse, and neither shortens reasoning. Keep one image
per grid. Untested: Qwen3.8, and self-hosted DeepSeek with the fix.

## Methods worth reusing
- `experiments/E068_selfhost_image_gap/scripts/perception_probe.py`: 60 ARC grids x 2, image only, asked for
  size/colours/corners/transcription. ~30 min self-hosted on 4x RTX PRO 6000, $0.64 on the API. The fast check for
  any vision-path change.
- `scripts/nll_image_ablation.py`: real vs grey images, same token stream, paired per rollout.
- Before suspecting numerics, diff the serving code's attention and masking semantics against the model's reference
  implementation.
