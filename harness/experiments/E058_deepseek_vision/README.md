# E058: vision for DeepSeek V4.1-Flash-1017 (text + token-aligned images in the best harness)

User request 2026-10-27: enable vision for DeepSeek like the friend's Qwen notebook (arc2-2026-harness-8d63d5), with
deliberate image sizing.

## Does DeepSeek have a vision tower?
Yes. `deepseek-ai/DeepSeek-V4.1-Flash-1017` config: `DeepseekV41ForCausalLM` with `vision_config` (32-layer ViT, hidden
1024, 16 heads, patch 14, `downsample_ratio` 3, `max_image_tokens` 1024, `min_pixels` 295,936). The DeepSeek API and
OpenRouter both accept image input. (E042's "no vision tower" referred to the synthetic/pruned test checkpoints.)

## Image size: why the Qwen notebook used 24 px, and what DeepSeek needs
- Qwen notebook: `pairs-plain` layout (one image per training pair, input left / output right, 2 blank cells apart,
  flat squares, no lines; from their ARC-AGI-3 code, default 16 px there), `vision_cell_pixels=24` in the recipe.
  Their own help text: "A Qwen token covers 32x32 pixels, so 32 draws one cell per token." 24 px is NOT aligned
  (3 cells per 4 tokens); no reason is given in the code. It reads as a token-budget compromise (a 30x30 pair image is
  ~1,860 tokens at 32 px, ~1,046 at 24, ~465 at 16; up to 16 images per task). Images supplement the numerical grids.
- DeepSeek, measured on the API (11 sizes, 1-token replies): one LLM token = one **42x42-px tile** (14-px patches,
  3x3 merge); charged tokens = tiles_w * tiles_h + tiles_h + 2 (a token per tile row plus two markers); images are
  rescaled to >= ~169 tiles (544^2 px) and <= ~1,024 tiles, and a non-integer rescale breaks cell/tile alignment
  (5x30 at 42 px became 32x6 tiles).
- Design (`grid_image.deepseek_cell`): cell side 42*k px with k = ceil(sqrt(169/cells)), so each cell is exactly k x k
  tiles and nothing is rescaled (30x30: k=1, 932 tokens; 3x3: k=5, 242 tokens). One image per grid (a side-by-side
  30x30 pair would exceed 1,024 tiles), flat cells without lines or margins, a text label before each image
  (numbered from 0 like the prompt), the JSON prompt unchanged and stated to be authoritative.

## Perception probe (`perception_probe.py`, 20 eval grids 5x13..30x30, $0.1)
| Mode | shape right | exact | cells right | out tokens |
|---|---|---|---|---|
| thinking off | 1/20 | 0/20 | 33.5% | 0.5k |
| thinking low | 3/20 | 0/20 | 36.6% | 17k |

Alignment confirmed (no-thinking token counts == prediction 20/20; thinking calls add a constant 26). DeepSeek cannot
read ARC grids from images: it gets the row/column counts wrong for 17-19 of 20 grids (Qwen3.8: 85-94% of cells, E033).

## Harness option
`evals/run_deepseek_tools.py --grid-format text+image --image-cell 0` (opt-in; config key `grid_format`): the full JSON
prompt, an image note, then one labelled image per training input/output and test input (`text_image_content`).
Test: `tests/test_deepseek_tools.py` `TextImageContentTest`. Smoke (1ae2feb7, 3 test inputs, 9 images): correct,
7 turns, 12.4k output tokens, 94% cache hits, $0.018.

## Pilot (pending): 30 random tasks (seed 58, `pilot_tasks.txt`) x 2 attempts, E046 best harness + images
(`run_pilot.sh`, launched off-peak by `launch_pilot_offpeak.sh`), compared with E046 `e046p_note_all` on the same
tasks (`compare.py`).

## Results (scored 2026-10-27 from the result files; runs finished overnight, retries done 15:23 UTC)
Both arms are the full 120 eval tasks x 2 attempts (the files keep the `pilot30` tag), text+image vs the same harness
text-only, same tasks, paired bootstrap over tasks (`compare.py`).

| model (API) | arm | pass@1 | pass@2 | out tokens/attempt | turns |
|---|---|---|---|---|---|
| DeepSeek V4.1-Flash-1017 (E046 best harness) | text (`e046p_note_all`) | 77.6 | 88.6 | 29.6k | |
| | text+image (`e058_pilot30`) | 81.1 | 90.7 | 27.0k | |
| Qwen3.8-Flash-Next (E048 harness, Alibaba) | text (E048 `sub80s40` + retry) | 59.8 | 72.9 | 35.0k | 9.5 |
| | text+image (`e058q_pilot30`) | **67.6** | **77.4** | 29.0k | 8.6 |

- Flash-Next: pass@1 +7.8 pp [95% CI +2.2, +13.8], pass@2 +4.4 pp [-1.7, +11.2], 17% fewer output tokens.
- DeepSeek: pass@2 +2.1 pp [-2.9, +7.5] (8 tasks better, 6 worse), pass@1 +3.5, 9% fewer output tokens.
- Images help although neither model can transcribe a grid from an image (perception probes: DeepSeek 0/20 exact,
  ~35% of cells; Flash-Next with thinking 1/20 exact, 49% of cells, `perception_probe_qwenflash.jsonl`): they work as
  a supplement to the exact JSON values, not as a replacement (E033: images instead of JSON, 1/55 vs 27/55).
- Kaggle cost not measured: each 30x30 grid adds ~900 prompt tokens (Qwen, 32 px cells) or ~930 (DeepSeek, 42 px), up
  to ~16 images per task, which adds prefill and KV on 4xL4. E128 as shipped has no vision tower (dropped in pruning).
