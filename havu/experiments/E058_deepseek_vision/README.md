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
