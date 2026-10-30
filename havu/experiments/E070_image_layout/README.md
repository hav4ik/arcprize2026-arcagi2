# E070: image layout for the DeepSeek image harness (Manh's --vision-layout)

2026-10-30. Teammate suggestion (Manh, from Qwen3.8 vision work): "send both input and output in the same image better
than split input and output"; his harness has `--vision-layout` with sheets (several pairs in one image) and pairs
(one image per pair). User: "Can you quickly check on the deepseek api? maybe this would be a stronger harness?"

## Arms (DeepSeek API, deepseek-flash = V4.1-Flash-1017, 120 eval tasks x 2 attempts, off-peak)
Baseline `e058_pilot30`: E046 best harness + `--grid-format text+image --image-cell 0` (one DeepSeek-aligned image
per grid; pass@2 90.7, pass@1 81.1, $4.61). Arms change only the training images (`--image-layout`, harness option
added for E070; default `grid` unchanged); test inputs stay one image each, as in Manh's renderer:
- `pair` (Manh's `pairs-plain`): one image per training example, input left, output right, two white cells apart.
- `sheet` (Manh's `sheets`, grids only): all training examples in one image, one per row; pages only past DeepSeek's
  1,024-tile cap (never needed on the 120 eval tasks).
Rendering (`arc_harness/grid_image.py` `rows_cell`, `render_rows`, `sheet_pages`): flat cells without lines on white;
cell 42*k px (k x k tiles per cell, smallest k reaching 169 tiles) when one tile per cell fits, else 28 or 14 px
(ViT-patch aligned); padded with white to whole 42-px tiles, so the API never rescales (checked on all 120 tasks: every
image 169-1,024 tiles). The text label before each image gives the sizes; the note says where input and output are.
Image tokens per task: grid 3,502, pair 2,557, sheet 1,441 (sheets draw large tasks at 14-28 px per cell).

Differences from Manh's renderer: no title, legend or size headers drawn in the image (our labels are text parts);
cells sized to DeepSeek tiles rather than a fixed 16/24 px.

`scripts/run.sh pair|sheet`, `scripts/analyze.py` (paired vs E058 via `experiments/E058_deepseek_vision/compare.py`).
Tests: `evals/tests/test_deepseek_tools.py` `E070ImageLayoutTest`.

## Results (2026-10-30, 16:07-16:36 UTC, off-peak; 0 errors; paired over the same 120 tasks)

| arm | pass@2 | pass@1 | pass@2 vs E058 [95% CI] | tasks better / worse | out tok / attempt | prompt tok / attempt | $ |
|---|---|---|---|---|---|---|---|
| E058 one image per grid (`e058_pilot30`) | 90.7 | 81.1 | | | 27.0k | 477k | 4.61 |
| `pair` (input and output in one image) | 88.8 | 78.7 | -1.9 [-6.9, +2.8] | 6 / 8 | 28.0k | 487k | 4.76 |
| `sheet` (all examples in one image) | 84.2 | 75.8 | -6.5 [-12.5, -1.1] | 5 / 12 | 28.0k | 447k | 4.70 |
| for scale: text only (E046 best, `e046p_note_all`) | 88.6 | 77.6 | | | 29.6k | | 10.11 |

`sheet` vs `pair`: -4.6 [-9.6, +0.4]. pass@1 is task-weighted (compare.py); the harness log's own pass@1 lines read
81.2 (pair) and 74.6 (sheet).

By cell size (pass@2, arm vs E058 on the same tasks):
- `pair`: 83 tasks at >= 42 px per cell 84.9 vs 87.8; 37 tasks at 28/14 px 97.3 vs 97.3. Smaller cells did not
  cost anything; the small loss is on normal-size tasks.
- `sheet`: 109 tasks at 28/14 px 84.4 vs 91.3; 11 tasks at >= 42 px 81.8 vs 84.8. It loses at both sizes, so it is
  not only resolution.

## Conclusion
Manh's finding from Qwen3.8 (input and output in one image beats separate images) does not transfer to DeepSeek
V4.1-Flash-1017 on the API: `pair` ties E058's per-grid images within the 2-3 pp noise band (slightly lower), `sheet`
is worse (CI excludes 0). Neither shortens reasoning. Keep E058's one image per grid. `--image-layout` stays opt-in
(default `grid`); it may still be worth a check on Qwen3.8 or self-hosted DeepSeek with the causal-attention fix
(E068), which this run does not cover.
