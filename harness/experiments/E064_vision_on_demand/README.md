# E064: vision on demand for DeepSeek V4.1-Flash-1017

2026-10-28. User request: does the harness let the model choose when to look at an image (no: E058 puts every grid
in the first message), and what do the reasoning traces suggest? Then: "let's try all of the improvements".

## Trace review of E058 (DeepSeek API, 240 attempts per arm; `scripts/trace_review/`)
- With images the model reasons in objects and colours (colour words 5.5 vs 0.09 per 1k reasoning chars), writes 12%
  less reasoning (median 58k vs 66k chars) and makes fewer tool calls (13.4 vs 15.2); it thinks a little before its
  first call (0.9k vs 0.1k chars) and keeps referring back to the images late in the attempt.
- Image wins (17 tasks, 8 also won by Qwen3.8-Flash-Next): the picture gives the gestalt (rooms, frames, which shapes
  belong together); code checks the cells. Image losses (9 tasks): a confident visual impression the cells contradict
  ("images look heavily symmetric", 344/865 cells off; "one continuous spiral", 4 components), then long chases.
- In 42-45% of attempts the model wants to "visualize" a grid it computed (mask, diff, candidate) and prints ASCII;
  it almost never tries matplotlib/PIL (<1%).

## Harness options (opt-in, `evals/run_deepseek_tools.py`; defaults and the pinned Kaggle copies unchanged; 53 tests)
- `--show-tool`: a `show_grid(code)` tool; `code` is evaluated in the session (one grid or a list of up to 4; bools
  count as 0/1) and the result comes back as labelled E058-aligned images. At most 12 calls per attempt. `shows`
  recorded.
- `--check-image`: the first accepted `submit_answer` is not accepted yet: the answer is drawn next to its test
  input(s) and the model resubmits (same code) or fixes it. Once per attempt, only with turns and 15% of the budget
  left; if the model never resubmits, the checked answer is kept (`answer_source` `image_check`). `image_check`
  recorded (turn, changed).
- `--image-note strict` (text+image): the note adds "confirm anything you see in the images (a symmetry, a count,
  which cells connect, sizes or positions) in code on the JSON grids before building on it".
- `--image-in {tool,user}`: images inside the tool result (DeepSeek API, InferenceNet accept it) or, `user`, a text
  tool result plus one user message with the images after the turn's tool results (DeepInfra rejects image parts in
  tool messages: 422 "Input should be a valid string"; self-hosted chat templates may too).
- Lazy mode = plain JSON prompt + `--show-tool`.
- OpenRouter's `DeepSeek` provider is refused for this account ("Paid model training violation (account settings)"),
  so arms beyond the DeepSeek balance run on DeepInfra with their own DeepInfra baseline.

`scripts/run.sh <base|show|check|strict|lazy>` (E046 best harness + `--grid-format text+image --image-cell 0`, 120
tasks x 2 attempts; `VIA=or` = OpenRouter DeepInfra with `--image-in user`); `scripts/analyze.py`.

## Results: DeepSeek API (vs E058 `e058_pilot30`, same harness with up-front images; paired over 120 tasks)

| arm | pass@2 | pass@1 | out tok / task-attempt | image use | $ |
|---|---|---|---|---|---|
| E058 images (baseline) | 90.7% | 81.1% | 27.0k | – | 4.61 |
| `show` (+ show_grid) | 87.4% (−3.3 [−8.8, +1.7]; 5 better / 8 worse) | 80.1% | 27.4k | 0.33 calls / attempt, used in 26% | 4.87 |
| `check` (+ image check) | 88.3% (−2.4 [−7.5, +2.2]; 5 / 8) | 78.9% | 26.8k | fired in 99%, answer changed after it 3x (1 right) | 4.57 |

- Both are ties with the baseline (within the 2-3 pp band); E058's own +2.1 over text was not significant either.
- What the model shows itself: mostly its final answer, often next to the test input (`[test_inputs[0], ans[0], ...]`),
  at 80-100% of the way through the attempt. It reaches for the tool on hard tasks (65% correct, 36k tokens vs 84% and
  24k without), but looking at its own answer almost never changes it: the forced check is a rubber stamp
  ("Looks consistent with training. Submit."). Its perception is too weak (E058 probe: 0/20 exact transcriptions) to
  spot cell-level errors in its answer.

## Results: OpenRouter DeepInfra (vs `e064or_base` = the E058 harness on DeepInfra; 0 errors after retries)

| arm | pass@1 | pass@2 | out tok / task-attempt | turns | $ |
|---|---|---|---|---|---|
| E046 best, text only (E047 `e047or_ctrl`, 3 samples) | 75.0% | 85.6% | 29.2k | 14.7 | |
| `base`: E058 up-front images | **78.0%** | 86.5% | 28.0k | 13.7 | 2.72 |
| `strict`: + strict image note | 77.0% | 86.9% (+0.4 [−4.2, +5.4]; 6 / 6) | 28.7k | 14.4 | 2.76 |
| `lazy`: JSON + show_grid, no up-front images | 74.0% | 84.0% (−2.5 [−8.3, +3.3]; 7 / 10) | 31.0k | 16.0 | 2.96 |

- E058's up-front images replicate on a second provider: pass@1 +3.0 over text, pass@2 +0.9, −4% tokens (DeepSeek API:
  +3.4 / +1.7 pooled).
- `strict` ties `base`. `lazy` (show_grid used in 30% of attempts, 0.51 calls/attempt) loses the image gain and costs
  +11% tokens: images help because the model sees them before it starts reasoning, not as an on-request tool.

## Conclusion
Keep E058 (E046 best + `--grid-format text+image --image-cell 0`, up-front images, default note). `--show-tool`,
`--check-image`, `--image-note strict` and lazy mode add nothing; they stay opt-in. `--image-in user` is needed for
providers (and probably self-hosted chat templates) that reject image parts in tool results. All-harness comparison:
`scripts/pass2_table.py` (pass@2 over all sample pairs, token mean / p50 / p90 / max, input, turns).
