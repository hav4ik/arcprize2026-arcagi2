# E033: Qwen3.8-27B with image inputs instead of JSON grids (no-tool baseline)

Run 2026-10-18, stopped early at 21:05 UTC to free the box for E032. Box: Vast 56598413, 2× RTX PRO 6000 Blackwell Max-Q, vLLM 0.31.0, official Qwen/Qwen3.8-27B-FP8 with the vision encoder, data parallel 2, prefix caching on, port 8001 via SSH tunnel.

## Image rendering: one visual token per cell
- Qwen3.8's vision encoder uses 16×16-pixel ViT patches and merges 2×2 of them into one LLM token, so one LLM token covers 32×32 px (`config.json`: patch_size 16, spatial_merge_size 2).
- `evals/arc_harness/grid_image.py` draws each cell as a solid 32×32 square in the standard ARC palette, with no grid lines or borders, so cell edges fall exactly on patch edges.
- Qwen's processor enlarges images below 65,536 px (256×256) by a non-integer factor, which would break the alignment. The server runs with `--mm-processor-kwargs '{"size": {"shortest_edge": 1024, "longest_edge": 16777216}}'`.
- **Verified** with `data_tools/check_image_tokens.py`: every image costs exactly rows×cols + 2 prompt tokens (vision start/end) for 1×1 … 30×30 grids.
- A 16-px cell (one ViT patch per cell) needs even grid sides, otherwise the processor resizes; not used.

## Perception check (6 eval grids, transcribe the image as JSON)
| Thinking | Exact grids | Cells correct |
|---|---|---|
| off (temp 0) | 0/6 | 59–99%, median ~85% |
| xhigh (temp 1.0) | 1/6 | 72–100%, median ~94% |

## No-tool A/B (ARC Prize prompt; the image arm replaces every JSON grid by its image plus "(R rows x C columns…)" and a colour key)
Settings: one chat call, tier xhigh, temp 1.0 / top_p 0.95 / top_k 20, max_tokens 96k, 1 attempt, Kaggle's eval copy (172 pairs). Commands: `evals/run_openai_arcprize.py --grid-format {json,image} --save-reasoning`. Results: `evals/results/e033_qwen_notools_{json,image}.jsonl`.

| Arm | Done | Correct on the 55 pairs both finished | Output tokens on those pairs |
|---|---|---|---|
| JSON | 55/172 (stopped) | **27/55** | 61.3k mean |
| Image | 172/172 | **1/55** (1/172 overall) | 15.2k mean |

- **Format problem:** the prompt's "respond in the format of the training output examples" has no text format in the image arm. The model answers with rows of space-separated digits, which the ARC Prize parser rejects. `evals/arc_harness/digit_grid_parse.py` parses those rows; the image numbers above already use it.
- **Even parsed, the answers are wrong.** 42 of 71 early answers had the right shape, and their cell accuracy was mean 65%. The model also thinks ~4× less than with JSON.
- **Conclusion so far:** at one token per cell, Qwen3.8-27B cannot read ARC grids precisely enough from images. Perception errors (~6–15% of cells) alone make exact answers unlikely.

## Not run (box handed to E032)
- Images + an explicit "answer as a JSON list of rows" instruction (removes the format confusion).
- Images + JSON together (do images add anything on top of exact values?).
