# E068: why images help Flash-1017 on the DeepSeek API but not self-hosted

2026-10-30. User request: "Let's spend some time to understand why. If we even tried with original weights, then it
might be the prompting problem?" (E066: candidate 1 image 69.7 vs text 75.1 pass@1; the full model self-hosted with
images 72.4 on 221/240 attempts vs the API's 81.1.)

## Paired re-analysis of E066's result files (no new runs; ARC Prize weighting per task-attempt)
On the 221 task-attempts the self-hosted full-model image run (`e066_full_ref`) finished:

| run | pass@1 | vs API image [95% CI] | pass@2 (107 tasks) |
|---|---|---|---|
| full, API, image (E058) | 82.2% | | 90.5% |
| full, self-hosted, image | 78.6% | −3.6 [−8.6, +0.9] | 86.8% |
| full, API, text (E046) | 78.9% | −3.3 [−9.0, +2.3] | 90.0% |
| full, self-hosted, text (`e066_full_ref_txt`, 136 of them) | 85.7% | −2.6 [−9.9, +4.8] | 94.1% (59 tasks) |
| candidate 1 (new E128 2-bit), image | 71.6% | −10.6 [−16.2, −5.0] | 81.3% |
| candidate 1, text | 76.4% | −5.8 [−10.8, −0.7] | 86.4% |
| old tool E128 2-bit (B32's model), text | 76.4% | −5.8 [−11.3, −0.4] | 86.0% |
| E72-152 2-bit, image / text | 70.7% / 71.7% | −11.5 / −10.5 | 79.9% / 82.7% |

Image effect (image − text, paired):
- on the 136 attempts both self-hosted full runs finished: API +5.5 [−1.5, +12.5], self-hosted −2.2 [−8.8, +4.8];
  interaction −7.7 [−16.9, +0.7];
- candidate 1 over all 240 attempts: −5.4 [−10.7, +0.3] vs the API's +3.5; interaction −8.9 [−16.0, −1.6].

Self-hosted text is not below the API (85.7 vs 82.7 on those 136; E043's older chat-path text reference was −6, p 0.09),
so the loss is specific to the image path. Both self-hosted models (full and candidate 1) show it.

## Ruled out
- **Prompt text and tokens:** byte-identical text; per-turn prompt token counts equal the API's (E066: 47,154 of 47,155
  turns; the fork's renderer reproduces the streams, 80/80).
- **Reasoning history on the chat path:** each turn's prompt grows by the previous completion plus the tool result
  (median +324 tokens self-hosted vs +307 API; never negative, even after 4k-token thinking turns), so earlier
  `reasoning_content` stays in the prompt as on the API (`scripts/reasoning_kept.py`; paired tables: `scripts/on_selfhost_set.py`, `scripts/interaction.py`).
- **Router image bias:** `bias_vl` applies on image-span positions only, in DeepSeek's reference
  (`inference/model.py` 821, 1252) and in the fork (`nvidia/router.py` 91-97).
- **Fork vision code vs the reference** (wheel `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11`, the one in E043's lock):
  preprocessing (`common/mm_preprocess.py`: resize plan, min_pixels 295,936, grey pad, normalisation, patch order),
  ViT (`deepseek_v4/vision.py`: 2D RoPE, full attention within one image, images encoded one at a time), 3x3 unfold
  aligner, and the span layout (`nvidia/vl_model.py`: aligner rows in reading order, learned start/newline/end) match
  line for line.
- The model does react to the images self-hosted: colour words per 1k reasoning characters 2.2 (API image 5.5; text
  runs 0.02 self-hosted, 0.09 API), and its first turns describe stripes, rings and colours. Lower in both harnesses, so
  partly a style difference.

## Open suspects (numerics / runtime)
FP8 KV cache on image positions (`--kv-cache-dtype fp8` in E043's serve.sh), the MM encoder attention backend or the
ViT's TP weight split (QKVParallelLinear / MergedColumnParallelLinear loading fused `wqkv` / `w1`), prefix-cache reuse of
image spans across turns, and the sampler (temperature 1.0 / top_p 1.0 vs the API's unknown thinking-mode sampler).

## Tests (scripts/)
1. `perception_probe.py`: 60 ARC training grids x 2 samples, the image alone (E058 rendering, no JSON or size label),
   asked for size, colours, corners and a transcription; thinking, effort low / 50, max_tokens 16,000.
   - **DeepSeek API** (`results/probe_api.jsonl`, $0.64): answered 76.7% (the rest hit the 16k cap), size 34.2%, colours
     69.2%, corners 66.7%, exact 25.8%, cell accuracy 98.3% when the shape was right. By grid size, exact / size:
     <=36 cells 63.3 / 70.0%, 37-144 30.0 / 43.3%, 145-400 6.7 / 13.3%, >400 3.3 / 10.0%.
   - Self-hosted: pending (needs a box with the full model + vision).
2. `nll_image_ablation.py --images real|grey`: E066's teacher-forced NLL of dev image rollouts with every image swapped
   for a same-size grey PNG (same token stream, paired); adds first-turn NLL. Pending.
Follow-ups if (1) shows a gap: the probe with `--kv-cache-dtype auto`, then `--no-enable-prefix-caching`.

## Status (2026-10-30 ~05:00 UTC)
Self-hosted tests on hold (user, via dsv4-pruning: "not now"). They need a box with the full DeepSeek-V4.1-Flash-1017
checkpoint (~510 GB download) and the vision tower, plus E066's dev data (`data/e066/tok_img_dev_s`,
`e065_image.parquet`) for test 2. Vast 58332344 (the user's) holds only B32's text-only E128 export. dsv4-pruning ranks
the numerics suspects the same way: FP8 KV on image positions, then prefix-cache reuse of image spans, then the MM
encoder backend.
