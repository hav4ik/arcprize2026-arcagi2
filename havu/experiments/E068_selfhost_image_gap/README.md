# E068: why images help Flash-1017 on the DeepSeek API but not self-hosted

2026-10-30. User request: "Let's spend some time to understand why. If we even tried with original weights, then it
might be the prompting problem?" (E066: candidate 1 image 69.7 vs text 75.1 pass@1; the full model self-hosted with
images 72.4 on 221/240 attempts vs the API's 81.1.)

## Summary (as of 10:30 UTC; the `causal`, `vitdp` and `noprefix` arms are still running)
1. **It is not the prompt.** Text and tokens are byte-identical to the API's, earlier reasoning stays in the prompt,
   and self-hosted text is as good as the API's. The loss is specific to images, and both self-hosted models show it.
2. **It is perception.** Shown one rendered grid with no text, the self-hosted full model reads it much worse than the
   API on the same pictures: grid size 14% vs 34%, corner colours 41% vs 67%, exact transcription 8% vs 26% (paired
   CIs exclude zero). The images still carry some signal self-hosted (swapping them for grey raises the NLL of the
   API's own reasoning), just much less.
3. **Not the sampler** (temperature 0.6 / top_p 0.95 reads the same as 1.0 / 1.0), **not FP8 KV** (the stack only runs
   DeepSeek's native mixed FP8/FP4 KV format), not the Engram image mask, not the router image bias, not the vision
   preprocessing or ViT/aligner math.
4. **Lead suspect: the fork attends bidirectionally over image spans in the language model; DeepSeek's reference is
   causal.** Being tested now (`causal` arm).

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
Offline (code and E066 traces):
- **Prompt text and tokens:** byte-identical text; per-turn prompt token counts equal the API's (E066: 47,154 of 47,155
  turns; the fork's renderer reproduces the streams, 80/80).
- **Reasoning history on the chat path:** each turn's prompt grows by the previous completion plus the tool result
  (median +324 tokens self-hosted vs +307 API; never negative, even after 4k-token thinking turns), so earlier
  `reasoning_content` stays in the prompt as on the API (`scripts/reasoning_kept.py`; paired tables:
  `scripts/on_selfhost_set.py`, `scripts/interaction.py`).
- **Router image bias:** `bias_vl` applies on image-span positions only, in DeepSeek's reference
  (`inference/model.py` 821, 1252) and in the fork (`nvidia/router.py` 91-97).
- **Fork vision code vs the reference** (wheel `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11`, the one in E043's lock):
  preprocessing (`common/mm_preprocess.py`: resize plan, min_pixels 295,936, grey pad, normalisation, patch order),
  ViT (`deepseek_v4/vision.py`: 2D RoPE, full attention within one image, images encoded one at a time), 3x3 unfold
  aligner, and the span layout (`nvidia/vl_model.py`: aligner rows in reading order, learned start/newline/end) match
  line for line. The checkpoint's 266 vision/aligner/delimiter tensors are BF16 (no quantization).
- **Engram at image positions:** the reference masks image tokens out of the n-grams and zeroes their Engram gate
  (`engram_mask = ~image_mask`); the fork does the same (`nvidia/model.py` 584-606: `image_sentinel_mask(input_ids)`,
  also on the lookback tokens). The VL model sets `requires_raw_input_tokens = True`, so the token ids reach this mask
  and the router on the multimodal path (vLLM would otherwise pass embeddings only). E043's patches pass Engram ids
  through unchanged when no row mask is loaded (`e043_obs.engram_ids`).
- The model does react to the images self-hosted: colour words per 1k reasoning characters 2.2 (API image 5.5; text
  runs 0.02 self-hosted, 0.09 API), and its first turns describe stripes, rings and colours.

On the box (below): the sampler, and FP8 KV as a configurable cause.

## Tests
1. `scripts/perception_probe.py`: 60 ARC training grids (15 each with <=36, 37-144, 145-400, >400 cells; seed 0) x 2
   samples; the image alone, drawn exactly as the E058 harness draws it (`data_uri(grid, aligned_cell(grid,
   "deepseek"))`, no JSON, no size label), asked for rows, cols, colours, corners and the grid as JSON. Thinking on,
   effort low (API) / 50 (self-hosted), max_tokens 16,000, 64 workers self-hosted. `--temperature/--top-p` for
   self-hosted runs (default 1.0 / 1.0).
2. `scripts/nll_image_ablation.py --images real|grey`: E066's teacher-forced NLL (`nll_eval_mm.py`, chat path,
   `prompt_logprobs`, unique cache_salt per request) of 32 API image rollouts, with every image replaced by a
   same-size grey PNG in the grey run (same resize plan and token stream, so the runs pair position by position).
   Adds first-turn NLL (`n_t1`, `nll_t1`). The 32 rollouts: `scripts/make_dev_subset.py` (nll_eval_mm.py's own
   `--per-task --n-seq 32 --seed 0 --max-len 60000` pick, 1 MB zstd).
3. `scripts/box_run.sh <arm>...`: serves one setting at a time and runs the tests (`box_chain2.sh`, `box_chain3.sh`:
   the order actually run). `scripts/box_report.py`: tables, paired by grid vs the API and per rollout between tags.

## Self-hosted results (Vast 58332344, 4x RTX PRO 6000 WS)
Full DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, 510 GB), E066 full_ref's stack and flags: E043
lock wheel `...vision11` with the E043 patches, TP4 + EP, `--kv-cache-dtype fp8`, block 128, prefix caching, CUDA graphs
FULL (the server drops to FULL_DECODE_ONLY: the indexer backend has no full-graph prefill), max-model-len 262,144,
64 seqs, 8,192 batched tokens, engram CPU offload, `--limit-mm-per-prompt '{"image": 64}'`, no masks (`ctrl loaded:
experts=None engram=None`). Raw files: `results/box/`.

**Perception probe (% of 120 calls; paired by grid vs the API, 95% CI over 60 grids):**

| run | answered | size | colours | corners | exact | exact by size <=36 / 37-144 / 145-400 / >400 | cell acc (shape ok) | 16k cap hits | out tok |
|---|---|---|---|---|---|---|---|---|---|
| API | 76.7 | 34.2 | 69.2 | 66.7 | 25.8 | 63 / 30 / 7 / 3 | 98.3 | 28 | 8.8k |
| `ref` (E066 settings) | 60.8 | 14.2 | 56.7 | 40.8 | 7.5 | 30 / 0 / 0 / 0 | 90.0 | 47 | 10.2k |
| `ref_t06` (temperature 0.6, top_p 0.95) | 63.3 | 13.3 | 58.3 | 47.5 | 7.5 | | 92.3 | | 9.2k |
| `causal` | pending | | | | | | | | |
| `vitdp` | pending | | | | | | | | |
| `noprefix` | pending | | | | | | | | |

`ref` − API: size −20.0 [−29.2, −11.7], colours −12.5 [−24.2, −0.8], corners −25.8 [−37.5, −13.3], exact −18.3
[−26.7, −10.0]. Within each run, sample 0 and sample 1 score alike (`ref` exact 8.3 / 6.7, API 23.3 / 28.3).

**Image-ablation NLL (32 API image rollouts, 348k generated tokens, per token):**

| tag | NLL generated | NLL first turn |
|---|---|---|
| `ref_real` | 0.7521 | 0.6462 |
| `ref_grey` | 0.7560 | 0.6641 |
| `causal_real` / `causal_grey` | pending | |

`ref`: grey − real +0.0095 [+0.0066, +0.0127] on all generated tokens, +0.0235 [+0.0150, +0.0320] on the first turn.
The self-hosted vision path carries information, since the images make the API's reasoning more predictable. There is
no API number to compare against (the API returns no prompt logprobs), so this cannot show how much is lost; the probe
can.

**FP8 KV is not configurable here, and unlikely to be the cause:** the fork maps `--kv-cache-dtype auto` to fp8 and
accepts only DeepSeek V4.1's native mixed FP8/FP4 KV format (`fp8_ds_mla`, `deepseek_v4_1/attention.py`
`_resolve_dsv41_kv_cache_dtype`: "DeepSeek V4.1 native attention only supports its mixed FP8/FP4 KV cache"), DeepSeek's
own cache layout. The planned `kvauto` arm was therefore a duplicate of `ref`; that server ran `ref_t06` instead. The
Lightning Indexer cache is MXFP4 ("Using MXFP4 indexer cache"); irrelevant for the probe's ~1-2k-token prompts.

## Lead suspect: bidirectional attention over image spans
- **The fork:** its config class sets `is_mm_prefix_lm = vision_n_layers > 0`
  (`transformers_utils/configs/deepseek_v41.py:66`; the comment: "The vision variant needs the mm-prefix plumbing
  (atomic image-span prefill + sparse-SWA window widening)"). Every attention path then lets image tokens attend to
  later tokens of the same image: "Vision variant: image spans are visible bidirectionally, widening prefill SWA index
  rows by up to max_image_tokens columns" (`deepseek_v4_1/attention.py:235`), plus `use_mm_prefix` in the FlashInfer
  sparse and FlashMLA backends and `sparse_swa.py`. vLLM logs "Forcing --disable_chunked_mm_input for models with
  multimodal-bidirectional attention".
- **DeepSeek's reference** (`inference/model.py` from the HF repo; md5 0f88edd2, the same file on the box) is causal:
  `get_window_topk_idxs` gives each prefill query "its own causal window", and compressed blocks are masked to those
  before the query. Image positions differ only in the router bias (`bias_vl`) and the Engram mask. Its "image spans
  must lie inside the first (start_pos 0) chunk" assert serves the embedding merge, not attention.
- If the model was trained causal (as the reference runs it), every image token's hidden states on the fork are ones
  it never saw in training: degraded, not destroyed, perception, as measured.
- **Test, arm `causal`:** ref + `--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input` (keeps the
  atomic image-span prefill). vLLM applies `--hf-overrides` with `config.update()` after the config class is built
  (`transformers_utils/config.py:819`), so the override replaces the constructor's value. The missing "Forcing
  --disable_chunked_mm_input" warning is not proof: vLLM prints it only when chunked mm input is not already
  disabled, and this arm disables it explicitly.
- Relevance: Manh's server (the Kaggle image candidate, `kaggle/dsv41_tools/CANDIDATE_E066_IMG.md`) is a build of the
  same fork; check its `is_mm_prefix_lm` before any image submission. dsv4-infer owns that harness and server config.

## Operations log (for the next box run)
- Box: Vast 58332344, 4x RTX PRO 6000 Blackwell WS, $5.74/h, traffic $0.0052/GB (the 510 GB download cost ~$2.7).
  Handed over by dsv4-pruning at ~08:30 UTC with the user's approval; to be destroyed after the run (user: "kill the
  rtx6000 nodes after finishing").
- Download: `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017` with the read-only HF key (passed on stdin, never on
  the command line): 08:36-09:02, 26 min, ~165 MB/s for the first 10 min, then ~400 MB/s (E066's box: 12.5 min).
- Server start: ~9 min to the first answer (checkpoint load 20-30 s, then compile, CUDA graphs, FlashInfer autotune);
  ~6-8 min on later restarts.
- Probe on the self-hosted server: ~30 min for 120 calls at 64 workers. Decode is ~830 tok/s over ~53 streams, ~16
  tok/s each, so a call that runs to the 16k cap takes ~17 min and two waves of long calls set the time. NLL: 4-5 min
  per 32 rollouts (4 workers).
- The local uplink is ~70 KB/s: copying E066's 316 MB dev set would have taken over an hour. Send subsets
  (`make_dev_subset.py`, 1 MB) and fetch large data on the box from HF.
- Do not find processes by name with a pattern that also appears in the ssh command itself: an `awk '/[p]robe.../'`
  over `ps` matched the remote shell running the command (its heredoc held the same text) and killed it. Use pids
  (`pgrep -P <parent>`) and upload scripts as files instead of heredocs.
- Box time used: ~3 h by the end of the planned arms (~$17 plus ~$3 traffic). API spend: the probe, $0.64.

## Status
Running on node 5 (chain 3): `causal` (probe + NLL real/grey), then `vitdp`, then `noprefix`. Plan agreed with the user:
if `causal` brings the probe within ~3 pp of the API, stop before `vitdp`, copy the results and destroy the box;
otherwise run both.
