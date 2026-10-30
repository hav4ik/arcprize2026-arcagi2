# E068: why images help Flash-1017 on the DeepSeek API but not self-hosted

2026-10-30. User request: "Let's spend some time to understand why. If we even tried with original weights, then it
might be the prompting problem?" (E066: candidate 1 image 69.7 vs text 75.1 pass@1; the full model self-hosted with
images 72.4 on 221/240 attempts vs the API's 81.1.)

## Summary (2026-10-30; box runs done at 11:24 UTC)
1. **It is not the prompt.** Text and tokens are byte-identical to the API's, earlier reasoning stays in the prompt,
   and self-hosted text is as good as the API's. The loss is specific to images, and both self-hosted models show it.
2. **It is perception.** Shown one rendered grid with no text, the self-hosted full model reads it much worse than the
   API on the same pictures: grid size 14% vs 34%, corner colours 41% vs 67%, exact transcription 8% vs 26% (paired
   CIs exclude zero). The images still carry some signal self-hosted (swapping them for grey raises the NLL of the
   API's own reasoning), just much less.
3. **Not the sampler** (temperature 0.6 / top_p 0.95 reads the same as 1.0 / 1.0), **not FP8 KV** (the stack only runs
   DeepSeek's native mixed FP8/FP4 KV format), not the Engram image mask, not the router image bias, not the vision
   preprocessing or ViT/aligner math.
4. **Main cause found: the fork attends bidirectionally over image spans in the language model; DeepSeek's reference
   is causal.** One server flag fixes it (`--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input`):
   probe exact 7.5% -> 18.3% (API 25.8%), small grids now match the API, and the images make the API's reasoning
   markedly more predictable. Every self-hosted image result so far (E066 candidates, full_ref) ran with the bug.
5. **A smaller residual gap remains on larger grids** (exact -7.5 [-15.0, -0.8] vs the API, all of it on grids with
   more than 36 cells). Not the indexer precision (the reference also uses FP4) and not leftover widening (every
   widening path is gated on the flag).
6. **The vision tower's TP split is not the cause of the size/exact gap:** replicating it per GPU (`vitdp`, run on the
   bidirectional path; it was meant to run on top of `causal` but `box_run.sh` defines it on top of `ref`) leaves
   exact and size unchanged (+0.8 / +0.8 vs `ref`). It did raise the answered rate (60.8 -> 75.8%) and colour-set
   accuracy (+15.0 [+5.8, +24.2]) to the API's level, so the sharded ViT's numerics may cost something; `causal` +
   unsharded ViT is untested.
7. **The same bug was fixed upstream** in vLLM PR #57152 (merged 2026-09-16) for upstream's later code; on the E043
   fork the override is equivalent (port: `patches/apply_pr57152_port.py`).

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
| `ref_t06` (temperature 0.6, top_p 0.95) | 63.3 | 13.3 | 58.3 | 47.5 | 7.5 | 30 / 0 / 0 / 0 | 92.3 | 44 | 9.2k |
| `causal` (causal image attention) | 67.5 | 25.8 | 60.8 | 51.7 | 18.3 | 67 / 7 / 0 / 0 | 98.4 | 39 | 9.4k |
| `vitdp` (`ref` + vision tower unsharded; bidirectional, NOT on top of `causal`) | 75.8 | 15.0 | 71.7 | 47.5 | 8.3 | 33 / 0 / 0 / 0 | 90.7 | 29 | 9.6k |

`vitdp` − `ref`: size +0.8 [−3.3, +5.0], colours +15.0 [+5.8, +24.2], corners +6.7 [−2.5, +15.8], exact +0.8
[+0.0, +2.5]; `vitdp` − API: size −19.2, exact −17.5, colours +2.5.
`ref` − API: size −20.0 [−29.2, −11.7], colours −12.5 [−24.2, −0.8], corners −25.8 [−37.5, −13.3], exact −18.3
[−26.7, −10.0]. `ref_t06` − API: the same (exact −18.3 [−26.7, −10.8]). `causal` − API: size −8.3 [−15.0, −1.7],
colours −8.3 [−18.3, +1.7], corners −15.0 [−24.2, −5.8], exact −7.5 [−15.0, −0.8]; on the 54 calls `causal` finished
first, API / `ref` / `causal` exact 48.1 / 16.7 / 40.7%. Within each run, sample 0 and sample 1 score alike (`ref`
exact 8.3 / 6.7, API 23.3 / 28.3): a second identical request (which can reuse the first one's cached image prefix) is
not worse, so prefix-cache reuse is not a suspect and `noprefix` was dropped. Conditional on answering (not hitting
the 16k cap), exact is API 33.7%, `causal` 27.1%, `ref` 12.3%.

**Image-ablation NLL (32 API image rollouts, 348k generated tokens, per token):**

| tag | NLL generated | NLL first turn |
|---|---|---|
| `ref_real` | 0.7521 | 0.6462 |
| `ref_grey` | 0.7560 | 0.6641 |
| `causal_real` | 0.7489 | 0.6335 |
| `causal_grey` | 0.7552 | 0.6616 |

`ref`: grey − real +0.0095 [+0.0066, +0.0127] on all generated tokens, +0.0235 [+0.0150, +0.0320] on the first turn.
`causal`: grey − real +0.0160 [+0.0111, +0.0217] and +0.0410 [+0.0296, +0.0527], i.e. ~1.7x the image information.
`causal_real` − `ref_real` −0.0081 [−0.0115, −0.0052] (first turn −0.0200 [−0.0265, −0.0135]), while
`causal_grey` − `ref_grey` is only −0.0015 [−0.0025, −0.0006] (first turn −0.0025, n.s.): the attention change matters
only when the images carry content.
The self-hosted vision path carries information, since the images make the API's reasoning more predictable. There is
no API number to compare against (the API returns no prompt logprobs), so this cannot show how much is lost; the probe
can.

**FP8 KV is not configurable here, and unlikely to be the cause:** the fork maps `--kv-cache-dtype auto` to fp8 and
accepts only DeepSeek V4.1's native mixed FP8/FP4 KV format (`fp8_ds_mla`, `deepseek_v4_1/attention.py`
`_resolve_dsv41_kv_cache_dtype`: "DeepSeek V4.1 native attention only supports its mixed FP8/FP4 KV cache"), DeepSeek's
own cache layout. The planned `kvauto` arm was therefore a duplicate of `ref`; that server ran `ref_t06` instead. The
Lightning Indexer cache is MXFP4 ("Using MXFP4 indexer cache"); irrelevant for the probe's ~1-2k-token prompts.

## Cause: bidirectional attention over image spans
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
- **Fix, arm `causal`:** ref + `--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input` (keeps the
  atomic image-span prefill). Every widening path is gated on the flag: `sparse_swa.py` 449-450 and 501, the V2
  runner's `compute_mm_prefix_ranges` (`v1/worker/gpu/model_states/default.py` 213-216), the FlashInfer sparse /
  FlashMLA backends; the remaining `max_image_tokens` use in `deepseek_v4_1/attention.py` 458 is for CED prefill
  only. vLLM applies `--hf-overrides` with `config.update()` after the config class is built
  (`transformers_utils/config.py:819`), so the override replaces the constructor's value. The missing "Forcing
  --disable_chunked_mm_input" warning is not proof: vLLM prints it only when chunked mm input is not already
  disabled, and this arm disables it explicitly.
- Relevance: Manh's server (the Kaggle image candidate, `kaggle/dsv41_tools/CANDIDATE_E066_IMG.md`) is a build of the
  same fork; check its `is_mm_prefix_lm` before any image submission (a server log with "multimodal-bidirectional"
  runs the buggy path). dsv4-infer (owns that harness and server config) and dsv4-pruning (E066) were told at 11:00
  UTC. E066's image-harness numbers (candidate 1 image −5.4 vs text, E72-152 image, image dev-NLL screening) were all
  measured with the bug and are worth re-measuring with the flag.
- **Upstream fix:** vLLM PR #57152, "[Bugfix][Model] Restore causal image SWA for DeepSeek V4.1" (merged 2026-09-16,
  merge commit 9f9e1dac), fixes the same bug against upstream's later layout (`models/deepseek_v41/`, in-kernel
  `prefill_left_visible`/`right_visible` widening also gated on `vision_n_layers`). It does not apply to the E043
  fork as is. On the fork the flag is the only gate, so `--hf-overrides '{"is_mm_prefix_lm": false}'` gives the
  PR's causal window (dsv4-pruning checked it at runtime too). `patches/apply_pr57152_port.py` is the PR's config
  hunk plus `max_image_tokens = 0`, for the fork's site-packages (tested on copies; not applied anywhere). Manh's
  newer fork (the Kaggle image server) may carry upstream-style paths, where the override alone may not be enough.
- Indexer precision is not a suspect: the reference also quantizes the Lightning Indexer's queries and keys to FP4
  (`fp4_act_quant` in `Indexer.forward`); the fork's "MXFP4 indexer cache" matches.

## Operations log (for the next box run)
- Box: Vast 58332344, 4x RTX PRO 6000 Blackwell WS, $5.74/h, traffic $0.0052/GB (the 510 GB download cost ~$2.7).
  Handed over by dsv4-pruning at ~08:30 UTC with the user's approval; handed back at 11:26 UTC with the checkpoint
  in place for two user-approved E066 follow-ups (dsv4-pruning destroys it afterwards; user rule: "kill the rtx6000
  nodes after finishing").
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
- Box time used: 08:30-11:26 UTC, ~3 h (~$17 plus ~$2.7 traffic). API spend: the probe, $0.64.

## Lessons
- **Compare the serving code with the model's reference implementation before suspecting numerics.** The bug was a
  semantic choice in the fork (bidirectional image spans, borrowed from DeepSeek V4 Vision-Exp), visible in one config
  line; the earlier audit compared vision preprocessing and the ViT but not the language model's attention masks.
- **A task-free perception probe is the fast, sensitive check for any vision-path change** (~30 min self-hosted,
  $0.64 on the API): it separated perception from reasoning, and its per-size breakdown showed the fix's residual.
  Image-ablation NLL (real vs grey) shows whether images carry information, but not how much is lost without an API
  reference.
- **Read the arm's code path before planning it:** `--kv-cache-dtype auto` silently maps to fp8 on this fork, so the
  planned BF16-KV arm was a duplicate; the server log line "Using DeepSeek V4.1's native mixed KV cache format" and
  `_resolve_dsv41_kv_cache_dtype` showed it.
- **Check each arm's exact flags against the plan:** `vitdp` was described as "on top of `causal`" but `box_run.sh`
  defined it on top of `ref`; it still answered its question (the ViT split), but not the combined one.
- **A missing warning is not evidence:** vLLM's "multimodal-bidirectional" warning disappears once
  `--disable-chunked-mm-input` is explicit. Verify flags from the "non-default args" line or a config assertion.
- **ARC accuracy of the fix is not measured yet** for the full model; candidate 1's re-eval (dsv4-pruning,
  `evals/results/..._e066_c1f_img.jsonl`) was at 222/240 attempts at 16:40 UTC.

## Status
Done on the box (11:24 UTC): `ref`, `ref_t06`, `causal`, `vitdp`; `kvauto` impossible (see above), `noprefix` dropped
(sample 0 vs 1 shows no prefix-cache effect). Results and filtered server logs are in `results/box/` and
`results/box/runs/`. Node 5 went back to dsv4-pruning at 11:26 UTC with the full checkpoint in place (not destroyed:
at 10:56 UTC the user approved two E066 follow-ups on it, an expert-usage check with the fix and candidate 1's
image-harness re-eval; dsv4-pruning destroys the box after those).
Open:
- an ARC eval of the image harness with the fix (self-hosted full model and candidate 1), which decides whether
  images help on Kaggle (candidate 1's re-eval is dsv4-pruning's);
- the residual gap on larger grids; next cheap test: `causal` + `--mm-encoder-tp-mode data` (~35 min).
