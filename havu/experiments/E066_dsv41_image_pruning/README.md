# E066: DeepSeek-V4.1-Flash-1017 pruning for the image harness (E128 uniform and per-layer counts)

User request 2026-10-28: two candidates, both for the E058 image harness (E046 best + one token-aligned image per grid)
with the vision tower kept:
1. E128 (uniform 128 experts per layer), keep set recomputed on image-harness traffic;
2. E128 with per-layer expert counts, 112-128 experts per layer on average.

Workflow (as E043): prune and screen on training data only, then the user picks candidates to measure.

## Data: training rollouts only (user rule: never the 120 eval tasks for pruning)
- Source: model-serving's E065 image set, HF `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e065` @ `a2a1c2d4`
  (`rollouts/e065_images_*.parquet`). The local copies (`evals/results/..._e065_*`) packed by
  `scripts/make_e065_parquet.py` are identical to it: 6,070 of 6,070 rows have identical messages, same 5,994 rollouts.
- 994 arc2_train tasks, 1,430 nvarc_b0, 3,570 nvarc_b1 (multi-test tasks are one rollout). 0 task ids and 0 parents
  among the 120 eval tasks; every parent is an ARC training task.
- Split (`E043 split_dev.py`, sha1(family) % 10, family = NVARC parent or training task id):
  calibration 5,497 rollouts / 167.9M tokens (90.7M generated, 16.65M image), dev 497 / 14.5M (1.48M image),
  85 dev families. Token dirs: `data/e066/tok_img_calib_s/shard*`, `data/e066/tok_img_dev_s/shard*` (not synced).
- The text twin (`e065t_*`) is not used: image and text statistics are never mixed.

## Token streams with images (`scripts/tokenize_image_rollouts.py`)
- Rendered with the model's `encoding/encoding.py`; tools from the stored harness schemas (multi-test changes the
  text); each `<｜deepseek_image｜>` (id 129264) expanded to its image span, n_h * (n_w + 1) + 2 positions, all id
  129264 (verbatim port of `inference/image_processor.py`'s resize plan).
- Check against the API's per-turn usage (`E043 verify_tokenization.py`): prompt tokens equal on 47,154 of 47,155
  turns (the exception: one rollout whose turn hit the 64K per-turn cap and was continued); completion equal on all
  but 9 turns (+-1 re-tokenization, as in E043).
- The vLLM fork consumes the same stream: `scripts/test_mm_requests.py` renders 80 request bodies through the fork's
  own `_normalize_messages` + `encode_messages` (from the wheel) + image expansion: 80/80 identical (2.57M tokens,
  237K image tokens). On the box the observer's input_ids are checked again (`class_stats.py aggregate`).

## Engram
- The fork treats image positions as dead for Engram: they break n-grams (like a sequence start) and their gate is
  zeroed. `E043 engram_census.py --image-dead` (new, opt-in) does the same.
- Census on calibration: 151.3M text positions, 99.4 GiB of distinct rows touched. Masks `data/e066/engram/masks_img`.
- Dev hit rates (13.0M text positions): the shipped tool 64 GiB table (E043, text tool traffic) 94.37% of row
  lookups / 91.45% on generated tokens; a new 64 GiB table from image traffic 94.57% / 91.43%; 96 GiB 97.5% / 96.1%.
  **Decision: both candidates reuse the shipped tool 64 GiB table** (`export_pruned.py --engram-from` the E128 repo).

## Expert statistics and keep sets (`scripts/class_stats.py`)
- Classes per scored position: prompt, gen (reasoning target), block (tool-call target, `</think>` to EOS), image
  (image-span input; the router adds `bias_vl` there). Score per layer = sum of class weights x native amplitude.
- E043's tool rule is `--w-enc 1,1,10,1 --w-dec 0,1,10,0`: with it image tokens count only in layers 0-19, because
  layers 20-39 were scored on generated targets only. Candidate 1 screens image weight 0 vs 1 in layers 20-39.
- Per-layer counts: greedy on the kept share (norm, freq, or their mean), within [floor, cap]; E043's analysis found
  the proxies disagree on layers 36-39 (one expert holds most of the norm), so dev NLL picks the allocation and the
  average (smallest in 112-124 within noise of candidate 1).
- `scripts/test_class_stats.py`: with E043's weights the keep sets equal E043's own pipeline (`reap_aggregate` +
  `special_stats --region` + `select --weight 10`) at E96 and E128; full-sequence alignment and the image class hold;
  a corrupted step is rejected.

## Final plan (user, 2026-10-28; both harnesses added 2026-10-29)
Both candidates: 2-bit GPTQ experts (E128 settings), the shipped tool 64 GiB Engram table, vision tower kept, DSpark
dropped. All choices use calibration and dev rollouts from E065 training data only.

**Both harnesses (user, 2026-10-29: "so the model would be more universal").** Calibration, dev screens and GPTQ
use equal shares of the E065 image set and its text twin (`e065t_*`, the E046 best harness without images; HF
`rollouts/e065_text_*`, local copies verified identical: 6,069 rows; 1 error row). Both are tool-harness traffic, so
this is not a tool/no-tool mix. Text twin: 5,993 rollouts, 167.3M tokens (prompt counts equal the API's on 52,766 of
52,768 turns); dev split = the same 85 families as the image set (paired). On the box: 12M image tokens
(`replay_mm.py`, chat path) + 12M text tokens (E043 `replay_reap.py`, token ids, from a 14M subset
`data/e066/tok_txt_calib_sub`), summed in `class_stats.py aggregate`; `nll_masks_mm.sh` scores each mask on NSEQ image
+ NSEQ text dev rollouts (same families); `gptq_candidate.sh dump` captures 0.7M + 0.7M tokens. Measurement: each
model on both harnesses (`eval_arc_img.sh`, HARNESS=img or txt; txt uses --token-exact as on Kaggle).

**Candidate 1 = E128 uniform for the image harness.** 128 experts in every layer, keep set from the image replay.
Dev screen (64 dev rollouts, one per family, paired against the full model): class weights enc `1,1,10,1`, dec
`0,1,10,w` with image weight w = 0 or 1, plus the shipped E128 keep set as a reference. Lowest dev NLL on generated
and tool-block targets wins. If the shipped keep set is not worse within noise, candidate 1 = shipped E128 experts +
vision tower (its GPTQ tensors are reused).

**Candidate 2 = per-layer counts at fixed quality.** Same class weights as candidate 1. Per-layer counts in
[96, 160] (160 = the patched router-kernel limit), multiples of 8, total budget B experts:
1. sensitivity sweep: from candidate 1 (128 everywhere), cut one layer at a time to 64 experts and score 16 dev
   rollouts; 2 baseline runs give the noise. 42 runs.
2. each layer's proxy curve (kept share of its score vs count) is scaled to pass through its measured point; greedy
   allocation in steps of 8 at B = 4,480 / 4,640 / 4,800 / 4,960 (averages 112 / 116 / 120 / 124);
3. full dev screen of each budget; candidate 2 = the smallest B whose dev NLL (generated and tool-block) is within
   noise (2x the baseline run-to-run difference) of candidate 1; re-checked after GPTQ.

**Measurement (the only use of the 120 eval tasks, after selection):** the full model self-hosted, candidate 1 and
candidate 2, each 120 tasks x 2 attempts in the E058 image harness (`scripts/eval_arc_img.sh`).

## Box runbook (4x RTX PRO 6000 is enough; 8x runs the evals in parallel)
1. E043 Reproduce steps 1-2 (env, patches, base checkpoint @ 1bae0e10). Copy `data/e066/tok_img_*_s`,
   `data/e066/e065_image.parquet`, the shipped tool Engram mask (`data/e066/engram/e043tool_top64gib.npz`), the shipped
   E128 keep set, and the E043 + E066 scripts.
2. Full model with the vision tower: `VISION=1 GPUS=0,1,2,3 TP=4 PORT=18001 CTRL=/dev/shm/e043/ctrlB.json serve.sh imgB`.
3. Preflight: `replay_mm.py --tokens 200000 --obs-dir .../obs_pre` then `class_stats.py aggregate` (stream check);
   `nll_eval_mm.py --n-seq 4` with the full model (token check).
4. Replay ~15M calibration tokens with the frozen tool Engram mask: `replay_mm.py --tok data/e066/tok_img_calib_s/shard*
   --parquet ... --tokens 15000000 --seed 1 --engram-keep e043tool_top64gib.npz`; `class_stats.py aggregate`.
5. Candidate 1 screen (`nll_eval_mm.py --per-task --n-seq 64`, runtime masks).
6. Candidate 2: sensitivity sweep, calibrated allocation, budget screen (sweep script and calibrated greedy: to write).
7. 2-bit GPTQ for both: `replay_mm.py --dump-x --dump-x-frac 0.1 --expert-keep <keep> --tokens 1500000`, then
   E043 `gptq_experts.py --keep <keep>` (per-layer lists work); dev NLL with `E043_GPTQ_DIR`.
8. Measure: full model, candidate 1, candidate 2 with `scripts/eval_arc_img.sh` (chat path; `--token-exact` has no
   image path).
9. Export: `export_pruned.py --no-mtp` (keeps vision.*, aligner.*, image_start/end/newline) + `--engram-from` the E128
   repo, then `make_lowbit_export.py`. Candidate 2 also needs per-layer `n_routed_experts` in the config, the exporter
   and the fork (not written yet; needed only to ship it).

## Box run (Vast 58209054, 4x RTX PRO 6000 Workstation at 400 W, PCIe 5.0, 755 GB RAM, 2.2 TB disk at 3 GB/s)
- Setup 2026-10-29: the image's demo vLLM (Qwen3.5-9B) stopped (supervisorctl + its orphan PIDs); checkpoint on disk
  (83/83 files, sizes = HF); `scripts/setup_env.sh`; full model + vision: `VISION=1 ... serve.sh imgB` (TP4 + EP4,
  util 0.90, `--limit-mm-per-prompt '{"image": 64}'`), engine init 280 s.
- Preflight: 7 image rollouts (200K tokens, 27,927 image positions): observer input_ids = offline stream; dev loss
  path matches the offline token at every position.
- Calibration replays: image 374 rollouts / 12.0M tokens (52 min, ~3.8K tok/s, 1.13M image positions); text 451 /
  12.0M (47 min). Combined stream check passes on all 825; every layer saw >= 377 of 384 experts.
- Image-position routing kept by 128-expert sets (image statistics): shipped E128 71.8%; recomputed, image weight 0
  78.0%; weight 1 83.3% (generated tokens 85.1 / 85.6 / 85.4%).

### Candidate 1 screen (dev NLL on generated targets, 32 image + 32 text dev rollouts, full experts = 0 ref)
| mask (128 per layer) | image NLL | text NLL | tool-block NLL image / text |
|---|---|---|---|
| full (2 runs) | 0.75221 / 0.75241 | 0.71902 / 0.71905 | 0.1049 / 0.1151 |
| shipped E128 | 0.81920 (+8.9%) | 0.77858 (+8.3%) | 0.1249 / 0.1346 |
| image weight 0 | 0.81701 | 0.77661 | 0.1242 / 0.1338 |
| **image weight 1 (candidate 1)** | **0.81655 (+8.55%)** | **0.77659 (+8.0%)** | **0.1240 / 0.1334** |

Paired bootstrap over rollouts, weight 1 - shipped: image -0.0027 [-0.0038, -0.0015], text -0.0020 [-0.0031, -0.0010],
tool-block (both) -0.0011 [-0.0019, -0.0002]. Weight 1 - weight 0: -0.0002 [-0.0007, +0.0003] (tie; weight 1 keeps
more image routing). The shipped set is worse, so candidate 1 gets its own GPTQ. `results/c1_screen.jsonl`.

### Candidate 2: layer sweep and budget screen
- Sweep (`results/sweep.jsonl`, `sens.json`): candidate 1 with one layer cut 128 -> 64, 8 image + 8 text dev rollouts,
  dev NLL increase (1e-3 nats/token): layers 12 (17.3), 17 (14.7), 19 (12.1), 4 (10.4) most sensitive; 20-34 cheap
  (0.2-2.2, except 23: 4.9); 36-39 2.4-4.8 (the norm proxy called them nearly free); layer 2 negative (-3.1).
  Re-check on 32 fresh rollouts (`sweep_check.jsonl`): layer 2 -3.20 [-3.89, -2.46], 12 +16.1, 25 +0.31, 26 +1.45
  (vs 0.17: small costs carry ~1e-3 of rollout noise at 16 rollouts); merged into `sens_merged.json`.
- Calibrated allocations (blend shape, floor 96, cap 160, steps of 8) move experts from layers 0-2, 7, 14, 18, 20-34
  (mostly to the floor) to 3-6, 11-13, 17, 19, 23, 37-39.
- Budget screen (`results/c2_screen.jsonl`, same 64 dev rollouts as candidate 1; dev NLL vs candidate 1, paired
  bootstrap):

| mask (experts) | generated targets | tool-block targets |
|---|---|---|
| calibrated blend, avg 112 (4,480) | +0.0120 [+0.0106, +0.0135] | +0.0033 [+0.0013, +0.0052] |
| calibrated blend, avg 116 (4,640) | +0.0015 [+0.0003, +0.0028] | -0.0021 [-0.0031, -0.0011] |
| calibrated blend, avg 120 (4,800) | -0.0055 [-0.0067, -0.0044] | -0.0042 [-0.0053, -0.0032] |
| calibrated blend, avg 124 (4,960) | -0.0105 [-0.0116, -0.0094] | -0.0057 [-0.0072, -0.0043] |
| calibrated norm, avg 120 | -0.0063 [-0.0073, -0.0052] | -0.0043 [-0.0059, -0.0030] |
| calibrated freq, avg 120 | -0.0042 [-0.0054, -0.0030] | -0.0040 [-0.0052, -0.0028] |
| uniform 120 | +0.0144 [+0.0132, +0.0156] | +0.0038 [+0.0030, +0.0046] |

  Run-to-run noise (full twice): +0.0001 [-0.0003, +0.0005]. At equal memory (4,800) the calibrated allocation beats
  uniform by 0.020 nats/token, a third of the full-model gap; the proxy-only estimate (E043 analysis/layer_alloc,
  +3 to +7 uniform-equivalent experts) understated it.

- Smallest budget within noise of candidate 1: avg 116 fails (blend +0.0015 [+0.0002, +0.0028], norm +0.0021);
  **avg 118 blend passes: -0.0020 [-0.0032, -0.0008]** (image -0.0024, text -0.0016; tool-block -0.0032) -> candidate
  2 = `calblend_f96c160s8_avg118`, 4,720 experts (-7.8% vs candidate 1), per layer
  `96 96 96 160 160 136 152 96 120 112 128 160 160 160 96 96 136 160 96 160 96 96 96 160 96 96 96 96 96 96 96 96 96 96 96 120 104 136 120 160`.

### 2-bit GPTQ (`gptq_candidate.sh`, E128 recipe, 0.71M image + 0.73M text calibration tokens, 144,013 rows)
- GPTQ dirs: candidate 1 47 GB, candidate 2 43 GB (`/workspace/dsv4/gptq/c{1,2}` on the box; ~6 min each on 4 GPUs).
- Candidate 1 2-bit dev NLL (`results/q_screen.jsonl`): image 0.84815 (native 0.81655, +3.9%; +12.8% vs full), text
  0.80680 (native 0.77659, +3.9%; +12.2% vs full); tool-block 0.1385 / 0.1497. As E043's tool E128 2-bit (+11.7%).

### Context limit on the image harness (found by the reference run)
The chat path requests max_tokens = min(64,000, remaining output budget) every turn and has no `--context-limit`, so
at the default max-model-len 131,072 any turn whose prompt passes ~67K tokens gets a 400 error. On the API image run
(E058, no cap), 27 of 240 attempts had such a turn and 11 of them were correct, while no attempt's real prompt + output
ever passed 131K (text E046: 31 / 15 correct, 1 incorrect). So shrinking max_tokens to the room left costs nothing,
and not shrinking loses ~6% of correct attempts. E066 serves every eval at max-model-len 262,144 (the reference's
errored attempts are re-run there). Suggested harness fix (dsv4-infer's file): `--context-limit` on the chat path.

### Invalid: the first candidate 1 eval ran without its expert mask
`results/tools_persistent_grid_lean2sub_dsv41_low_e066_c1_{img,txt}.jsonl` (2026-10-29 12:17-16:35 UTC, ~200/240
attempts each) measure the wrong model. Do not use them.
- `nll_masks_mm.sh` reset the control file to "no expert mask" at the end of the 2-bit dev NLL (12:13:57 UTC). The
  eval server then logged `ctrl loaded: experts=None`, and the evals started at ~12:17.
- With `E043_GPTQ_DIR` set and no mask, the router chooses from all 384 experts: the 128 kept ones are GPTQ, the other
  256 are rounded to 2 bits by Humming RTN at load (`e043_humming.precomputed` returns None for them).
- Fixes:
  - `nll_masks_mm.sh` now restores the control file's previous content.
  - Candidate 1 is evaluated from its own export (`serve_eval.sh c1x`: GPTQ-only, no runtime mask needed).
  - The runtime-mask server (`serve_eval.sh e72q`) checks the server's last `ctrl loaded` line before evals start.

### Final candidates (user, 2026-10-29)
- **Candidate 1:** uniform E128, export `c1_gptq`, 132,060,122,105 bytes, content check OK. HF:
  `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit`.
- **Dynamic count:** floor 72, cap 152, step 16, average 120 (`calblend_f72c152s16_avg120`, 6 distinct counts). It
  replaces the step-8 floor 72 / cap 160 layout the user picked from the chart, with the user's conditional OK:
  - same dev NLL vs C1: -0.0104 all / -0.0115 image (step 8: -0.0103 / -0.0117);
  - Kaggle KV 3.38M vs 3.27M (dsv4-infer, dummy weights): Humming's repack holds extra memory per distinct count.
  HF: `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E72-152-avg120-2bit`.
- Exports: `scripts/export_candidate.sh` (E043 `export_pruned.py --no-mtp --no-routed-experts [--per-layer]` +
  `make_lowbit_export.py`). The Engram is the HF E112 repo's 16 row chunks, mask-hash checked. Uploads:
  `scripts/upload_candidate.sh` (cards in `scripts/cards/`).
- Box 1 chain `scripts/chain_box1_main.sh` runs: C1 export parity vs `c1q`, upload, E72 capture + GPTQ + server + dev
  NLL, then export + upload.
- Evals run per box from the local machine with `scripts/run_pair.sh` (image first, the text twin once the image run
  has 200 finished attempts): E72-152 on box 1 (`e72_img`, `e72_txt`); candidate 1 on box 4 (Vast 58284661, from its
  HF export, sha-checked; `c1x_img`, `c1x_txt`).
  - Box 4 setup: `box4_setup.sh` in the scratchpad; the steps are `setup_env.sh`, HF download, the sha check, then
    `serve_eval.sh c1x`.
  - Vast's vllm template image starts a demo vLLM (Qwen3.5-9B) under supervisord that holds ~89 GiB per GPU. Stop it
    with `supervisorctl stop vllm` and then kill its orphaned EngineCore and workers by PID; the supervisor stop alone
    leaves them running.
- FP4 Engram (teammate's idea, for host RAM): `scripts/fp4_engram_chunked.py` builds `export/engram64_fp4` (rel. MSE
  ~1.39%, as in E043). dsv4-infer puts it on the next-version list as a host-RAM safety margin (~30 GiB on Kaggle); it
  does not pay in throughput.
  - **Pilot (`scripts/fp4_pilot.sh`, 2026-10-29):** the C1 2-bit export served with MXFP4-rounded Engram rows vs the
    shipped FP8 Engram, paired dev NLL on the same 64 rollouts:
    - image: generated 0.84825 -> 0.84813, tool-call 0.13855 -> 0.13820;
    - text: generated 0.80641 -> 0.80686, tool-call 0.14936 -> 0.14960.
    - All changes are below 0.0005, within noise: free on both harnesses.
  - **Format errors on 1M real rows** (`scripts/engram_format_error.py`, relative L2 vs the FP8 table):
    - MXFP4 (OCP scale rule; vLLM PR 61530, NVIDIA's layout) 12.0%; MXFP4 best-of-two scale 11.8%; both 4.25 bits.
    - NVFP4 9.5% and INT4 with a BF16 scale per 32 9.8%, both 4.5 bits; INT4 with a UE8M0 scale per 32 14.7%.
    - MXFP6 1.2% at 6.25 bits.
  - Recommendation: MXFP4 with best-of-two scales. Same storage and decoder as the PR, 64 -> ~34 GiB.

## Final ARC-AGI-2 results (2026-10-29, 120 eval tasks x 2, ARC Prize scoring, `scripts/compare_evals.py`)

> **2026-10-30: every image result in E066 ran with a serving bug (E068).** The fork sets `is_mm_prefix_lm =
> vision_n_layers > 0`, so the language model attended bidirectionally over image spans; DeepSeek's reference is
> causal. Fixed with `--hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input`; on the full model the
> grid-reading probe rises from 7.5% to 18.3% exact (API 25.8%; `experiments/E068_selfhost_image_gap`). The image rows
> below, the image dev-NLL screens and the image half of the calibration (expert statistics and GPTQ activations)
> are affected; text rows, text calibration and the Engram census are not. Re-check: `scripts/fix_c1_chain.sh`
> (fixed-path replays, keep-set shift vs candidate 1) and `scripts/run_c1xc_local.sh` (candidate 1 image re-eval).

| model | harness | pass@1 | pass@2 | output tok/attempt | tool-call error rate |
|---|---|---|---|---|---|
| candidate 1 (new E128, 2-bit) | text | 75.1 | 85.8 | 38.9K | 14.4% |
| candidate 1 (new E128, 2-bit) | image | 69.7 | 80.0 | 37.5K | 15.4% |
| E72-152 (per-layer 72-152, avg 120, 2-bit) | text | 69.0 | 79.6 | 35.9K | 7.3% |
| E72-152 | image | 69.0 | 78.3 | 36.9K | 9.3% |
| old tool E128 2-bit (B32's model; E046 run 7, 10-25) | text | 73.7 | 84.6 | 40.8K | 15.1% |
| full model, API | text | 77.6 | 88.6 | 29.6K | 2.7% |
| full model, API (E058) | image | 81.1 | 90.7 | 27.0K | 3.6% |
| full model, self-hosted (221/240 finished) | image | 72.4 | 82.8 | 24.4K | 5.2% |

Paired differences (95% bootstrap CI over tasks):
- Candidate 1 text vs B32 text: pass@1 +1.5 [-3.1, +6.0], pass@2 +1.2 [-3.8, +6.2]. A tie, slightly ahead.
- Candidate 1 text vs the full model's API text run: -2.5 / -2.8 (CIs include zero).
- Candidate 1 vs E72-152 on text: +6.2 [+0.6, +11.7] / +6.2 [+0.4, +12.1].
- **Candidate 1 image vs its own text run: -5.4 [-10.6, -0.4] / -5.8 [-11.2, -0.8].** Images hurt the pruned model on the
  self-hosted stack.
- The full model self-hosted (image) vs the API (image): -4.4 [-9.3, +0.2] on the same tasks.
  - Ruled out as the cause: prompt text (byte-identical) and prompt tokens including image tokens (identical to the
    API on every task), image preprocessing (the fork's `load_image` is a line-for-line port of DeepSeek's reference),
    and sampling (temperature 1.0 / top_p 1.0, the model card values).
  - Untested suspects: FP8 KV cache on long image prompts, kernel numerics, the API's thinking-mode sampling.
- Decisions:
  - E72-152 retired: below B32 on both harnesses, and below its own better dev NLL.
  - Candidate 1 on the text harness is the next-submission option: B32's setup with the model swapped.
  - The image harness is on hold until the self-hosted image gap is explained.
- Kaggle: candidate 1 is a private Model `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pyTorch/default/1`
  (137 files, 132.06 GB, verified, shared with chankhavu). Tonight's submission (user) was B32-M.
- Trace reviews (`trace_review/report_*.md`): no mechanical damage in either candidate.
  - Candidate 1's tool errors are identifier typos at the old E128 rate, growing with attempt length (5.8% -> 17%).
  - E72-152 tends to submit the first rule that fits the training pairs without checking test-only features.

## Engram A/B on B32's model (2026-10-30, Vast 58332344, 4x RTX PRO 6000; `scripts/b32_engram_ab.sh`, `b32_engram_fp8next.sh`)

User's question: does MXFP4 Engram cost ARC accuracy on the model and harness we know best? Setup:
- Model: B32's tool E128 2-bit (HF `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`), text only.
- Harness: B32's pinned harness and flags, tool code in the local executor on the box (`scripts/b32_local_eval.py`).
- Server: the fork at TP4, fp8 KV, prefix caching.
- Runs: 120 eval tasks x 2 attempts per arm, each arm alone on the box. Only the Engram table differs: the shipped FP8
  (`export/b32_e128`) or the same rows rounded to MXFP4 (`export/b32_fp4`; served weights checked to hold 16 distinct
  FP8 bytes, the E2M1 grid, vs 254 in FP8).
- Scoring: locally with the **Kaggle** solutions file (`scripts/score_b32_engram.py`). The GitHub copy used by the
  harness `correct` field differs on 4a21e3da, f560132c, b6f77b65.

| Engram (same box, 120 tasks) | pass@1 | pass@2 | attempt 0 | attempt 1 | tasks fully solved (pass@2) | output tok/attempt |
|---|---|---|---|---|---|---|
| FP8 (shipped) | 69.9 | 81.4 | 69.7 | 70.1 | 95 | 41.4K |
| MXFP4 | 69.8 | 78.9 | 67.6 | 71.9 | 91 | 41.4K |

- Paired MXFP4 - FP8 (95% bootstrap CI over tasks): pass@1 -0.1 [-4.9, +4.6], better/worse 19/19; pass@2 -2.5
  [-8.8, +3.8], 9/12. **No measurable effect.** A pass@2 loss of up to ~5 points can't be excluded.
- The 48 GiB rows + MXFP4 arm was postponed (user); its table is built and on HF (below).
- Mixed baselines misled: before the control ran, MXFP4 looked -9.5 vs E052 arm A (CI excluding zero) and -4.7 vs a
  combined FP8 baseline. All of that was setup difference. Compare Engram variants on the same box only.
- **E052's 76.5 is not reproduced here:** same model and harness, FP8 Engram, attempt 0: 69.7 here vs 76.5 in E052 arm A
  (V6-era stack, 4090s). E046 run 7 (Kaggle key) has 70.0. The cause is open: the stack, or E052 being a high draw
  (one run each, CIs ~+/-5).
  - Against the stack explanation (model-serving): the full model's text run on this same wheel matched the API
    (85.7 vs 82.7 on 136 attempts).
- MXFP4 needs vLLM PR 61530 (+ the row_map patch) to save host RAM; until then it only proves the format is safe.
- Engram tables backed up: HF dataset `chankhavu/dsv41-flash1017-e066-engram-tables` (private): `b32_tool64_mxfp4/`
  and `b32_tool48_mxfp4/`, verified.
- **Further Engram pruning** (`scripts/engram_subset.py`; rows ranked by the training census only):
  - Coverage check on B32's own eval traces (`scripts/engram_evalcheck.py`, measurement only, nothing selected).
    92.2% of eval Engram lookups fall in the 64 GiB table (training census: 94.3%).
  - Lookups kept by smaller budgets, training vs eval:

    | budget | training | eval | eval, generated tokens |
    |---|---|---|---|
    | 56 GiB | 99.92% | 98.59% | 98.00% |
    | 48 GiB | 99.44% | 97.00% | 95.71% |
    | 40 GiB | 98.50% | 95.11% | 93.01% |
    | 32 GiB | 97.18% | 93.03% | 90.02% |

  - The training census understates the loss on unseen tasks by 2-4x.
  - Rows never seen in the training census (26.9M, 10%) still serve 1.1% of eval lookups.
  - Advice: 56 GiB at most without an ARC A/B, 48 GiB only after one, nothing below 40 GiB.
- Node 5 handed to model-serving for the image-harness research (user, 2026-10-30 08:10 UTC); results and logs in
  `results/b32_engram/`.

## Status
- 2026-10-28: CPU prep done (data, tokens, census, scripts, tests). No box yet; no API spend.
- 2026-10-29: box up, replays done, candidate 1 picked; candidate 1 GPTQ capture and the layer sweep running.
- 2026-10-29 afternoon: final candidates fixed (above); boxes 2 and 3 destroyed; the C1 eval was found unmasked and
  re-planned; the box 1 chain and the local evals are running.
- 2026-10-29/30 night: all four evals finished; both boxes destroyed (logs in `results/box1`, `results/box4`).
- 2026-10-30: Engram A/B on B32's model done: MXFP4 = FP8 on ARC (pass@1 -0.1, pass@2 -2.5, CIs include zero); node 5 handed to model-serving.
