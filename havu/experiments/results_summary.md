# ARC-AGI-2 eval results: all runs (recomputed from result files)

Built 2026-10-11 from `evals/results/*.jsonl`. Every score below was recomputed from the stored `pred` grids against the
GitHub ground truth (`data/arc-agi-2-github/data/evaluation`, 120 tasks / 167 test pairs). For E001a–c the reference is
NVIDIA's `expected_output`. READMEs were used only to identify runs and settings, and to cross-check. Each record is the
latest one per (task, pair, attempt), as the resume logic writes them. The recomputed `correct` matched the stored flag on
every record.

## Main table

Scores are percentages, task-weighted (ARC Prize scoring). Per-attempt values are in parentheses. "—" means not applicable.

| Model | Harness / tools | Key settings | Full pass@1 | Full pass@2 | Pilot pass@1 | Pilot pass@2 | Tokens / attempt | Attempts | Cost | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **DeepSeek, no tools: official public eval** | | | | | | | | | | |
| DeepSeek V4.1-Flash (E001d) | Single call, ARC Prize prompt | max effort, max_tokens 200k | **59.9** (65.1 / 60.3 / 54.2) | **74.6** (mean of 3 attempt pairs; a0+a1 79.3) | **57.4** (66.7 / 57.8 / 47.8) | **72.6** (a0+a1 80.0) | 55.7k (median 49.2k) | 3 | ~$16.7 est. (27.9M output tokens × $0.60/M) | 17 connection-error attempts were retried, so all 501 are complete. Majority vote over 3: 68.6 |
| DeepSeek V4.1-Flash (E005) | Single call, ARC Prize prompt | high effort | — (paused, 2/167 pairs) | — | — (0 pilot pairs) | — | 8.0k | 1 | ~$0.01 | Paused: 1/2 correct. Short pairs only, not representative |
| DeepSeek V4.1-Flash (E005) | Single call, ARC Prize prompt | low effort | — (paused, 7/167 pairs) | — | — (2/26 pairs) | — | 10.1k | 1 | ~$0.04 | Paused: 6/7 correct. Short pairs only |
| DeepSeek V4.1-Flash (E003) | Single call, ARC Prize prompt | max effort | — (4 tasks) | — | — | — | 57.4k | 2 | ~$0.28 | Own 4-task subset: pass@1 62.5 (50 / 75), pass@2 75.0 |
| DeepSeek V4-Pro (E003) | Single call, ARC Prize prompt | max effort | — (4 tasks) | — | — | — | 67.4k | 2 | ~$1.07 (at $1.98/M) | Own 4-task subset: pass@1 50.0, pass@2 50.0 |
| **DeepSeek, no tools: NVIDIA validation subset** (different set: 120 problems, 1 test input each, NVIDIA prompt; not comparable with the rows above) | | | | | | | | | | |
| DeepSeek V4.1-Flash (E001a) | Single call, NVIDIA prompt | max effort, 200k | 52.9‡ (52.5 / 53.3) | 60.8‡ | — | — | 81.3k | 2 | ~$11.7 est. | ‡ NVIDIA set |
| DeepSeek V4.1-Flash (E001b) | Single call, NVIDIA prompt | high effort, 200k | 40.8‡ (45.0 / 36.7) | 56.7‡ | — | — | 72.0k | 2 | ~$10.4 est. | ‡ NVIDIA set |
| DeepSeek V4-Pro (E001c) | Single call, NVIDIA prompt | max effort, 200k | 40.0‡ | — | — | — | 74.7k | 1 | ~$17.7 est. | ‡ NVIDIA set |
| **DeepSeek, with tools (E004, persistent Python sandbox, harness v3)**: full eval, 120k total / 64k per turn / 24k final budget | | | | | | | | | | |
| V4.1-Flash, **E004-GL** (candidate) | grid mode | lean2 prompt, low effort | 66.0 (66.7 / 70.0 / 64.4 / 62.8) | 76.4 (mean of 6 attempt pairs, 75.3–77.8; a0+a1 77.5) | 71.7 (70.0 / 76.7 / 73.3 / 66.7) | 82.8 (a0+a1 80.0) | 49.5k (median 41.4k) | 4 | $22.60 rec. | Majority vote over 4: 76.9 |
| V4.1-Flash | grid mode | lean2, high effort | **68.3** (67.1 / 69.4) | **78.2** | **73.3** (80.0 / 66.7) | **86.7** | 55.2k | 2 | $12.33 rec. | |
| V4.1-Flash | grid mode | lean2, max effort | 65.3 | — | 63.3 | — | 65.0k | 1 | $7.22 rec. | |
| V4.1-Flash | grid mode | lean2, low effort, **100k total cap** | 61.9 | — | **73.3** | — | 45.5k | 1 | $5.20 rec. | The prospective cap run |
| V4.1-Flash | code mode (`submit_solution`, checked on train pairs) | lean2, high effort | 61.9 | — | 56.7 | — | 58.9k | 1 | $6.66 rec. | |
| V4.1-Flash | code mode | lean2, max effort | 60.7 | — | 60.0 | — | 65.2k | 1 | $7.17 rec. | |
| V4.1-Flash | grid mode | **plain** ARC Prize prompt, low effort | 51.2 | — | 51.1 | — | 61.0k | 1 | $6.61 rec. | Prompt ablation |
| V4.1-Flash | grid mode | **plain** ARC Prize prompt, max effort | 47.6 | — | 37.8 | — | 86.2k | 1 | $9.06 rec. | Prompt ablation |
| **DeepSeek, with tools: E004a pilots** on E004's own hard-biased subset (20 pairs / 15 tasks; only 2 tasks overlap the standard pilot), 200k total / 8k final | | | | | | | | | | |
| V4.1-Flash | grid, v1 | plain, max | — | — | — | — | 103.9k | 1 | $1.32 | Subset: 13/20 pairs, 66.7 task-weighted |
| V4.1-Flash | code, v1 (buggy check) | plain, max | — | — | — | — | 97.2k | 1 | $1.25 | Subset: 12/20, 60.0 |
| V4.1-Flash | code, v2 | plain, max | — | — | — | — | 104.4k | 1 | $1.30 | Subset: 12/20, 60.0 |
| V4.1-Flash | grid, v2 | lean, max | — | — | — | — | 74.0k | 1 | $0.95 | Subset: **15/20, 76.7** |
| V4.1-Flash | code, v2 | lean, max | — | — | — | — | 91.0k | 1 | $1.19 | Subset: 15/20, 73.3 |
| V4.1-Flash | grid, v2 | lean, high | — | — | — | — | 57.2k | 1 | $0.78 | Subset: 14/20, 66.7 |
| V4.1-Flash | code, v2 | lean, high | — | — | — | — | 75.7k | 1 | $0.99 | Subset: 12/20, 56.7. For reference, E001d on this subset: a0 10/20 (53.3), mean of 3 attempts 48.9 |
| V4.1-Flash | stateless sandbox (v0) | max | — | — | — | — | 61.6k | 2 | $0.16 | 2 easy tasks: 4/4 attempts correct |
| **Qwen** | | | | | | | | | | |
| Qwen3.8-27B, OpenRouter DeepInfra (bf16) | Single call, ARC Prize prompt | high effort, 200k | — (pilot only) | — | 20.0 (6/26) | — | 35.2k | 1 | $1.75 rec. | 14/26 attempts returned empty content (mid-thought `stop`) |
| Qwen3.8-27B, OpenRouter Reka / Alibaba diagnostics | Single call | high; Reka thinking ~2k, Alibaba ~16k (both fixed) | — | — | — (11 attempts, 4 tasks) | — | 4.3k / 2.9k / 20.7k | 1 | $0.33 rec. | 0/11 correct. These runs test providers, not scores |
| Qwen3.8-27B-NVFP4, self-hosted 4×RTX 6000 (SGLang) | Generation API, ARC Prize prompt | medium tier, 48k budget, temp 0.6, top_p 0.95, top_k 20 | 9.7 (9.7 / 9.6) | 13.3 | 12.8 (14.4 / 11.1) | 14.4 | 39.1k | 2 | rental: not stated in README; ~$10 if the run's 115.8 min are billed at the box's $5.42/h | 53% of attempts forced to answer; 42.5% had a mid-thought EOS |
| Qwen3.8-27B-NVFP4, self-hosted (official settings) | Generation API, ARC Prize prompt | xhigh tier, 240k budget, temp 1.0, top_p 0.95, top_k 20 | **36.4** (35.8 / 36.9) | **48.2** | **40.0** (36.7 / 43.3) | **46.7** | 73.6k (median 67.5k) | 2 | rental ~$27 (incl. pilot) | The first 52 records are the pilot run (`_t1_pilot`). 0 errors, 3/334 forced answers |
| Qwen3.6-35B-A3B-NVFP4, self-hosted | Generation API, ARC Prize prompt | no tier (no system message), 240k, temp 1.0, top_p 0.95, top_k 20, presence penalty 1.5 | **interim (203/334 attempts)**: 6.1 (6.2 / 6.0) on 101–102/167 pairs | 10.2 on the 90 pairs with both attempts done | 10.0 (6.7 on 25/26 pairs / 13.3) | 13.3 (25 pairs) | 45.6k (biased low: short attempts finish first) | 2 (in progress) | rental, run still in progress; no $ in README | Lower bound with unfinished pairs counted wrong: pass@1 3.8. 0 errors, 0 forced answers so far |
| **Nemotron** | | | | | | | | | | |
| Nemotron 3.5 Lightning, OpenRouter CoreWeave (bf16) | Single call, ARC Prize prompt | high effort, 200k | **0.0** (0/167) | — | **0.0** (0/26) | — | 68.0k (median 64.9k) | 1 | $2.36 rec. (+$0.03 smoke run) | 167/167 completed. The model is not capable enough; this is not a harness problem (E008) |
| Nemotron 3 Super 120B-A12B, OpenRouter DekaLLM (fp8, "degraded") | Single call, ARC Prize prompt | high effort, temp 1.0, top_p 0.95, 200k | — (pilot only) | — | 0.0 on only 5/26 pairs (4 tasks) | — | 49.2k (5 completed) | 1 | $0.11 rec. | **5/26 completed, 21/26 provider errors** (network or upstream) on long streams. Inconclusive |

### Definitions
- **Task-weighted ARC Prize scoring:** each task scores the fraction of its test pairs solved. The score is the mean over tasks.
- **Full** = all 120 official public-eval tasks / 167 test pairs. **Pilot** = the first 15 tasks in sorted task-id order (`0934a4d8` … `247ef758`), which have 26 pairs.
  - E004a used its own 20-pair, hard-biased subset, which shares only 2 tasks with the pilot. Its scores appear in Notes on that subset, not in the pilot columns.
  - For runs that covered every task, the pilot columns are computed from the same records.
- **pass@1** = the mean, over attempt indices, of the single-attempt score.
- **pass@2:**
  - With 2 attempts: a pair counts if either attempt is correct.
  - With more than 2 attempts: the mean over all attempt pairs. The attempts 0+1 value is noted.
  - "—" for 1-attempt runs.
- **Coverage:** an incomplete run is scored only on its finished pairs: a task's score is computed over its covered pairs, and only tasks with at least one covered pair are averaged. The coverage is flagged in the row.
- **Tokens / attempt** = mean completion tokens over finished attempts (latest record). For multi-turn tool runs this is the total output across all turns of the rollout.
- **Cost:**
  - "rec." = the sum of recorded `cost_usd` (OpenRouter) or `est_cost_usd` (tool harness; off-peak list price, input included) over every record in the file, including superseded or errored records, since those were billed.
  - "est." = recorded output tokens × list price (Flash $0.60/M, Pro $1.98/M off-peak). It ignores input (~$0.001 per call) and the unrecorded partial output of errored calls.
  - "rental" = vast.ai box time from the E008 README. API cost was $0.

## Notes and caveats

- **The pilot is about as hard as the full set, but noisy.** Across the 11 runs that cover both, pilot minus full pass@1 averages +0.8 pt. Single runs range from −9.8 (grid/plain/max) to +11.4 (cap-100k).
  - Flash E001d: pilot 57.4 vs full 59.9.
  - E004-GL: 71.7 vs 66.0.
  - Qwen xhigh: 40.0 vs 36.4.
  - With 15 tasks, one task moves the pilot score by 6.7 pt. Use the pilot to catch large gaps, such as Qwen vs Flash or Nemotron at 0, not to rank close configs.
- **Single samples vary by about ±3–5 pt.** E004-GL's 4 samples span 62.8–70.0, and E001d's 3 samples span 54.2–65.1.
  - Compare means over samples. Grid/lean2/high (68.3, 2 samples) vs E004-GL (66.0, 4 samples) is within that noise.
  - Tool runs use a 120k total budget; E001d used one 200k call.
- **Interim and incomplete runs:**
  - **Qwen3.6 is interim** (snapshot of 203/334 attempts, 2026-10-11 14:39 UTC). Short attempts finish first, so its tokens/attempt is biased low and its accuracy may shift.
  - **Nemotron 3 Super** is inconclusive: 21/26 attempts were DekaLLM or network failures.
  - The Qwen3.8 OpenRouter pilot lost 14/26 attempts to empty content. Its 20.0 likely understates the model: the self-hosted xhigh run gets 40.0 on the same pilot.
  - Paused runs (E005) and subset runs (E003, E004a) are listed for completeness only.
- **Public vs semi-private, and contamination.** All official-set numbers here are on the *public* eval.
  - ARC Prize reports V4 Flash 0731 at 63.9% public / 61.4% semi-private, and Qwen3.8-27B at 42.4% semi-private (vs our 48.2% public pass@2).
  - The public eval predates V4.1, and Flash listed real eval task ids in E004 traces, so its absolute scores may be inflated. Comparisons within one model are unaffected.
- **Cross-checks against the READMEs and the TSV.** Every recomputed score matches the current README tables within 0.5 pt. Some numbers are stale:
  - **E001d attempt 0.** The TSV E001d headline (pass@1 63.1, pass@2 78.1) and the E004 README's "E001d attempt 0 = 63.1%" predate the 2026-10-03 retry of the 17 errored attempts. The current file gives a0 65.1 and a0+a1 79.3, as noted in the E004 README and the TSV.
  - **E001d cost.** The E008 README lists E001d at "~$10 per sample". Recorded tokens imply ~$5.6 per sample, since $10.5 was the original 2-attempt run.
  - **Tokens.** Code/lean2/high recomputes to 58.9k tokens/attempt vs 57.8k in the E004 README.
