# E029: where V4.1-Flash-1017 saves tokens, and what is left to cut ($0, offline)

This is a read-only analysis of the saved transcripts. No API calls were made.
- **Runs compared:** old = V4.1-Flash, E009 sub80s40 (501 attempts, 3 samples); new = 1017, E029 sub80s40 (334 attempts, 1 sample). Same harness, `persistent-v4`.
- **Cross-check:** s20x60, old E010 (333 attempts) against new E029 (334 attempts).
- **Scripts:** `scripts/trace_features.py` builds `trace_features.jsonl`, one row per attempt with per-turn lists. It reuses E026's `text_feats`, `SOLVE_DEF` and grid-line definitions. `scripts/trace_stats.py <cfg>` writes `stats_sub80s40.txt` and `stats_s20x60.txt`, which hold every number below.
- **Table cells:** mean / median per attempt.
- **"Paired":** the 127 pairs (114 on s20x60) where both models have at least one correct attempt. Each model's correct attempts are averaged per pair. CIs are a 2,000× task bootstrap.

**Definitions**
- **Phases**, assigned per turn:
  1. first turn, i.e. up to the first tool call (every attempt calls a tool in turn 1);
  2. exploration: turns before the first `def solve/transform…` cell;
  3. hypothesis/debug: from that turn until the turn that first produces the submitted grid;
  4. verification tail: everything after that, including the submit turn.
- **"Submitted grid exists":** the first tool output that contains the exact submitted grid. This covers 76% of correct new attempts. Otherwise it is the last cell that defines or assigns the name used in the submit expression (24%).
- **Wasted turns:** an overlay rather than a phase: turns whose code raised an error, and turns truncated by length.

## 1. Where the 22% went (sub80s40)

The mean drop is −8.9k tokens (40.7k → 31.7k). It splits as:
- −3.5k within correct attempts;
- −3.5k within failed attempts;
- −2.0k from the mix, because more attempts are correct and correct attempts are cheaper.

| Phase | old all | new all | Δ (share of Δ) | Paired both-correct old → new | Paired Δ [95% CI] (share) |
|---|---|---|---|---|---|
| Total | 40.7k / 37.6k | 31.7k / 27.5k | −8.9k | 32.5k → 25.1k | **−7.5k [−9.5, −5.4]** |
| 1. First turn | 3.1k / 0.2k | 1.3k / 0.2k | −1.8k (20%) | 3.3k → 1.4k | −1.9k [−2.8, −1.2] (26%) |
| 2. Exploration | 22.5k / 13.6k | 16.6k / 11.5k | −6.0k (67%) | 14.5k → 13.0k | −1.6k [−3.4, +0.2] (21%) |
| 3. Hypothesis / debug | 14.1k / 8.9k | 12.6k / 8.3k | −1.4k (16%) | 13.7k → 9.7k | **−4.0k [−5.9, −2.2] (53%)** |
| 4. Verification tail | 1.0k / 0.4k | 1.2k / 0.4k | +0.2k (−2%) | 1.0k → 1.1k | +0.1k [−0.2, +0.4] |
| Reasoning tokens | 37.2k | 28.1k | | 29.3k → 21.8k | −7.6k (102%) |
| Visible tokens (content + code) | 3.5k | 3.7k | | 3.2k → 3.3k | +0.1k |
| Overlay: turns with code errors | 1.7k | 1.2k | | 1.4k → 1.0k | −0.4k (n.s.) |
| Overlay: truncated turns | 1.1k | 0.1k | | 0 → 0 | |

- **The new model thinks less per turn; it does not take fewer turns.**
  - **Per turn:** turns went 11.5 → 11.9 paired (n.s.), and tool calls 12.7 → 13.6. Tokens per turn fell from 3.1k to 2.3k, and the largest turn from 12.5k to 9.1k.
  - **Long turns:** turns over 16k tokens fell from 0.30 to 0.15 per attempt. They held 23.5% of all tokens before and 13.2% now.
  - **Thinking distribution:** p90 thinking per turn went 8.1k → 5.6k and p99 25.8k → 17.8k.
- **First turn:** the mean fell 58% (3.1k → 1.3k) and the median stayed about 0.2k, so the cut is in the tail: p90 first-turn thinking went 9.9k → 4.1k.
- **Where in the attempt:**
  - Over all attempts, most of the drop is in exploration (67%). Much of that comes from failed attempts (old failed attempts spent 39.6k on exploration, new ones 25.4k), together with the mix shift.
  - On the same solved pairs, most of it is in the hypothesis/debug loop (53%) and the first turn (26%).
- **Verification tail:** it was already small in the old model and did not shrink.
- **Characters per reasoning token:** 3.02 → 3.18 (+5%). Word counts fall slightly less than token counts.
- **s20x60 cross-check:** total −6.2k. By phase:
  - first turn −2.6k (42%);
  - exploration −2.1k (34%);
  - hypothesis −1.9k (30%);
  - tail +0.4k.

  Paired over 114 pairs: −6.4k [−7.8, −4.9].

## 2. Strategy markers per 1k reasoning words (sub80s40)

| Marker | old all | new all | new correct | new failed | Paired Δ [95% CI] |
|---|---|---|---|---|---|
| Reasoning words per attempt | 19.0k | 15.1k | 12.1k | 23.4k | −3.3k [−4.3, −2.3] |
| Hand-transcribed grid rows | 3.41 / 1.63 | 2.13 / 0.65 | 2.11 | 2.16 | −1.4 [−2.1, −0.9] |
| … copied from earlier tool output | 0.96 | 0.62 | 0.58 | 0.74 | −0.3 |
| … copied from the prompt | 0.88 | 0.46 | 0.52 | 0.33 | −0.4 |
| (r,c) tuples in prose | 34.8 | 30.2 | 29.1 | 33.1 | −2.4 [−5.3, +0.8] n.s. |
| Est. tokens in cell-coordinate lines (E026 definition) | 9.5k | 7.2k | 5.2k | 12.6k | −2.0k [−2.8, −1.2] |
| "check this in code/python" (explicit tool intent) | 1.01 | 0.75 | 0.85 | 0.49 | −0.30 |
| "let's compute/count/check/trace" with no code mention | 2.03 | **2.54** | 2.65 | 2.24 | **+0.57 [+0.43, +0.73]** |
| "hmm" | 8.02 | **4.59** | 4.54 | 4.71 | −3.2 |
| "actually" | 4.77 | 3.57 | 3.58 | 3.55 | −1.2 |
| "wait" | 3.21 | 3.21 | 3.38 | 2.74 | 0.0 |
| "maybe" | 8.06 | 7.92 | 7.34 | 9.50 | 0.0 |
| Re-verification phrases | 1.34 | 1.26 | 1.36 | 0.99 | −0.1 n.s. |
| Give-up phrases ("given the time", "best guess") | 0.11 | 0.06 | 0.03 | 0.14 | −0.03 |

- **Prose after "let me check in code":**
  - Old: 2.6k tokens per statement-turn, 10.4k per attempt. New: 1.3k and 4.9k (paired −4.5k).
  - Read 8 traces: the statement usually comes at the end of a long prose turn (median 85% of the way through), not at the start. The turn decides to code only after it has reasoned at length.
  - A looser regex that also matched bare "let's compute" mostly caught mental arithmetic, so it was dropped.
- **Re-reading grids:**
  - Copying rows that code already printed is rare: below 1 row per 1k words in both models.
  - Cell-coordinate reasoning is the larger cost: about 7.2k tokens per new attempt, 23% of reasoning characters (old 9.5k).
- **Overall:** the new model hedges less ("hmm" −43%, "actually" −25%) and transcribes fewer rows. But it states more mental "let's compute" steps, and its cell-coordinate share is unchanged at about 29% of characters in big turns (old 28%).
- **s20x60:** the same shifts ("hmm" 8.2 → 4.7, "actually" 5.0 → 3.8, grid rows 3.5 → 2.3, mental-compute 2.0 → 2.6).

## 3. Code use (sub80s40)

| | old all | new all | Paired Δ |
|---|---|---|---|
| Tokens to first code run | 3.1k / 0.2k | 1.3k / 0.2k | −1.9k |
| Tokens to first code touching `test_input` | 25.1k / 20.3k | 17.8k / 13.4k | **−7.6k** (the turn number is unchanged: 6.8) |
| Tokens to first `def solve` | 25.7k / 19.6k | 21.1k / 16.9k | −4.7k |
| Tokens until the submitted grid exists | 37.3k | 30.0k | −7.5k |
| Tool calls | 12.7 | 13.6 | +0.4 n.s. |
| Tool-output characters (without budget notes) | 9.2k | 10.0k | +0.6k |
| Code characters | 8.8k | 9.3k | +0.4k |
| `print()` calls / debug prints (mismatch/diff/where) | 29.6 / 1.7 | 32.8 / 1.9 | +2.4 / +0.2 |
| Code cell comparing solve() to the train outputs before submitting | 96% | 96% | |
| Last such check printed all-pass: correct / failed attempts | 83% / 31% | 82% / 44% | |

The new model reaches the same milestones in the same number of turns but with about 30% fewer tokens. It runs slightly more code and prints more. Checking every train pair in code before submitting was already universal and still is.

## 4. Waste remaining in new traces (sub80s40, 10.6M tokens)

| Bucket | Share of all new tokens |
|---|---|
| Correct attempts: first turn | 3.3% |
| Correct attempts: exploration | **30.6%** |
| Correct attempts: hypothesis / debug | **23.1%** |
| Correct attempts: tail after the submitted grid exists | 2.6% |
| Failed attempts: up to the last progress | **37.4%** |
| Failed attempts: after the last progress | 3.0% (old: 8.5%) |
| Overlap: turns with >8k thinking (5.9% of turns) | **35.9%** (old: 10.2% of turns, 49.5% of tokens) |
| Overlap: cell-coordinate and grid-row lines | about 23% of reasoning characters (29% inside the big turns) |

"Last progress" means the latest of: the first `def solve`, the first appearance of the eventually submitted grid, or a new best True count in a train check.

- **Tail after the answer exists, correct attempts:**
  - Per attempt: mean 1.15k, median 0.49k, p75 1.30k, p95 4.61k.
  - Only 5% of attempts exceed 5k and 1% exceed 10k. Net of a 150-token submit turn, the tail is **2.4%** of all tokens; s20x60 gives 2.0%.
  - So the "verification tail" is not where the tokens are.
- **Failed attempts after the last progress:** mean 3.5k, median 0.33k, p95 20.4k. Budget notes and the new model's behaviour cut this from 8.5% to 3.0% of tokens.
- **True solution in failed attempts:** it never appeared in any tool output of a failed attempt (0/90), so no failure "had the answer and moved past it".
- **Where the remaining tokens are:**
  - long reasoning turns, which are prose hypothesis search with cell coordinates;
  - the 27% of attempts that fail, which hold 40% of tokens (mean 47.5k).

## 5. Offline what-ifs (new sub80s40)

Savings are upper bounds: a cut turn's work may move into later turns.

"At risk" means correct attempts with a cut that falls before their answer existed. A cut changes those attempts; it does not necessarily make them wrong.

| Rule | Tokens saved | Correct attempts at risk (of 244) | Notes |
|---|---|---|---|
| Oracle: stop when the eventually submitted grid first exists | 3.6% | 0 | Upper bound for any early-stop rule |
| Stop at the first test grid after an all-pass train check | 3.4% | 22–36 change their answer | Train-check detection is heuristic, so the count is noisy. Revisions made after the train pairs pass do matter |
| Per-turn thinking cap 8k | 14.4% | 106 | |
| Per-turn thinking cap 12k | 7.3% | 63 | |
| Per-turn thinking cap 16k | 4.0% | 35 | s20x60: 3.3%, 15/215 |
| Per-turn thinking cap 24k | 1.3% | 10 | |
| First-turn thinking cap 2k | 2.5% | 37 | |
| First-turn thinking cap 4k | 1.8% | 25 | Turn 1 >4k: 74% accuracy vs 73% otherwise, so not harmful |
| First-turn thinking cap 8k | 0.8% | 13 | |
| Hard total cap 60k | 3.4% | 10 lost | |
| Hard total cap 50k | 8.8% | 26 lost | |
| Hard total cap 40k | 17.2% | 45 lost | |
| Abandon if no all-pass train check by 40k | 16.4% | 37 lost | |
| (pass@2) Skip attempt 1 when attempt 0's last check all-passed | 31.6% | pass@2 −3.6 pp (s20x60: −6.7 pp) | An adaptive-k lever, not trace shortening |

- **Long turns:** attempts with a turn above 16k thinking tokens are less accurate: 64% vs 75% (s20x60: 41% vs 67%). Some of these are hard pairs, not runaway prose.
- **s20x60 as evidence:** it is effectively a lower cap (60k, 20k soft target). It loses 8.5 pp pass@1 for −16% tokens, consistent with the hard-cap rows above.

## 6. Levers, ranked by estimated savings and risk

1. **Code at the start of a long turn, not the end. Estimated 4–7% saving at a moderate risk that needs testing.**
   - **Where the tokens are:** 36% of tokens sit in the 6% of turns with >8k thinking. Their explicit "check it in code" decision comes at a median 85% of the way through.
   - **Harness change:** a per-turn thinking cap of about 12–16k, followed by a nudge to "run code now to test the current hypothesis". This is the DeepSeek analogue of Qwen's `--turn-think-cap`. It needs a small change: today a `length` finish without a tool call just continues with no note.
   - **Offline estimate:** 4.0% (16k) to 7.3% (12k) saved. It touches 35–63 of 244 correct attempts before their answer exists.
   - **Next step:** A/B on one sample (~$3.6).
2. **Do cell-coordinate work in code. Up to about 10% saving if half of it moved; speculative, low-moderate risk.**
   - Cell-coordinate and grid-row lines are still about 7.2k tokens per attempt (23% of reasoning). Mental "let's compute/trace" statements rose 25%.
   - Prompt line: "derive coordinates, counts and paths with run_python; do not trace cells in prose".
   - E026 found that the task summary did not move this cost, so a prompt instruction or a cap is needed.
3. **First-turn cap of about 4k. 1.8% saving, low value.**
   - The new model already cut the first-turn tail by 58%.
   - Long first turns are not associated with failure, so a cap gains little.
4. **Failed attempts (40% of tokens). Only accuracy-costly levers are left.**
   - The tokens after the last progress are just 3%. Hard caps or abandon rules cost about 1 correct attempt per 0.35% saved; at 60k the cap saves 3.4% and loses 10 correct attempts.
   - Adaptive k (skipping attempt 1 after an all-pass attempt 0) saves 32% but costs 3.6 pp pass@2. Keep k = 2.
5. **Do not add early-stop or verification-tail rules.**
   - Even an oracle saves only 3.6%.
   - Implementable rules flip 22–36 correct attempts, because revisions made after the train check passes are productive.

## Caveats
- **Samples:** the new model has 1 sample per config, the old one has 3 (sub80s40) or 2 (s20x60). Unpaired comparisons mix difficulty with sampling, so the paired both-correct rows are the difficulty-controlled numbers.
- **Heuristic markers:** phase boundaries, train-check detection and "last progress" come from regexes over code and tool output (`persistent-v4` has no `--train-check` labels).
  - Answer location uses grid extraction from tool output. Grids that were printed truncated or with `.` substitution fall back to the def/assign turn, which makes tails slightly longer.
  - Stop-rule accuracy counts depend on which printed grid is taken as "the" test output (shown as a range).
- **What-ifs are static:** they replay fixed traces. A capped model may redistribute its thinking, so the savings are upper bounds and "at risk" is not "lost".
- **Token-to-word ratio:** characters per reasoning token rose 5% (3.02 → 3.18), so per-1k-word rates and token counts are not interchangeable across models.
- **Model attribution:** all old-model transcripts are V4.1-Flash and all new ones are 1017, by date (see README).
