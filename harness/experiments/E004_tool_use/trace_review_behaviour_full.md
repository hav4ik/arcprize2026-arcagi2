# E004 full-eval trace review: effort, mode, wasted tokens and harness problems

This review is read-only. It made no API calls and covers the `lean2` full runs (`tools_persistent_{grid,code}_lean2_deepseek-flash_{low,high,max}_full_a0.jsonl`, harness v3, 120k total output budget, 64k per turn, 24k reserved for the final turn, 40 turns at most).

- **Data:** statistics use **attempt 0 only**: 167 pairs per config, 5 configs, 835 attempts. There is no code-low run.
  - Attempt-1 records that were already complete (grid-low 100, grid-high 85) are used only to measure noise. They are biased toward short attempts, because the fast ones finish first.
- **Scoring:** "pass@1" is task-weighted, as in the README. "Pairs" are raw counts out of 167.
- **Method:** heuristics were computed over every trace:
  - verification cells = a `run_python` cell that references `train` and `output` and contains a comparison;
  - "pass" = the cell's output contains `True` and no `False`;
  - "printed" = the grid occurs in a tool output, in JSON or digit-row form.

  The pass/fail labels are noisy, so they are used only for broad splits.
- **Traces read in detail (14):**
  - 53fb4810 grid-max vs grid-low;
  - 4c3d4a41.1 grid-low vs grid-high;
  - 142ca369 grid-high;
  - 3a25b0d8.0 grid-high vs code-high;
  - 4e34c42c.0 code-high;
  - 0934a4d8 code-high;
  - 6ffbe589 grid-low;
  - the final turns of 88e364bc.1 (low), 269e22fb (high), 6e4f6532 (low) and 7ed72f31.1 (low);
  - the first turns of the 17 attempts whose first turn hit the 64k per-turn cap.

## 0. Summary table (attempt 0)

| | grid-low | grid-high | grid-max | code-high | code-max |
|---|---|---|---|---|---|
| pass@1 (task-weighted) | 66.7% | **67.1%** | 65.3% | 61.9% | 60.7% |
| correct pairs | 106 | 112 | 107 | 102 | 100 |
| answered (pred ≠ None) | 141 | 127 | 115 | 124 | 111 |
| precision of answers | 75% | 88% | **93%** | 82% | 90% |
| no answer | **26** | 40 | 52 | 43 | 56 |
| wrong answer | 35 | 15 | 8 | 22 | 11 |
| mean / median output | 49.9k / 42.9k | 54.8k / 48.2k | 65.0k / 60.8k | 58.9k / 50.9k | 65.2k / 56.9k |
| mean tokens, correct attempts | 35.7k | 39.6k | 45.1k | 40.3k | 43.1k |
| median turns | 12 | 12 | 8 | 11 | 8 |
| median tokens per turn | 3.0k | 3.7k | 5.9k | 3.9k | 6.4k |
| median largest turn | 12.8k | 14.9k | 24.6k | 16.9k | 29.3k |
| share of output in turns ≥32k | 7% | 10% | 27% | 8% | 35% |
| first call: median / mean tokens | 169 / 954 | 199 / 2.6k | 232 / 11.0k | 263 / 4.0k | 6.8k / 18.2k |
| first call after >20k tokens | 0 | 2 | 37 | 11 | 61 |
| turn 1 hit the 64k cap | 0 | 0 | 5 | 0 | 12 |
| tool calls per 10k tokens | 2.81 | 2.27 | 1.55 | 2.32 | 1.46 |
| verification cells per attempt (per 10k) | 6.8 (1.37) | 5.8 (1.07) | 4.7 (0.72) | 6.6 (1.13) | 4.6 (0.70) |
| attempts with ≥1 train check | 98% | 99% | 100% | 99% | 98% |
| "wait" per 10k tokens | 13.2 | 13.7 | 18.0 | 13.7 | 17.4 |
| median wall time | 206s | 226s | 316s | 249s | 311s |

**Noise level.** On the 100 (low) and 85 (high) pairs that already have both attempts, the same config disagrees with itself on 20% and 25% of pairs. Low and max disagree on 35 pairs (17 against 18), while independent Bernoulli draws would give about 30. Per-pair differences are therefore mostly sampling noise. The patterns below are in *how* each config fails, and they are consistent across configs.

## 1. Why low and high effort match or beat max with tools

**Max is the most precise, but it runs out of budget.**
- When max answers it is right 93% of the time, against 75% for low. But it leaves 52 pairs unanswered, twice as many as low (26).
- Under a fixed 120k budget these two effects cancel: 106 vs 107 correct pairs.

**Max does fewer, longer experiments.**
- It makes 1.55 tool calls per 10k tokens against 2.81 at low, and has half as many verification cells per token.
- Its median turn is 2× longer, and 27% of its output is in turns of 32k tokens or more (7% at low).
- It says "wait" 18 times per 10k tokens against 13. The tool loop is the efficient part, and max spends its budget reasoning between calls.

**The plain-prompt pathology comes back at max.**
- Although lean2 says to use code, 37 grid-max and 61 code-max attempts reason for more than 20k tokens before their first call. 17 first turns hit the 64k per-turn cap.
- In those turns the model announces code repeatedly and keeps going. code-max 13e47133 mentions code 38 times, the first at 2% of the turn, with 143 "wait"s.
- The worst case is grid-max 53fb4810. Turn 3 is a 64k-token turn with no tool call: 226k characters that drift into listing ARC task IDs from memory ("1bc43 (xor boxes)…"). The attempt ends at 101k tokens with no answer. grid-low solves the same pair in 15.7k tokens and 8 calls, verifying both training pairs in code.
- Long first turns are not themselves fatal. Max attempts with a first turn over 20k are right 78% of the time, because they are mostly easy tasks solved by hand. The cost is that they leave less budget for the hard ones.

**Same pairs, same result, fewer tokens.** On the 89 pairs that both low and max solve:
- median output is 26.5k (low), 29.9k (high) and 41.9k (max);
- median calls are 9 / 8 / 6;
- the median time to the first call is the same, about 200–270 tokens.

So low spends about 37% fewer tokens for the same answers.

**The two efforts fail differently** (§4): max fails by budget exhaustion, low by committing early to a rule that fits train. Low's main weakness is wrong answers (35), not unanswered pairs.

## 2. Why code mode is worse than grid mode

Same-effort comparison at high: code is 10 pairs worse (102 vs 112). It has 3 more unanswered pairs and 7 more wrong answers. Grid-high also loses 5 pairs to copying errors that code mode cannot make (§5), so the gap in the reasoning itself is about 15 pairs. Pooled over high and max, grid-only wins outnumber code-only wins 45 to 28 (sign test p ≈ 0.06), so the effect is borderline but has the same sign at both efforts.

**`submit_solution` never rejected anything.**
- There were 227 submissions, and every one was accepted on the first try: 0 training rejections and 3 non-train errors.
- In every submitted attempt the model had already run its function on `train` and `test_input` in `run_python`.
- The check only repeats what the model already does in grid mode, so it adds no accuracy.
- Writing the submission costs little: submission code is 0.9% of output characters, and tokens from the first submission to the end are 1.6%.

**Code mode has no way to give a best guess.**
- Of 49 unsubmitted code-high attempts, all 49 ran into the final turn. Only 1 of 49 (and 0 of 58 at max) was correct.
- Grid mode can stop and write a grid whenever it judges the answer good enough. Several grid-only wins depend on that:
  - **142ca369 grid-high:** the rule fails train ex0, which involves dot reflectors, but the test has no dots. The model answers anyway and is right. A code submission could never pass.
  - **0934a4d8 code-high:** the reflection rule passes all of train at 8.9k tokens. The lean2 instruction to check assumptions on `test_input` then shows that two hidden columns have no mirror partner. The model searches for 90k more tokens, never submits, and ends with nothing. grid-high answers the same pair at 46.7k and is right.
  - Similar cases where the heuristic shows a passing train check but no submission: 4e34c42c, 3dc255db, cbebaa4b.1 (code-high hit 40 turns), 221dfab4.1, 80a900e0, 6ffbe589.
- In 16 of the 25 grid-high-only wins and 16 of the 20 grid-max-only wins, code ended with no submission (stop `final` or `length`).

**Code mode also reasons longer before its first call.**
- Code-max's median first call comes at 6.8k tokens, against 232 for grid-max. 61 code-max attempts and 11 code-high attempts reason for more than 20k first (grid: 37 and 2).
- The likely cause is the code-mode sentence appended after "Your response:", which reads as a request for the full solution first.

**Accepted but wrong:** 17 at high and 9 at max. The rule fits train but the test is out of distribution. This is the same failure as wrong answers in grid mode, so the gate does not help.

## 3. Where the low-effort tokens go

Total grid-low output is 8.34M tokens.

| outcome | attempts | mean tokens | share of all output |
|---|---|---|---|
| correct | 106 | 35.7k | 45% (about 9% of that is spent after the correct grid was first printed) |
| wrong answer | 35 | 57.8k | 24% |
| **no answer** | **26** | **97.4k** | **30%** |

**What the no-answer attempts look like.**
- 22 run into the final turn and 4 are parse or copy failures (§5).
- They are genuine dead ends: a median of 19 turns and 17.5 calls of hypothesis cycling, not single long turns. Only 3 of them ever printed the correct grid.
- The final turn produces 8 answers and 0 correct ones at low, at a cost of 419k tokens (5% of output).

**Lowering the budget does not help.** Simulated cap: an attempt keeps its outcome if it finished under the cap, otherwise it counts as unanswered.

| cap | grid-low pass@1 | mean tokens |
|---|---|---|
| 60k | 56.7% | 39.7k |
| 80k | 63.7% | 45.3k |
| 96k (no final turn) | **66.7%** | **48.1k** |
| 120k (current) | 66.7% | 49.9k |

- The low-effort tail is still productive: attempts that end between 60k and 96k are right about 50% of the time.
- An early stop of the form "no passing train check by 60k" would save 1.5M tokens but lose 10 correct pairs. The same rule at 80k saves 0.7M and loses 3.
- Capping also hurts pass@2. grid-low plus grid-high scores 79.2% at 105k tokens per pair, 78.8% at a 96k cap and 74.6% at an 80k cap.

**Changes that save tokens without losing accuracy:**
1. **Remove or replace the forced final turn.** At low it yields 0 correct answers out of 30 and costs 5% of output; across all configs it yields 5 correct out of 231 at about 3M tokens. If kept:
   - **Append budget warnings to tool output** at about 70k and 90k: "N tokens left: print solve(test_input) and answer now, or give your best guess". This stays inside the current user turn and still allows a tool call.
   - **Cap the turn when `max_turns` triggers it.** That turn currently gets `remaining` tokens (up to 64k). grid-low 195c6913.0 spent exactly 64,000 tokens there, and 6ffbe589 spent 40.6k; neither produced an answer.
2. **Hand the answer over from the session instead of retyping it** (§5). This saves about 3k tokens per correct attempt spent re-checking and copying, and recovers 5 pairs at low.

The wrong-answer tokens (24%) are where accuracy is lost at low. Cutting them needs better rule selection, not a smaller budget (§4).

## 4. Per-pair differences between low and max

**Low solves, max misses (17 pairs).** In 17 of 17, max ended with **no answer**, all at 96–120k tokens.
- 8 hit the 120k cap mid-turn. The rest went into the final turn and failed.
- Low solved them with 9.5k–87k tokens.
- Examples:

  | pair | low | max |
  |---|---|---|
  | 53fb4810 | 15.7k | 101k, 2 calls |
  | b0039139.1 | 9.5k | 96k |
  | 9385bd28.1 | 48k | 120k, 5 calls |

- 8 of these 17 were also solved by no-tool E001d on both attempts. They are not hard tasks; max simply spends its budget before converging.

**Max solves, low misses (18 pairs).**
- **12 are wrong answers given early**, often in under 30k tokens:

  | pair | low tokens | low calls |
  |---|---|---|
  | 7b5033c1 | 890 | 3 |
  | 7ed72f31.1 | 9.4k | |
  | 4c3d4a41.1 | 17.5k | |
  | 247ef758.1 | 21k | |
  | 88e364bc.1 | 27.8k | |
  | d35bdbdc.2 | 28.8k | |

- Most of these passed a train check and then accepted the first rule that fit, even though the test called for a different variant.
  - 4c3d4a41.1 low: the rule passed both training pairs at turn 3. The model then wrote "Let me just trust code — it matched", and some cells were wrong.
  - grid-high spends 3 more long turns working out which part of the shape moves, and gets it right.
- **4 are harness or copy losses:**
  - 71e489b6.0: unparsed row-per-line format;
  - 7c66cb00.0: duplicated row;
  - 7ed72f31.1: extra row;
  - 88e364bc.1: 14 cells miscopied.
- **3 are budget exhaustion:** 195c6913.0, 6ffbe589 and a25697e4.0.

**Pattern.** Low under-thinks the choice of rule and max over-thinks the analysis. The two error sets barely overlap, which is why pairing two efforts works for pass@2:

| combination | pass@2 | tokens per pair |
|---|---|---|
| grid-low + grid-high | 79.2% | 105k |
| grid-high + grid-max | 79.0% | 120k |
| grid-low + grid-max | 76.1% | 115k |
| code-high + code-max | 70.7% | 124k |

A cheap prompt fix aimed at low's main error: once train passes, ask "name one materially different rule that also fits train; if it gives a different test output, pick the one that uses structure present in `test_input`". 13 of 15 wrong grid-high answers and 25 of 35 wrong grid-low answers had passed a train check.

## 5. Harness and parsing problems

1. **The final turn is broken. This is the main problem.**
   - "Finalize now" is sent as a new user message, with `tool_choice="none"` and the tools still listed.
   - In 144 of the 231 final turns (62%), the model writes a tool call as plain text (`<｜｜DSML｜｜ invoke name="run_python">…`), which cannot be parsed as a grid.
   - About 60% of final-turn reasoning restarts cold in a terse style: "We need solve ARC-like puzzle. Need infer transformation. We have training examples…". This fits DeepSeek dropping earlier `reasoning_content` once a new user turn begins (likely, not verified).
   - The model is left with only its code cells. Result: 5 correct out of 231 final turns.
2. **The final turn triggered by `max_turns` has no tight cap.** `cap = min(max_tokens, remaining)` applies when `index == max_turns-1`, giving up to 64k. Six grid-low attempts hit 40 turns; their final turns used 64,000, 40,566 and 34,744 tokens, among others, with 0 correct.
3. **Copying the final answer by hand loses correct answers in grid mode.** In each case below the tool had printed the correct grid, but the answer the model typed was wrong:

   | config | pair | what went wrong |
   |---|---|---|
   | grid-low | 7c66cb00.0 | duplicated row, 31×24, so invalid |
   | grid-low | 7ed72f31.1 | extra row, 29×28 |
   | grid-low | 88e364bc.1 | 14 cells wrong, copied from numpy `[6 6 6 …]` rows into JSON |
   | grid-low | 6e4f6532.0 | 2 cells wrong |
   | grid-high | 36a08778.1 | duplicated row, 31×30 |
   | grid-high | 31f7f899.0 | a row of 18 values |
   | grid-high | 269e22fb.0 | 7 cells wrong, copied from digit strings |

   - Max copies more carefully: 97% of its answers equal a printed grid, against 89% at low.
4. **Parser misses valid answers** (`run_deepseek_arcprize.parse`):
   - **Row-per-line lists with no outer brackets:** `[0, 0, …]\n[0, 3, …]` (58490d8a.0 and 8b7bacbf.1 at high, 71e489b6.0 at low). The backscan finds one row, a 1-D list, so the result is None.
   - **A trailing `],\n]`:** 195c6913.1 at low; its answer was wrong in any case.
   - **A JSON grid followed by an explanation:** 195c6913.1 at max, which was correct and not parsed.
   - A lenient fallback (take the last valid JSON grid, then the last block of bracketed rows) recovers 1 / 2 / 1 correct pairs at low / high / max and never changes a parsed answer.
   - Together with item 3, fixing the answer hand-off moves pass@1 for grid-low from **66.7% to 69.2%** (+5 pairs) and for grid-high from **67.1% to 70.0%** (+5 pairs). grid-max moves from 65.3% to 65.7%.
5. **Code mode has no exit for a best guess,** and the check never rejects (§2). Its fallback ("last text grid") only runs in the broken final turn.
6. **A truncated non-final turn is followed by no message.** On `finish == "length"` the harness `continue`s with two assistant messages in a row and nothing in between.
   - At max this happens 73 times (18 of them at the 64k per-turn cap). The continuation is usually coherent.
   - One cold restart was seen: grid-max 53fb4810 T4, "The user is asking me to solve an ARC-like puzzle. I've already done a lot of analysis in my head".
   - A short user nudge ("Turn limit: call run_python with your current hypothesis") would be clearer.
7. **Bookkeeping.**
   - Attempt-1 records are appended to the `*_full_a0.jsonl` files.
   - `evals/report.py` averages over all records, and early attempt-1 completions are short (grid-low attempt 1 averages 40k tokens against 50k for attempt 0). The table therefore looks cheaper and more accurate than the truth until attempt 1 finishes.
   - Report attempt 0 separately or filter by attempt.

## 6. Recommendations

1. **Fix the answer hand-off in grid mode.** This is free accuracy (+2.5–2.9 pp at low and high).
   - Add a `submit_answer(expr)` tool, or a rule "set `answer` in the session". The harness then reads the grid from the session (validated) and never parses retyped text.
   - Keep text parsing as a fallback with the lenient parser.
2. **Replace the forced final turn.**
   - Put budget warnings inside tool outputs (at about 70k and 90k), so reasoning is kept and a tool call is still allowed.
   - If a forced final turn stays: send no tools, disable thinking, give it 2–4k tokens, and treat a session `answer` or `solve(test_input)` as the fallback.
   - Apply the same cap when `max_turns` triggers the final turn.
   - This saves about 5–7% of output, and a cap at 96k showed no loss in accuracy at low.
3. **Use low or high effort in grid mode, and spend extra budget on a second attempt at a different effort, not a larger cap.**
   - grid-low plus grid-high gives 79.2% pass@2 at 105k tokens per pair.
   - Lowering the cap costs accuracy at every effort.
   - Drop code mode, or give it a best-guess exit, for example `submit_solution(code, best_guess=True)` that skips the training check when the budget is low.
   - Then target low's error, premature commitment to the first rule that fits train, with a single "alternative rule" check after train passes.
