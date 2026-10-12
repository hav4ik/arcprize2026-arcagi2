# E004 trace review: how Flash uses the Python tool, and what goes wrong

Read-only review of `evals/results/tools_persistent_*.jsonl` and their `.messages.json.gz` traces. No API calls were made. Runs were still in progress, so all numbers come from one snapshot of **109 completed records** (smoke file excluded). Pilot pairs are not task-weighted. The baseline is E001d attempt 0 on the same pairs.

## Snapshot

| file | prompt | harness | n | correct | mean out | median out | hit 200k | E001d a0 same pairs | E001d mean out | tool calls/att |
|---|---|---|---|---|---|---|---|---|---|---|
| code_max_pilot | plain | v1 | 20 | 12/20 | 97.2k | 86.3k | 4 | 10/20 | 58.1k | 6.7 |
| grid_max_pilot | plain | v1 | 20 | 13/20 | 103.9k | 90.8k | 5 | 10/20 | 58.1k | 3.6 |
| code_max_pilot2plain | plain | v2 | 11 | 9/11 | 47.6k | 31.1k | 0 | 8/11 | 38.1k | 4.9 |
| code_max_pilot2 | lean | v2 | 12 | 12/12 | 44.3k | 40.4k | 0 | 9/12 | 37.0k | 9.2 |
| grid_max_pilot2 | lean | v2 | 14 | 13/14 | 32.5k | 29.0k | 0 | 9/14 | 48.0k | 7.2 |
| code_high_pilot2 | lean | v2 | 15 | 11/15 | 37.9k | 26.1k | 0 | 9/15 | 51.9k | 9.5 |
| grid_high_pilot2 | lean | v2 | 17 | 14/17 | 36.7k | 28.1k | 0 | 10/17 | 50.0k | 10.2 |

- **Plain prompt:** with tools it costs about 1.7× the no-tool baseline (97–104k vs 58k per attempt). Accuracy rises only slightly.
- **Lean prompt:** cuts output to 30–45k per attempt and beats E001d attempt 0 on the same pairs in every configuration.
- **Effort:** lean at `high` uses about as many tokens as lean at `max` and is somewhat less accurate.

**Traces read in detail (17):**
- 7b5033c1 in five configurations;
- 31f7f899 grid/high/lean;
- 6e4f6532 grid/max/plain;
- 3dc255db code/max/plain and grid/high/lean;
- 2b83f449 code/max/plain;
- dfadab01 code/max/plain, grid/max/plain and grid/high/lean;
- aa4ec2a5 code/high/lean;
- 13e47133 pairs 0 and 1, code/max/plain v1;
- e12f9a14 grid/max/plain;
- a251c730 code/max/lean.

Aggregate statistics below cover all 109 records.

## 1. Workflow

**First tool call**
- *Plain prompt:* the model solves the task by hand first.
  - The first call comes after a median of 47.6k (code) and 86.2k (grid) output tokens.
  - 35 of 40 v1 attempts spend more than 20k tokens before calling the tool.
  - Many first-turn traces announce "let me use code" several times and then keep reasoning. In e12f9a14 grid/max/plain the first such mention comes at 1% of the turn, and 8 mentions follow. The turn then runs 191,808 tokens with no tool call.
- *Lean prompt:* the median first-call position drops to 1.7k–15k tokens. Some traces open with a 100–300-token turn whose only content is "let me inspect with code".

**What it computes**
- Typical cell sequence:
  1. shapes, colour counts and connected components of every pair;
  2. a hypothesis function plus a loop over `train` printing `True` or `False` and the number of differing cells;
  3. debugging cells that print a diff grid;
  4. `solve(test_input)`, printed;
  5. the submission or the final answer.
- Lean traces use a median of 6–8 calls per attempt, plain traces 3–6. Every turn makes exactly one call: there are no parallel calls in the data.
- Calls are well formed. 34 cells raised errors: ValueError 13, IndexError 7, TypeError 7, AssertionError 6, SyntaxError 3. The model fixes each one in the next cell, usually in under 600 tokens ("Bug: col is already the color int. Fix.", dfadab01 code/max, turn 2).

**Grids built by hand**
- *Plain:* grid rows written out in the reasoning are still common, at 8–10 rows per 10k tokens. The worst case is 230 rows in dd6b8c4b grid/max.
- *Lean:* 3–6 rows per 10k tokens. Hand work moves into the middle turns instead: the model reasons over coordinates printed by the tool. For example, 3dc255db grid/high/lean has a 53k-token turn 7 with 104 "wait"s and about 1,500 coordinate references, which ends with "Final grid printed. Let me code it to be safe."

**Retyping the task data into code is common and harmful**
- 117 code literals reproduce training grids and 22 reproduce test inputs, even though `train` and `test_input` are preloaded.
- Some cells rebind the preloaded variables, for example `train = []` (7b5033c1 code/max v1, turn 1) and `train = None` (a251c730 code/max/lean, turn 1).
- Retyping caused at least four failures or large token sinks:
  - **6e4f6532.0 grid/max/plain, turn 3:** the retyped `test` duplicated row 18, giving 31×28 instead of 30×28. The model noticed ("The test grid is 31×28, not 30×28 as I assumed") but trusted its own copy. The final grid has 31 rows, fails validation, and `pred` is None.
  - **13e47133.1 code/max v1, turn 3:** `test_input` was rebound to a 31-row retyped copy. Under v1 the submission then ran on the bad copy and was rejected three times ("Return a rectangular 1-30 by 1-30 grid"). The model concluded "the REAL test input is (31, 30)" and spent more than 100k tokens before running out of budget.
  - **a251c730.0 code/max/lean, turn 5:** "train is lost (I set it to None). I need to reconstruct training examples manually" cost a 22k-token turn.
  - **aa4ec2a5 code/high/lean, turn 1:** a retyped grid had ragged rows, which raised an "inhomogeneous shape" error. The model then switched to the preloaded `train`.

**Verification on train**
- Nearly every v2 attempt checks every training pair in `run_python` before answering.
- In code mode, all 43 v2 submissions were accepted on their first try. The only rejections were 4 in v1 (13e47133) and 2 in the smoke run. `submit_solution` therefore works as a final gate, not a feedback loop.
- One real rejection gave a training-pair diff (13e47133.0 v1). The model followed it with a 63k-token turn that made no tool call and ran out of budget.
- In grid mode the final text grid is almost always copied from a printed tool output, and no copying errors were found. A few plain grid/max answers were built by hand without printing; one of them, 6e4f6532, is wrong.

## 2. Where the output tokens go

| config | before 1st call | between calls | after last call |
|---|---|---|---|
| code/max/plain v1 | 61% | 25% | 14% (all from runs that hit the budget) |
| grid/max/plain v1 | 81% | 12% | 6% |
| code/max/plain v2 | 78% | 22% | 0% |
| code/max/lean | 55% | 45% | 0% |
| grid/max/lean | 36% | 60% | 5% |
| code/high/lean | 21% | 79% | 0% |
| grid/high/lean | 11% | 84% | 5% |

Of output characters, 80–96% are reasoning and 3–20% are tool-call code. Grid literals inside code make up 0–4%.

Waste that the harness or prompt could remove:
1. **Hand analysis before the first call** under the plain prompt: 50–190k tokens.
2. **Budget exhaustion with no answer.** 9 of 40 v1 attempts hit 200k. The "Finalize now" turn has 8,192 tokens with thinking still on. In all 9 the model kept reasoning, reached the length limit and emitted no grid.
   - In both 13e47133 code runs, the correct test grid had already been printed by the tool (pair 1: message 14; pair 0: messages 8 and 10), yet `pred` is None.
   - That is 1.8M tokens, about 20% of all v1 output, spent for a score of zero.
3. **Retyping grids,** and the long repair turns it triggers: roughly 20k–100k tokens in the cases listed above.
4. **Long middle turns** (10–55k tokens) in which the model hand-simulates the test output or re-derives coordinates the tool could print.
5. **Re-verification after the answer is fixed.** For example, 7b5033c1 code/high, turn 2, recounts by hand cells the code had already counted.

## 3. Failure modes on wrong attempts (snapshot: 25 wrong)

| mode | count | examples |
|---|---|---|
| Budget exhausted, no grid (harness finalisation) | 9 | e12f9a14 ×4, 13e47133 ×2 (correct grid already printed), a251c730, 6e4f6532.1, 2b83f449 (all v1) |
| Simpler rule that fits train but breaks on a test-only feature (premature acceptance) | 6 | 7b5033c1 ×5: grouping by colour instead of walking the snake path, because the test path revisits colours (18 cells wrong); 31f7f899 grid/high/lean: assumed symmetric bars, and the test has an asymmetric one (1 cell) |
| Ambiguity noticed, wrong branch chosen | 7 | dfadab01 ×4: whether out-of-bounds or overlapping stamps are dropped; 3dc255db ×3 (code/high/lean, code/max/plain v2, and code/max/plain v1, which also hit the red flag below): bar length as extent or cell count, with clipping |
| Red flag rationalised away | (1, same attempt as above) | 3dc255db code/max/plain v1: the test output came out 12×14 while every train output matched its input size ("go with the natural rule output") |
| Partial sub-rule, a few cells off | 2 | 2b83f449 code/max (2 cells), code/high/lean (6 cells): placement of the border 3s |
| Transcription error in retyped input | 1 | 6e4f6532.0 grid/max/plain: duplicated row, invalid grid |

**Hasty answers under lean:**
- grid/max/lean 7b5033c1 used 857 tokens and 2 cells and accepted the first rule that fit. The same configuration with code/max/lean, turn 1, considered "reading order vs chain order: in training both give same", checked whether all cells form one path and got the answer right.
- So brevity alone is not the problem: the model needs to look for a second rule that also fits train. Wrong lean answers are cheap, from 0.9k to 85k tokens.

No parsing failures were found. `parse` picked up the grid even when explanatory text followed it (31f7f899 grid/high/lean, 3e6067c3.1 grid/max/lean).

## 4. Harness and prompt issues

1. **Finalize turn cannot finish** (`solve()`, final-turn branch).
   - The turn keeps `thinking` enabled with `max_tokens` 8192. In 9 of 9 cases the model spent all of it on reasoning and wrote no content (e.g. `grid_..._max_pilot` e12f9a14.0, turn 2).
   - There is also no fallback to grids the tool already produced.
2. **Unbounded single turn.**
   - The per-turn cap is `remaining - final_tokens` (191,808 on turn 1), so a single reasoning turn can consume the whole budget. e12f9a14 grid ×2 never called the tool.
   - The non-final `finish == "length"` path (9 events) also appends nothing before the next request. In every observed case the next turn was the final one. When a tool call was truncated (13e47133.1, turn 13), the harness answered "Tool skipped".
3. **v1 rebinding bug, confirmed twice:**
   - 7b5033c1 v1 (`train = []`, then an unchecked acceptance, which happened to pass the real train pairs);
   - 13e47133.1 v1 (rebound `test_input`, rejected three times, budget exhausted).

   v2 fixes the submission check. Rebinding in the session still misleads exploration, and a lost `train` cannot be restored (a251c730 lean, `train = None`).
4. **Ambiguous rejection message** in `submit_solution`. When every training pair passes but the test output is invalid, the model sees only a `ValueError` traceback. Nothing says that the training pairs passed or that the failure is on the test input (13e47133.1 v1, turns 9, 11 and 12).
5. **Failure feedback is too large.** Rejection feedback is the full predicted and expected grid of every failing pair, truncated at 16k characters. Four 30×30 failures overflow it and the model gets truncated JSON. A cell-level diff would be smaller and clearer. This is rare in the data: only one diff-style rejection was seen.
6. **Preloaded data is mentioned only in the tool description.** The user prompt never says the grids are already loaded as `train` and `test_input`, and the lean suffix ("never construct, copy... grids") did not stop retyping: 7 of 15 code/high/lean attempts contain retyped grids.
7. **Prompt conflict in code mode.** The ARC Prize text ends "Respond in the format of the training output examples ... Your response:", and the code-mode instruction is appended after "Your response:". In one trace the model debates which answer the grader reads (13e47133.1 v1, turn 6: "the grader uses the code. I'll present the output...").
8. **`submit_solution` description rewards stopping early.** "If all pass... ends this attempt" gives no incentive to test alternative rules or to check that the test input stays within the rule's assumptions.

## 5. Recommendations

**Harness:**
1. **Fix finalisation.**
   - In the final turn, disable thinking (or set effort low) and ask only for the grid.
   - Fall back first to `transform` or `solve` in the session applied to the host `test_input`, then to the last test-shaped grid printed by a tool. That would have recovered 13e47133 ×2.
2. **Cap each turn**, at about 24–32k tokens for the first turn and 16k for later ones. On a non-final `length` finish, append a user nudge: "Turn limit: call run_python now with your current hypothesis." With the lean prompt this should eliminate the runs that hit 200k. Then lower `max_total_output` to about 120k.
3. **Protect the task data.**
   - Re-inject `train` and `test_input` before every cell, or expose `get_task()` and `restore()`.
   - Have the tool output warn when a cell assigns `train =` or `test_input =` or contains a grid literal identical to a task grid.
4. **Sanity-check submissions before accepting them.** Return warnings rather than rejections, with one chance to revise:
   - the output shape relation differs from the one shared by all train pairs (it would have flagged 3dc255db code/max);
   - the output uses colours never seen in train outputs or the test input;
   - the transform raises an error on the test input. Say "all train pairs passed; the test input fails: ..." instead of the bare traceback.
5. **Make rejection feedback compact:** per pair, the shape match, the number of wrong cells, the first 20 `(r, c, pred, expected)` triples and a short diff mask.
6. **Use a second candidate for pass@2.**
   - Let `submit_solution` accept an optional `alternative` transform, or ask once after acceptance: "Is there a materially different rule that also fits every training pair? If so submit it."
   - Score the alternative as attempt 2. 9 of the 10 wrong v2 answers involved an alternative rule the model had named or that fits train (7b5033c1 ×4, dfadab01 ×2, 3dc255db ×2, 31f7f899).
   - Separately, for selection over independent attempts, vote on the test output among accepted programs.

**Prompt (lean v2):**
- Move the data note into the user prompt: "The grids above are already loaded as `train` and `test_input`. Never retype them or assign to these names."
- Add a test-input check: "Before answering, list the assumptions your rule makes (symmetry, one object per colour, output size, ordering) and assert them on `test_input` in code. If the test input breaks one, find a rule that also covers it."
- Add a tie-break rule: "If two rules fit all train pairs, prefer the one that uses structure present in the test input."
- Replace "Keep your reasoning brief" with an explicit protocol: "first call within ~2k tokens; one hypothesis per cell; do not simulate outputs by hand; print diffs."
- In code mode, drop "Respond in the format of the training output examples" or follow it with "Your answer is the submitted code; do not print the grid."

**Budget:** run E004b with the lean prompt at max effort. A 120k total output budget with a 32k first-turn cap should hold the mean near 30–40k per attempt. That is about 35% below E001d, and the snapshot shows higher accuracy on the pilot pairs.
