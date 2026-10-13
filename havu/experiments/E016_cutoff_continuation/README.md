# E016: Rescuing cut-off attempts with a "time is up" no-thinking submit

## Question
In E010 s20x60 (Flash, low effort, `lean2sub`, 20k soft / 60k hard), 24 of 333 full-set attempts hit the 60k cap with no answer (`answer_source` None, `stop_reason` output_budget; the session `answer` fallback found nothing). If the harness sends one more short turn saying "time is up, submit now" with thinking disabled, how many attempts produce an answer, and how many are correct?

## Method
Script: `continue_cutoffs.py`. Tests: `test_continue_cutoffs.py` (offline). Summary: `analyze.py`, which writes `results/summary.md`. Raw rows: `results/continuations.jsonl`.

- **Messages.** Each trace (`trace_path`) is the exact list sent to DeepSeek: the prompt, assistant turns with `reasoning_content` and `tool_calls`, and tool results with the budget notes already appended. The tests check that the prompt equals `build_prompt + lean2sub` and that the turn and tool-call counts match the record. In 23 of the 24 traces the last message is a truncated assistant turn that holds only reasoning, with no content and no tool calls.
- **Replay.** Each attempt gets a fresh `SandboxSession` with the task data, set up as the harness does. Every executed tool cell is re-run in order; skipped calls from truncated turns are left out.
  - 232 cells were replayed with **0 divergences** from the original outputs (budget notes stripped before comparing).
  - There were no state resets. One cell raised an error, and it had raised the same error originally.
- **Continuation arms.** Each arm sends one request: `tool_choice` auto, both tools available, and a 2,048-token cap (8,192 for the thinking arm).
  - **toolnote** (main arm): drop the truncated reasoning-only turn and append `[Time is up: ... Call submit_answer now with your best answer ...]` to the last tool result, the same way the budget notes are attached.
  - **usermsg**: keep the trace exactly as sent and append a user message with the same text. DeepSeek accepts both forms.
  - **toolnote_think** (comparison arm, thinking on at low effort): the same messages as toolnote.
- **Extra turn.** If the reply runs code instead of submitting, or its submit is rejected, the note `[Time is up. Call submit_answer now ...]` is appended and one more turn is sent in the same thinking mode.
- **Scoring.** The harness's own `submit_answer` and the official exact-match check are used. Fallback order: submit, then the session `answer`, then `parse`/`lenient_parse` of the final text.
- **Zero-cost baseline (probe).** After replay, read the session's `answer` and run every user-defined function that takes one argument on the train and test inputs (2 s alarm per call). The best function is chosen by training-pair accuracy, and the comparison happens on the host.
- **Timing.** The requests were sent on Monday at 04:47–04:50 UTC, off-peak, so the price multiplier is 1.

## Results (24 attempts)
| Arm | Answered | via submit / session | Correct | Input tokens (cache hit) | Output | Cost |
|---|---|---|---|---|---|---|
| toolnote (thinking off) | 22 | 15 / 7 | **0** | 749k (515k) | 13.3k | $0.045 |
| usermsg (thinking off) | 20 | 14 / 6 | **0** | 770k (568k) | 14.0k | $0.041 |
| toolnote_think (thinking on, 8k) | 17 | 15 / 2 | 1 (28a6681f.0#1) | 2,482k (2,468k) | 104k | $0.072 |
| probe baseline (no API) | 0 had `answer` | — | 0 | — | — | $0 |

Total recorded spend was **$0.157**, including a 1-record API test, against the $3 cap. The per-attempt table is in `results/summary.md`.

### Effect on the s20x60 full-set scores
The rescued answers are substituted into the 2-sample s20x60 results:

| | pass@1 mean (task-weighted / pair-weighted) | pass@2 (task-weighted / pair-weighted) |
|---|---|---|
| s20x60 as recorded | 62.43% / 61.38% | 76.53% / 75.45% |
| + no-thinking rescue (either arm) | unchanged | unchanged |
| + thinking rescue | 62.85% / 61.68% | unchanged |

The thinking arm's one rescue does not change pass@2, because the other attempt on that pair was already correct. Of the 24 cut-off pairs, 9 were solved by the other attempt and 8 had the other attempt cut off as well.

## Findings
1. **With thinking disabled, DeepSeek drops all earlier `reasoning_content` from the prompt.**
   - The prompt tokens show it. For 142ca369.1#0, the original last turn had a 45.7k prompt; the thinking-off continuation had 12.5k, while the thinking-on continuation had about 96k, including the cut-off turn's ~26k of reasoning.
   - So a thinking-off "time is up" turn sees only code and tool outputs. It cannot draw on the half-formed hypothesis that was in the reasoning.
2. **There is almost nothing in the session to submit.** Cut-off attempts are attempts where the model never found a rule.
   - No session had `answer` set.
   - Only 1 of 24 (0934a4d8.0#1) had a function that fits every training pair, and its output on the test input was not a valid grid.
   - No function in any session gives the correct test output.
3. **Results by arm.** The no-thinking continuation is cheap (about $0.002 per attempt) and returns *an* answer 83–92% of the time. It is always wrong: typically it writes `answer = <quick heuristic>` and submits it. Thinking on with 8k tokens rescued 1 of 24, by luck: the submitted rule fit 2 of 3 training pairs.

## Recommendation
Do not add a "time-up + no-thinking submit" step to the harness for accuracy. At this cap it rescues 0 of 24, and the attempts it targets are unsolved rather than unsubmitted. The existing session-`answer` fallback already catches a set but unsubmitted `answer`. It fired 2 times in s20x60, and both of those answers were wrong as well.

The step is harmless and cheap if an answer is always wanted, for example for distillation records or for Kaggle, where an empty attempt scores the same as a wrong one. It is worth adding only for those reasons.

To recover long-tail pairs, the lever is budget, not hand-off. E010's s25x80 and E009's sub80s40 regain those pairs, and so would a larger soft target.
