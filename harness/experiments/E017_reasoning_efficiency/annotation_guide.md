# E017 phase 1B: annotation guide for trace readers

You annotate DeepSeek V4.1-Flash attempts from our best ARC-AGI-2 tool harness (grid mode, low effort, persistent Python
session with `train` and `test_input` preloaded, `submit_answer(expr)` hands over the answer, budget notes after every
tool output). The goal is to find where reasoning tokens are wasted or go wrong, and which intervention would fix it.

Inputs per attempt `Sxx` (all under `experiments/E017_reasoning_efficiency/`):
- `packets/Sxx.md`: outcome, rule references from earlier analyses, path of a correct DeepSeek program for the task,
  submitted answer vs truth (for wrong ones), and an offline **replay timeline**: every `run_python` cell was re-run and
  after each cell any function named `solve*`/`transform*`/`apply*`/`predict*` was probed on the train pairs and test.
  Use it to place "first train-passing solution" and "first correct test state" precisely. Functions with other names
  or inline checks are not probed, so also read the tool outputs.
- `rendered/Sxx.txt`: the trace, turn by turn, with tokens per turn and cumulative % of the attempt. Long runs of
  digit-only lines in reasoning were collapsed (count shown); that collapsed text is itself grid transcription.
- The task JSON (train + test with outputs) and the reference program, paths in the packet.

Read the packet, then the whole trace (skim repetitive stretches, but cover all of it; `grep -n` for "Wait", "Actually",
"Hypothesis", "contradict", "doesn't match", "budget" helps). Work out the true rule yourself from the task JSON and the
reference program if the references are unclear. Do not run any model or paid API. You may run Python on the task
JSON with `.venv/bin/python -I` if it helps to check a claim (never execute code from the traces on the host).

## Output: one JSON object per attempt, one line each, appended to `annotations/<your reader id>.jsonl`

Fields (keep strings short; quotes verbatim from the trace, max ~200 chars each):

```
{"sid": "S00", "task_id": "...", "pair": 0, "attempt": 2, "run": "sub80s40", "bucket": "...", "outcome": "wrong|correct|none", "tokens": 30201,
 "true_rule": "one or two sentences",
 "model_rule": "the rule the model finally applied, one or two sentences",
 "token_share": {"perception": 0, "hypothesis_search": 0, "code_writing": 0, "debugging": 0, "verification": 0, "post_success": 0, "other": 0},
 "n_hypotheses": 0,
 "correct_idea_first_pct": null, "correct_rule_complete_pct": null, "first_train_pass_pct": null,
 "perception_errors": ["..."],
 "checked_all_train_before_submit": "yes|no|partial|n/a",
 "train_check_would_catch": "yes|no|n/a",
 "contradiction_ignored": false, "contradiction_quote": "",
 "test_input_examined": "yes|partial|no", "test_needs_beyond_train": "yes|no|unclear", "test_detail": "",
 "failure_mode": "none|missed_idea|partial_rule|train_overfit|execution_bug|perception|budget_exhausted|ambiguous|gave_up",
 "failure_detail": "",
 "best_intervention": "a|b|c|d|e|f|g|none", "second_intervention": "a|b|c|d|e|f|g|none", "intervention_rationale": "",
 "wasted_tokens_pct": 0, "waste_detail": "",
 "early_signals": "what in the first ~30% of the trace predicted the outcome",
 "quotes": [{"pct": 12, "text": "..."}, {"pct": 80, "text": "..."}],
 "notes": ""}
```

Definitions:
- `token_share` (percent of this attempt's output tokens, sums to 100; estimate from per-turn token counts):
  - `perception`: reading, describing or transcribing grids in prose; listing cells or coordinates; counting by hand.
  - `hypothesis_search`: proposing, testing and abandoning rule ideas in prose, including reasoning over tool output.
  - `code_writing`: planning and writing code that implements a hypothesis (code itself included).
  - `debugging`: fixing exceptions or implementation bugs (not rule errors).
  - `verification`: checking a candidate rule against train or test assumptions *before* a train-passing solution exists.
  - `post_success`: everything after the first train-passing solution (re-checks, re-deriving, test sanity, hesitation).
  - `other`: e.g. budget deliberation, giving up.
- `n_hypotheses`: distinct rule ideas the model seriously pursued (not mere mentions).
- `correct_idea_first_pct`: % of the attempt's tokens where the true core idea first appears in the reasoning (null if
  never). `correct_rule_complete_pct`: when the full correct rule (all sub-rules) is first stated or coded (null if never).
  `first_train_pass_pct`: when a solution reproducing every train pair first exists (replay timeline or tool output).
- `train_check_would_catch` (wrong answers only): would a rule check on all train pairs have exposed the error? "no"
  means the submitted rule fits every train pair (train-overfit / out-of-distribution test).
- `contradiction_ignored`: the model saw evidence against its rule (a failing pair, a cell it could not explain, an
  assumption violated on test_input) and explained it away or proceeded anyway.
- `test_input_examined`: did it inspect test_input for properties the rule depends on (new colours, legend/key, object
  counts, conditions not seen in train)? `test_needs_beyond_train`: does the test require a case train does not show?
- `failure_mode`: `train_overfit` = rule passes all train pairs but is wrong on test; `partial_rule` = core idea right
  but a sub-rule/condition wrong; `missed_idea` = core concept wrong; `execution_bug` = right rule, code wrong.
- Interventions (pick the single one most likely to have turned this attempt correct, or for correct attempts, the one
  that would most cut tokens without losing the answer):
  - (a) computed task summary / perception aid in the prompt (shapes, colour counts, objects, diffs, symmetries);
  - (b) helper library in the session (objects/bbox/mirror/`check(solve)` with compact per-pair diffs);
  - (c) a "state the rule in words first; list the roles of objects / legend / colours" prompt step;
  - (d) verifier-gated retry with a fresh, diverse attempt (e.g. restart when no train-passing solution by N tokens);
  - (e) critic/judge choosing between candidate answers;
  - (f) more budget;
  - (g) nothing helps (capability limit).
- `wasted_tokens_pct`: share of tokens that did not contribute to the final answer or to ruling out a wrong idea
  (e.g. re-transcribing grids already printed, re-verifying after success, circular re-derivation).

Be concrete and evidence-based. Short verbatim quotes with their % position are the evidence. When finished, reply with
a 5-10 line summary of what you saw across your attempts (patterns, surprises), not a repeat of the JSON.
