# Annotation guide (E029, OLD vs NEW DeepSeek-Flash trace strategy)

Read-only, $0. Do not call any model API, do not start runs, never read or print `SECRETS.yaml` / `ENDPOINTS.yaml`.
Write only your own annotation file `experiments/E029_deepseek_flash_1017/reading/<LETTER>.jsonl`. Touch nothing else.

## What you are reading
Two models on the same tool harness (sub80s40: Python `run_python` tool with `train` / `test_input` preloaded,
`submit_answer(expr)` hands a session expression over as the answer; 80k output-token hard limit, "aim to finish within
40k" budget notes appended to every tool output, a "Budget nearly spent: submit now" note from 60k), same prompt
(`lean2sub`: inspect with code, never copy grids by hand, check every train pair with code, check test assumptions,
keep reasoning brief, submit `solve(test_input)` when train passes).
- **OLD** = DeepSeek V4.1-Flash (E009, before 2026-10-17).
- **NEW** = DeepSeek V4.1-Flash-1017 (E029).

Rendered traces: `/tmp/e029_render/<task>_p<pair>_{OLD,NEW}.txt`. Each starts with a per-turn table (output tokens per
turn, cumulative). Reasoning text carries interpolated position markers `[~tok 12.3k]` (cumulative output tokens,
approximate within a turn); use them for every token position you report. Tool outputs over 5k chars are truncated in
the middle. The task JSON (train pairs, test input AND test output) is
`/home/vu/Workspace/ARC-AGI-2/data/arc-agi-2-github/data/evaluation/<task>.json`; look at it so you know the true rule
(e.g. load it with `.venv/bin/python -I -c ...` and print compact grids).

Files can be large (up to 270k chars). Read in chunks (Read with offset/limit) and use grep for markers
(`### ASSISTANT turn`, `### TOOL OUTPUT`, `True`, `budget`, `submit_answer`, `def solve`). You need not read every
hand-transcribed row: skim those, but estimate their size in tokens (about 3 chars per token).

Groups (why the pair was selected): SAVE = both models solved it, NEW used far fewer tokens; GAIN = NEW solved, OLD failed
in all 3 attempts; WASTE = NEW solved but used > 50k tokens; REGRESS = NEW failed both attempts, OLD solved at least one.

## For EACH trace write one JSON object (one line)
```
{"task": "...", "pair": 0, "group": "SAVE|GAIN|WASTE|REGRESS", "model": "OLD|NEW", "attempt": 0, "correct": true,
 "tokens": N, "turns": N,
 "turn1_tokens": N,                 # output tokens before the first tool call
 "first_solve_tok": N|null,         # cum tokens when a candidate solve()/transform is first run on all train pairs
 "train_pass_tok": N|null,          # cum tokens at the tool output where the FINAL submitted rule first passes all train pairs
 "submit_tok": N,                   # cum tokens at submit_answer (or at the end if none)
 "post_pass": {"tokens": N|null, "what": "short: what it did between train_pass_tok and submit (test audit, re-run train, hand-check test output, doubts about alternative rule, nothing)"},
 "perception": {"mode": "code|hand|mixed", "hand_reading_tokens_est": N, "note": "how it looks at grids: prints + reads cells in prose? object tables in code? diffs?"},
 "hypotheses": {"n_rules_tried": N, "idea_tok": N|null,   # where the core correct idea first appears
                "abandon_style": "code-falsified|prose-falsified|patched with special cases|never abandoned|n/a",
                "note": "short"},
 "debugging": "short: prints/diffs per pair vs mental simulation of its own code; how many fix iterations",
 "verification": {"what": "short", "n_checks": N, "redundant_checks": N, "redundant_tokens_est": N},
                    # redundant = re-checking something code already verified (re-running train after pass, hand-checking a code-produced grid, re-deriving the rule)
 "budget_reaction": "short: n/a if it never passed 40k; else what the notes changed (ignored / sped up / submitted a failing rule / set answer)",
 "loops": {"present": true|false, "tokens_est": N, "note": "re-deriving the same thing, oscillating between two rules"},
 "final_answer": "solve(test_input) after full train pass | submitted with a failing train pair | session fallback | other: ...",
 "failure_cause": "for wrong traces only: wrong rule / right rule, bug / ran out of budget / misperception / test-only case not handled; else null",
 "waste": [{"pattern": "<code from list below>", "tokens_est": N, "tok_range": "12.3k-18.0k", "note": "short"}],
 "evidence": ["<= 4 items: '[~tok 12.3k] <your own paraphrase of what the model does there>'; code or tool output may be quoted"],
 "notes": "2-4 sentences on its strategy"}
```
Waste pattern codes (use these so counts can be aggregated; several per trace allowed; only list items >= ~1k tokens):
- `prose_before_code`: long theorising before the first tool call or before any experiment that could test it.
- `hand_reading`: reading/transcribing printed grids cell by cell in prose (when code could have answered).
- `mental_simulation`: executing geometry or its own code in its head instead of running it.
- `rule_restatement`: re-stating or re-summarising the same rule again without new information.
- `dead_end_prose`: pursuing a wrong hypothesis in prose for long after a cheap code test could have killed it.
- `post_pass_reverify`: after code passed all train pairs, re-verifying train or re-deriving the rule.
- `test_hand_check`: checking the code-produced test output by hand/in prose (as opposed to a quick code check).
- `alt_rule_doubt`: after pass, long deliberation about alternative interpretations for the test.
- `loop`: oscillating or re-deriving the same thing repeatedly.
- `code_bug_churn`: many fix iterations for implementation bugs (not rule errors).
- `other`: describe in note.

Then per pair one extra line:
`{"task": "...", "pair": 0, "group": "...", "model": "COMPARE", "speed_to_code": "...", "perception": "...", "hypotheses": "...",
  "debugging": "...", "verification": "...", "budget": "...", "where_tokens_saved_or_lost": "breakdown with token estimates",
  "why_outcome_differs": "for GAIN/REGRESS: the decisive difference; else what made NEW faster/slower",
  "notes": "2-4 sentences"}`

**Do not copy the model's reasoning text verbatim** (no quoted passages from `[reasoning]` blocks, not even short ones):
writing out reasoning passages trips a safeguard that kills the reader. Describe what happens in your own words, with
token positions. Code from tool calls and tool outputs may be quoted briefly.

Be concrete and honest; estimates are fine but say when unsure. Use only the trace text and the task JSON as evidence.
