# E026: How Qwen3.8-27B solves ARC-AGI-2 with tools, compared with DeepSeek V4.1-Flash (read-only analysis)

Written 2026-10-16. No model calls, no runs, $0. The DeepSeek-only analysis is in `deepseek_strategy.md` (another agent).

## Summary

**Data.**
- **Regex and turn statistics** on every attempt of five runs, all on the 167 eval pairs:
  - Qwen: E020 K3, x96p, x96pc8 ("c8") and E012;
  - DeepSeek: E017 V3 and E009 sub80s40.
- **Close reading** by Claude readers working to one annotation guide: 37 pairs, 58 Qwen traces and 35 DeepSeek traces. The pairs were picked so that the gap shows up:
  - 14 pairs DeepSeek V3 solved on both attempts and Qwen failed in both x96p and c8;
  - 10 pairs where one of the two Qwen configs solved it;
  - 8 pairs both models solved;
  - 5 pairs only Qwen c8 solved.

  Reading counts are therefore conditional on that selection and are not population rates.
- **Main comparison:** Qwen c8 or x96p (xhigh, 96k, summary + helpers + roles, program answers) against DeepSeek V3 (low effort, 80k/40k, summary + helpers + roles, `submit_answer`). This is the closest harness match available. The main differences left are the answer mode, the effort tier, c8's think cap, and an extra 16k of budget for Qwen.

**1. At the same harness, Qwen's gap is half speed and half generalisation, not finding a train fit.**
- On attempt 0, c8 reaches a train-passing answer as often as DeepSeek: 77% vs 78%.
- When it does, the answer is right less often: P(correct | train pass) is 72% vs 84%.
- DeepSeek gets 35 pairs right that c8 gets wrong:
  - 18 where Qwen's program passed train but was wrong on test;
  - 17 where Qwen never passed train. In 7 of those 17, DeepSeek's own answer also failed the train check (a hand-built or hand-patched grid).
- At the Kaggle budget (K3, 48k) speed dominates: 47 of the 58 DeepSeek-only pairs are Qwen attempts with no train pass, and 49% of K3 attempts never submit at all.

**2. Qwen reasons in prose for much longer before it tests anything.** On the same 167 pairs (medians):

| | Qwen x96p | Qwen c8 | DeepSeek V3 |
|---|---|---|---|
| First turn, before any tool call | 25.6k | 8.2k (the cap fires) | 4.3k |
| First `def solve` written at | 38k | 38k | 20k |
| First code run on `test_input` at | 36k | 27k | 18k |
| Total tokens | 56k | 53k | 34k |

- In x96p, 57% of all thinking words come before the first tool call; DeepSeek's share is 19%.
- This is an effect of the **xhigh tier**. The first turn is 0.5–1.4k at medium (with or without the summary) and 20.6k at xhigh without the summary.
- Qwen also keeps reasoning after it decides to use code. After writing "let me compute it in code", x96p continues for a median 10.7k characters in the same turn; DeepSeek continues for 2.3k.

**3. Hand perception goes wrong, and Qwen drops correct ideas because of it.**
- Qwen's thinking holds more coordinate tuples (61–64 per 1k words vs 42) and more paragraphs about reading grids (41% vs 27% by a keyword heuristic).
- Readers rated hand transcription "heavy" in 32 of 58 Qwen traces vs 10 of 35 DeepSeek traces.
- DeepSeek hand-reads too (it has more "row N:" lines), so volume alone does not separate the two models. What separates them:
  - Qwen re-reads grids that code has already printed.
  - A hand miscount makes it **abandon the right idea**: 221dfab4, 4c3d4a41, 89565ca0, 8e5c0c38.
  - DeepSeek writes "Let me just compute in code" and does it.

**4. Qwen's train-passing programs fit the train pairs through incidental details.** Readers' main cause for "passes train, wrong on test" (11 Qwen traces were test-only cases, 4 overfit):
- Qwen picks the minimal or coincidental sub-rule:
  - a constant shift of 1, where the true rule is "rest on the template";
  - a phase anchored to index 0, where it is "distance from the key";
  - "only the caps change", where the whole content is swapped.
- It adds one local patch per exception, where DeepSeek finds one structural cause.
- It notices a test anomaly and argues it away. c8 on c7f57c3e: "the center color varies (3, 6 — treated as unchanged interior)".
- A syntax lint does **not** catch this. Literal tables, constant comparisons and magic numbers do not differ between right and wrong train-passing programs, for either model.

**5. Other differences.**
- **Style:** DeepSeek uses more hedging words (25 vs 15 per 1k words: "maybe", "hmm", "actually"), but readers judged Qwen to loop more ("heavy" in 18 of 58 traces vs 3 of 35). Qwen writes about 2.5× more markdown.
- **Code:** Qwen's code has more bugs (unpacking dicts, numpy truth values, off-by-one), which cost debug turns. It writes its own extractors where `objects()` would do. Helper and `check()` use rates are similar: `check` in 51% of c8 attempts vs 54% of V3.
- **Rejections:** Qwen almost never gets a submission rejected (13 of 334 K3 attempts), because it submits only after `check` passes. Failure handling therefore happens inside `run_python` loops.

**6. What SFT on DeepSeek rollouts should change, in priority order:**
1. **Go to code early and keep turns short.** In DeepSeek's correct V3 traces the longest turn is 11.8k at the median and the first tool call comes at about 4k; also train in the short reactions to tool output. This is the cheapest gain at a 48k budget.
2. **Use code output as the source of truth.** Never re-type printed grids; settle a doubtful count with a code cell rather than by hand.
3. **Model rules at the object or relation level and audit the test.** Look for one structural explanation for the exceptions. Print the test prediction and check each test-only assumption.
4. **Score candidate rules by train diff counts in code** instead of hand-designing special cases.

**Do not imitate:**
- hand-built or hand-patched answer grids (4% of V3 attempts are correct without a train-passing program);
- "designer intent" repairs of input cells (53fb4810);
- hard-coded lattice coordinates;
- DeepSeek's rare 20–50k prose turns.

**Format and length.**
- DeepSeek traces are 1.05× longer in Qwen tokens; the median correct trace is 24.6k, p75 43k, p90 62k, so about 75% fit a 48k cap.
- Answers must be converted from `submit_answer("answer")` to a self-contained `submit_program`. 94% of correct V3 attempts have a train-passing answer program, so this is feasible.
- Effort tier: DeepSeek ran at low effort, so Qwen's medium tier, not xhigh, is the natural target for training and inference.

## Details

### Data and method
| run | model | harness | pairs × attempts | pass@1 (attempts in file) |
|---|---|---|---|---|
| E020 K3 | Qwen | xhigh, 48k/24k, program, summary + helpers + roles, rich feedback, think cap 8192 | 167×2 | 37% |
| E020 x96p | Qwen | xhigh, 96k/48k, same without the cap | 167×1 | 49% |
| E020 x96pc8 ("c8") | Qwen | x96p + think cap 8000 / grace 2048 | 167×1 | 56% |
| E012 | Qwen | medium, 48k/24k, `submit_answer`, no summary or helpers | 167×2 | 25% |
| E017 V3 | DeepSeek Flash | low, 80k/40k, grid mode, `submit_answer`, summary + helpers + roles | 167×2 (+24) | 69% |
| E009 sub80s40 | DeepSeek Flash | low, 80k/40k, no summary or helpers | 167×3 (2 used) | 65% |

**Parsing.**
- Qwen: raw chat-template transcripts (`*_traces/*.txt.gz`), split per assistant turn into thinking, prose, tool calls and tool response. Turn token counts come from `turn_log`; their sum matches `usage.completion` exactly.
- DeepSeek: `*.messages.json.gz`, using `reasoning_content` per turn plus `events` usage. The reasoning text is complete, at about 3 characters per token.

**Phase heuristic.** Each thinking paragraph gets one label:
- perception: more than 30% digit rows, or 3 or more "row N:" lines, or 6 or more (r,c) tuples;
- otherwise by keyword, in order: test → verify → code → hypothesis → other.

Each label's share is weighted by characters. It is coarse: hand pattern sketches such as `0 # #` count only through their coordinates.

**Scripts** (scratch, not in the repo): `feat.py` (per-attempt features), `testinsp.py`, `lint.py`, `style.py`, `turn1.py`, `defer.py`, `render.py` (condensed renderings for the readers).
**Reader output:** `annotations_g{1..4}.jsonl` (126 objects).

### 1. Qwen's strategy in phases (x96p and c8; K3 is c8 cut off at 48k)
1. **Perception and transcription.** Qwen reads the summary (it cites it about 6–9 times per attempt; 51 of 58 read traces rely on it) and then re-derives shapes by hand.

   x96p on 8e5c0c38, turn 1, 37.5k tokens, before any code:
   > "6. (14,16): Row 13: 0 0 0 / Row 14: # # # / Row 15: 0 # # → pattern E: 000/###/0##"

   DeepSeek on the same pair, turn 1, 390 tokens:
   > "Let me examine actual changes. Let me use code."

   Its code was a diff of every training pair. Qwen thinking is 41% perception paragraphs; DeepSeek's is 25–28%.
2. **Hypothesis generation.** This happens mostly in prose inside the giant first turn (x96p's median first turn is 25.6k tokens; 7666fa5d's single turn was 94k). The right idea often appears early, then is lost or diluted.

   cb2d8a2c, x96p, turn 9, at about 88k of 96k, with no code for the rule:
   > "What if the left delta = the number of 1s in A + 1: Ex1: 1+1 = 2 ✓!! Ex3: 2+1 = 3 ✓!!!"

   6e4f6532, c8, turn 21 (about 71k):
   > "This is the rule!!"

   DeepSeek had the same insight by about 25k.
3. **Verification.** Qwen checks hypotheses by hand against the printed grids and the summary. Hand checks with errors reject correct ideas.

   221dfab4, c8, turn 6:
   > "So d-anchor hypothesis fails at r9"

   This was a miscount; the d-anchor rule is correct. 2c181942, c8, turn 11, after the cap forced code:
   > "that must have been a hand-counting error ... OK, let's trust the code."
4. **Coding.** The first `def solve` comes at about 38k (DeepSeek: 20k), at 86–89% of the attempt's tokens, a similar fraction to DeepSeek's.
   - The code is longer and buggier. Readers saw 3–4 debug turns from simple bugs on d8e07eb2, 8b9c3697 and 9aaea919, which DeepSeek's code mostly avoided.
   - Helper use is about the same: `objects()` in 10–21% of attempts vs 18%, `check()` in 51–63% vs 54%. Several readers saw Qwen write its own feature extractor instead of using `objects()`.
5. **Test application.**
   - **Train-passing programs:** Qwen ran code on `test_input` before submitting in 92–98% of them, and printed `solve(test_input)` in 64–70%. DeepSeek printed it in 87% (it must compute the answer grid in the session).
   - **Printing is not inspecting.** For Qwen, P(correct) is the same whether or not it printed. Readers saw Qwen notice the test-only feature and dismiss it (c7f57c3e, 4c3d4a41).

     4c3d4a41, x96p, turn 5:
     > "Both 6s vanish! Is this consistent with the training behavior? ... the rule is verified against the training examples, and that's what matters."
6. **Reaction to failed checks.**
   - **Few rejections:** only 9–13 attempts per run ever had a submission rejected, because Qwen submits after `check` passes.
   - **Inside `run_python`,** fixes are targeted at bugs. Rule failures lead to a new local patch, not a new model: "So the processing order is reverse row-major (bottom-up, right-left)? Plausible." (dfadab01, x96p, turn 5).
   - **Budget:** when it runs out, Qwen rushes. 7666fa5d, x96p, turn 2, after a 94k first turn: "Budget is nearly exhausted. I need to hurry: submit the best program."
7. **Looping and hedging.**
   - Repeated lines are rare by regex (about 1%).
   - Readers rated re-deriving the same thing "heavy" in 18 of 58 Qwen traces vs 3 of 35 DeepSeek traces.
   - DeepSeek's hedging words are commoner per word (25 vs 15 per 1k: `maybe` 7.8 vs 2.9, `hmm` 8.9 vs 5.5). DeepSeek hedges in short bursts and then acts.
8. **What the think cap (c8) changes.**
   - It cuts the first turn to 8k and makes Qwen print things. Several fixed hand misreads came from code that the cap forced (2c181942, 4c3d4a41, 9385bd28, edb79dae, 8e5c0c38).
   - It does not give direction. The note often fired 5–10 times per attempt, and each time Qwen ran one small probe and then wrote another 8k of prose.

### 2. Contrast on the same pairs, and which differences go with Qwen failures
Outcome table, attempt 0, 167 pairs (tp = train pass):

| | c8 | x96p | K3 (48k) |
|---|---|---|---|
| both right | 82 | 79 | 59 |
| DeepSeek right, Qwen tp but wrong | 18 | 18 | 11 |
| DeepSeek right, Qwen no tp | 17 | 20 | 47 |
| Qwen right, DeepSeek wrong | 11 | 2 | 3 |

Features of right vs wrong Qwen attempts (c8 / x96p / K3):
- **Median longest turn:** in x96p it is 21.6k (right) vs 37.4k (wrong); the cap removes this signal in c8 and K3.
- **No `solve` ever written:** 0% of right attempts; 15%, 38% and 56% of wrong ones.
- **`check()` used:** 68–80% of right attempts vs 23–47% of wrong ones.
- **Perception share:** 36–38% (right) vs 43% (wrong).

For DeepSeek too, wrong attempts have longer prose turns (median longest turn 19.2k vs 11.8k) and use `check` less (38% vs 62%). The same habits mark failure in both models; Qwen simply shows them more often and earlier.

Divergences from the reading, with the pairs where each decided the outcome:

| divergence | DeepSeek | Qwen | pairs |
|---|---|---|---|
| Data before theory | prints a diff or change table in turn 1–2 (≤3.3k) | 6–94k of prose first (x96p) | 7666fa5d, 64efde09, a32d8b75, 9385bd28, c7f57c3e |
| Search method | brute force in code: d8 + translation matching, enumerating tilings with a seam score, ranking candidates by train diff counts | hand-designed special cases; debugging its own matchers | 446ef5d2, 6e4f6532, b6f77b65, 64efde09 |
| Rule level | relational or object level: rest on the template, distance from the key, swap whole contents, one anchor explanation | minimal or coincidental: constant offset, absolute index, cap-only change, one patch per exception | 4c3d4a41, 221dfab4, c7f57c3e, dfadab01, 9385bd28, 2d0172a1 |
| Trust in code | "Let me just compute in code" | re-reads printed grids; a hand miscount drops the right idea | 221dfab4, 89565ca0, 8e5c0c38, 58f5dbd5, 247ef758 |
| Ambiguity | names it ("This is genuinely ambiguous from training", 291dc1e1 V3) and settles it with a principle | picks one ("by symmetry probably clockwise") | 291dc1e1, 4c3d4a41 |
| Cheap first test | falsifiable test within 2k, then a coordinate change (r+c, r−c) | undirected feature sweeps | 7666fa5d, 269e22fb |

Where Qwen did better, mostly in c8 (both-solved and Qwen-only pairs):
- Its programs derive lattice positions; DeepSeek hard-codes them (9aaea919, d8e07eb2).
- It trusted the train diffs over a "repair the noise" prior (53fb4810, where both DeepSeek attempts repaired input cells).
- It used a marker cell count where DeepSeek used a train-coincident bbox extent (3dc255db).

### 3. Implications for SFT
**Worth imitating** (each seen in several DeepSeek traces):
1. **A code-first opening:** a diff or change table, per-colour masks, extracted stamps. Then one hypothesis per short turn. The median DeepSeek turn after tool output is 0.9k tokens.
2. **Treat code output as ground truth** and compute any count in a cell.
3. **Score candidate rules numerically** against all train pairs; for small search spaces, enumerate.
4. **Relational or object-level rules,** one structural cause for all exceptions, and explicitly named ambiguity.
5. **Count-audit the printed test output** (dfadab01: "126 is not a multiple of 8").

**Not worth imitating, or not learnable in program mode:**
- **Hand-built or hand-patched answers.** In V3, 14 of 231 correct attempts have no train-passing program. 5 of the 14 DeepSeek-only wins read had such an attempt 0: 2d0172a1, 6e4f6532, 89565ca0, a25697e4, b6f77b65.

  89565ca0, V3, turn 23:
  > "OK this is too fragile. Given budget, I'll just hardcode the test answer based on my manual geometric analysis, and submit."

  In 4 of these 5, attempt 1 has a train-passing program, so take that rollout instead. b6f77b65 has none.
- **"ARC avoids noise" repairs** of input cells that training never changes (53fb4810, both attempts).
- **Hard-coded absolute lattice coordinates** in `solve` (9aaea919, d8e07eb2). They passed here but generalise worse.
- **DeepSeek's own long prose turns.** The median longest turn in correct traces is 11.8k, p90 24.7k; 64efde09 had 50k and 8b9c3697 22.7k. These are what Qwen already overdoes.

**Hard to imitate or needing conversion:**
- **Answer interface.** DeepSeek ends with `submit_answer("answer")`; 241 of 334 submissions are the literal `answer`. Qwen's harness needs a self-contained `solve` in `submit_program`.
  - Rebuild the final program from the session cells.
  - Re-check it host-side on the train pairs (and test, if training data has labels).
  - Rewrite the last turn.
- **Tool-call format.** DeepSeek uses API JSON tool calls (99% one call per turn; 85% of turns have empty visible content). Re-render them into Qwen's `<tool_call><function=run_python><parameter=code>` XML with thinking in `<think>`. This is a mechanical conversion.
- **Harness text.** DeepSeek's budget notes ("aim to finish within 40k (hard limit 80k)") and prompt (`lean2roles`) differ from Qwen's. Re-render the tool outputs with Qwen's harness text, or train at the matching budget.
- **Length.** DeepSeek traces retokenise to 1.045× in Qwen's tokenizer (40 correct V3 traces). Correct traces are 24.6k / 43.5k / 61.7k at p50 / p75 / p90, so about 75% fit the Kaggle 48k cap as-is.
- **Effort tier.** DeepSeek's traces are low effort. Qwen's xhigh tier is what triggers the 20–25k first turns, so train (and run) with the medium or no-tier system line, not xhigh.
- **Style is already close.** Qwen's no-tool xhigh thinking opens in the same telegraphic register as DeepSeek's no-tool traces: "We need solve puzzle. Need produce final output…" against "We need solve ARC-like puzzle. Need infer transformation." Both switch to "Let me…" prose in tool mode. SFT has to change strategy, not register.

**Data-filtering suggestions:**
1. Keep only rollouts whose final answer comes from a program that passes train. This drops about 6% of V3's correct attempts (14 of 231).
2. Prefer traces whose first tool call comes at ≤5k and whose longest turn is ≤16k.
   - Do not drop long-turn traces outright: hard tasks need them.
   - Truncate them, or down-weight them.
3. Drop or down-weight traces with:
   - hard-coded coordinates;
   - edits to input cells that no train pair changes;
   - "Given budget, I'll just…" guesses.
4. Over-sample tasks where Qwen fails and DeepSeek succeeds with relational rules (the 14 + 21 + 16 gap pairs here are eval tasks, so use the same kind of selection on training-set rollouts, E025).
5. Use eval-task traces only for analysis, never for training (AGENTS.md contamination rule).
6. Filter on test correctness as well as on train pass. In the 5 pairs that only Qwen c8 solved, DeepSeek traces 53fb4810 a0 and a1, 3dc255db a1, and dbff022c a0 and a1 all pass train but are wrong on test. They show the failure modes to avoid:
   - a "designer intent" repair. 53fb4810, V3 a0, T6: "The anomaly in test is a strong hint that the designer wants us to fix it."
   - a train-coincident rule, kept after dismissing the right alternative on a misread. 3dc255db: bbox extent instead of cell count.
   - a fixed legend orientation. dbff022c: the legend is read from the top or left, but the test legend sits on the bottom edge.

### Caveats
- The reading sample (37 pairs) was chosen to show divergence, so its outcome counts over-represent DeepSeek wins. Population numbers above come from the regex/turn statistics over all 167 pairs.
- Each trace had one reader, and phase labels are heuristic. Quotes were checked by the readers against the traces; I did not re-verify every one.
- The harnesses are not identical:
  - effort tier: Qwen xhigh vs DeepSeek low;
  - budget: 96k vs 80k;
  - answer mode: program vs grid;
  - c8's cap rewrites turn structure.

  The speed numbers mix model and tier effects. The medium-tier rows (E012 vs sub80s40: first turn 0.5k vs 0.3k, first solve 24.9k vs 19.7k, accuracy 26% vs 66%) show that at a short first turn the remaining gap is mostly rule quality and debugging.
