# Qwen answer-format audit (2026-10-11)

Question: are any correct answers lost to the answer format or the ARC Prize parser (`evals/arc_harness/arcprize.py parse()`, applied to the text after the last `</think>`)?

**Answer: no.** Across 450 attempts in two runs, 0 incorrect attempts contain the correct grid anywhere, in any format, in either the answer or the thinking. `\boxed` is never used. Keep the parser.

## Runs audited
- **Qwen3.6-35B-A3B (NVFP4), `gen_none_240k_t1pp15`:** a snapshot of the first 116 finished lines; the run is still in progress. 14 correct, 102 incorrect.
- **Qwen3.8-27B (NVFP4), `gen_xhigh_240k_t1`:** finished, 334 lines. 120 correct, 214 incorrect.
- **Re-parse check:** re-parsing every `content` with `parse()` reproduces every stored `correct` flag.
- **Positive control:** the thinking-trace detectors find the solution grid in the thinking of 8/14 (3.6) and 65/120 (3.8) correct attempts. The lenient search therefore does detect grids when they are present.
- **Audit scripts:** `/tmp/qaudit/` (scratch; not part of the repo).

## Answer-format distribution (all attempts)

| Format of `content` | Qwen3.6 (116) | Qwen3.8 (334) |
|---|---|---|
| Bare JSON grid only, sometimes with a label such as `OUTPUT:` | 98 (12 correct) | 334 (120 correct) |
| Bare JSON grid plus prose explanation | 8 (1 correct) | 0 |
| JSON grid inside a ```` ```json ```` / ```` ```python ```` fence, with prose | 8 (1 correct) | 0 |
| JSON grid inside a code fence, no prose | 2 (0 correct) | 0 |
| `\boxed{...}` | **0** (0 in the thinking either) | **0** (0 in the thinking either) |
| Digit rows, markdown table, or space-separated grid only | 0 | 0 |
| No grid at all (`pred` is None) | 0 | 0 |

Other observations:
- Every answer contains exactly one top-level JSON grid. Multi-line pretty-printed grids: 17 (3.6) and 110 (3.8).
- No answer has text after the grid. The only extra grid-like fragments are 3.6's 1-D row lists inside prose (for example, a `[0, 4, 4]` tile). These are not grids, so the backscan ignores them.
- Qwen3.6 sometimes mimics the prompt layout before the grid, for example `--Example 0-- OUTPUT:` or `--Example 0 OUTPUT--`. The parser handles this.
- **Unclosed thinking:** in 3.8, 3 attempts hit the budget and got a forced `</think>`. All 3 still produced a grid (2 wrong, 1 correct). In 3.6, all 116 closed `</think>` on their own.

## Lenient re-parse of the incorrect attempts (would-be-correct counts)

| Rule | Qwen3.6 (102 incorrect) | Qwen3.8 (214 incorrect) |
|---|---|---|
| (a) Any JSON grid anywhere in content equals the solution | 0 | 0 |
| (a') Last *valid* grid in content, instead of the last bracket only | 0 | 0 |
| (b) Digit-row / markdown / `|0|1|` / space-separated / code-block rows in content contain the solution | 0 | 0 |
| (c) Solution in the thinking but not the answer (not a parse bug) | 0 | 0 |
| (d) `\boxed` regex broke and the fallback missed the grid | n/a, `\boxed` never used | n/a |
| (e) Correct grid present but not the last grid | 0 | 0 |
| (f) Transposed solution present | 0 | 0 |
| (f) String digits (`"0"`) normalising to the solution | 0 (no string-digit grids exist) | 0 |
| **Total recoverable** | **0** | **0** |

## Wrong-answer shapes (not a parse issue, for context)

| Shape of the wrong answer | Qwen3.6 (102) | Qwen3.8 (214) |
|---|---|---|
| Right shape, wrong cells | 67 (7 within 2% cell error, 15 within 5%) | 152 (29 within 2%, 64 within 5%) |
| One row short, same width | 14 | 17 |
| Transposed shape (not the transposed solution) | 2 | 4 |
| Other wrong shape | 13 | 24 |
| Jagged rows | 6 | 17 |

**"One row short" is the one true output-writing loss.** These attempts have the right rule but drop one row while typing a large grid:
- **Exact hits:** the answer equals the solution with exactly one row deleted.
  - Qwen3.6: 1 case (3e6067c3 p1 a1, 30×30). All 30 solution rows appear in its thinking.
  - Qwen3.8: 4 cases (3e6067c3 p0 a0, 64efde09 p0 a1, 80a900e0 p0 a1, 800d221b p0 a1).
- **Near hits:** 1–2 cells off after realigning the dropped row. Qwen3.8 has 2 more (80a900e0 p0 a0, 2c181942 p0 a1).
- **Not recoverable by parsing:** the full grid never appears contiguously anywhere. A prompt tweak could help instead, for example asking for one row per line, or for the row count to be stated and checked.

## Eight-plus failed Qwen3.6 outputs, read fully (answer plus the last 2–3k characters of thinking)

All nine are clean bare-JSON (or fenced JSON) answers that the parser read correctly. Every failure is a wrong rule, a misapplied rule, or a transcription slip. None is a format failure.

1. **38007db0 p1 a0** (25×7, 2 cells off). The rule is nearly right, but one 3×3 motif is wrong: it predicts `8 8 3` where `8 3 8` is expected. The thinking ends in a loop: "All checks pass. Output matches expected format. Proceeds." repeated many times, then the answer is given verbatim. Cause: a rule-detail or perception error.
2. **221dfab4 p1 a0** (30×30 expected, 29×30 given). The rule is wrong. It maps "2->4, 1->3" only on the beam rows and says "Rows 10-29: same as input"; the real rule also recolours shapes outside the beam. It then drops a row while retyping the 30-row grid.
3. **78332cb0 p0 a1** (5×23 expected, 23×5 given). Wrong stacking orientation: "Square grids -> ascending… Output matches 23x5 grid." It stacks the blocks vertically, not horizontally, and the block order also differs. This is not a transposed solution.
4. **20a9e565 p0 a0** (2×21 expected, 12×3 given). The rule is wrong: "stacking that tile vertically N times". The thinking shows doubt ("My tile `[0,4,4; 4,4,4; 4,4,4]` feels off") but commits anyway. The answer is prose, a fenced tile, then the bare JSON grid; it parses fine.
5. **142ca369 p0 a1** (18×18, 235 cells off). The rule is wrong: it fills diagonal stripes ("a plausible interpretation… I'm confident"). The answer is prefixed with `--Example 0-- OUTPUT:` copied from the prompt layout, and still parses.
6. **1818057f p0 a0** (22×22, 2 cells off). The rule is right. Its own check says "Changed indices 16, 17, 18 to 8" for row 17, but the row it typed has 8s at 17–19, and the answer repeats that row. Cause: a transcription or column-index slip. The fenced draft in the thinking has the same error, so no parser could recover it.
7. **4c7dc4dd p0 a0** (5×5, 25 cells off). An admitted guess: "So 0->0, 1->2, 8->6 is a good guess." Wrong rule.
8. **53fb4810 p0 a0** (25×18, 2 cells off). The rule is nearly right. It keeps the `4 3 / 6 2` alternation down to the bottom row, but the solution flips the phase at the end (rows 23–24). It spends its last 3k characters re-counting row lengths, not checking that detail.
9. **3e6067c3 p1 a1** (30×30 expected, 29×30 given). The rule is right, and every one of the 30 rows is verified in the thinking ("Length 30… Matches input"). The answer, after a `--Example 0 OUTPUT--` label, is the solution with row 22 missing. This is the transcription loss described above.

Common pattern in the thinking: long "Proceeds. / Output matches requested format. / Done." loops before `</think>`. The model often writes partial drafts ("Rows 0-4: same as input") and types the full grid for the first time in the answer. That is where rows get dropped or shifted.

## Recommendation
- **Keep the parser.** It loses nothing on either run. Both models answer with a single JSON grid; they never use `\boxed` or digit-row formats.
- **Watch the latent `\boxed` weakness, but don't change the parser for it.** The non-greedy `\boxed\{(.*?)\}` would break on `\boxed{\begin{bmatrix}…}`. The backscan also returns None if the text ends in a non-grid `]` or `}`, rather than searching further back. Neither case occurs here. If a future prompt asks for `\boxed`, re-audit then.
- **Optional prompt-side gain (small):** 1 (3.6) and 4 (3.8) attempts are exact solutions minus one row, plus about 2 near-exact. That is about 1–2% of attempts, lost to typing large grids rather than to parsing. A cheap test would ask for one row per line, or for the row count to be stated and checked before the answer. It is not worth changing the official ARC Prize prompt for the main eval.
