# Sample A: wrong-shape outputs on Flash-easy tasks (Nemotron 3.5 Lightning, high)

10 attempts. Flash solves each of these tasks 3/3. All 10 Nemotron attempts have `finish=stop` at 62-82k completion tokens
(budget 200k), so none was truncated: the model stopped on its own. Char offsets refer to the decompressed reasoning trace.

## Per-attempt

**981571dc p0** (30x30 grid symmetric about the diagonal and both axes, with 0-holes; rule: fill each hole from its mirror cell).
- Perception: dimensions eventually right, but it starts with "around 25-30 rows, each of length 30-35". It treats every row as a 1-D digit string.
- Rule: never found it. The words "symmetr" and "palindrom" never appear. It spends ~130k chars fitting per-row arithmetic to the hole contents ("(L * R + i) mod 10", "linear congruential generator").
- End: gives up and returns the input: "Given the absolute time limit, I will output the test input as the output, assuming the pattern is identity." (132k)
- Shape: while copying the 30-row test input it dropped input rows 17-18, giving 28x30. The holes are also unfilled, so the content would be wrong even at the right size.

**8e5c0c38 p0** (22x22; rule: delete cells so that each object becomes left-right mirror-symmetric).
- Perception: "Input is a 25x25 grid" (it is 22x22). It merges rows when transcribing: input row 5 is quoted with the contents of rows 4 and 5 mixed together.
- Rule: never found it ("mirror"/"symmetr": 0 hits). It cycles through row shifts, gravity, column shifts and rotation (~100k chars).
- End: "Given the extreme time pressure..." then "I'll output a 22x22 grid of 1s" as a placeholder (135k). The answer text admits the pattern was "not fully identifiable".
- Shape: it intended 22x22 but emitted rows of 20-21 ones, so the parsed shape is 22x20. Long constant rows are miscounted.

**8e5c0c38 p1** (same task, second test).
- Perception: "The grids are 21x22 (21 rows, 22 columns)". It uses 1-indexed rows, which shifts row numbers by one.
- Rule: never found it. It analyses horizontal segments row by row: "Remove all non-background cells that have the background color immediately to their right..."
- End: "Given the absolute dead end... I'll output the test input with all non-4 cells removed (all 4s). That's my final guess."
- Shape: an all-4 grid with one row missing (21x22).

**20a9e565 p1** (sequence of growing shapes; the output is the next element of the series, 8x17).
- Perception: calls the 29x29 test input "Test Input: 30x30" (108k). It gives Ex2's output as "18 rows" (it has 17). Its early row indices are 2-3 off.
- Rule: never found it. It tries "keep the main structure number", boundary tracing and row-wise left/right values. It dumps a full all-zero grid at ~140k.
- End: "Given the extreme time pressure, I'll output a matrix that ... matches the test input's height (14)".
- Shape: the rule was wrong (non-zero rows x 2), and although it said "14x2" it emitted 15x2.

**58490d8a p0** (count the shapes of each legend colour and draw them in the 0-box; answer 9x11).
- Perception: "Input is a 20x25 grid" (20x22); "Ex 2 input width 10" (it is 20). It never names the 0-box as the output template.
- Rule: partial. It sees that values sit at odd positions and that "the number of such columns in a row equals the count of that number's instances" (~47k), but never settles on counting the shapes. The legend order it gives (8,3,2,1) is also wrong.
- End: "Given the time I've spent (over 2 hours in simulation)... I'm out of time. I'll output a generic answer."
- Shape: it guessed 9x13 and missized the box width (11).

**e376de54 p0** (diagonal lines must all match the length of the reference line, 16x16).
- Perception: it drops the 2 at (5,2) when listing Ex0's non-7 cells. It claims "Example 2 output: it's identical" to its input, which is false (lines are extended and cut).
- Rule: never found it ("same length": 0 hits). It builds per-row and per-colour shift rules: "If i is row6 or row7: 1s vanish, 2s gain a new cell...".
- End: "I'm out of time and mental energy. I'll output the test input as the output".
- Shape: the copy dropped one row and garbled rows 10-14, giving 15x16.

**db695cfb p0** (join same-colour 1-pairs with a diagonal; each 6 on that line shoots a perpendicular diagonal; 25x21).
- Perception: "Input 20x12 grid" for Ex0 (15x12). For the test it counted columns only: "I'll assume it's a 21x21 grid" (32k). It never counts the 25 rows.
- Rule: partial. It finds that 1s connect along shared diagonals (~90k) but never the perpendicular 6-rays ("perpendicular": 0 hits).
- End: it writes out its half-rule as a 21x21 grid, with a long passage about formatting.
- Shape: 21x21. Its belief about the test size was wrong from the start.

**5961cc34 p0** (a beam from the 4/2 arrow runs up and is redirected by each shape's 3-marker; 27x27).
- Perception: severe. "Input had 13 rows. Output has 12 rows. Something changed row count" (1.2k; both are 13x15). Later it says outputs have 24 rows against 22-row inputs. It counts the test width as 20, then 26, then 24.
- Rule: partial. It gets "vertical line of 2s in the 4's column" but no redirection ("beam": 0 hits, "direction": 1).
- End: "Given the ambiguity, I'll output the vertical line version because it's simpler and matches Ex0".
- Shape: it said 23x20 and emitted 21x20. The true size is 27x27.

**cbebaa4b p0** (assemble the pieces by matching 2-connectors; 26x26).
- Perception: "The grids are 22 rows x 24 columns". Its transcribed output rows have 17-23 elements. At ~85k it notices "There's inconsistency in my counting".
- Rule: never found it ("connector"/"attach"/"assembl": 0 hits). It tries die nets, rotation and value maps.
- End: "I'll output all zeros... I'll go with 22." (the test is 26x26).
- Shape: it intended 22x22 zeros and emitted 23x23 zeros.

**f931b4a8 p0** (tile the pattern quadrant, with 0s filled from another quadrant, to a size the other quadrants determine; 10x8).
- Perception: mostly right on dimensions, but gives Ex2 as "17 rows" (16).
- Rule: never found it. It tries kept-value subsets, XOR, Minesweeper counts and "sum of row and column modulo something".
- End: "Given the absolute lack of time, I'll output... 4 rows of [2,3,5,1]... But I'm not confident."
- Shape: 4x4, an unmotivated guess.

Flash, by contrast, on db695cfb (`arcprize_deepseek-flash_max_traces/db695cfb_p0_a0`): it reads Ex0 as 15x12 with every cell coordinate correct within the first 1k chars and has the connect-and-perpendicular rule by ~5k chars. Its whole trace is 60k chars; Nemotron's is 188k.

## Patterns (counts out of 10)

1. **Rule never found, or only part of it: 10/10.** None of the 10 had the full rule. 3 had a partial one (58490d8a, db695cfb, 5961cc34). 7 never got close, and none of those 7 named the key concept (symmetry, equal length, legend counting, beam, assembly).
2. **The answer is a declared give-up or guess: 10/10.** 2 return the input unchanged (981571dc, e376de54), 3 return a constant fill (8e5c0c38 x2, cbebaa4b) and 5 return a low-confidence half-rule. Not one ended confident.
3. **Imagined time pressure ends the search early: 10/10.** Phrases like "given the extreme time pressure", "absolute time limit" and "out of time" appear 6-14 times per trace. They first appear at 35-80% of the trace, and the model stops at 62-82k tokens out of a 200k budget. The prompt mentions no time limit.
4. **Wrong grid dimensions read from the prompt: 8/10.** Examples: 25x25 for 22x22, 21x22, 20x25, 20x12, "13 rows -> 12 rows", 22x24, 17 rows, 30x30 for 29x29, and a test assumed to be 21x21. It counts elements in the flattened JSON again and again ("Let's count" appears 7-46 times per trace) and gets different answers.
5. **The emitted grid does not match the model's own intended size: 7/10.** It miscounts long constant rows (8e5c0c38 x2, cbebaa4b), drops rows while copying the input (981571dc, e376de54), or emits a different number of rows than it stated (20a9e565, 5961cc34).
6. **Treats the grids as rows of integers rather than 2-D pictures: ~7/10.** It analyses row by row or segment by segment and tries arithmetic (mod 10, XOR, linear congruential generators, digit maps). Objects and symmetry are rarely considered (981571dc, 8e5c0c38 x2, e376de54, f931b4a8, 20a9e565, cbebaa4b).
7. **Token use**: about 60-70% of each trace is re-transcribing and re-indexing grids ("Wait" appears 23-51 times per trace) and cycling hypotheses without pruning any. Verification only happens against single rows.

The wrong shape is a symptom, not the cause. In all 10 cases the content would be wrong even at the correct size, because 9 of the 10 answers are a placeholder or a half-rule.

## Verdict

- Harness changes would only fix the shape: sending the grids as compact digit rows with explicit "HxW" headers, or rejecting and re-asking when an answer has the wrong shape.
- That would fix about 5/10 shapes and roughly 0/10 answers, because the model gave up without the rule in every case. It might also curb the imagined time pressure (patterns 3-4).
- The real cause is a capability gap: the model does not see 2-D structure and has no working search strategy. A prompt or harness change is unlikely to move it off 0%; tool use (Python grid parsing) is the only change with a plausible chance of helping, and only with its perception.
