# E017 phase 1B: how Flash reasons in the tool harness, and where the tokens go (no API cost)

## Question
In our best tool harness (grid mode, low effort, `lean2sub`, `submit_answer`, budget notes; E009 sub80s40 and E009/E011 sub96n), where does DeepSeek V4.1-Flash spend its reasoning, where does it go wrong, and which harness or prompt changes would give more correct answers per token?

## Data and method
- **Population:** all 835 scored attempts of the two full-set runs:
  - sub80s40: 3 samples, 501 attempts;
  - sub96n: 2 samples, 334 attempts.
  - Outcomes: 559 correct, 221 wrong answer, 55 no answer.
  - **Wrong answers take 37% of all output tokens** (mean 58.9k per attempt). Correct attempts take 51% (mean 32.4k), no-answer attempts 13% (mean 82.4k).
- **Offline replay of every attempt** (`scripts/replay_probe.py`, local sandbox, no model calls):
  - every `run_python` cell is re-executed in a fresh session;
  - after each cell, every function named `solve*`/`transform*`/`apply*`/`predict*` is probed on the train pairs and the test, and `answer`/`ans` is compared with the truth (hashes only, so the truth never enters the session);
  - 99.1% of re-run cells reproduce their original output exactly.
  - **Probe blind spot:** functions with other names (`simulate`, `solve_panel`), tuple returns and inline checks are missed. It finds a train-passing function in 76% of correct attempts, so treat its train-pass counts as lower bounds.
  - Output: `replay.jsonl`, `signatures.json`.
- **Reading sample:** 50 attempts (`scripts/sample.py`, seed 2017, one attempt per task within a bucket). Each was annotated by a Claude reader against `annotation_guide.md`, working from the rendered trace (`rendered/`), a context packet (`packets/`: rule references, a correct DeepSeek program, the answer diff and the replay timeline) and the task JSON.

  | Bucket | Drawn | Annotated |
  |---|---|---|
  | (i) wrong answer on a pair another attempt solved | 20 | 18 |
  | (ii) failed on a pair no config ever solved | 10 | 10 |
  | (iii) correct and over 50k tokens | 10 | 10 |
  | (iv) correct and under 20k tokens | 10 | 10 |

  - Bucket (ii) covers 5 of the 7 never-solved pairs.
  - Bucket (iv) was chosen as same-task contrasts where possible. Eight tasks have both a wrong and a correct-short attempt.
  - **Two attempts are missing:** S14 (b6f77b65) and S18 (3dc255db). The reader was stopped twice by an API content filter, so they are not annotated.
- **Per-attempt annotations:** `annotations.jsonl`, 48 lines with verbatim quotes. Each records:
  - token share by category;
  - when the correct idea first appears;
  - perception errors;
  - whether it checked train, and whether that check would have caught the error;
  - contradictions it ignored;
  - whether it used test_input;
  - failure mode and best intervention.
- **Caveats:**
  - Each attempt has a single reader.
  - Token shares are readers' estimates from per-turn token counts.
  - Intervention votes are opinions, and one of them (d) is contradicted by the population simulation below.

## Findings

### 1. Tokens go to prose discovery, not to code or post-success checking

Token share by category (token-weighted, %):

| Bucket | Perception (reading or transcribing grids, hand counting) | Hypothesis search in prose | Code writing | Debugging | Verification before a pass | After the first train pass | Other | Reader-estimated waste |
|---|---|---|---|---|---|---|---|---|
| Wrong, solvable (18) | 18 | 58 | 10 | 2 | 5 | 2 | 4 | 47 |
| Never solved (10) | 18 | 53 | 10 | 3 | 5 | 2 | 9 | 46 |
| Correct, long (10) | 19 | 51 | 11 | 6 | 8 | 4 | 1 | 46 |
| Correct, short (10) | 28 | 35 | 13 | 4 | 13 | 8 | 0 | 23 |
| **Population-weighted estimate** | **20** | **51** | 11 | 4 | 7 | **4** | 3 | **42** |

- **"Re-deriving after success" is already small.** In the replay, the first train-passing function appears at a median 95% of a correct attempt's tokens. Only 6.8% of correct-attempt tokens come after it. E009's `submit_answer` hand-off already fixed this.
- **Debugging is small too:** 0.4–0.5 tool errors per attempt.
- **What remains is the discovery phase.** About 70% of tokens go to reading grids and searching for hypotheses in prose before any code tests them. Readers judged about 42% of all tokens wasted: re-transcribing grids already printed by code, circular re-derivation, and prose geometry that a code cell settles.
- **Long prose turns predict failure** (population, replay):

  | Longest single turn | Correct |
  |---|---|
  | under 10k | 81% (n=333) |
  | 10–20k | 66% |
  | 20–30k | 56% |
  | 30k or more | **26%** (n=76) |

  - Turns over 15k tokens hold 28% of all tokens, and 53% of tokens in no-answer attempts.
  - Hand-transcribed grids show the same gradient. Attempts with 40 or more digit-row lines in their reasoning were 51% correct (60k tokens on average); attempts with none were 76% correct (33k).
  - Both of these are correlations: harder tasks cause both.
  - Quotes: S19 spent 98% of its tokens in one reasoning turn ("No code at all in the first 98%"). S36: "I'm going in circles." (53%). S32: 37% of its tokens went to hand transcription before the first tool call.

### 2. The right core idea usually appears early; the full rule is what's missing
- **When the true core idea first appears** (median % of the attempt):
  - wrong-solvable: 16%;
  - never-solved: 23%;
  - correct-long: 16%;
  - correct-short: 12%.
- **The full correct rule** was reached in 2/18 wrong-solvable and 0/10 never-solved attempts, against 20/20 correct ones.
- So failures are rarely "missed idea". As in E007, the model has the concept and fails at a sub-rule, a role binding, or a test case train doesn't show:

  | Failure mode (28 failing attempts) | Count |
  |---|---|
  | **train_overfit** (rule fits every train pair, wrong on test) | **12** |
  | partial_rule | 7 |
  | missed_idea | 4 |
  | ambiguous (train underdetermines it) | 2 |
  | perception | 1 |
  | execution_bug | 1 |
  | gave_up | 1 |

- **Ideas found and then dropped**, often because of a hand miscount:
  - S01: "Interesting pattern by area pairs!" (39%), dropped after miscounting an area as 11 instead of 7.
  - S03: "each object is a 'mirror' that reflects the incoming ray" (53%), dropped.
  - S28: "The 8s form a shape that acts as a 'stamp'" (7%), rejected at 69% ("Not stamping.") because its check ignored copies clipped at the border.
  - S21: it saw the dots being eaten at 50% and the figure hopping onto dots at 74%, with no budget left.

### 3. Perception errors are common and costly
- **Frequency:** 24 of 28 failing attempts and 16 of 20 correct attempts had at least one perception error: a misread cell, a miscount, a wrong object size, or a misread train output.
- **Errors that decided the outcome:**
  - S17 misread a column of 5s as 0s, which led it to reject the true push/gravity rule.
  - S01's area miscounts (above).
  - S08's 3×3 template detector could not see a frame clipped by the grid edge. Its printed answer still showed `7 2 7`, and it wrote "The printed ans looks right."
  - S27 misread ex1's 7-domino, against its own printout two turns earlier.
- **Contrast with the correct attempts.** The short correct attempts tabulated objects with code early. S42 recorded bars as `[a,b]` spans and S48 used connected components, and both generalised to the odd test object. Their wrong siblings, S00 and S08, used rigid templates and symmetric radii.

### 4. The test usually needs a case train never shows, and the wrong attempts don't audit for it
- **The test is almost always novel.** In 155 of 167 pairs the test input falls outside the train inputs' range on at least one cheap feature (size, colours, number of components, border-touching objects, object sizes; `scripts/test_novelty.py`). Readers judged that the test needs something beyond train in 23/28 failing attempts, and also in 7/10 short-correct ones.
- **The difference is what the attempt did about it.**
  - All 10 short-correct attempts inspected test-specific properties after passing train. S42 printed each test bar's centre, saw 7.5, and switched from radii to spans. S48 noticed the 2-row clipped object and chose a component detector. S46 printed per-object counts on the test and checked the pairing was unique.
  - The train-overfit failures trusted their train-perfect solver. S05: "let me not manually verify; the algorithm matched training." S08: "The printed ans looks right."
- **Contradictions were noticed and set aside** in 23/28 failing attempts, against 2/20 correct ones:
  - S06: "All training pairs reproduce with my rule (ex0 differs only in an irrelevant border detail…)";
  - S19: "the test A does NOT match the 'axis+pointer' structure… my inference of the rule must be wrong", then submitted anyway;
  - S36 (correct by luck): "ex1 fails… But for the test, only one frame";
  - S04: "Given ex1 fails any simple rule, my answer will likely be wrong";
  - S22: "Clearly scene A far arm going down-right is wrong", then submitted at the cap.
- **Calling things noise is an early warning sign.** S04 labelled the marker dots "noise" at 17% ("I think noise is random. Let's move on."), and they were the key. S05 called the 9 blocks "unrelated". S12 called the template "irrelevant to shape".

### 5. Population signatures, and why early stopping or restarting does not pay
- **Train pass predicts success.**
  - P(correct) is 83% when a probed train-passing function ever exists, against 42% when none does. The second figure is inflated by probe misses.
  - Attempts that ended before 75% of the cap with a probed train-passing function: 399/463 correct (86%).
- **Late attempts are mostly forced guesses.**
  - 196 attempts (23%) passed 75% of their cap with no probed train-passing function.
  - They use **43% of all output tokens**, and only 32 of them (16%) end correct.
  - They contain 117 of the 221 wrong answers. These are mostly guesses submitted at or after the "Budget nearly spent" note (S10: "did well on 3/4"; S28: "this guess is almost certainly wrong").
- **Even so, stopping or restarting them costs accuracy.** Correct attempts reach their first train pass late (deciles 73–99% of their tokens). Among attempts still running at X tokens with no train pass yet:

  | X | Eventually correct |
  |---|---|
  | 20k | 58% |
  | 30k | 51% |
  | 40k | 43% |
  | 60k | 26% |

- **Restart simulation** (`scripts/restart_sim.py`): an attempt with no train pass by X is replaced by a fresh sample of the same pair from the 5-attempt pool, given the rest of an 80k budget.
  - The result is −7 to −12 pp pass@1 at +5 to +17% tokens, for every X from 20k to 50k.
  - The simulation is optimistic for restarts: a fresh attempt's success is credited only when it finishes within the remaining tokens, and the pool is the same pair.
  - So **intervention (d), which readers chose for 4 attempts, is rejected on population data.** The "no train pass yet" signal arrives too late to separate slow successes from dead ends.
- **A judge has little to choose from.** Only 3 of the 221 wrong attempts ever held the correct test grid in their session, and S19 is the vivid case: it computed the right panel at 88% and rejected it as "ad hoc". An LLM critic choosing between an attempt's own candidates (e) has little to work with. This matches E011, where a diverse second attempt did not beat a duplicate.

### 6. What each failure needed

Readers' best-intervention votes:

| Intervention | 28 failing | 20 correct (to cut tokens) | Total |
|---|---|---|---|
| (a) computed task summary / perception aid | 8 | 13 | 21 |
| (c) state roles and legend first | 8 | 1 | 9 |
| (b) helper library with `check(solve)` and diffs | 3 | 6 | 9 |
| (d) verifier-gated restart | 4 | 0 | 4, rejected (section 5) |
| (e) critic or judge | 4 | 0 | 4 |
| (g) capability limit | 1 | 0 | 1 |
| (f) more budget | 0 | 0 | 0 (it was the second choice 3 times) |

- **A train check would have caught** the error in 10/28 failing attempts, but not in 18/28. Those 18 are mostly train-overfit or ambiguous cases. This matches E011, where `--verify-submit` bought nothing.
- **Do the two attempts on never-solved pairs fail the same way?** For 88e364bc (S24, S25), yes: both use the same stop rule, which train underdetermines (one is labelled ambiguous, the other train_overfit). abc82100 (S28, S29) and a32d8b75 (S26, S27) fail for the same conceptual reason in each pair: an unassigned legend role, and the 4-versus-7 key.
  - S24 and S25 go wrong only on whether a diagonal slide may squeeze past a wall corner. The only fix readers see there is (g) or (e).

## Proposals (ranked by expected gain per token)
Each is opt-in and testable as a variant of sub80s40. The decision test is the same for all:
- 2 full-set samples, paired against sub80s40's 3 samples with `evals/paired_vs_baseline.py` (pass@1, pass@2, tokens per attempt, correct-rollout p50/p90);
- the replay probe (`scripts/replay_probe.py`) re-run on the new traces, to check the mechanism metrics: grid-like reasoning lines, longest-turn tokens, contradiction-like submits, first-train-pass position.
- **Cost:** 334 attempts × ~41k output × $0.60/M ≈ $8.2, plus ~15% input ≈ **$9.4 per variant** (sub80s40 recorded $4.50 per sample). A combined run of P1+P2 would be about $9.5 too. The AGENTS.md rule applies: ask before anything over $10.
- **No dev-set screening.** E009 showed that 56-pair dev winners regress on the full set.

### P1. Computed perception pack in the prompt (intervention a): `--task-summary`
- **Evidence:**
  - top vote (21/48);
  - perception is about 20% of tokens and 28% in short-correct attempts;
  - 24/28 failing attempts had perception errors, and in at least 4 of them the error decided the outcome (S01, S08, S17, S27);
  - grid-like reasoning lines track failure (76% → 51% correct).
- **Price:** it costs input tokens, which are mostly cache hits at $0.003/M after turn 1.
- **Spec:** host-side code, no model involved, appended after the ARC Prize prompt, with a hard cap of about 3k tokens.
  - **Per pair, one header line:**
    `Ex0: in 10x12 bg=2 -> out 10x12 bg=2 | 5 cells changed (4->8 x5) | colours in {2:96,4:24} out {2:96,4:19,8:5}`
  - **Per input grid, an object table:** 8-connected same-colour components plus multi-colour non-background components, at most 12 rows, the rest summarised per colour. Each row is `id colour size bbox=(r0,c0)-(r1,c1) holes=k border=yes/no shape=<D8-canonical hash>`.
    - The shape hash lets "same shape up to rotation" be read off directly.
    - Separator lines or panels are reported when found, as `panels: 3x1 split by colour 5 at rows 4,9`.
  - **Per pair, matching:** each input object is labelled unchanged / moved (dy,dx) / recoloured a→b / transformed (D8 op) / removed; output objects with no match are listed as new.
  - **For the test input:** the same object table, plus a line `Outside train range: size 30x30 (train ≤ 22x22); new colour 6; 2 objects touch the border (train: 0); component count 14 (train 5–9)`. This uses the features in `scripts/test_novelty.py`.
  - **One prompt line:** "A computed summary of the grids follows; trust it over reading the grids by hand, and use code for any count."
- **Cheap pre-test (free):** for the 28 annotated failures, check whether the summary's facts contain the decisive quantity. Readers named several: area pairs (S01), hole components (S04), per-object counts (S11), cell count kept unless clipped (S22, S23), clipped frame (S08), "test is 30x30" (S26).

### P2. Roles-first and test audit, in the prompt (intervention c, plus the audit habit of the correct attempts): `--prompt lean2roles`
- **Evidence:**
  - train_overfit is the largest failure mode (12/28);
  - contradictions were ignored in 23/28 failing attempts against 2/20 correct;
  - all 10 short-correct attempts audited the test after passing train;
  - early "noise"/"irrelevant" labels preceded failure (S04, S05, S12);
  - 9 best votes and 10 second votes for (c).
- **Price:** about 1–2k extra output tokens per attempt.
- **Spec:** append to `lean2sub`:

  > Before writing solve(), write a short ROLES note: what each kind of object or colour does (template, marker, key/legend, obstacle, container, background). Do not call anything noise or irrelevant unless code shows it is removed or unchanged in every training output. When solve() reproduces every training output, print what is different about test_input compared with all training inputs (sizes, colours, objects clipped by the border, counts, key entries, shapes not seen) and say in one line how solve handles each difference; if your rule ignores some non-background cells of test_input, explain them before submitting. If any training pair still fails, or a test difference is unhandled, do not submit: fix the rule.

- **Optional combination with P1:** the harness can print the "Outside train range" line itself on the first `submit_answer` call, so the audit costs no reasoning when nothing is novel.

### P3. Session helper library with compact `check(solve)` (intervention b): `--helpers`
- **Evidence:**
  - 9 best votes and 13 second votes;
  - E015 found that BFS components, bounding boxes and mirrors are re-implemented in nearly every program;
  - hand geometry dominates the long correct attempts (S38, S33, S31, S39);
  - S13 failed on a hand-written search bug, and S28 rejected the right idea because its own check ignored border clipping.
- **Spec:** preloaded in the session as plain numpy, and listed in the tool description in about 300 tokens:
  - `objects(g, by='colour'|'nonbg', diag=True)` → list of dicts (colour, cells, bbox, size, holes, touches_border, shape_key);
  - `bbox(mask)`, `crop(g, bbox)`, `d8(g)` (all 8 transforms), `match_d8(a, b)`, `panels(g)` (split by separator lines);
  - `stamp(g, pattern, at, clip=True)`;
  - `ray(g, start, dir, stop=lambda v: v!=bg)`;
  - `check(fn)` prints one line per train pair, e.g. `pair 1: 3 cells differ (r,c got/exp): (4,9) 8/4, (5,8) 8/4, (5,10) 8/4`, or `shape 5x5 vs 6x6`, then `ALL PASS` (8 diffs at most).
  - One prompt line: "Use `check(solve)` after every change to solve."
- **Two side benefits:** `check` gives the replay probe a canonical name to look for, and it makes `--verify-submit`-style confidence (82% vs 39% precision, E011) available for answer selection.

### P4. LLM-free invariant critic on submit (cheap e): `--invariant-check`
- **Evidence (`scripts/invariant_critic.py`, offline on the 780 answers):**
  - Invariants that hold on every train pair: same shape, input non-background cells kept, only background cells change, some change happens, palette ⊆ input palette, colour counts conserved, output not smaller.
  - Applied to the submitted test answer, they flag **19 wrong and 3 correct** (precision 0.86, 9% recall on wrong answers).
  - The full 13-invariant set flags 37 wrong and 52 correct, so it is unusable.
  - **Caveat:** the subset was selected on these same answers, so there is a winner's-curse risk.
- **Spec:**
  - on the first accepted `submit_answer`, compute the kept invariants;
  - on a violation, reject once with one line, e.g. `[check: in every training pair the input's non-background cells are unchanged; your answer changes 6 of them. Fix, or resubmit the same expression to confirm.]` (the same mechanics as `--verify-submit`);
  - it fires on about 3% of attempts.
- **Expected gain:** at most +1 pp pass@1, if about half of the flagged answers get fixed, for about 1–2k tokens on the 3% of attempts it fires on. It is cheap enough to bundle with P1.

### P5. Code before long prose (process nudge): `--prompt lean2code`
- **Evidence:** correlational only. Correctness drops from 81% to 26% as the longest single turn grows; turns over 15k hold 28% of all tokens; readers repeatedly flag 10–35k-token prose turns that one code cell would have settled (S06, S19, S36, S38, S16).
- **Spec:**
  - append "Test each idea with code within a few hundred words of thinking: print a compact table or diff rather than reasoning over grids in your head; never transcribe grid rows into your reasoning.";
  - optionally lower `--max-tokens` per turn from 64k to 24k so that a runaway turn ends, while keeping the hard total.
- **Rank:** last, for two reasons. E009's `lean2brief` brevity prompt cost accuracy, and a turn ending at its cap without a tool call needs care: the harness then continues without a user turn, and that path is untested at scale.

### Not recommended (from this data)
- **(d) restart when there is no train pass by X:** simulated at −7 to −12 pp pass@1 for X from 20k to 50k.
- **(e) an LLM judge among an attempt's own candidates:** the correct grid almost never appears (3/221).
- **(f) more budget:** E011 already found the long tail. Attempts that pass 75% of the cap with no train pass are 16% correct.
- **Hard per-attempt caps on late attempts:** about 19% of all tokens are spent beyond 50% of the cap in late no-pass attempts. Cutting there would save those tokens but lose up to 32 correct attempts (about −4 pp), the same trade-off as E009's cap table.

## Per-attempt annotations (detail in `annotations.jsonl`)
Column meanings:
- **idea@%**: where the true core idea first appears, as % of the attempt's tokens;
- **pass@%**: where a solution first reproduces every train pair;
- **waste%**: the reader's estimate of wasted tokens;
- **interv**: best and second-best intervention.

| sid | task.pair | bucket | tokens | failure mode | idea@% | pass@% | waste% | interv |
|---|---|---|---|---|---|---|---|---|
| S00 | 31f7f899.0 | wrong solvable | 30k | train_overfit | 10 | 96 | 55 | b/e |
| S01 | 62593bfd.0 | wrong solvable | 84k | missed_idea | 39 | – | 55 | a/d |
| S02 | 36a08778.0 | wrong solvable | 45k | train_overfit | 31 | 99 | 55 | c/e |
| S03 | 142ca369.1 | wrong solvable | 67k | missed_idea | 21 | – | 60 | d/c |
| S04 | 8698868d.0 | wrong solvable | 84k | missed_idea | 32 | – | 60 | d/a |
| S05 | a47bf94d.0 | wrong solvable | 37k | train_overfit | 6 | 93 | 30 | e/a |
| S06 | 271d71e2.0 | wrong solvable | 75k | partial_rule | 18 | – | 35 | d/f |
| S07 | 20270e3b.1 | wrong solvable | 69k | partial_rule | 6 | – | 45 | c/a |
| S08 | d35bdbdc.2 | wrong solvable | 16k | perception | 3 | 89 | 45 | b/a |
| S09 | 4c416de3.0 | wrong solvable | 49k | train_overfit | 6 | 93 | 30 | c/e |
| S10 | 2d0172a1.0 | wrong solvable | 84k | partial_rule | 38 | – | 55 | a/c |
| S11 | b0039139.0 | wrong solvable | 22k | train_overfit | 38 | 99 | 35 | a/c |
| S12 | dfadab01.0 | wrong solvable | 27k | train_overfit | 4 | 98 | 22 | c/d |
| S13 | 446ef5d2.1 | wrong solvable | 65k | execution_bug | 14 | – | 45 | b/a |
| S15 | 16de56c4.0 | wrong solvable | 43k | train_overfit | 1 | 71 | 30 | e/d |
| S16 | fc7cae8d.0 | wrong solvable | 91k | train_overfit | 15 | – | 60 | d/c |
| S17 | 4c3d4a41.1 | wrong solvable | 18k | train_overfit | 20 | 98 | 35 | a/c |
| S19 | 7491f3cf.0 | wrong solvable | 58k | train_overfit | 22 | 99 | 45 | e/c |
| S20 | 5545f144.0 | never solved | 77k | gave_up | 23 | – | 55 | a/g |
| S21 | 5545f144.0 | never solved | 76k | partial_rule | 50 | – | 40 | c/f |
| S22 | 88bcf3b4.1 | never solved | 78k | train_overfit | 17 | 96 | 45 | a/f |
| S23 | 88bcf3b4.1 | never solved | 89k | partial_rule | 3 | – | 55 | a/b |
| S24 | 88e364bc.0 | never solved | 22k | ambiguous | 41 | 87 | 30 | e/g |
| S25 | 88e364bc.0 | never solved | 33k | train_overfit | 18 | 96 | 20 | g/c |
| S26 | a32d8b75.0 | never solved | 71k | partial_rule | 74 | – | 45 | a/c |
| S27 | a32d8b75.0 | never solved | 65k | ambiguous | 23 | 95 | 30 | c/a |
| S28 | abc82100.0 | never solved | 63k | missed_idea | 7 | – | 55 | c/b |
| S29 | abc82100.0 | never solved | 81k | partial_rule | 33 | – | 55 | c/b |
| S30 | f931b4a8.1 | correct long | 55k | none | 83 | 99 | 45 | a/b |
| S31 | db0c5428.0 | correct long | 85k | none | 10 | 100 | 42 | b/a |
| S32 | 9385bd28.0 | correct long | 58k | none | 6 | 91 | 30 | a/b |
| S33 | a25697e4.1 | correct long | 64k | none | 30 | – | 40 | b/a |
| S34 | 4a21e3da.0 | correct long | 66k | none | 26 | 96 | 50 | a/c |
| S35 | c7f57c3e.0 | correct long | 55k | none | 13 | 99 | 40 | a/b |
| S36 | e8686506.0 | correct long | 58k | none | 18 | – | 65 | a/d |
| S37 | 89565ca0.0 | correct long | 64k | none | 62 | 98 | 55 | a/c |
| S38 | 2c181942.0 | correct long | 52k | none | 10 | 94 | 40 | b/a |
| S39 | 0934a4d8.0 | correct long | 65k | none | 1 | 10* | 50 | a/b |
| S40 | 16de56c4.1 | correct short | 19k | none | 10 | 96 | 18 | a/b |
| S41 | 4a21e3da.0 | correct short | 19k | none | 15 | 76 | 22 | c/a |
| S42 | 31f7f899.0 | correct short | 17k | none | 61 | 72 | 30 | a/b |
| S43 | dfadab01.0 | correct short | 18k | none | 2 | 96 | 20 | a/b |
| S44 | 4c3d4a41.1 | correct short | 18k | none | 16 | 99 | 30 | a/b |
| S45 | 36a08778.0 | correct short | 15k | none | 29 | 93 | 30 | b/– |
| S46 | 8698868d.0 | correct short | 16k | none | 6 | 93 | 20 | a/b |
| S47 | fc7cae8d.0 | correct short | 20k | none | 45 | 96 | 20 | b/a |
| S48 | d35bdbdc.2 | correct short | 14k | none | 4 | 97 | 30 | a/b |
| S49 | e8686506.0 | correct short | 18k | none | 9 | 100 | 12 | b/a |

\* S39 passed train at 10% with a rule that cannot apply to the test (its hole sits on the border). About 70% of its tokens then went to recovering the hidden transpose rule.

**Notes for the reference rules:**
- **142ca369:** R08 found the E007 and E015 rules both wrong. Only the L-shapes emit rays; a ray bounces off bars and dots and takes their colour. This was checked with an independent simulator.
- **abc82100:** R18 found the anchor offset missing from the rule. Its rule reproduces the train pairs and the test.
- **88bcf3b4:** R16 added clipping at the grid edge.
- **a32d8b75:** R17 found the 4 sets the corner and the 7 sets the rotation, with crossed placement.
- **88e364bc:** the E015 reference program uses the same diagonal stop rule as the failing attempts, so it would fail this pair as well.

## Files
- **Scripts** (`scripts/`), run with `.venv/bin/python -I`. None makes paid calls.

  | Script | What it does |
  |---|---|
  | `index_attempts.py` | Indexes all 835 attempts → `attempts_index.jsonl` |
  | `replay_probe.py` | Offline replay with the probe → `replay.jsonl` and `replay_sample.jsonl`; needs the local sandbox URL |
  | `sample.py` | Draws the reading sample → `sample.jsonl` |
  | `render.py` | Renders sampled traces → `rendered/` |
  | `packets.py` | Builds reader context packets → `packets/` |
  | `aggregate.py` | Merges annotations → `annotations.jsonl`, plus summaries |
  | `signatures.py` | Population timeline signatures → `signatures.json` |
  | `restart_sim.py` | Restart-policy simulation |
  | `invariant_critic.py` | Critic test → `invariant_flags.json` |
  | `test_novelty.py` | Test-versus-train feature ranges |

- **Annotations:** `annotation_guide.md` (the readers' instructions and schema), `annotations/R*.jsonl` (raw, per reader) and `annotations.jsonl` (merged).
