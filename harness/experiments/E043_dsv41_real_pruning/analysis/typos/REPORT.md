# E043 typo analysis: why expert-pruned DeepSeek-V4.1-Flash-1017 misspells identifiers

Read-only, offline, CPU-only analysis of the tool-harness traces (2026-10-26). Scripts and outputs are in this folder. `report_tables.py` regenerates every table into `out/tables.md` (T1–T11).

## Summary
- **The extra errors are surface-form errors, not reasoning errors.**
  - Identifier errors per 100 `run_python` calls: full model 0.2–0.4; E128 8.2, E112 18.7, E96 22.8, E72 31.3.
  - Logic errors (Value/Index/Key) stay at the full-model level: 1.1–1.7 vs 0.9–1.4.
  - 53% (E128) to 86% (E96) of SyntaxErrors carry a typo or self-repair marker on the failing line, or a `{` closed by `]`.
  - The full model makes 13 typos in 60,084 training-rollout calls; E128 makes 293 in 5,374 calls (about 250×).
- **A typo is a token-boundary event inside a multi-token word.**
  - 90–94% of intended names are multi-token in context, against 14–18% of all identifier uses.
  - The first token is right in 86–92% of typos (uniform-character null: 42–46%).
  - The divergence starts exactly at a token boundary in 73–88%, at token index 1 in 82%.
  - Hazards per 10k positions:

    | model | word-internal boundary splitting two letters (`com\|ps`) | `_` boundary | first token |
    |---|---:|---:|---:|
    | E128 | 26.6 | 1.3 | 0.1 |
    | E96 | 84 | 10 | 0.8 |
- **Mechanism (strong evidence): the continuation token is not conditioned on the fragment that was actually emitted.**
  - BPE splits the same name differently depending on the character before it: ` comps` = `Ġcomp|s`, `(comps` = `( com|ps`.
  - The model emits the first piece of one split and the continuation of another (`com`+`s`).
  - After a merged punctuation+letter token it restarts the word (`(m`+`mask`, `.m`+`most_common`).
  - This split-mixing explains 31–49% of all typos, against a 0.2% chance rate.
  - `comps` after a space: 0 typos in 5,971 occurrences. After punctuation: 15% (E128) to 81% (E96) of attempts are typos.
  - Re-tokenized copies are 4.3× (E72) to 12.8× (E112) riskier than token-identical copies.
- **The model notices its slip immediately.**
  - It writes `X if False else Y` in-line repairs in 7–9% of calls (full model 0.7–1.2%), plus `?` markers and restarts. These markers cause most SyntaxErrors.
  - Reading its own output works; emitting the continuation fails.
  - Prose shows it too: near-miss misspellings in reasoning are 3.5× (E128) to 11× (E72) the full model's rate.
- **Cost is turns, not tokens.**
  - Turns that follow an identifier typo: E128 5.3%, E112 13.0%, E96 15.3%, E72 17.7%. These turns hold only 1.4–4.7% of tokens (E72 10.3%).
  - 23–40% of next calls keep the typo, and about half of typos repeat one written earlier.
  - With task and model controlled, attempts with typos do not fail more often: two-way fixed-effects β = +0.012 (95% CI −0.045 to +0.063).
- **A harness fix works today.**
  - Rule: after an identifier error, rewrite the unknown name to the unique close match (edit distance ≤ 2) and re-run once.
  - Replayed in a local session, the cell then completes in 61% (E128), 52% (E112) and 45% (E96) of reproduced identifier errors. That avoids about 0.75 (E128) to 2.3 (E96) error turns per attempt.
  - The rewrite equals the model's own next-turn fix in 89–93% of cases; working code is never touched.

## 1. Data and method

| label | run (prefix `tools_persistent_grid_lean2sub_`) | attempts | run_python calls |
|---|---|---:|---:|
| API-full | `deepseek-flash_low_e030_hdrmin` | 334 | 4,327 |
| SH-full | `dsv41_low_e043_full_ref_sub80` (162 errored records dropped) | 172 | 2,003 |
| E128 | `dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80` | 334 | 5,374 |
| E112 | `..._e112t_..._sub80` attempt 0 + `..._sub80_r2` (as attempt 1) | 334 | 6,717 |
| E96 | `..._e96t_w10_u2u3_gptq_eng64_sub80` | 334 | 7,618 |
| E72 | `..._e72_w30_eng64_rescue2` (different config: 60k output cap, soft budget 20k, strong wording, ×30 tool weight) | 167 | 3,100 |
| DEV-E128 | `devs_e128t_t1` / `_minp05` / `_topp95` | 33 / 33 / 32 | 134 / 128 / 134 |
| TRAIN | full-model rollouts `e031_arc2_train`, `e031_nvarc_b0`, `e036_nvarc_b1`, `e036_nvarc_b1x` | 8,754 | 60,084 |

Excluded: `mt64`, `e128t_mt*/pp*/final_all/ctxsmall*`, E112 main attempt 1, and `_r3`.

Scripts:

| script | role |
|---|---|
| `extract.py` | compacts each attempt (turns, reasoning, calls, outputs, exception, usage) |
| `ident.py` | one row per error call: typo, intended name, edit/token class, history of the intended name; plus one row per identifier occurrence (denominators) |
| `splitmix.py` | split-mixing test vs chance |
| `synerr.py`, `synerr2.py` | SyntaxError classes, bracket pairs, merged-bracket test, failing-line content |
| `reasoning_oov.py` | near-miss misspellings in reasoning |
| `tokclass.py` | candidate token classes as prediction targets |
| `repair.py` | static pre-execution rewrite |
| `replay.py` | retry-once repair, replayed in a local session |
| `report_tables.py` | builds every table |

How the intended name is recovered:
- First choice is the model's own fix: the typo line is aligned with the most similar line of the next 1–2 calls.
- Otherwise, the nearest name by OSA (optimal string alignment) edit distance. Candidates: names in the attempt, preloaded names and builtins, or `dir()` of the module/type named in the error.
- Ties, first-character mismatches at distance 2, and `_`-private names are rejected.
- `typo` means the name was never defined in the attempt and is not simply a missing import or a wrong API. Unresolved cases are kept separate (T2).
- The token results hold on the high-confidence subset: first token right 84–94%, boundary 68–90%. They also hold without the `comps` family: 82–90% and 65–84%.

Tokens: V4.1 `tokenizer.json`, applied to the call's code with the preceding character included. DSML code parameters are raw text, so code tokens are the code's own tokens.

Caveats:
- **Canonical re-tokenization, not sampled ids.** About 1 in 8 completions is sampled non-canonically. "Restart" readings such as `(m`+`mask` are inferred; `--token-exact` runs could log the sampled ids.
- **Only typos that raise are observed.** Rates are lower bounds.
- **The replay ran on Python 3.11 / numpy 2.1 / scipy 1.17; the sandbox ran Python 3.10.** 94–99% of recorded identifier exceptions reproduce: E128 410/438, E112 1243/1255, E96 1714/1740, E72 963/969.
- **The sandbox prints "Did you mean" in 0 outputs.** Its IPython plain traceback drops the suggestion.

## 2. Taxonomy (T1–T3)

T1, per 100 run_python calls:

| model | any err | identifier | NameError | AttrError | Import/Module | bad kwarg | SyntaxError | other TypeError | Value/Index/Key | `if False else` | `?` in code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 2.2 | 0.23 | 0.09 | 0.12 | 0.02 | 0 | 0.14 | 0.62 | 1.06 | 0.99 | 0.05 |
| SH-full | 2.6 | 0.40 | 0.15 | 0.25 | 0 | 0 | 0.10 | 0.50 | 1.40 | 1.20 | 0.05 |
| TRAIN | 1.8 | 0.30 | 0.10 | 0.14 | 0.05 | 0 | 0.10 | 0.37 | 0.91 | 0.69 | 0.02 |
| E128 | 14.5 | 8.22 | 5.47 | 1.71 | 0.35 | 0.69 | 3.18 | 1.10 | 1.69 | 7.18 | 0.65 |
| E112 | 30.6 | 18.70 | 10.56 | 3.65 | 3.41 | 1.09 | 9.68 | 0.73 | 1.28 | 8.23 | 2.90 |
| E96 | 38.3 | 22.84 | 11.21 | 3.62 | 7.42 | 0.59 | 13.09 | 0.80 | 1.35 | 8.73 | 4.32 |
| E72 | 46.3 | 31.26 | 11.71 | 6.32 | 10.94 | 2.29 | 12.94 | 0.94 | 1.10 | 8.68 | 4.03 |

T2, identifier errors classified as typos:
- E128 293/442, E112 871/1262, E96 1142/1747, E72 547/989.
- Unresolved (mostly garbled): 81 / 319 / 476 / 384.
- Wrong-API errors are barely elevated: 0–13 vs 4 (API-full).
- TRAIN full model: 192 identifier errors, of which only 13 are typos. The rest are state/scope (62), wrong API (59) and missing import (14).

T3, edit types of pruned-model typos:

| edit type | share |
|---|---:|
| one-character deletion | 37–48% |
| doubled-character insertion | 7–18% |
| segment repeat | 3–12% |
| tail replaced | 9–17% |
| truncation | 6–11% |
| substitution | 5–8% |
| transposition | 0 |

Median intended length: 5 characters.

Top typos, pruned models pooled; tokens are the intended name's tokens in context:

| typo | intended | n | intended tokens |
|---|---|---:|---|
| `coms` | `comps` | 648 | `com` `ps` (` comps` = `Ġcomp` `s`) |
| `scpy` / `scip` / `scpi` / `scopy` / `scscipy` | `scipy` | 166 / 57 / 12 / 12 / 13 | `Ġsc` `ipy` |
| `dequeue` / `deq` | `deque` | 94 / 20 | `de` `que` (` deque` is a single token) |
| `zzip` | `zip` | 59 | `(z` `ip` |
| `mmost_common` | `most_common` | 43 | `.m` `ost` `_` `common` |
| `scipy.nd` / `nd` | `ndimage` | 42 / 20 | `.nd` `image` |
| `dbool` | `dtype=bool` | 18 (+29 mapped to `bool`) | `,d` `type` `=` `bool` |
| `ymmin` | `ymin` | 24 | `ym` `in` (` ymin` = `Ġy` `min`) |
| `ddiff`, `mmask` | `diff`, `mask` | 19, 17 | `(d` `iff`, `=m` `ask` |

E128 has no `scipy` typos (6 ModuleNotFoundErrors in total).

SyntaxError line content (`synerr2.py`):

| model | n | `?` marker | `if False` | word restart (`d dtype=`) | garbled import | `{`→`]`/`)` | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| E128 | 170 | 30 | 35 | 20 | 3 | 2 | 80 |
| E112 | 647 | 178 | 67 | 51 | 95 | 129 | 127 |
| E96 | 994 | 299 | 110 | 26 | 144 | 275 | 140 |
| E72 | 399 | 111 | 53 | 29 | 42 | 53 | 111 |
| TRAIN | 59 | 10 | 13 | 0 | 0 | 0 | 36 |

Examples: `from scpy? no`, `from scipy import ndlabel if False else None`, `bl=np.full((5,5),c,d dtype=int)`.

Bracket mismatches:
- 86–97% are a set/dict comprehension opened with `{` and closed with `]` or `)`: E112 129/139, E96 275/284, E72 53/62; full model 1 in 4,327 calls.
- Example: `C={(r+o[0]+d[0],c+o[1]+d[1]) for r,c in p['C']]`.
- Merged bracket tokens are not implicated: 43–57% of carets fall on a merged token like `])`, vs 47–49% of all closing brackets.

## 3. Token-level mechanism (T4, splitmix)

T4:

| model | typos | intended multi-token | base | first token right | at boundary | alnum\|alnum | per 10k alnum\|alnum boundaries | per 10k `_` boundaries | per 10k first tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E128 | 293 | 94% | 18% | 92% | 88% | 75% | 26.6 | 1.3 | 0.1 |
| E112 | 871 | 90% | 17% | 86% | 73% | 66% | 55.8 | 5.9 | 1.0 |
| E96 | 1142 | 94% | 15% | 91% | 76% | 70% | 84.2 | 10.0 | 0.8 |
| E72 | 547 | 94% | 14% | 91% | 76% | 71% | 120.3 | 5.8 | 0.6 |

- Uniform-character null for the first token: 42–46%.
- Divergence token index, pruned models pooled: index 0: 299, 1: 2,341, 2: 194, 3: 19.
- TRAIN's 13 full-model typos look different: 54% have the first token wrong, and split-mixing explains 0.

Token events (`textutil.token_event`):

| event | E128 | E112 | E96 | E72 | example |
|---|---:|---:|---:|---:|---|
| continuation lost leading chars | 39% | 34% | 43% | 31% | `com`+`s`; `sc`+`py` |
| restart (re-emit word after first piece) | 20% | 17% | 8% | 14% | `(m`+`mask`, `ym`+`min` |
| continuation replaced | 12% | 20% | 22% | 24% | `.fl`+`ravel`; `de`+`queue` |
| tokens dropped / truncated at boundary | 7% | 5% | 7% | 11% | `,d`+`bool` |
| extended after last token | 13% | 6% | 5% | 4% | `placements2` |
| first token wrong | 6% | 13% | 9% | 7% | `np.b`+`unique` |

Split-mixing test:
- Definition: the typo = the first k pieces of the current-context split + a token-aligned suffix of the word's split in any context.
- Explained: E128 139/284 (49%), E112 368/856 (43%), E96 449/1128 (40%), E72 166/543 (31%). Chance from all one-edit variants: 0.17–0.23%.
- Most explained cases need a different segmentation (95 / 257 / 402 / 120); the rest re-emit the word (44 / 110 / 43 / 46).

`comps` by preceding character (typos / correct uses):

| context | E128 | E112 | E96 | E72 |
|---|---:|---:|---:|---:|
| space (`Ġcomp\|s`) | 0/1458 | 0/1734 | 0/2007 | 0/772 |
| `=`, `(`, `,` (`com\|ps`) | 67/391 | 198/74 | 305/73 | 94/51 |

Typos per 10k identifier uses by preceding character (pruned models pooled):

| preceding character | typos per 10k uses |
|---|---:|
| space | 8.8 |
| `=` and other operators | 26.8 |
| `.` | 23.8 |
| `(` `[` `{` | 14.5 |
| `,` | 8.5 |
| line start | 3.1 |

Self-repair:
- `X if False else Y` appears in 386 E128 calls, of which 234 ran.
- Example: `dil=binary_donia if False else binary_dialil if False else binary_deleration if False else binary_dilation(mash if False else mask, structure=st8)`.

Prose (`reasoning_oov.py`), near-miss misspellings per 10k reasoning words:

| model | per 10k words |
|---|---:|
| API-full | 3.7 |
| SH-full | 3.3 |
| E128 | 12.7 |
| E112 | 29.2 |
| E96 | 29.5 |
| E72 | 39.2 |
| DEV-E128 | 7.9–9.1 |

- Examples: `seeparators` (`Ġseep|ar|ators`), `Seggments`, `neeighborhood`, `mslides`, `sturtle`.
- 77–80% have the first piece wrong, because most prose words are single `Ġword` tokens. The wrong short fragment is then completed with the intended word's ending (`Ġb`+`umbers`). This is consistent with the same failure, but that is a hypothesis.

## 4. Copied vs fresh (T5)

Typos per 10k uses of multi-token identifiers:

| model | fresh | copy, same token ids | copy, re-tokenized |
|---|---:|---:|---:|
| E128 | 52.5 | 17.2 | 116.4 |
| E112 | 128.6 | 33.2 | 425.2 |
| E96 | 160.3 | 59.9 | 496.0 |
| E72 | 213.9 | 106.6 | 461.0 |

Copy and propagation facts:
- 82–84% of typos are copies, 75–79% from earlier code. 16–18% are never-seen API names (`most_common`, `popleft`, `scipy`).
- The previous occurrence is close: median 106–126 tokens back; same turn in 53–67%; 3 or more turns back in only 6–10%.
- 46–51% of typo errors repeat a typo already written in earlier code.
- In 26–45% of typos, the misspelled form had already been written in reasoning, in the same turn or an earlier one.

## 5. Context and turn (T6)

- Typos per 100 calls rise with turn for E128 (2.5 at turns 0–2 → 7.8 at turns 20+). E112, E96 and E72 peak at turns 6–9.
- Per word-internal boundary, the hazard falls with context: 102 / 81 / 54 / 39 / 31 per 10k for contexts <15k, 15–25k, 25–40k, 40–60k and >60k tokens.
- Within an attempt, first half vs second half of calls (per 100 calls): E128 4.7 vs 6.7, E112 14.6 vs 13.1, E96 18.6 vs 13.5, E72 20.4 vs 17.5.
- No evidence of a context-length cause.

## 6. Cost (T7, T8)

Next call after a typo:

| model | typo gone | next call completes | same error again | near-copy of the code | repair-turn tokens (median / mean) |
|---|---:|---:|---:|---:|---:|
| E128 | 61% | 55% | 27% | 80% | 439 / 549 |
| E112 | 66% | 46% | 24% | 73% | 379 / 530 |
| E96 | 71% | 43% | 20% | 56% | 278 / 399 |
| E72 | 77% | 39% | 14% | 53% | 301 / 555 |

Share of turns and tokens:

| model | turns after any error | turns after a typo | tokens in those turns | 40-turn stops |
|---|---:|---:|---:|---:|
| API-full | 2.1% | 0% | 0% | 8 |
| E128 | 13.7% | 5.3% | 1.4% | 11 |
| E112 | 29.2% | 13.0% | 4.0% | 33 |
| E96 | 36.9% | 15.3% | 4.7% | 58 |
| E72 | 44.2% | 17.7% | 10.3% | 5 |

Outcome on the same pair:
- Within a model, discordant attempt pairs where the failing attempt had more typos vs the winning one: E128 14 vs 5 (p = 0.06), E112 8 vs 15, E96 15 vs 19.
- Pruned models pooled, pair and model fixed effects:
  - any typo in the first 6 calls: β = +0.012 (CI −0.045 to +0.063);
  - typos per call: β = −0.03 (CI −0.23 to +0.17).
- Typos cost turns, context and turn-cap stops. They do not detectably change per-attempt success.

## 7. Harness fix (T9)

**(A) Static pre-execution rewrite**
- Rewrites 63–71% of typo calls. The rewrite agrees with the model's own fix in 90–95% (E128 54/58, E112 107/113).
- Bare-name rewrites on calls that ran fine: 65 for E128, 53 of them in `if False` dead branches. On the full model: 7 of 60,084 TRAIN calls, 6 in dead branches.
- Method rewrites misfire on 106 full-model calls that ran fine (`.start`/`.stop` of slices and regex matches, `.success`). Use bare names only, if at all.

**(B) Retry once after the error, replayed**

| model | reproduced | retry proposed | cell completes | = model's fix | error turns avoided per attempt |
|---|---:|---:|---:|---:|---:|
| E128 | 410 | 299 (73%) | 250 (61%) | 67/72 | 0.75 |
| E112 | 1243 | 814 (65%) | 647 (52%) | 136/146 | 1.94 |
| E96 | 1714 | 1075 (63%) | 774 (45%) | 160/174 | 2.32 |
| E72 | 963 | 551 (57%) | 309 (32%) | 62/70 | 1.85 |
| DEV-E128 (3 runs) | 34 | 27 | 25 | 5/6 | |

- When the cell still fails, it mostly exposes the next genuine bug (TypeError, IndexError).
- No retry is proposed for garbled or ambiguous names (`ndlabel`, `permairs`, `np.b`).
- False-repair risk: the rewrite differs from the model's own fix in 7–11% of known cases (`pople`→`popleft` vs the model's `pop`; `dint`→`dict` vs `int`). The repair is error-triggered only. A partial first run's side effects repeat on the retry.

Recommended implementation:
- Trigger on NameError, AttributeError, ImportError, ModuleNotFoundError, or an unexpected-keyword TypeError.
- Rewrite to the unique close match and re-run, with up to 3 rewrites.
- Tie-breaks: prefer a candidate the typo is a deletion of, then the longest common prefix.
- Add a numpy `dtype` keyword rule.
- Prefix the output with `[auto-corrected: coms -> comps]`; this also curbs propagation.
- `replay.py:propose` is a starting point for `evals/run_deepseek_tools.py`.

## 8. Token weighting (T10, T11)

Verified in `special_stats.py`:
- `aggregate --region` marks every generated target from `</think>` (128822) through EOS: the whole tool-call block, DSML plus code.
- `select --weight 10` gives `contrib_dec + 9 × block`.
- `contrib_dec` uses all positions, prompt included, on layers 0–19, and only generated targets on layers 20–39.
- So code is ×10 and reasoning ×1. 42% of TRAIN positions are not generated.

Class shares in TRAIN, as targets:

| class | code (11.2M tokens) | reasoning (112.7M tokens) |
|---|---:|---:|
| WIC-alnum (word-internal continuation token: 2nd+ token of a word, splitting two alphanumerics) | 6.02% | 7.82% |
| WIC any (incl. `_`) | 6.83% | 7.87% |
| WIC-alnum in a re-tokenized copy | 1.16% | 0.47% |
| WIC in a fresh word | 1.25% | 1.41% |
| tokens of copied words | 50.1% | 51.0% |
| def/import sites | 6.69% | – |

DEV, API-full and E128 code has the same profile: WIC-alnum is 6.2–6.7% of code tokens.

Coverage of the typo loci (T11):

| class | share of typos, E128 / E112 / E96 / E72 | enrichment |
|---|---:|---:|
| WIC-alnum | 75 / 66 / 70 / 71% | 11–12× |
| re-tokenized copy WIC | 31 / 34 / 35 / 23% | 20–30× |
| fresh-word WIC | 12 / 9 / 8 / 11% | 7–10× |
| copied-word tokens | 83 / 82 / 84 / 83% | 1.7× |
| def/import sites | 2 / 14 / 27 / 32% | 0.3–4.8× |

Ranking of weighting rules:
1. **WIC-alnum targets in code and reasoning, extra weight about 10–30.**
   - Generic definition: the target token starts strictly inside a `[A-Za-z_]\w*` word, with alphanumerics on both sides of the boundary.
2. **Re-tokenized-copy WIC as an add-on** (×30–100). Most specific, but only 1.2% of code tokens.
3. **Punctuation-merged first piece** (`(m`, `.m`, `,d`) followed by a WIC target. Aimed at restarts; use only if (1) leaves them.
4. **Import/def sites.** Relevant to E112 and below only.
5. **All copied-word tokens.** Too broad; not recommended.

The new statistics are CPU-only:
- The per-token observer records (ids, 40×6 experts, weights, amplitudes) and the token streams are on HF `chankhavu/dsv41-flash1017-e043-pruning-artifacts` (`obs/`, `tok/`; the README lists 20 GB of raw observer records).
- The class masks are functions of token ids and roles, so `special_stats.py aggregate` with a class mask in place of `--region` produces the new statistics.
- GPUs are needed only for GPTQ and the eval.
- First diagnostic: what share of WIC-alnum contribution do the E128/E112/E96 keep sets retain, vs 94.9% of tool-block contribution for E112?

## 9. Proposals, ranked

| # | proposal | where | expected effect | evidence |
|---|---|---|---|---|
| 1 | Retry once with a unique close match, plus an auto-corrected note | harness, now | −57% / −51% / −44% identifier-error turns (E128 / E112 / E96); 0.75–2.3 turns per attempt; 89–93% agreement with the model's fix; working code untouched | replayed |
| 2 | Keep-set retention of WIC-alnum contribution | CPU, after the HF download (README: 20 GB; needs approval) | diagnostic for 4–5 | – |
| 3 | Prompt nudge to PEP8-spaced code | prompt, now; dev split first | `=`-led names are 3× riskier than space-led; `comps` after a space never fails; untested | lead-char analysis, `comps` test |
| 4 | Re-select experts with WIC-alnum (+ re-tokenized copy) weighting | stats CPU; GPTQ and eval need the box | about 70% of typo loci at 6% of tokens; size unknown until 2 | T11 |
| 5 | Healing LoRA, loss up-weighted on WIC targets | GPUs | trains the failing transition; the model already detects its slips | self-repair |
| 6 | Copy guard logits processor (boost continuations of identifiers already in context) | serving, engineering | up to the 83% of typos that are copies; risk to new names; untested | T5 |

Not recommended:
- min_p / top_p sampling: it doesn't remove the typos (E043 dev test).
- Merged-bracket-token fixes: errors land on merged bracket tokens at the baseline rate.

## 10. Facts vs hypotheses

**Measured:**
- All rates and counts (denominators in `out/tables.md`).
- The `comps` asymmetry.
- Split-mixing vs chance.
- First-token-correct vs null.
- Replay outcomes.

**Inferred:**
- The sampled tokens.
- That the failure is the continuation's conditioning on the emitted fragment's characters (sub-word composition).
- That the lost experts are low-total-contribution specialists (testable by proposal 2).
- That WIC weighting would fix the typos (needs proposals 2 and 4).

## Files

- Scripts: `common.py`, `textutil.py`, `dump_api_names.py` (miniconda Python with scipy), `extract.py`, `ident.py`, `splitmix.py`, `synerr.py`, `synerr2.py`, `reasoning_oov.py`, `tokclass.py`, `repair.py`, `replay.py`, `report_tables.py`.
- `out/`: `tables.md`, `key_numbers.json`, `splitmix.json`, `synerr*.json`, `reasoning_oov.json`, `tokclass_*.json`, and `ident_errors_*`, `repair_*`, `replay_*` pickles.
- The large `attempts_*` and `occ_*` pickles (about 1 GB) were moved to the session scratchpad. To regenerate them, run `extract.py`, then `ident.py <models>`.