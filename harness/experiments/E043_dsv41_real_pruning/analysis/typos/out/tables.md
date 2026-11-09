## T1. Error taxonomy (per 100 run_python calls)
| model | attempts | calls | any error | identifier (Name/Attr/Import/Module/kwarg) | NameError | AttributeError | Import+ModuleNotFound | kwarg TypeError | SyntaxError | other TypeError | Value+Index+Key | timeout | 'if False else' self-repair | '?' in code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 334 | 4327 | 2.22 | 0.23 | 0.09 | 0.12 | 0.02 | 0.00 | 0.14 | 0.62 | 1.06 | 0.23 | 0.99 | 0.05 |
| SH-full | 172 | 2003 | 2.60 | 0.40 | 0.15 | 0.25 | 0.00 | 0.00 | 0.10 | 0.50 | 1.40 | 0.25 | 1.20 | 0.05 |
| E128 | 334 | 5374 | 14.50 | 8.22 | 5.47 | 1.71 | 0.35 | 0.69 | 3.18 | 1.10 | 1.69 | 0.26 | 7.18 | 0.65 |
| E112 | 334 | 6717 | 30.56 | 18.70 | 10.56 | 3.65 | 3.41 | 1.09 | 9.68 | 0.73 | 1.28 | 0.16 | 8.23 | 2.90 |
| E96 | 334 | 7618 | 38.34 | 22.84 | 11.21 | 3.62 | 7.42 | 0.59 | 13.09 | 0.80 | 1.35 | 0.18 | 8.73 | 4.32 |
| E72 | 167 | 3100 | 46.26 | 31.26 | 11.71 | 6.32 | 10.94 | 2.29 | 12.94 | 0.94 | 1.10 | 0.06 | 8.68 | 4.03 |
| DEV-E128-t1 | 33 | 134 | 10.45 | 6.72 | 5.22 | 0.00 | 0.00 | 1.49 | 1.49 | 0.75 | 0.75 | 0.00 | 3.73 | 1.49 |
| DEV-E128-minp05 | 33 | 128 | 12.50 | 12.50 | 10.16 | 0.00 | 0.00 | 2.34 | 0.00 | 0.00 | 0.00 | 0.00 | 6.25 | 0.00 |
| DEV-E128-topp95 | 32 | 134 | 8.96 | 6.72 | 3.73 | 0.75 | 0.00 | 2.24 | 1.49 | 0.75 | 0.00 | 0.00 | 4.48 | 0.00 |
| TRAIN | 8754 | 60084 | 1.78 | 0.30 | 0.10 | 0.14 | 0.05 | 0.00 | 0.10 | 0.37 | 0.91 | 0.08 | 0.69 | 0.02 |

### SyntaxError classes (per 100 calls) and bracket pairs
| model | invalid-syntax | bracket-mismatch | unmatched-closer | never-closed | forgot-comma | string | incomplete-input | indentation | { closed by ] or ) | all mismatches | caret on merged bracket token | baseline merged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 0.07 | 0.02 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 1 | 1 | 0.00 | 0.48 |
| SH-full | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |  | 0.47 |
| E128 | 1.54 | 0.17 | 0.04 | 0.20 | 0.61 | 0.30 | 0.06 | 0.02 | 2 | 9 | 0.55 | 0.49 |
| E112 | 4.87 | 2.07 | 0.34 | 0.19 | 1.44 | 0.25 | 0.15 | 0.04 | 129 | 139 | 0.57 | 0.48 |
| E96 | 7.13 | 3.73 | 0.34 | 0.22 | 0.89 | 0.37 | 0.13 | 0.05 | 275 | 284 | 0.43 | 0.47 |
| E72 | 6.61 | 2.00 | 1.03 | 0.29 | 1.97 | 0.45 | 0.13 | 0.06 | 53 | 62 | 0.56 | 0.48 |
| DEV-E128-t1 | 0.75 | 0.00 | 0.00 | 0.00 | 0.75 | 0.00 | 0.00 | 0.00 | 0 | 0 |  | 0.53 |
| DEV-E128-minp05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |  | 0.48 |
| DEV-E128-topp95 | 0.75 | 0.00 | 0.00 | 0.00 | 0.75 | 0.00 | 0.00 | 0.00 | 0 | 0 |  | 0.47 |
| TRAIN | 0.04 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 9 | 0.43 | 0.47 |

## T2. Identifier errors: classification
| model | identifier errors | typo | typo, high-confidence intended | unresolved | state/scope | missing import | wrong API | unavailable module | use-before-def | typo share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 10 | 1 | 1 | 2 | 2 | 0 | 4 | 0 | 1 | 0.10 |
| SH-full | 6 | 0 | 0 | 1 | 3 | 0 | 1 | 0 | 1 | 0.00 |
| E128 | 442 | 293 | 227 | 81 | 41 | 7 | 13 | 1 | 3 | 0.66 |
| E112 | 1262 | 871 | 614 | 319 | 35 | 1 | 13 | 0 | 10 | 0.69 |
| E96 | 1747 | 1142 | 758 | 476 | 87 | 8 | 10 | 0 | 6 | 0.65 |
| E72 | 989 | 547 | 408 | 384 | 34 | 3 | 0 | 0 | 3 | 0.55 |
| DEV-E128-t1 | 12 | 6 | 4 | 5 | 0 | 0 | 0 | 0 | 0 | 0.50 |
| DEV-E128-minp05 | 16 | 12 | 6 | 4 | 0 | 0 | 0 | 0 | 0 | 0.75 |
| DEV-E128-topp95 | 9 | 1 | 1 | 6 | 0 | 2 | 0 | 0 | 0 | 0.11 |
| TRAIN | 192 | 13 | 10 | 62 | 14 | 3 | 59 | 30 | 7 | 0.07 |

## T3. Character-level edit type of typos (share of typos with a recovered intended name)
| model | typos | deletion | insertion(doubled) | segment-repeat | insertion | substitution | transposition | truncation | chunk-deletion | tail-replaced | multi-edit | len intended (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 1 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 2 |
| E128 | 293 | 41 | 18 | 12 | 4 | 5 | 0 | 6 | 3 | 9 | 1 | 5 |
| E112 | 871 | 37 | 17 | 6 | 6 | 8 | 0 | 6 | 2 | 14 | 3 | 5 |
| E96 | 1142 | 48 | 7 | 3 | 6 | 7 | 0 | 9 | 2 | 16 | 1 | 5 |
| E72 | 547 | 40 | 11 | 5 | 5 | 8 | 0 | 11 | 2 | 17 | 1 | 5 |
| DEV-E128-t1 | 6 | 0 | 0 | 50 | 17 | 0 | 0 | 0 | 33 | 0 | 0 | 5 |
| DEV-E128-minp05 | 12 | 75 | 0 | 17 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| DEV-E128-topp95 | 1 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| TRAIN | 13 | 23 | 0 | 0 | 8 | 23 | 0 | 23 | 8 | 15 | 0 | 4 |

### Most frequent typos (pruned models pooled)
| typo | intended | count | intended tokens (in context) | models |
|---|---:|---:|---:|---:|
| coms | comps | 648 | com | ps | E112,E128,E72,E96 |
| scpy | scipy | 166 | Ġsc | ipy | E112,E72,E96 |
| dequeue | deque | 94 | de | que | E112,E128,E72,E96 |
| zzip | zip | 59 | (z | ip | E112,E128,E72,E96 |
| scip | scipy | 57 | Ġsc | ipy | E112,E72,E96 |
| mmost_common | most_common | 43 | .m | ost | _ | common | E112,E128,E72,E96 |
| scipy.nd | ndimage | 42 | .nd | image | E112,E72,E96 |
| tol | tolist | 37 | .t | ol | ist | E72,E96 |
| dbool | bool | 29 | ,b | ool | E112,E128,E72,E96 |
| ymmin | ymin | 24 | ym | in | E112,E72,E96 |
| coms2 | comps2 | 23 | com | ps | 2 | E112,E72,E96 |
| nd | ndimage | 20 | Ġnd | image | E112,E72,E96 |
| deq | deque | 20 | de | que | E112,E128,E72,E96 |
| ddiff | diff | 19 | (d | iff | E112,E72,E96 |
| dbool | dtype=bool | 18 | ,d | type | = | bool | E112,E128,E72,E96 |
| flravel | ravel | 18 | ravel | E112,E96 |
| mmask | mask | 17 | =m | ask | E112,E128,E72,E96 |
| tollist | tolist | 16 | tol | ist | E72,E96 |
| scscipy | scipy | 13 | Ġsc | ipy | E112,E72,E96 |
| coms_info | comps_info | 13 | com | ps | _info | E128,E96 |
| get_coms | get_comps | 13 | =get | _com | ps | E112,E128,E96 |
| scpi | scipy | 12 | Ġsc | ipy | E112,E72,E96 |
| scopy | scipy | 12 | Ġsc | ipy | E112,E72,E96 |
| coms_mask | comps_mask | 11 | com | ps | _mask | E112,E96 |
| se | set | 10 | set | E96 |

## T4. Token-level locus of typos vs per-position opportunity
| model | typos | intended multi-token % | base: occurrences multi-token % | first token correct % | diverges at a token boundary % | at alnum|alnum boundary % | hazard per alnum|alnum boundary (x1e4) | hazard per _/symbol boundary (x1e4) | hazard per first token (x1e4) | hazard inside-token (per occ, x1e4) | hazard after last token (per occ, x1e4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 1 | 100.0 | 17.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E128 | 293 | 93.9 | 17.8 | 92.5 | 88.1 | 74.7 | 26.6 | 1.3 | 0.1 | 0.6 | 0.7 |
| E112 | 871 | 89.9 | 16.9 | 86.0 | 73.2 | 66.2 | 55.8 | 5.9 | 1.0 | 2.8 | 0.8 |
| E96 | 1142 | 93.6 | 15.3 | 90.6 | 76.2 | 70.4 | 84.2 | 10.0 | 0.8 | 3.5 | 0.9 |
| E72 | 547 | 94.0 | 13.9 | 91.2 | 75.5 | 70.6 | 120.3 | 5.8 | 0.6 | 5.0 | 1.0 |
| DEV-E128-t1 | 6 | 83.3 | 17.4 | 83.3 | 50.0 | 0.0 | 0.0 | 0.0 | 1.1 | 2.3 | 3.4 |
| DEV-E128-minp05 | 12 | 91.7 | 17.4 | 91.7 | 83.3 | 66.7 | 49.7 | 0.0 | 1.0 | 1.0 | 2.1 |
| DEV-E128-topp95 | 1 | 100.0 | 14.6 | 100.0 | 100.0 | 100.0 | 9.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| TRAIN | 13 | 53.8 | 15.5 | 30.8 | 30.8 | 15.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Token-level event (canonical tokens of the intended name in context; pruned models pooled and per model)
| model | typos | continuation-lost-leading-chars | continuation-replaced | continuation-replaced(same 1st char) | restart(repeat-emitted-chars) | dropped-1-token(s) | dropped-2-token(s) | truncated-at-boundary | continuation-truncated | continuation-extended | extended-after-last-token | first-token-wrong | unaligned | null: P(first token) if chars uniform % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E128 | 293 | 39 | 9 | 3 | 20 | 0 | 2 | 5 | 3 | 0 | 13 | 6 | 0 | 42 |
| E112 | 871 | 34 | 12 | 8 | 17 | 0 | 1 | 4 | 5 | 0 | 6 | 13 | 0 | 46 |
| E96 | 1142 | 43 | 13 | 9 | 8 | 0 | 0 | 7 | 5 | 0 | 5 | 9 | 0 | 46 |
| E72 | 547 | 31 | 15 | 9 | 14 | 0 | 0 | 11 | 9 | 0 | 4 | 7 | 0 | 45 |
| DEV-E128-t1 | 6 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 17 | 0 | 41 |
| DEV-E128-minp05 | 12 | 75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 8 | 0 | 57 |
| DEV-E128-topp95 | 1 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 |
| API-full | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 50 |
| TRAIN | 13 | 0 | 0 | 8 | 0 | 0 | 0 | 15 | 8 | 0 | 15 | 54 | 0 | 72 |

### Divergence token index (pruned pooled)
token 0: 299, token 1: 2341, token 2: 194, token 3: 19

## T5. Copy status of the intended identifier: typo hazard per 10k occurrences (multi-token identifiers only)
| model | fresh: typos | fresh: occ | fresh: /10k | copy, same tokens: typos | copy, same tokens: occ | copy, same tokens: /10k | copy, re-tokenized: typos | copy, re-tokenized: occ | copy, re-tokenized: /10k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 0 | 7868 | 0.0 | 0 | 46355 | 0.0 | 1 | 9434 | 1.1 |
| E128 | 42 | 7997 | 52.5 | 122 | 71058 | 17.2 | 111 | 9540 | 116.4 |
| E112 | 112 | 8708 | 128.6 | 292 | 87975 | 33.2 | 379 | 8914 | 425.2 |
| E96 | 150 | 9359 | 160.3 | 470 | 78444 | 59.9 | 449 | 9052 | 496.0 |
| E72 | 85 | 3974 | 213.9 | 279 | 26179 | 106.6 | 150 | 3254 | 461.0 |
| DEV-E128-t1 | 0 | 209 | 0.0 | 2 | 1124 | 17.8 | 3 | 212 | 141.5 |
| DEV-E128-minp05 | 0 | 227 | 0.0 | 7 | 1240 | 56.5 | 4 | 214 | 186.9 |
| DEV-E128-topp95 | 0 | 222 | 0.0 | 1 | 762 | 13.1 | 0 | 217 | 0.0 |
| TRAIN | 2 | 105082 | 0.2 | 2 | 409899 | 0.0 | 3 | 120148 | 0.2 |

### Where was the intended name before? (typos, pruned models)
| model | typos | seen before % | in earlier code % | only in reasoning % | never in attempt (API/builtin) % | fresh & API/builtin name % | last seen same turn % | median tokens since last occurrence | p90 tokens since | last seen >= 1 turn back % | last seen >= 3 turns back % | same tokens as an earlier occurrence % | typo form also in reasoning (same/earlier turn) % | typo form already in earlier code % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E128 | 293 | 82.9 | 79.2 | 3.8 | 17.1 | 7.8 | 67.2 | 106.0 | 1547.0 | 15.7 | 5.8 | 53.9 | 25.6 | 48.1 |
| E112 | 871 | 82.0 | 74.9 | 7.1 | 18.0 | 12.2 | 61.7 | 112.5 | 2255.5 | 20.3 | 7.1 | 44.5 | 27.7 | 50.9 |
| E96 | 1142 | 84.2 | 79.1 | 5.1 | 15.8 | 9.8 | 58.8 | 126.0 | 1549.0 | 25.3 | 9.5 | 52.1 | 38.4 | 50.5 |
| E72 | 547 | 83.0 | 75.9 | 7.1 | 17.0 | 11.0 | 53.2 | 124.0 | 1605.2 | 29.8 | 10.4 | 65.6 | 44.8 | 46.3 |

### Typo hazard by the character before the identifier (pruned pooled, multi-token identifiers, per 10k occ)
| lead | typos (multi-token) | occurrences (multi-token) | per 10k multi-token occ | all typos per 10k occ (any length) | share of occ multi-token % | alnum|alnum boundaries per occ | mid-boundary typos per 10k mid boundaries |
|---|---:|---:|---:|---:|---:|---:|---:|
| op | 806 | 50552 | 159.4 | 26.8 | 16.3 | 0.2 | 119.4 |
| other | 12 | 757 | 158.5 | 22.2 | 12.9 | 0.2 | 58.9 |
| dot | 429 | 40737 | 105.3 | 23.8 | 19.3 | 0.2 | 80.0 |
| space | 535 | 65173 | 82.1 | 8.8 | 9.9 | 0.1 | 55.0 |
| open | 589 | 81575 | 72.2 | 14.5 | 19.0 | 0.2 | 67.1 |
| comma | 260 | 82030 | 31.7 | 8.5 | 23.5 | 0.2 | 24.7 |
| linestart | 10 | 3630 | 27.5 | 3.1 | 11.3 | 0.1 | 28.2 |

## T6. Context length and turn index: identifier-typo errors per 100 run_python calls
| model | ctx <15k | ctx 15-25k | ctx 25-40k | ctx 40-60k | ctx >60k | turn 0-2 | turn 3-5 | turn 6-9 | turn 10-19 | turn 20+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E128 | 3.6 | 3.8 | 7.1 | 7.3 | 6.7 | 2.4 | 4.4 | 6.2 | 6.5 | 7.5 |
| E112 | 10.6 | 14.4 | 13.1 | 14.2 | 12.9 | 8.2 | 13.0 | 15.9 | 14.0 | 12.6 |
| E96 | 14.3 | 18.1 | 14.5 | 13.6 | 11.9 | 9.4 | 15.9 | 19.2 | 17.5 | 12.5 |
| E72 | 16.9 | 20.7 | 16.6 | 6.8 | 21.7 | 7.9 | 18.2 | 22.0 | 20.6 | 15.6 |
| API-full | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |

### Per-boundary hazard (alnum|alnum typos per 10k such boundaries) by context length, pruned pooled
| context | mid-boundary typos | mid boundaries | per 10k |
|---|---:|---:|---:|
| <15k | 512 | 45931 | 111.47 |
| 15-25k | 572 | 66892 | 85.51 |
| 25-40k | 486 | 87808 | 55.35 |
| 40-60k | 280 | 69117 | 40.51 |
| >60k | 136 | 43574 | 31.21 |

### Within attempt: typo rate in the first vs second half of an attempt's calls (attempts with >= 6 calls)
| model | first half /100 calls | second half /100 calls | calls |
|---|---:|---:|---:|
| E128 | 4.55 | 6.52 | 5250 |
| E112 | 13.76 | 12.44 | 6649 |
| E96 | 17.45 | 12.83 | 7529 |
| E72 | 19.66 | 16.02 | 3051 |

## T7. Recovery: next call after an identifier typo
| model | typos | with a next call | typo gone in next call % | next call completes % | next call errors again % | same error repeated % | next call is a near-copy (sim>=0.8) % | repair-turn tokens (median) | repair-turn tokens (mean) | 'Did you mean' in output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 1 | 1 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 | 677.0 | 677.0 | 0 |
| E128 | 293 | 291 | 60.1 | 55.0 | 44.7 | 27.5 | 80.1 | 438.0 | 545.2 | 0 |
| E112 | 871 | 864 | 65.3 | 45.5 | 54.2 | 24.5 | 73.6 | 381.5 | 533.6 | 0 |
| E96 | 1142 | 1132 | 70.0 | 43.7 | 56.2 | 20.1 | 55.6 | 278.5 | 385.4 | 0 |
| E72 | 547 | 541 | 74.9 | 38.1 | 61.9 | 15.3 | 53.0 | 305.0 | 567.6 | 0 |
| DEV-E128-t1 | 6 | 6 | 50.0 | 33.3 | 66.7 | 50.0 | 100.0 | 564.5 | 524.2 | 0 |
| DEV-E128-minp05 | 12 | 12 | 58.3 | 66.7 | 33.3 | 33.3 | 83.3 | 256.0 | 351.8 | 0 |
| DEV-E128-topp95 | 1 | 1 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 | 232.0 | 232.0 | 0 |
| TRAIN | 13 | 13 | 84.6 | 100.0 | 0.0 | 0.0 | 69.2 | 280.0 | 431.9 | 0 |

### Turns and tokens attributable to error recovery (all error calls; per attempt)
| model | attempts | turns/attempt | turns after an error % | turns after an identifier typo % | tokens in turns after an error % | tokens after identifier typo % | 40-turn stops |
|---|---:|---:|---:|---:|---:|---:|---:|
| API-full | 334 | 13.9 | 2.1 | 0.0 | 1.0 | 0.0 | 8 |
| SH-full | 172 | 12.7 | 2.5 | 0.0 | 0.8 | 0.0 | 0 |
| E128 | 334 | 17.1 | 13.7 | 5.1 | 4.2 | 1.3 | 11 |
| E112 | 334 | 21.0 | 29.2 | 12.4 | 9.3 | 3.9 | 33 |
| E96 | 334 | 23.6 | 36.9 | 14.4 | 11.3 | 4.3 | 58 |
| E72 | 167 | 19.7 | 44.2 | 16.6 | 25.7 | 8.0 | 5 |
| DEV-E128-t1 | 33 | 5.2 | 10.0 | 3.5 | 3.6 | 1.5 | 0 |
| DEV-E128-minp05 | 33 | 4.9 | 9.9 | 7.5 | 2.1 | 1.7 | 0 |
| DEV-E128-topp95 | 32 | 5.2 | 7.2 | 0.6 | 1.7 | 0.1 | 0 |

## T8. Do attempts with more typos fail more often on the same pair?
| comparison | discordant pairs where the failing attempt had the higher typo rate | ... the winning attempt had the higher rate | p (sign) | early (first 6 calls) typos: fail > win | win > fail | p early |
|---|---:|---:|---:|---:|---:|---:|
| API-full: attempt vs attempt, same pair | 0 | 0 |  | 0 | 0 |  |
| SH-full: attempt vs attempt, same pair | 0 | 0 |  | 0 | 0 |  |
| E128: attempt vs attempt, same pair | 14 | 5 | 0.0636 | 4 | 4 | 1 |
| E112: attempt vs attempt, same pair | 8 | 15 | 0.21 | 2 | 9 | 0.0654 |
| E96: attempt vs attempt, same pair | 15 | 19 | 0.608 | 11 | 13 | 0.839 |
| E72: attempt vs attempt, same pair | 0 | 0 |  | 0 | 0 |  |
| pruned models pooled, all attempts of a pair (all win x loss attempt pairs) | 486 | 338 |  | 268 | 219 |  |
| all eval models pooled (all win x loss attempt pairs) | 978 | 498 |  | 523 | 301 |  |

### Linear probability model, pruned models pooled, pair and model fixed effects (two-way demeaned)
- any identifier typo in the first 6 calls (0/1): beta = +0.012 (95% cluster-bootstrap CI -0.045 to +0.063); 1169 attempts, 167 pairs.
- typo errors per call: beta = -0.033 (95% cluster-bootstrap CI -0.228 to +0.174); 1169 attempts, 167 pairs.

## T9. Auto-repair simulation
### (A) Static pre-execution rewrite (bare names, module attributes, method names, imports)
| model | calls | calls with a rewrite | bare-name rewrites on fine calls in an 'if False' line | ... on calls that ran fine | ... of which a bare-name rewrite | typo calls | typo name rewritten % | rewrite = model's own fix (where known) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 4327 | 12 | 0 | 9 | 0 | 1 | 0.0 | 0/0 |
| SH-full | 2003 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0/0 |
| E128 | 5374 | 497 | 53 | 124 | 65 | 293 | 71.0 | 54/58 |
| E112 | 6717 | 1184 | 60 | 129 | 78 | 871 | 63.0 | 107/113 |
| E96 | 7618 | 1556 | 50 | 140 | 77 | 1141 | 66.2 | 136/143 |
| E72 | 3100 | 835 | 19 | 65 | 35 | 544 | 69.9 | 56/62 |
| DEV-E128-t1 | 134 | 6 | 1 | 1 | 1 | 6 | 50.0 | 0/0 |
| DEV-E128-minp05 | 128 | 13 | 2 | 3 | 3 | 12 | 83.3 | 2/2 |
| DEV-E128-topp95 | 134 | 2 | 1 | 1 | 1 | 1 | 100.0 | 1/1 |
| TRAIN | 60084 | 174 | 6 | 106 | 7 | 13 | 46.2 | 1/1 |

### (B) Retry once after the error, replayed in a local Python session
| model | identifier-error calls replayed | replay reproduces the recorded exception | retry proposed | retry proposed % | cell completes after retry | completes % of reproduced | 2+ rewrites needed | still failing: top exceptions | first rewrite = model's own fix | disagreements (typo->rewrite vs model fix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| API-full | 9 | 8 | 2 | 25.0 | 0 | 0.0 | 0 | TypeError:2 | 0/1 | x0->xs vs c0 |
| SH-full | 4 | 2 | 0 | 0.0 | 0 | 0.0 | 0 |  |  |  |
| E128 | 438 | 410 | 299 | 72.9 | 250 | 61.0 | 38 | TypeError:19, NameError:9, IndexError:6, ValueError:6 | 67/72 | deq->dq vs deque; mn->min vs mxn; hh->nh vs h; pople->popleft vs pop |
| E112 | 1255 | 1243 | 814 | 65.5 | 647 | 52.1 | 105 | NameError:43, TypeError:40, AttributeError:24, ImportError:14 | 136/146 | axdist->axisdist vs dist; coms2->comps2 vs coms; men->min vs mid; coms->comps vs copy |
| E96 | 1740 | 1714 | 1075 | 62.7 | 774 | 45.2 | 115 | TypeError:65, ImportError:65, ModuleNotFoundError:54, AttributeError:50 | 160/174 | coror->corner vs cor; coms->comps vs cs; comp->comps vs coms; dint->dict vs int |
| E72 | 969 | 963 | 551 | 57.2 | 309 | 32.1 | 75 | TypeError:56, ModuleNotFoundError:48, AttributeError:44, NameError:38 | 62/70 | sc->gc vs sys; cmmax->cmmmax vs cmax; dint->dict vs int; comp->comps vs coms |
| DEV-E128-t1 | 9 | 9 | 6 | 66.7 | 6 | 66.7 | 0 |  | 0/1 | dint->dict vs int |
| DEV-E128-minp05 | 16 | 16 | 15 | 93.8 | 15 | 93.8 | 0 |  | 4/4 |  |
| DEV-E128-topp95 | 9 | 9 | 6 | 66.7 | 4 | 44.4 | 0 | NameError:2 | 1/1 |  |

## T10. Candidate token classes (as prediction targets) in generated text
| model | segment | tokens | word % | word-first % | WIC % | WIC-alnum % | WIC-copied % | WIC-alnum-copied % | WIC-retok-copy % | WIC-alnum-retok-copy % | WIC-fresh % | copied-word-token % | retok-copy-token % | def-site % | def-site-WIC % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | code | 11240225 | 55.92 | 49.10 | 6.83 | 6.02 | 5.58 | 5.01 | 1.32 | 1.16 | 1.25 | 50.13 | 6.43 | 6.69 | 1.06 |
| TRAIN | reason | 112686426 | 57.96 | 50.09 | 7.87 | 7.82 | 6.46 | 6.43 | 0.48 | 0.47 | 1.41 | 51.01 | 1.68 | 0.00 | 0.00 |
| API-full | code | 1045437 | 55.80 | 48.08 | 7.72 | 6.68 | 6.63 | 5.83 | 1.18 | 1.00 | 1.10 | 51.62 | 4.78 | 6.84 | 1.36 |
| API-full | reason | 7946906 | 60.45 | 52.13 | 8.32 | 8.26 | 7.11 | 7.07 | 0.50 | 0.49 | 1.21 | 54.61 | 1.65 | 0.00 | 0.00 |
| SH-full | code | 445187 | 56.19 | 48.50 | 7.69 | 6.65 | 6.58 | 5.76 | 1.20 | 1.03 | 1.11 | 51.73 | 5.14 | 7.09 | 1.36 |
| SH-full | reason | 3060973 | 58.75 | 50.29 | 8.46 | 8.40 | 7.16 | 7.11 | 0.53 | 0.52 | 1.31 | 52.16 | 1.75 | 0.00 | 0.00 |
| E128 | code | 1444363 | 56.80 | 48.85 | 7.95 | 6.74 | 7.15 | 6.10 | 0.83 | 0.73 | 0.80 | 53.63 | 3.49 | 6.34 | 1.20 |
| E128 | reason | 10096962 | 63.20 | 54.14 | 9.05 | 9.00 | 7.87 | 7.83 | 0.43 | 0.43 | 1.18 | 57.87 | 1.43 | 0.00 | 0.00 |
| E112 | code | 1698621 | 56.62 | 49.32 | 7.30 | 6.49 | 6.59 | 5.91 | 0.64 | 0.56 | 0.71 | 53.90 | 2.84 | 7.27 | 1.28 |
| E112 | reason | 9953100 | 64.07 | 54.62 | 9.45 | 9.39 | 8.01 | 7.97 | 0.42 | 0.41 | 1.43 | 58.20 | 1.41 | 0.00 | 0.00 |
| E96 | code | 1744946 | 57.08 | 50.14 | 6.94 | 6.24 | 6.16 | 5.59 | 0.63 | 0.56 | 0.78 | 54.16 | 2.93 | 6.65 | 1.11 |
| E96 | reason | 8128347 | 64.80 | 55.10 | 9.70 | 9.63 | 8.25 | 8.19 | 0.44 | 0.43 | 1.45 | 58.22 | 1.51 | 0.00 | 0.00 |
| E72 | code | 636864 | 57.08 | 50.82 | 6.25 | 5.43 | 5.34 | 4.65 | 0.62 | 0.54 | 0.91 | 53.45 | 3.20 | 6.82 | 1.05 |
| E72 | reason | 3592885 | 64.51 | 57.30 | 7.21 | 6.97 | 5.84 | 5.61 | 0.28 | 0.28 | 1.37 | 58.20 | 1.09 | 0.00 | 0.00 |
| DEV-E128-t1 | code | 23967 | 56.07 | 48.91 | 7.16 | 6.24 | 6.25 | 5.54 | 1.16 | 0.98 | 0.92 | 50.26 | 6.33 | 7.13 | 1.15 |
| DEV-E128-t1 | reason | 176022 | 58.46 | 49.75 | 8.72 | 8.67 | 7.31 | 7.28 | 0.46 | 0.46 | 1.40 | 50.49 | 1.64 | 0.00 | 0.00 |

## T11. Coverage of typo loci by candidate classes vs class share of tool-block tokens (TRAIN rollouts)
| class | share of TRAIN tool-block tokens % | enrichment E112 | enrichment E128 | enrichment E72 | enrichment E96 | share of typos % E112 | share of typos % E128 | share of typos % E72 | share of typos % E96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WIC any (incl. _ boundaries) | 6.8 | 9.8 | 11.1 | 10.4 | 10.5 | 67.2 | 75.4 | 71.1 | 71.5 |
| WIC in a fresh word (not seen before) | 1.2 | 7.1 | 9.8 | 8.6 | 6.5 | 8.8 | 12.3 | 10.8 | 8.1 |
| WIC-alnum (target = word-internal continuation at alnum|alnum) | 6.0 | 11.0 | 12.4 | 11.7 | 11.7 | 66.2 | 74.7 | 70.6 | 70.4 |
| WIC-alnum in a re-tokenized copy | 1.2 | 29.6 | 26.4 | 19.7 | 30.4 | 34.4 | 30.7 | 22.9 | 35.3 |
| all tokens of copied words | 50.1 | 1.6 | 1.7 | 1.7 | 1.7 | 82.0 | 82.9 | 83.0 | 84.2 |
| all tokens of re-tokenized copies | 6.4 | 7.1 | 5.9 | 4.4 | 6.3 | 45.5 | 38.2 | 28.5 | 40.3 |
| def/import-site tokens | 6.7 | 2.1 | 0.3 | 4.8 | 4.0 | 14.1 | 1.7 | 32.4 | 26.6 |
