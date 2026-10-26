# E052: next-candidate code (fix list + opt-in levers) vs V6 code, on 2x Vast 4x4090

2026-10-26. Same pipeline as E050 (V6 settings: 36 workers, KV gate 0.6, multi-test + budget 0.5, context limit 163840,
util 0.92, ATTEMPTS=1, all 120 Kaggle eval tasks, no deadline pressure), effort 50, only the code / opt-ins differ.
Control: E050 `eff50` (V6 pinned code + effort env edit, box 1). Analysis: `experiments/E050_effort_ab/analyze.py
--a <arm>/out --b <control>` (arm a is labelled "eff25" in its output; pairs matched by test-input content).

| arm | code / opt-ins | task pass@1 | multi (49) / single (71) | out tok/task | turns | no answer |
|---|---|---|---|---|---|---|
| E050 eff50 (control) | V6 (harness a8a5b665, solver 4b344011) | 71.4% | 66.7 / 74.6 | 42.2k | 19.2 | 7 |
| **A `new_ar`** | fixes (harness 13ebdb70, solver 24aaa376) + auto-repair | **76.5%** | **77.2** / 76.1 | 42.9k | 19.3 | 5 |
| B `new_ar_bg` | A + best-guess nudge (0.8 / 3 turns), solver 82010f8a | 74.6% | 74.5 / 74.6 | 42.2k | 17.9 | 5 |
| C `new_ar_pp` | A + post-pass note (I8), harness 882c817f, solver c760fae0 | running | | | | |

- A vs control: +5.1 pp [-1.8, +12.2], task wins 17/9 (p=0.17), tokens x1.017 [0.94, 1.11]. B vs control: +3.2 pp
  [-4.0, +10.4]. B vs A: -1.9 pp [-8.2, +4.3], tokens x0.984, -1.4 turns. Same-config noise (run 7 a0 vs a1): 2.5 pp.
- Reading: the new code is at least no harm and probably a gain, concentrated on multi-input tasks (A1 per-pair partial
  answers, A6 multi train checks, A9 context-capped budget notes); not significant on 120 tasks.
- Auto-repair: A 134 proposed / 134 applied / 94 retries ok (70%); no measurable token or turn effect at n=120. Kept.
- Best-guess nudge (I7): fired in 24 attempts, 18 submitted after, **0 of 24 correct**; -1.4 turns, -1.6% tokens.
  Not adopted.
- Box-image credentials: arm A's server_cmd.json held the Vast image's HF_TOKEN (solver before the redaction fix);
  redacted on copy. The live solver redacts TOKEN/KEY/SECRET/PASS env values.
