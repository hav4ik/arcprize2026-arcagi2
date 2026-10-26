# DeepSeek-V4.1 E128 on Kaggle 4xL4: inference findings, 2026-10-25/26

Inference-side work of the dsv4-infer session (E042 serving, E046 multi-test), from submit v3's 50.9 to the
final candidate dsv41-tools-submit-mtf Version 6. Details and raw numbers: `experiments/E042_dsv41_l4_synthetic/README.md`
(sections from "Deadline simulation" on), `experiments/E046_multi_test_rollout/README.md`, pod timelines in
`experiments/E046_multi_test_rollout/pods/`.

## 1. What went wrong with submit v3 (50.9)

- **KV thrash.** E128 x 48 streams on Kaggle's 1.44M-token KV pool fills the pool ~2 h into the run. Once running
  requests hold ~0.88 of KV, the cached prefixes of streams waiting between turns are evicted and every turn
  re-prefills 35-40k tokens: tok/s 380 -> 170-280, uncached prefill 130 -> 640 tok/s, prefix hits 97% -> 77%, with
  0 preemptions (measured on pod B with E128 x 56 on 1.66M, the same KV per stream as v3).
- **The private set has ~335 test pairs over 240 tasks** (teammate's LB probe; the public file has 259). The solver
  runs attempts per pair: ~29% more work than planned.
- The 12-h deadline simulator (`E042 scripts/deadline_sim.py`, thrash model `--thrash 0.88:0.6`, KV ratio 0.83 per
  context token, calibrated on pod B's thrash onset) predicted **52.7 for v3 with thrash vs 71.4 without**; v3 scored
  50.9. Kaggle rerun logs are not visible, so this is the best explanation, not a proof.

## 2. Submit Version 5 (dsv41-tools-submit, submitted 2026-10-25 ~23:55 UTC)

- v3's exact notebook and kit + one settings cell: **36 workers** (v3: 48), `--linear-backend humming` (Humming
  dense MXFP8), CUDA graphs <= 64 (`max_cudagraph_capture_size` 64), max-num-seqs 64, util 0.945. Kaggle KV
  1,554,183 tokens (+8% vs v3; pods give 1.66M at the same settings).
- **Graph cap 64 is harmless:** the capture sizes logged on Kaggle are [1, 2, 4, ..., 56, 64]; batches > 64 cannot occur.
- **Humming dense vs Ada dense parity** (`E042 scripts/mxfp8_dense_parity.py`, L4, V4.1 dense shapes, M 1-2048):
  relative error vs the exact fp32 product with bf16 inputs: Ada 2.6-2.7e-2 (W8A8: activations quantized to MXFP8),
  Humming 1.8-3.3e-3 (W8A16). Humming is ~10x closer to full precision.
- Kaggle 30-min bench of its settings (36 streams): 306-324 tok/s, hits 0.92-0.96, 0 preemptions.
- Pod run at Kaggle KV (4.8 h, eval set): KV peaked 0.691 (3.56 h), 0.57-0.69 from 1.8 h; 290-320 tok/s; no thrash.
  Per-attempt accuracy 70.2% vs E128's 8x-box reference 78.2% on the same pairs (114 attempts; noise or a small
  real gap under the Kaggle flags; see 7).
- Two adversarial audits (sub-agents) found no failure path; Version 6 of the same notebook (Ada dense instead of
  Humming) was the "fewest untested parts" fallback (KV 1,615,809).
- Projection with the simulator and the pod accuracy: **~65-68**.

## 3. Things that do not work / were rejected

- **KV CPU offload** (`--kv-offloading-size`): the fork's OffloadingConnector rejects V4.1's KV layout. First
  `tokens_per_block=8 not divisible by tokens_per_hash=32` (an 8-token non-prefix-cacheable group); with
  `--prefix-match-unit 8` the offload scheduler asserts `isinstance(kv_cache_spec, FullAttentionSpec)`
  (`get_sliding_window_size_in_chunks`). Needs vLLM patches.
- **KV gate at 0.75** (stop admitting new attempts above KV 0.75): too late. The gate stops admissions, but the
  running attempts' contexts keep growing; E128 multi-test x 52 reached KV 0.895 and thrashed at 1.65 h.
- **Fixed worker counts without a gate for multi-test:** x 44 thrashed at 2.0 h; x 36 peaked at KV 0.834 (0.05 from
  the edge) on the full pool and thrashed at 3.4 h on a 10% smaller pool.
- **Shorter budgets** (s20x60 etc.): lose on Kaggle at 259 pairs; at 335 pairs and 48 streams with thrash s20x60
  wins only because it never thrashes; at 36 streams the Kaggle harness ties or wins. Not adopted (E128 untested).
- **Multi-test prompt wordings** (other session, `--multi-prompt lean/early/first`): all tie on the API; "first"
  (only test input 0 in the prompt) saves no KV (the model prints the others).
- Indexer tile tuning (<= 2%), CED prefill (approximate), length-aware admission (noise), train-check precision
  tweaks (1 label in ~300), sibling transfer of a passing program to the other test inputs (67/99 right: a free
  extra answer at best).
- Budget 1.25x (`--multi-budget 0.25`): within noise of 1.5x, only -5% tokens.

## 4. One rollout per task (E046 `--multi-test`)

For tasks with several test inputs (~40% of tasks, ~55% of pairs): the prompt shows every test input, the session
preloads `test_inputs`, and the model submits a list of grids. Results, E128 on the 45 multi-pair eval tasks:

| E128, 45 multi-pair tasks / 92 pairs | pass@1 | pass@2 | tokens / task-attempt | turns |
|---|---|---|---|---|
| per pair (E043 baseline) | 69.0% | 78.3% | 77.8k | 36.4 |
| per pair, Kaggle flags (4 samples, box 2) | 67.5% (cells) | ~76-84% | 74.6-81.0k | 34-36 |
| multi-test, 80k budget (runs 1-2) | 58.7-60.3% | 69.6-76.1% | 46-48k | 19-21 |
| **multi-test + 1.5x budget (runs 3-5 pooled)** | **67.8%** (cells 70.7%) | **81.5%** | **~55k (-29%)** | **~24** |

- At the plain 80k budget, 24/90 multi attempts end without an answer (budget or turn cap): one budget must
  cover 2-3 test inputs and E128 is token-hungry. `--multi-budget 0.5` scales output budget, soft budget and turn
  cap by 1 + 0.5 (n - 1).
- **All 120 tasks, final harness (run 7):** task pass@1 73.7% / pass@2 84.6% vs per pair 72.8% / 82.5%; per pair
  74.0% / 85.6% vs 71.9% / 81.4% (paired 13 vs 6); -17% tokens. Single-test tasks are unchanged (same path).
- Full Flash on the DeepSeek API (other session): all 120 tasks 77.6% / 88.6% vs per pair 74.5% / 84.9%, -23%.
- E128 vs full Flash: ~4-8 pp lower and ~45% more tokens in both modes (pruning: 14% of tool calls are identifier
  typos), so the gap is not specific to multi-test.

Bugs found and fixed in the harness (`evals/run_deepseek_tools.py`):
- `submit_answer(code, n)` rejected a list of numpy grids (the prompt's own example): nested conversion
  (multi-test first, then single-test too after a run-7 fallback failure).
- A model-defined `eval()` in the session broke the answer cell: built-ins are called via `import builtins as _b`.
- **Context overflow:** a 1.5x-2x budget lets prompt + 64k max_tokens pass max-model-len 163,840 (400 errors,
  4 attempts in runs 3-4, n=2 and n=3 tasks). `--context-limit N` caps each turn's max_tokens to the room left and
  ends an attempt with < 1024 tokens of room as stop="context" through the session-answer / fallback-solve path.
  Validated on d35bdbdc (4/4 attempts, 0 errors) and with a forced 16k limit.

## 5. Kaggle reality: ~1 finished attempt per task

- 12 h on Kaggle delivers ~11M generated tokens (~300 tok/s); two attempts per task need ~20M. The simulator
  (`E042 scripts/multi_sim.py`, task-level draws of 240 tasks from the eval set, E128 records) finishes a second
  attempt for only ~1% of jobs per pair and ~2-9% with multi-test. **pass@1 is the relevant metric**; pass@2 numbers
  above are eval-set accuracy without a deadline.
- Multi-test's Kaggle gain is efficiency: every first attempt finishes (per pair leaves ~11 of 335 pairs without one)
  plus a few second attempts.

## 6. KV pressure, the gate, and pod stress tests (Runpod 4xL4 at Kaggle's KV)

Pods ran the full Kaggle pipeline (`local_run.py`, `pods/podrun.sh`) on the eval set with KV pinned to Kaggle's pool
(util 0.938 -> 1,556,850 tokens; 0.928 -> 1,403,009 = 90%):

| config (multi-test + 1.5x budget unless noted) | max KV | outcome |
|---|---|---|
| per pair x 36 (Version 5) | 0.691 | 0.57-0.69 from 1.8 h, 4.8 h clean |
| x 44 ungated (80k budget) | 0.93 | thrash from 2.0 h |
| x 36 ungated | 0.834 | no thrash on the full pool; thrash at 3.4 h on 90% KV |
| x 32 ungated | 0.692 | flat from 2.2 h |
| x 52 + gate 0.75 | 0.925 | thrash from 1.65 h |
| x 44 + gate 0.6 | 0.716 | 3.8 h clean; running 21-44; attempts/h equal to ungated x 36 |
| x 44 + gate 0.6, 90% KV | 0.804 | crossed 0.80 at 1.36 h, gate pulled it back to 0.62 (running 44 -> 25) |
| **x 36 + gate 0.6 (candidate)** | 0.714-0.792 | full pool 0.714, 90% KV 0.792, heavy-only (60 largest tasks) 0.731, heavy + 90% 0.775, throttled pod 0.695: 3-4 h each, no thrash, 0 preemptions |

- The gate (`ARC_KV_ADMIT_MAX`, min 16 running) bounds KV whatever the task mix by letting concurrency drift down
  when attempts are long; the cost is ~15-30% tok/s in heavy phases and refill bursts when several attempts finish.
- Failure-mode tests on the candidate, all PASS: **vLLM killed mid-run** -> "server died", restart in 175 s (pod),
  36 in-flight attempts re-queued, submission.json valid throughout; **deadline path** (stop-new 1 h, force 1.17 h):
  admission closed at 0.67 h, 19 streams cut at the force deadline (2 multi-test cuts kept session answers), final
  submission valid, exit 0 before the hard stop; **throttled pod** (190-215 tok/s): KV 0.695, 1 gate hold.
- Accuracy of finished attempts, pods tests 1-5: 327/424 = 77.1% vs E043 (8x box) 80.1% by `bench_accuracy.py`, but that
  script is flawed (see 10); matched properly the V6 config on L4 is -0.6 pp vs its own reference, i.e. no gap.
- On 90% pools the late KV peak creeps to 0.78-0.79 (the gate only stops admissions); thrash onset is ~0.82+.
- Host RAM stays at 105-114 GiB (Kaggle cgroup 173 GiB). The Engram table pins 23.6 GiB per rank.

## 7. Throughput facts

- Kaggle L4s are power-limited to 50 W (pods: 72 W, power-capped at ~1.55 GHz). Decode is memory-bound and matches:
  ~315 tok/s at 36 streams on both. **Prefill is ~25-30% slower on Kaggle**: saturated uncached prefill ~1,180 tok/s
  (Kaggle) vs ~1,490-1,710 tok/s (pods), so a 2048-token chunk takes ~1.5-1.7 s and stalls all decode.
- **Occasional full re-prefills (~0.2% of turns):** a stream's next prompt hits only the shared system prefix (640
  tokens) and re-prefills 50-106k tokens, freezing decode for 3-5 min. Two triggers: the token-exact resync path
  (full re-render after a rewritten / non-canonical turn) and a stream idle in a tool call (timeouts over-
  represented) losing its cached tail; the hit falls to 640 rather than to the divergence point, consistent with the
  hybrid (sliding-window) cache needing state at the exact resume position. ~1 per server-hour, ~2-4% of throughput,
  present in every config (v3 too). Fix later (render only from the rewritten turn; investigate retention).
- Remote eval boxes: `--token-exact` uploads ~1-2 MB per turn; one lossy SSH tunnel throttles a 96-stream run. Use
  several `ssh -C` tunnels with a comma-separated `--base-url`.

## 8. The final candidate: dsv41-tools-submit-mtf Version 6

- Version 5's notebook and **exact inputs** (no new dataset version: Kaggle mount errors cluster on fresh cache
  entries, discussions 751344 / 751598 / 752147) + settings cell (36 workers, multi-test, budget 0.5, context limit
  163840, gate 0.6) + cell 0b with the pinned harness a8a5b665 and solver 4b344011
  (`kaggle/dsv41_tools/build_frozen_mt.py`).
- Cell 0b checks sha256, required features and the settings: in the **commit run** a failure raises with
  "E046 CHECK FAILED"; in the **competition rerun** it restores v3's kit files and runs per pair at 36 workers
  ("E046 FALLBACK"). Smoke commit: "E046 CHECK OK", 6/6 submitted.
- Kaggle stress test `chankhavu/dsv41-tools-mtf-bench` V1 (same cells + bench settings: eval set, force 3.0 h;
  outputs in `experiments/E046_multi_test_rollout/kaggle_bench/`): "E046 CHECK OK"; server up in 590 s, KV 1,554,183;
  295-319 tok/s at full concurrency, hits 0.94-0.99; KV gate first held at 1.96 h (KV 0.609; pods 1.41 h), running
  29-35 while gated; **max KV 0.727**; 0 preemptions, 0 errors; admission closed at 2.48 h, force cut of 11 streams at
  3.00 h, final submission valid (120/120), solver exit 0. 90 pairs answered: 72.2% correct vs E043 75.6% on the same
  pairs (pod test 1: 75.4% vs 78.7%; 88 shared attempts 65 vs 68).
- ~~L4 per-attempt accuracy is ~3 pp below the 8x-box records~~ **(retracted, see 10)**: `bench_accuracy.py`
  compared multi-test runs with per-pair references and keyed pairs by index across two eval files that differ in
  5 tasks. Matched by test input, V6 on L4 is -0.6 pp (CI -3.9..+2.9) vs its own reference; Kaggle bench 73.6 vs 74.1.
- Projection after the reviews: **V6 ~69 (range ~63-74)**, expected **+2.5 pp over Version 5 (defensible +1..+5)**;
  V6 below V5 with probability ~0.2-0.35 from attempt noise. Both shift down together if the private set is harder.
- **Submitted 2026-10-26 ~10:30 UTC.** Kaggle's queue was ~7 h, so scores land ~19 h after submission.

## 9. Process notes

- Sub-agents ran the boxes (Vast accuracy runs, pod fleet, audits, discussion and license research); the main
  session decided. Six Runpod 4xL4 pods + two Vast 4x RTX PRO 6000 boxes in parallel.
- Kaggle discussion threads (read-only) on mount failures: ERRORED_MOUNTING_DATASET (poisoned cache, hits Models,
  clusters on new versions) and slow FUSE reads; mitigation: frozen inputs, submit early in the UTC day.
- DeepSeek-V4.1-Flash-1017 is MIT; winner license CC BY 4.0; the 2026 OSAID clause is unresolved (memory note).
  No public posting; Kaggle models / datasets / notebooks stay private.

- **Open fixes for the next candidate:** `kaggle/dsv41_tools/next_candidate_fixes.md` (adversarial reviews of V6; each fix needs a regression test).

## 10. Adversarial reviews of Version 6 (2026-10-26, seven sub-agents, no Kaggle / GPU / paid API)

| review | verdict | key points |
|---|---|---|
| code audit | no blocker / major | pushed cells = local; smoke settings cannot leak; deadlines = v3; 65 tests pass; the rerun fallback restores the real Kaggle kit (solver 62144e4a, harness 1ad5ee24) = Version 5's code |
| evidence audit | supports V6 over V5; claims corrected | pass@1 multi vs per pair is a tie (+0.8, CI -4.6..+6.0), the win is -17% tokens; the "3 pp L4 gap" was a script artifact; gated runs have 1-2 five-minute refill dips at KV ~0.6; hours 5-11 untested for both versions |
| answer path to submission.json | no mapping bug | every correct record reaches the right row/slot in every run (bench 65/65, pods, restart, deadline), 0 discrepancies |
| 12 h timeline and limits | no blocker | worst-case end 11.75 h; little token slack (sim 70.9 / 65.3 / 59.7 at 1.0 / 1.1 / 1.2x eval token demand); one restart per run |
| deep code audit + 44 new tests | no blocker / major | 155 existing + 44 new tests pass; context-limit arithmetic exact; single-test path unchanged |
| e2e fake-server dry run (rerun mode) | no invalid submission in any scenario | 240-task test file, heavy 4-6-input set, server deaths, 5 cell-0b corruptions, /metrics 404/hang/NaN; latent: a main-loop exception ends the run silently with fallback rows |
| V6 vs V5 decision | submit V6 now | E(V6 - V5) ~ +2.5 pp; on L4 multi-test beats V5's per-pair run on multi-input pairs (73.5 vs 62.9%, same pairs); fallback floor = V5 |

All gaps found (30 items, each with a required regression test) are in `kaggle/dsv41_tools/next_candidate_fixes.md`;
the audit's tests and e2e tools are in `kaggle/dsv41_tools/tests_mtf_v6/`.
