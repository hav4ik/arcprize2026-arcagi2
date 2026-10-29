# E064: B32's harness with Manh's server — external-server variant NOT USED

2026-10-28 ~18:20 UTC: started as "B32's harness on top of Manh's server notebook" (solver in an opt-in
external-server mode against his server on :18084). **Superseded the same hour by the user's plan B32-M** (B32's
notebook with Manh's environment, the solver launching his vLLM itself; see "B32-M" below). The files of the first
variant are kept, marked not used, and are not part of any candidate:
- `kaggle/dsv41_tools/pinned_manh/` (B32's pinned files + ARC_EXTERNAL_SERVER mode + 02b): NOT USED.
- `kaggle/dsv41_tools/build_manh_stack.py`, `kaggle/dsv41_tools/cells/manh_m{1,2,3,4}_*.py`: NOT USED.
- `kaggle/dsv41_tools/tests/test_solver_external.py` (8 tests, pass): tests of the unused variant.

## B32-M (rebuilt 2026-10-28 ~20:00 UTC with plan B + a 48-min smoke; not pushed, the coordinator reviews and pushes)
B32's notebook with Manh's environment layer: his image (Python 3.13), CUDA 13 and wheels (vllm 0.32.1rc1+g5e1bccd07c,
torch 2.14), the L4 replica's server config, 80 workers, gate 0.85; B32's code byte-identical; Manh's extra flags OFF.
- Builder `kaggle/dsv41_tools/build_b32m.py` (wraps build_frozen_mt.py with B32's REPLICATE.md options); cells
  `kaggle/dsv41_tools/cells/b32m_{env,manh_flags,contract,report}.py`; versions `kaggle/dsv41_tools/b32m_manh_versions.json`.
- Smoke build (= submission build): `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m/` (README_B32M.md: checks, wall
  time ~45-50 min in-notebook, weight-loading files, rebuild steps). Full build without smoke cells: `..._full/`.
- Plan B (cell 2r): his row-chunk fix a6c3ee8c2b patched into his installed nvidia/model.py, sha256-guarded (his
  Kaggle wheel g5e1bccd07c lacks it: KeyError 'layers.1.engram.embed.weight.rows_0' on the Kaggle copy).
- Smoke: kill switch 47:30 (stop-new 43, force 45), 3 smallest tasks, metrics every 30 s, timeline + windows in the
  report; ~48-50 min in-notebook, ~55 min session.
- Tests: `kaggle/dsv41_tools/tests/test_b32m.py` (9, pass); the whole dsv41 suite 100 pass.

## L4 E2E replica (E064b)
A 4-6 h local end-to-end replica of B32-M on Runpod L4 pod #2 (5z3srm3108wcwl: 4x L4 72 W, 24 vCPU, 250 GB RAM,
200 GB disk), with the E055 toolkit: B32's kit and B32-M's settings, in Manh's environment. Scripts are in `l4e2e/`.

**Setup (2026-10-28 20:00-21:00 UTC):**
- **Python 3.13** venv `/root/b32m/venv` (uv cpython 3.13.18). On Kaggle it's the image's Python 3.13.
- **Manh's wheels:** all 294 top-level wheels of `nguyennguyen599/vllm-wheels` **version 8** (files created 17:51-17:53
  UTC), downloaded one by one with kagglehub, version-pinned (`l4e2e/dl_kaggle_files.py`).
  - The CLI's whole-dataset download returned 404.
  - A first 12-thread attempt hit Kaggle 429s and was stopped; the rest were downloaded sequentially with backoff.
- **The vLLM wheel is identical in v8 and v9** (Manh's 20:08-20:10 upload):
  - vllm-0.32.1rc1.dev0+g5e1bccd07c, sha256 85e05736…f22;
  - its model.py is the base 5e1bccd07c (ead1e087…), so v9 does NOT contain the row-chunk fix.
- **Install:** `l4e2e/install_b32m.sh`, Manh's cell 4 commands unchanged, minus the /kaggle/working source installs
  (unsloth-zoo, unsloth, peft, quack-kernels). Every step succeeded offline from his wheels, numpy/scipy for the tool
  sandbox included. torch 2.14.0+cu130, vllm 0.32.1rc1.dev0+g5e1bccd07c, numpy 2.4.6, scipy 1.18.1
  (`/root/b32m/freeze.txt`).
- **Plan B:** cell 2r, extracted verbatim from the built notebook (`l4e2e/cell2r_from_b32m.py`, sha256 a6d3559a…), run
  with the venv's Python. Action **patched**: model.py ead1e087… → b2a04fc2… (= a6c3ee8c2b).
- **Model: the Kaggle Model copy**, assembled at `/root/dsv41/model-e128k` (`l4e2e/make_kaggle_copy.py`):
  - 113 files (49.6 GB: gptq/, configs, encoding/) hardlinked from the pod's HF copy after sha256 matched the Kaggle
    manifest (experiments/E043_dsv41_real_pruning/results/kaggle_e128_manifest.json);
  - the other 23 files (81.5 GB: resharded backbone, 16 Engram row-chunk files, index, serving/, README) downloaded
    from the Kaggle Model one by one;
  - the whole copy is then verified against the manifest's sha256.
  - Not enough disk for both copies, so the HF copy's backbone and Engram files on pod #2 were deleted.
- **Launcher:** `l4e2e/launch_b32m.sh`. The server args and env JSON are byte-equal to B32-M's cell 00; util is 0.938
  (the pods' equivalent of Kaggle's 0.945); the tool sandbox runs on the same 3.13 venv.

## Kaggle smoke V2 (2026-10-28, 20:50-21:26 UTC): PASS ("B32-M SMOKE OK")
- Settings: latest image pinning; GPU gate saw 4 GPUs on Python 3.13.15; row-chunk patch "patched" with the hash verified.
- Timeline: solver at +13.2 min (install about 5 min: cells 1-5 ran from 436 s to 731 s); server up after 1105 s, i.e. weights 139-193 s + CUDA-graph capture 541 s; first attempts at +31.8 min (B32: ~+14 min).
- Server: KV 3,174,039 tokens; weights 14.8 GiB/GPU.
- API contract OK: dsv41 at max_model_len 163840; thinking kept via reasoning_content and via reasoning; token ids; metrics.
- Attempts: 4/4 correct with train_pass (2 tasks, one with 2 test inputs); final submission written and valid.
- Host: peak Shmem 125 GiB, min MemAvailable 39.9 GiB (cgroup 173 GiB).
- GPUs: power limit 72.00 W, peak 71.9 W, 2040 MHz.
- Kernel `chankhavu/dsv41-tools-submit-b32m` Version 2 is the submission build (reruns ignore the smoke settings). Version 1 failed: no GPU with the pinned image.

**Run `b32m_x2t`** (launched 21:26 UTC; workload evalx2t_s2026; deadlines shifted 600 s: stop-new 10.67 h, force
11.00 h; util 0.938; 80 workers):
- **The Kaggle copy loads through the plan-B loader:**
  - server up after **555 s**, against 440 s for pod 1's manh_x2t_w80 (our build, Python 3.12) and 235 s for B32;
  - model loading 14.8 GiB per GPU in ~91 s;
  - **GPU KV cache 3,058,602 tokens**, the same as pod 1;
  - the Engram tables offloaded per rank have the same row counts as pod 1's HF-copy run (5,745,304 / 21,917,600 /
    42,116,852 / 66,335,128 on layer 1), so the row-chunk loader rebuilt the same tables.
- Hourly interims: `l4e2e/interim.sh <tag>` → `l4e2e/reports/interim_<tag>.txt`.

## Next version: host-memory guard (S23)
B32-M, like B32, has no active host-memory monitor: the KV gate watches GPU KV only. Headroom against Kaggle's 173 GiB
cgroup is ~28 GiB: peak unreclaimable 145.2 GiB over 11 h at x80; Shmem ~125 GiB is fixed at startup.
- An OOM kill of the vLLM server costs a ~18 min restart.
- An OOM kill of the kernel or solver fails the run.
Planned fix, with the design and tests: `kaggle/dsv41_tools/next_candidate_fixes.md` S23. The solver gates on anon + shmem
from memory.stat (soft limit: pause launches; hard limit: kill the largest sandbox).
- **Interims sent to the coordinator** (`l4e2e/interim2.py`; reports `l4e2e/reports/coord_2300.txt`, `coord_2335.txt`):
  - **Health at 2.14 h:** 0 error lines; cg unreclaimable host-memory peak 142.5 GiB (Kaggle 173); KV 0.57; 0 preemptions;
    0 host restores.
  - **gen tok/s by half hour, b32m / manh80 (pod 1) / B32 (old pod):**

    | pod hours | b32m | manh80 | B32 |
    |---|---|---|---|
    | 0-0.5 | 497 | 539 | 408 |
    | 0.5-1.0 | 600 | 615 | 464 |
    | 1.0-1.5 | 588 | 607 | 471 |
    | 1.5-2.0 | 606 | 618 | 431 |

    About 2% below pod 1 and +35% over B32.
  - **First attempts done by 2.08 h:** 78 / 82 / 70 (R1: 79).
  - **Paired first-attempt score** on the 82 tasks finished in all four runs: b32m 81.7, manh80 82.3, B32 86.4, R1 83.3.
    b32m − manh80 is −0.6 [−8.3, +6.2]; b32m − B32 is −4.7 [−12.5, +2.3].
  - **token_exact:** breaks 0.218 per turn, 0 rerenders, 0 resyncs, 0 parse fallbacks.
  - Verdict at 2.1 h: same speed, memory and early accuracy as pod 1's run of the same server config.

**Scope change (coordinator, ~00:00 UTC 10-29):**
- `b32m_x2t` runs the full ~11-12 h to the kill switch, as the CV side of a CV/LB gap check for the submitted Kaggle notebook chankhavu/dsv41-tools-submit-b32m **Version 3** (submitted ~23:45 UTC 10-28).
- Disk: the unused fork venvs /root/manh{,2}/venv and the build leftovers were deleted (pod #2 now at 83%).
- The final report adds a Kaggle-clock variant with B32-M's real Kaggle startup (`l4e2e/report_shift.py <shift_s>`): 1300-1400 s instead of the toolkit's 600 s.
- **4.53 h** (`l4e2e/reports/coord_4h5.txt`):
  - **Health:** 0 error lines; host-memory peak 143.1 GiB; KV 0.63; 0 preemptions; host restores starting (76 tok/s).
  - **gen tok/s by half hour, 2.0-4.5 h, b32m / manh80 / B32:** 583/582/368, 575/591/399, 564/573/373, 567/568/345,
    607/598/422.
  - **First attempts done:** b32m 193, manh80 194, B32 143, R1 163.
  - **Paired first-attempt score** on 191 common tasks: b32m 75.7, manh80 75.4, R1 76.0, B32 79.9.
    - b32m − manh80: +0.3 [−5.2, +5.4];
    - b32m − B32: −4.0 [−9.2, +0.9].
    - R1, which is B32's own stack, sits the same ~4 below candBf_x2t, so that run's draw is likely high.

### Final: b32m_x2t (CV of B32-M Version 3), finished 2026-10-29 07:42 UTC
Every attempt finished early: 480/480, first and second attempts on all 240 tasks, by pod **10.27 h** (stop-new 10.67 h).
**Raw 77.92** task-weighted (sd 3.1), 267/344 pairs.
- **Health over the whole run:**
  - 0 errors, OOMs or restarts;
  - **0 preemptions**; KV max 0.72 (never ≥ 0.80); host restores max 351 tok/s;
  - cg unreclaimable host memory peak **146.6 GiB** (Kaggle 173 GiB);
  - mean 557 gen tok/s from server-up (healthy reference 601); prefill share 13.6%; 20.1M tokens generated;
  - token_exact: 8,664 turns, 0.294 breaks per turn, 0 rerenders, 6 resyncs, 4 parse fallbacks;
  - stops: 443 submitted, 26 output_budget, 9 max_turns, 1 context.
- **Kaggle clock** (`reports/final_*.md`):

  | startup gap | central (k 1.45, d 0.94) | optimistic | pessimistic | attempts kept |
  |---|---|---|---|---|
  | toolkit default 600 s | **77.71** | 77.92 | 76.88 | 469/480 |
  | B32-M's real Kaggle start, 1300 / 1350 / 1400 s | **76.88** | 77.92 | 76.88 | 461 / 461 / 459 |

- **Paired** (Kaggle clock central, 600 s, 240 tasks):
  - vs B32 `candBf_x2t` (77.43): **+0.28** (sd 2.0);
  - vs pod 1's `manh_x2t_w80` (same fork config on our build): −0.49 (sd 2.1);
  - vs B32+R1 `b32r1_x2t`: −0.21 (sd 2.0).
  - A tie with all three.
- **First-attempt accuracy is lower than B32's**, made up for by second attempts. On 225 tasks finished in all four
  runs (task-weighted; multi-test list predictions scored per input): b32m 71.6, manh80 72.9, R1 74.5, B32 78.0.
  - b32m − B32 = **−6.1 [−11.4, −1.1]**, outside noise; b32m − manh80 = −1.5 [−6.9, +3.5].
  - Both fork runs trail B32 on first attempts. B32-M wins it back by finishing all 240 second attempts (B32: 106).
  - Same score, different composition: if Kaggle is slower than the pod, B32-M has more second attempts to lose before
    first attempts are cut.
- **Projected LB** (Kaggle-clock central, 600 s / real startup):
  - pre-registered rule (mean gap of V5/V6, −1.2): 76.5 / 75.7;
  - **with the new calibration (mean gap of V5, V6, B32: −3.6): 74.1 / 73.2**;
  - with B32's own gap (−8.5): 69.2 / 68.3.
- **Copied off:** the run (`l4e2e/runs/b32m_x2t`, incl. the traces tarball), the pod setup logs (`l4e2e/pod_logs/`), the
  reports (`l4e2e/reports/`). Secret scan: 252 files, 0 hits. Pod #2 left running and idle (GPUs empty, loggers
  stopped) for the user to decide.
