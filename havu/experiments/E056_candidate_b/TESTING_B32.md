# Candidate B32: test record

Every test run on Kaggle candidate B32 before submission, in the order of the team protocol
(`kaggle/SUBMISSION_TESTING_PROTOCOL.md`). B32 = `chankhavu/dsv41-tools-submit-candb` Version 2: DeepSeek-V4.1-Flash-1017
E128 2-bit tool model, one rollout per multi-input task, new harness code + auto-repair, 56 workers, KV gate 0.85 + queue
hold, 32 GiB KV CPU offload, fork PR #156 indexer skip, prefix-cache retention 512, Kaggle KV pool. Lineage and results:
`notes/2026-10-27_candidate_b32.md`; rebuild and re-run: `kaggle/dsv41_tools/REPLICATE.md`.

Status as of 2026-10-27 10:15 PDT: stages 0-5 passed (restart test on L4 PASS; 16 GiB twin on 4090 pending); stage 6
(Kaggle smoke of Version 2) running; stage 7 calibration pending.

## Stage 0. Frozen code
| item | value |
|---|---|
| solver | `kaggle/dsv41_tools/pinned_candb/solver_dsv41.py` 77fe80a3 |
| harness | `pinned_candb/run_deepseek_tools.py` 882c817f |
| train_check / auto_repair | 8fe36175 / auto_repair.py 2292369c, auto_repair_names.json 571f47eb |
| vLLM overlay (16 files) | `experiments/E056_candidate_b/kit/vllm_overlay_rw.tgz` 1a3a6679 (deterministic build, `scripts/build_kit.py --check`) |
| notebook | `kaggle/dsv41_tools/out/nb_candb32/` built by `build_frozen_mt.py` (command in REPLICATE.md, plus `--smoke-minutes 40 --smoke-stress --smoke-integrity experiments/E043_dsv41_real_pruning/results/kaggle_e128_manifest.json`) |
| inputs | Version 5's: competition, kit dataset `chankhavu/dsv41-tools-kit`, wheel kernel `chankhavu/dsv41-sm89-fork-py312-cu130`, model `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1` |

The replica, the stress tests and the smoke all ran these exact files (hash-checked on every box and in the notebook).
Bug found by freezing: the live harness had gained an import (`grid_image.deepseek_cell`, E058) that the Kaggle kit lacks;
a notebook built from live code would have failed every attempt. The builder now refuses unresolved imports.

## Stage 1. Unit and packaging tests (CPU)
- `kaggle/dsv41_tools/tests`: 76 tests OK, including commit / rerun / fallback paths of every notebook cell, the queue
  hold (S17), engine death as a connection error (S15), host-restored tokens counted as cached (S16), the smoke cells
  (deadlines, sampler, report, sandbox check against the real executor, input integrity with a corrupted file), the
  pins-dataset cell and the import guard.
- `evals/tests`: 241 OK (harness).
- Offload overlay CPU tests: 26 (v1) / 31 (v2) OK on every venv built.

## Stage 2. Component tests on dev boxes (4×4090)
| test | result | where |
|---|---|---|
| E054 #156 indexer skip | logits bit-identical on 74 shapes; decode step −5.7% / −6.5% at 36 / 48 streams | `experiments/E054_exact_cherrypicks/` |
| E051 KV offload byte-level parity | 8/8 cases, all 4 ranks, private pinned memory | `experiments/E051_kv_offload/` |
| E056 combined kit | installs through the notebook's own cells (16/16 shas); server starts with the offload line; parity 8/8 with the indexer skip on; late-phase A/B tie | `experiments/E056_candidate_b/README.md` |
| E051 late-phase replays | offload 874-971 tok/s vs ~95 without at 36-48 heavy streams; 16 GiB collapses at 56 heavy streams (host-pool cliff); 32 GiB clears 64 | E051 README |
| Sandbox limits | each tool session: 4 GiB address space, 64 MB files, 1,200 s CPU, oom_score_adj 1000; 6 GiB allocation blocked, runaway loop interrupted, memory returned | `kaggle/dsv41_tools/cells/cell2b_sandbox_check.py` (local run) |

## Stage 3. Local Kaggle replica (4×L4, full run)
`candBf_x2t` on Runpod 4×L4 hegn, 2026-10-27 04:08-15:08 UTC, 11.01 h, workload `evalx2t_s2026` (240 tasks / 344 pairs),
Kaggle KV pool, Kaggle deadlines −600 s, 32 GiB pool.

| metric | value |
|---|---|
| raw score (task-weighted) | 79.1 (attempt_1 only 77.0) |
| Kaggle-clock score | 77.4 |
| coverage | 240/240 first attempts (last at 8.63 h), 106 second attempts |
| accuracy of answered first-attempt pairs | 78.9% |
| throughput | mean 389 tok/s, 15.1M output tokens, decode batch mean 42.6 (27-56) |
| KV and offload | KV 0.85-0.94 from hour 1; 487 preemptions in bursts, absorbed; 1,000-4,000 tok/s restored from host; recomputed share flat (~3%) |
| memory | server peak 141 GiB (PSS anon + shared); growth slowing to ~0.1 GiB/h per worker; GPU memory flat |
| sensitivity | all first attempts finish unless Kaggle is > ×1.3 slower than the pod (measured ×1.1) |

Earlier B builds: the first B replica (28 GiB, before S17) exposed the gate-reopening bug at 2.4 h (stopped at 4.05 h);
the 16 GiB B+ and 32 GiB B+ replicas were stopped early for higher-value runs. Archives: `experiments/E055_kaggle_replica/runs/`;
full data: HF `livobereshka/arc-agi2-dsv41-e128-candb32-replica`.

## Stage 4. Targeted stress tests
| test | result | where |
|---|---|---|
| E057b heaviest-90-tasks, real solver, 16 GiB, 80 min (4090) | 56 vs 64 workers: 765 vs 774 tok/s, 0 vs 0 preemptions, peak in-flight context 2.68M / 2.62M (cliff ~4.8M), host ≤ 49% of the box | `experiments/E057b_workers64_stress/` |
| Hour-by-hour pressure in the replica | stage 3 | |

## Stage 5. Failure paths
| test | status | where |
|---|---|---|
| E059 server restart under load (forced TP1 worker kill at +1.5 h, 2-s memory sampling), 32 GiB on L4 | **PASS**: old memory released in ~16-56 s before the new server allocated; restart peak 1.05x steady (151.7 vs 144.4 GiB); all in-flight jobs re-queued (S15), 0 lost; server up again 140 s after restart start; >= 90% throughput 19 min after the kill | `experiments/E059_restart_under_load/README.md` |
| E059 twin, 16 GiB on 4090 57983124 | running | same |
| Solver failure paths (server death, restarts, re-queue, deadlines, garbled metrics, missing KV readings) | unit-tested (stage 1) | `kaggle/dsv41_tools/tests/test_solver.py` |

## Stage 6. Kaggle smokes (user-approved)
| run | what | result |
|---|---|---|
| Version 1 (B16, 2026-10-27 06:20 UTC, ~21 min, ~0.75 h quota) | 3 tasks; 30-min deadlines; memory/GPU/disk report | PASS: hashes, overlay, KV 1,554,183, offload line, 6/6 pairs, valid submission; host anon 10.8 + shmem 106.9 GiB at start vs 173 GiB; GPUs 72 W; allocator unset; /kaggle/working 19.5 GiB free (`kaggle_smoke_v1/`) |
| **Version 2 (B32, the notebook to submit, 16:58 UTC, ≤ ~42 min, ≤ ~1.4 h quota)** | input integrity (136 model files sha256, wheels, kit, competition files); 56-session sandbox check; 24 tasks with the GPU KV pool cut to ~333k tokens (evictions, host restores, preemption, gate, queue hold); force cut and final write; memory/GPU/disk/sandbox report and coverage line | running (`kaggle_smoke_v2_b32/`) |

## Stage 7. Score estimate
Kaggle-clock replica 77.4. Calibration: V5's replica 57.6 vs LB 58.19; V6's replica (running on wd44, Kaggle-clock point
~12:50 PDT) projects to ~52-56 vs LB 61.94. Realistic projection ~74 (80% range 67-80); see the notes file.

## Sign-off
- [ ] Version 2 smoke passed (integrity OK, sandbox OK, coverage shows host restores and preemptions, valid submission)
- [x] E059 restart test passed on L4 with 32 GiB (restart peak 1.05x steady); 16 GiB twin on 4090 pending
- [ ] User approves the submission
