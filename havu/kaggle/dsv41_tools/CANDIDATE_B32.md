# Candidate B32: start here

The Kaggle candidate approved for submission on 2026-10-27, and the base for the next candidates.

| | |
|---|---|
| Kaggle notebook | https://www.kaggle.com/code/chankhavu/dsv41-tools-submit-candb, **Version 2** (Version 1 is the 16 GiB build) |
| Submission description | `B32: DSv4.1-Flash E128 2-bit tool agent \| multi-test + new harness (auto-repair) \| KV CPU offload 32 GiB \| 56 workers, KV gate 0.85 + queue hold \| indexer skip, retention 512 \| local 11h L4 replica 77.4 (Kaggle-clock)` |
| Expected leaderboard | ~74 realistic (80% range 67-80); replica-based ~74-77 |
| Git tag | `b32-submission` on hav4ik/arcprize2026-arcagi2 |

## What it is
DeepSeek-V4.1-Flash-1017 pruned to 128 experts, 2-bit GPTQ, tool-tuned (E043), served by the SM89 vLLM fork with a
16-file overlay (kit 12 + fork PR #156 indexer skip + 4 KV-offload files), TP4 on Kaggle's 4xL4, driving a Python-tool
agent (one rollout per multi-input task, auto-repair, train-check answer ranking). Lineage: v3 (LB 50.97) → V5 (58.19) →
V6 (61.94) → new harness code → Candidate A → B → B32 (`notes/2026-10-27_candidate_b32.md` section 1).

## Files (frozen, hash-checked in the notebook)
| file | sha256 (8) |
|---|---|
| `pinned_candb/solver_dsv41.py` | 77fe80a3 |
| `pinned_candb/run_deepseek_tools.py` | 882c817f |
| `pinned_candb/train_check.py` | 8fe36175 |
| `pinned_candb/auto_repair.py`, `auto_repair_names.json` | 2292369c, 571f47eb |
| `experiments/E056_candidate_b/kit/vllm_overlay_rw.tgz` (16 files) | 1a3a6679 |
| built notebook | `out/nb_candb32/` |
The live `solver_dsv41.py` and `evals/run_deepseek_tools.py` have moved on (S20 watchdog, E058 image prompts): build new
candidates from pinned copies (the builder refuses imports the Kaggle kit cannot satisfy).

## Key settings
56 workers, `ARC_KV_ADMIT_MAX=0.85`, `ARC_ADMIT_MAX_WAITING=1`, multi-test budget 0.5, context limit 163840, auto-repair,
GPU util 0.945 (Kaggle KV pool 1,554,183 tokens), Humming dense, FULL CUDA graphs ≤ 64, max-num-seqs 64,
`--prefix-cache-retention-interval 512`, `--kv-transfer-config` OffloadingConnector `cpu_bytes_to_use 34359738368`
(32 GiB = 8 GiB per rank; pinned buffers round up to powers of two), private memory, checkpoint SWA policy, LRU.
Exact build command: `REPLICATE.md` section 3.

## Evidence
- Test record (every test, stage by stage): `experiments/E056_candidate_b/TESTING_B32.md`.
- Local replicas and calibration: `experiments/E055_kaggle_replica/` (V5 replica 57.6 vs LB 58.19; V6 replica projected ~63-64
  vs LB 61.94; B32 77.4 Kaggle-clock, raw 79.1; provenance audit and pre-registered calibration rule in its README).
- Full B32 replica data: HF dataset `livobereshka/arc-agi2-dsv41-e128-candb32-replica` (private).
- vLLM hang analysis: `notes/2026-10-27_vllm_hang_protection.md`.

## Known risks (accepted for this submission)
1. Upstream OffloadingConnector async-load scheduler deadlock under KV pressure (fixed in vLLM 0.31 #58947, not in the
   fork): silent stall, ~1-3% per run (never seen in ~49 h of our offload serving).
2. Host memory: ~141-145 GiB projected at 12 h vs Kaggle's 173 GiB; tool sandboxes are killed first on overrun (tested).
3. Engine-core or API-server wedges are not detected (no solver progress watchdog in B32).

## How to build on it
1. Follow `kaggle/SUBMISSION_TESTING_PROTOCOL.md` for every new submission.
2. Rebuild and re-run: `REPLICATE.md` (notebook build, local 4xL4 replica with `experiments/E055_kaggle_replica/toolkit/`).
3. Next-candidate list, in order of expected value: `next_candidate_fixes.md`
   - S20 progress watchdog on (`ARC_STALL_RESTART_S=600`) + cherry-pick vLLM #58947 (deadlock fix): removes risk 1 and 3;
   - S19 metrics reset after restart (logging), S18 host-capacity guard;
   - E057 8-bit two-shot all-reduce (decode +3-7%, prefill ~1.3x, planned);
   - analysis of B32's traces: 21 pairs without a first-attempt answer, value of second attempts (attempt_2 policy);
   - a second B32 replica draw (`launch_candBf2_t2.sh`, workload `evalt2_s2027`) is running for a tighter estimate.
