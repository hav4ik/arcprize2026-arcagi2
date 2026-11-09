# Replicating candidate B32 (DeepSeek-V4.1 E128 tool solver on Kaggle 4×L4)

How to rebuild and re-run our ARC Prize 2026 (ARC-AGI-2) candidate B32 from this repository. Lineage, results and
decisions: `notes/2026-10-27_candidate_b32.md`. Earlier history: `notes/2026-10-26_dsv41_kaggle_inference_findings.md`.

## 1. External artifacts (private during the competition)

| artifact | where | built by |
|---|---|---|
| Base model | DeepSeek-V4.1-Flash-1017 (MIT) | DeepSeek |
| Pruned + quantized model (E128 2-bit GPTQ, tool keep set) | HF `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit`, revision 9bce98c5 (121 files, 131,090,911,045 B); Kaggle model `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit` (resharded to ≤ 5 GB) | `experiments/E043_dsv41_real_pruning/` (Engram row census, REAP expert statistics, GPTQ, export, parity) |
| vLLM fork wheel | `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` from github.com/yhfgyyf/vllm-deepseek-v4-sm89 | upstream fork |
| Offline wheel set (198 packages) | Kaggle kernel output `chankhavu/dsv41-sm89-fork-py312-cu130` | `experiments/E042_dsv41_l4_synthetic/kaggle/build_kaggle.py` |
| Solver kit (harness snapshot, solver, stage_model, 12-file vLLM overlay) | Kaggle dataset `chankhavu/dsv41-tools-kit` (Version 5's `dsv41_kit.tgz`, copy in `kaggle_kit_v5/`) | `build.py` |
| B32 replica (traces, logs, workload, report, code) | HF dataset `livobereshka/arc-agi2-dsv41-e128-candb32-replica` | `experiments/E055_kaggle_replica/` |

## 2. Code that defines B32

| piece | file(s) |
|---|---|
| Tool harness (pinned 882c817f) | `pinned_candb/run_deepseek_tools.py` (live copy: `evals/run_deepseek_tools.py`; `evals/arc_harness/`, `evals/auto_repair.py`) |
| Solver (pinned 77fe80a3): vLLM server, workers, KV gate + queue hold, deadlines, restarts, submission | `pinned_candb/solver_dsv41.py` (live copy: `solver_dsv41.py`; tests in `tests/`) |
| Train check / answer ranking (8fe36175) | `pinned_candb/train_check.py` |
| vLLM overlay, 16 files (sha 1a3a6679): the kit's 12 + fork PR #156 indexer skip + 4 KV-offload files | `experiments/E056_candidate_b/kit/`, built by `experiments/E056_candidate_b/scripts/build_kit.py`; sources: `vllm_overlay_src/` (kit 12), `experiments/E054_exact_cherrypicks/overlay/` (#156), `experiments/E051_kv_offload/overlay/` (offload; v2 scheduler in `overlay_v2/`) |
| Notebook builder | `build_frozen_mt.py` (+ `cells/cell0b_pins.py`), tests `tests/test_notebook_cells.py` |

Run the tests: `.venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests` and
`.venv/bin/python -m unittest discover -s evals/tests` (the harness tests need a sandbox, see `evals/sandbox_docker/`).

## 3. Build the B32 notebook

```bash
P=kaggle/dsv41_tools/pinned_candb
KVT='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":34359738368,"offload_prompt_only":false,"swa_store_policy":"checkpoint","shared_region":false,"eviction_policy":"lru"}}'
EXTRA=$(python3 -c "import json,sys; print(json.dumps(['--kv-transfer-config', sys.argv[1]]))" "$KVT")
.venv/bin/python kaggle/dsv41_tools/build_frozen_mt.py --slug dsv41-tools-submit-candb --out kaggle/dsv41_tools/out/nb_candb32 \
  --workers 56 --budget 0.5 --gate 0.85 --retention 512 --util 0.945 --auto-repair --admit-max-waiting 1 \
  --smoke-minutes 40 --smoke-stress --smoke-integrity experiments/E043_dsv41_real_pruning/results/kaggle_e128_manifest.json \
  --solver $P/solver_dsv41.py --harness $P/run_deepseek_tools.py --train-check $P/train_check.py \
  --overlay experiments/E056_candidate_b/kit --server-extra "$EXTRA"
```
The submitted notebook itself is `out/nb_candb32/` (git tag `b32-submission`). Rebuilding with today's builder gives the
same cells except two inert differences: the gzip timestamp inside cell 0b's payload (same file hashes) and the smoke
report's hang-test section, which only runs when a hang-test cell wrote `smoke_hang.json`.

### B32.1 (B32 + failure handling; the handoff notebook)
Same model, overlay and settings as B32, plus the solver's S20 progress watchdog (600 s), S21 re-launch hold and S22
post-restart ramp (16 + 8 per minute). The pinned files ship as the private dataset `chankhavu/dsv41-candb1-pins`
(`--pins-dataset`; no base64 in the notebook). Its commit run is a 32-min smoke that freezes the engine on purpose
(`cells/cell2c_hang_test.py`) to exercise the watchdog, restart and ramp; the competition rerun does not.
```bash
P=kaggle/dsv41_tools/pinned_candb1
.venv/bin/python kaggle/dsv41_tools/build_frozen_mt.py --slug dsv41-tools-submit-candb1 --out kaggle/dsv41_tools/out/nb_candb1 \
  --pins-dataset dsv41-candb1-pins \
  --workers 56 --budget 0.5 --gate 0.85 --retention 512 --util 0.945 --auto-repair --admit-max-waiting 1 \
  --stall-restart 600 --conn-hold --restart-ramp 16,8 \
  --smoke-minutes 32 --smoke-stress --smoke-hang \
  --smoke-integrity experiments/E043_dsv41_real_pruning/results/kaggle_e128_manifest.json \
  --solver $P/solver_dsv41.py --harness $P/run_deepseek_tools.py --train-check $P/train_check.py \
  --overlay experiments/E056_candidate_b/kit --server-extra "$EXTRA"
# upload the pins folder once (private): kaggle datasets create -p kaggle/dsv41_tools/out/pins_dsv41-candb1-pins
```
The notebook keeps Version 5's inputs (competition, kit dataset, wheel kernel, model) and carries the pinned files in
cell 0b (or, with `--pins-dataset <slug>`, in a small attached dataset). The commit run is a smoke (B32: 40 min, ~46 min of session; B32.1: 32 min); the
competition rerun runs the full solver. The builder refuses pinned files whose imports the kit cannot satisfy.
Kaggle needs the competition attached to get 4×L4 (otherwise 2×T4).

## 4. Run a local Kaggle replica (4×L4, ~11 h)

1. **Box:** a 4×L4 pod with ≥ 256 GB host RAM and ≥ 200 GB disk (or 4×4090 for development). Install the stack with
   `experiments/E042_dsv41_l4_synthetic/scripts/setup_box.sh` → `box_install.sh`; download the model with the resumable
   HTTP downloader `experiments/E057b_workers64_stress/results/setup/dl_e128_http.sh`.
2. **Kit and venv:** `experiments/E046_multi_test_rollout/pods/podprep.sh` (kit, 12-file overlay, tool venv, data), then
   `experiments/E056_candidate_b/gpu/setup_candb_venv.sh` (venv copy + the 16-file overlay through the notebook's cells,
   all shas and the offload CPU tests checked). Stage the kit with the `pinned_candb/` files at their kit paths.
3. **Launch:** `experiments/E055_kaggle_replica/toolkit/replica_run.sh <port> <ip> <name> evalx2t_s2026 [ENV=VAL…]`
   (see `experiments/E055_kaggle_replica/launch_candBf_x2t.sh` for B32's settings). It copies the workload, sets
   Kaggle's KV pool (util 0.938 on pods), moves Kaggle's deadlines 600 s earlier, and starts memory logging.
4. **Watch:** `experiments/E055_kaggle_replica/poll_e055d.sh`-style polls, `toolkit/memdetail.sh` for per-process memory.
5. **Score:** `experiments/E055_kaggle_replica/toolkit/replica_report.py <run_dir> --solutions … --challenges … --idmap …`
   gives the raw task-weighted score, the Kaggle-clock score (Kaggle's ~1.1× slowdown, attempts after Kaggle's force
   deadline dropped), coverage and bottlenecks. Workloads: `toolkit/workloads/` (`make_workload.py`).

Expected: B32 raw 79.1, Kaggle-clock 77.4 (one draw, sd ~3); V5 with the same protocol 57.6 vs its leaderboard 58.19.

## 5. Experiments behind each piece
E042 (stack on L4), E043 (pruning, quantization, tool keep set), E046 (multi-test rollouts, pods), E050 (reasoning effort),
E051 (KV CPU offload), E052 (new harness code), E053 (vLLM upgrade options), E054 (exact cherry-picks), E055 (replicas
and toolkit), E056 (combined overlay, Kaggle smoke), E057b (workers stress), E059 (restart under load). Each folder has a
README; `experiments/experiments.tsv` is the registry.
