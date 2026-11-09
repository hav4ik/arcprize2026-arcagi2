# E072 harness-improvement loop: a job queue that keeps both Runpod 4xL4 pods busy

Started 2026-10-31 (UTC). Pods (never stop or delete; pod 3 `ccn5ns0b8emb3o` was handed to Codex on 2026-10-31, its loops here are stopped): pod 2 `5z3srm3108wcwl` (ssh root@213.173.105.27 -p 18070), pod 5
`u71eibokv8t34t` (root@213.173.105.24 -p 13984, V12 replica 2). Experiments: `BACKLOG.md`. Pairing: `paired.py`.

## What is on each pod (`/root/queue`)
- `bin/queue_runner.sh`: detached loop (setsid; `runner.pid`; stop: `touch /root/queue/runner.stop`; start: `bin/start_runner.sh`).
  It waits until no `e062pl/run_nb.py` process runs and every GPU is under 2 GB (3 checks, 15 s apart), takes the oldest
  `pending/*.job` (else writes a filler from `filler.job.tpl`: an incumbent V12 replicate on eval120_kaggle), runs `bin/run_job.sh`,
  moves the job to `done/` or `failed/` (failure backoff 2 / 10 min), and starts the next. It never kills a run it did not start.
- `bin/run_job.sh`: verify notebook sha256 (prefix) and the wheels set -> `mk_variant_nb.py` (env overrides; patches only the `_MT_EXPECT`
  entry of cell 0b that an override changes, statically verified; see R10) -> `stage_model.sh` (E128: missing/short files from Kaggle,
  token via the local watcher handshake, sha256-verified against `bin/e128_manifest.json`; when the disk is short and no notebook runs, it first
  deletes the img-causal model, which is re-downloadable from HF) -> `launch_nb.sh` (= phase_h `launch_nb_owner.sh`: rerun mode,
  fresh kenv, `env -i`, nbclient `run_nb.py`, `v12_sampler.py`, memlog, uv-cache cleanup, plus the job's env and workload) -> poll
  every 30 s (NOTEBOOK_DONE, runner gone, or the job's max Kaggle clock -> `stop_run.py`) -> `reap.py` (sampler/memlog/server by pid
  file, cmdline-checked) -> `score_job.py` -> `END_REASON`. Live `score.json` every 10 min.
- Data: `/root/nbruns/<run>/` (run = `q_<label>`), `/root/kwork_<run>/`, `index.tsv`, `logs/runner.log`.

## Job file (shell env, sourced after `pod.env`)
`JOB_LABEL`, `NB_PATH`, `NB_SHA256` (prefix ok), `WORKLOAD` (`eval120_kaggle`), `KT_WHEELS_DIR` (`$WHEELS_V11`), `WHEELS_EXPECT_COUNT`,
`WHEELS_EXPECT_VLLM`, `MODEL_KIND` (`e128`|`none`), `EXTRA_ENV="K=V K=V"` (no spaces in values), `MAX_KAGGLE_H` (default 7); optional
`KT_MODEL_OWNER/SLUG/DIR`, `KT_SRC_DIR`. Samples: `jobs/`. Enqueue: `./enqueue.sh pod3 jobs/X.job`. Base notebooks live in
`/root/queue/nb/base/` (V12: `manh_v12.ipynb`, c47be5a8). Copy new notebooks there before enqueueing.

## Data sync to this workspace (no agent turns)
`sync_queue.sh <pod> <port> <ip>` (two detached loops, every 5 min; stop with `runs/sync_<pod>.stop`) rsyncs each queued run into
`runs/<label>/` (score.json, nb.timeline, samp.*, `dsv41_logs/{attempts,metrics}.jsonl, solver.log`, submission.json, nb.ipynb at
the end) and `runs/<pod>/queue/{state.txt,index.tsv,runner.log}`.

## Workload
`workloads/eval120_kaggle_*`: the 120 originals of `evalx2t_s2026` (Kaggle-version eval), with solutions; `make_eval120.py` builds it
and the 60/60 dev/holdout halves (seed 2026, 4 strata by V12 replica difficulty; V12 pod replica on these 120: pass@2 81.8,
pass@1 76.0 task-weighted by this scorer; dev 81.7 / holdout 81.9). `bin/arcscore.py` = scorer (reconstructs attempt_1/2 like the
solver, Kaggle task-weighted).

## Pod 2 notes
- The 42 non-Engram E128 files (59.3 GB) were deleted earlier; 14 are prefetched (kept 14 GB free next to the running notebook), the
  rest download in the job's pre-step (about 10 min of idle GPU) after the img-causal model is deleted (disk 200 GB cannot hold both).
- No token is stored on any pod. `token_watcher.sh` (local, one loop per pod, logs `runs/token_<pod>.log`) answers `/root/queue/need_kaggle_token` by piping the token from SECRETS.yaml to `bin/dl_with_token.sh` over ssh.
- Pod 3 got Manh's vllm-wheels v11 by relay from pod 2 (the set V12 ran with), hardlink-deduplicated against the sft set.
