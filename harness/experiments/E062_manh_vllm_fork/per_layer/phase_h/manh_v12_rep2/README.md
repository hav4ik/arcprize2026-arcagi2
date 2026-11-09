# Manh V12 replica 2 (pod 5 `u71eibokv8t34t`): run-to-run variance of the 80.83 replica

User request (2026-10-31 via the coordinator): a second full replica of Manh's V12, the exact setup of `../manh_v12/README.md`.
- **Notebook:** `kaggle/dsv41_tools/out/manh_v12_363638159`, sha256 c47be5a8 unchanged (`EXTRA_ENV` empty: no variant patch).
- **Model:** the old tool E128, Kaggle `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit` v1 (136 files, sha256 vs the cell 1b manifest), downloaded through the local `token_watcher.sh` handshake (token on ssh stdin only, never on the pod's disk).
- **Inputs:** vllm-wheels v11 (294 wheels, vllm g8cf0c0664b, sha256 5039fd7680a4 = pod 2's), arc2-2026-src v23, cu13-install v2, dsv41-tools-kit v5, unsloth-main v15, all version-pinned downloads (`E072_harness_loop/bin/new_pod_inputs.sh`).
- **Workload:** evalx2t_s2026 (240 tasks, 344 pairs), rerun mode (`KAGGLE_IS_COMPETITION_RERUN=1`), to the notebook's own end (job `jobs/rep2_v12_pod5.job`, queue runner on the pod, run `q_rep2_v12_pod5`).
- **Host difference:** this pod has 251 GB host RAM (pods 2/3: 503 GB); L4 x4 as before. Compare decode speed at matched load (`E072_harness_loop/rep2_compare.py`).
- **Data:** `data/` in the same layout as `../manh_v12/data` (synced every 5 min by `E072_harness_loop/rep_dash_loop.sh`).

## Blocker found at start 0 (2026-10-31 03:24 UTC): this pod's memory cgroup is 124 GB
- Pod 5 shows 251 GB in `free`, but `/sys/fs/cgroup/memory/memory.limit_in_bytes` = 123,999,997,952 (115.5 GiB; pod 2: 250 GB). `max_usage_in_bytes` reached the limit.
- Start 0 loaded cold (2,056 s: Engram tables read from a cold disk; pod 2 took 245 s), reached "GPU KV cache size: 1,608,578 tokens" (= pod 2's), then VllmWorker-3 died right after the SimpleCPUOffload buffers (4 x 10 GB pinned) were allocated at host usage ~106 GiB: a cgroup OOM kill (no traceback, "exit code: None").
- V12's host RAM is 133-148 GiB (Kaggle-eq; pod 2 replica), above this cgroup. The exact V12 setup cannot run on this pod.
- Model integrity: all 136 files sha256-verified (VERIFY_ALL 136/136); wheels 294 (vllm sha 5039fd7680a4); inputs as pod 2's.
