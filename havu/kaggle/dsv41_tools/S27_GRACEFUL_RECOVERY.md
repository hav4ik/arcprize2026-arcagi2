# S27: graceful recovery on top of PR #2 (S26), for Manh's notebook

This branch is PR #2's head (`havu-s24-supervised-server`) plus **one commit**, `8a761258b`. It changes
`pinned_candb_ext/solver_dsv41.py` (the solver your notebook's cell 0b ships; sha `11bfaaaa` -> `3d6f93ee`) and the
live `solver_dsv41.py` the same way.

**Apply:** `git cherry-pick 8a761258b` on your branch, then rebuild cell 0b from `pinned_candb_ext/`.

## Why (pod kill test, 2026-10-29 22:06 UTC)

We SIGKILLed one TP worker with 63 requests running, under a restart supervisor.
- The API server exits on its own 13 s later. The supervisor relaunched the server, and it was up again in 150 s.
- In the second before it exits, the dying server answers **every in-flight request with a plain HTTP 500**,
  `The server had an error while processing your request`. Only server.log has `EngineDeadError`.
- The S26 solver counted that 500 as a failed try (`ARC_JOB_TRIES=2`), so every in-flight attempt used one try. One
  attempt was lost: its retry went 0.3 s later into the same dying server.
- A second death would lose every attempt then in flight.

## The three changes

1. That 500 is in `CONN_ERRORS`: a free re-queue, like `EngineDeadError`.
2. `ARC_CONN_TRIES` defaults to **12** when `ARC_EXTERNAL_SERVER_STATUS` is set (4 otherwise). Each restart costs every
   in-flight job at least one connection re-queue.
3. `start_external()` waits while the status is not `up`. A restarted server can answer `/v1/models` on the same port
   before your supervisor writes "up"; accepting it early would spend one of the restarts.

**Tests:** test_solver `ExternalServer`, `SupervisedServer` and `MainLoop` pass on this branch (24 tests).

## Server-cell edits that go with it (no code)

`kaggle/dsv41_tools/MANH_SAFETY_EDITS.md` on main
(https://github.com/hav4ik/arcprize2026-arcagi2/blob/main/havu/kaggle/dsv41_tools/MANH_SAFETY_EDITS.md):
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`;
- util 0.935 -> 0.92;
- pin `nguyennguyen599/vllm-wheels/11`;
- don't change `ARC_WORKERS` without `_MT_EXPECT`.

With this commit, `ARC_JOB_TRIES` / `ARC_CONN_TRIES` in the settings are no longer needed (they are harmless).

Our own full candidate (your v6 server cell + our restart supervisor + this solver logic, util 0.90, 64 workers):
`kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/README.md` on main, Kaggle kernel
`chankhavu/dsv41-tools-submit-b32m-fix`.
