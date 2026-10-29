# B32-M fix: tonight's candidate (2026-10-29)

Notebook sha256 `bec71fb1e3f1`, kernel `chankhavu/dsv41-tools-submit-b32m-fix`. This is B32-M V3 (LB 46.5) with the
causes found today removed, plus **graceful recovery from server failures**.
- Model: the E128 text model already on Kaggle (`chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1`).
- Harness: B32's, byte-identical (`882c817f`).
- Server: Manh's v6 server cell (no `--supervise`), with the margins below.
- Diff against his v6: `review/`.

```
python3 kaggle/dsv41_tools/build_img_manh.py --text --manh-nb <Manh v6 pull> --slug dsv41-tools-submit-b32m-fix \
  --max-num-seqs 64 --gpu-util 0.90 --kv-offload-gib 10 --exec-timeout-s 900 --pin-dataset nguyennguyen599/vllm-wheels/11
```

| B32-M V3 | This build | Why |
|---|---|---|
| DCP4, Manh's comm flags off | TP4, no DCP, his production flags (decode flags checked: numerically identical on vs off) | his tuned Kaggle config; DCP4's fp8 combine was lossy and 0.72x speed on Kaggle |
| wheel g5e1bccd07c (no prefill-memory fix) | vllm-wheels **v11** pinned (g8cf0c0664b = v10 gf506afa210, which has the fix, + his mHC kernel speed-up) | the likely OOM trigger was re-prefill of long requests |
| util 0.938, ~1.1 GiB VRAM free | util **0.90**, ~2.0-2.5 GiB free (measured on pods) | room for long-prefill spikes |
| 80 workers, offload 32 GiB | **64** workers/seqs, offload **10 GiB/GPU** | host RAM ~150 of 173 GiB |
| B32 solver: launches into a dead port, burns retries, 19 min restarts, no hang detection | **our recovery:** see below | B32-M's 46.5 fits a run that stopped answering at Kaggle hour 4-6 |
| exec timeout 300 s | **900 s** | a slow step no longer kills the engine |
| 80-task load-test commit, no answer checks | light smoke (2 tasks), fails on no answers, crash lines, RAM headroom < 5 GiB, missing or short model files | quota; fail closed |

## Graceful recovery (cell 2s supervisor + solver 1ea2b25d)

- **Detect:** a daemon thread watches the server pid and confirms a death twice. It also catches a hang: 25 min
  with no generated tokens while requests run.
- **Signal:** it writes the status "restarting". The solver (S26) stops launching at once. In-flight attempts are
  **re-queued for free**, including the dying server's plain HTTP 500s ("The server had an error while processing your
  request"), which the kill test showed would otherwise cost each attempt a try. `ARC_CONN_TRIES=12` covers several
  outages.
- **Clean stop:** it kills the server's process group and waits until the group and every GPU compute process are
  gone, so two engines' RAM never overlap. A relaunch that dies in startup is cleaned up the same way.
- **Relaunch:** it reruns his exact launcher command with a fresh `--output-root dsv41_logs/r<N>`. It waits for
  `/v1/models`, then writes "up". The solver waits for "up", then resumes.
- **Limits:** at most 2 restarts, and only with >= 60 min before stop-new; otherwise "failed", and the solver finishes
  with what it has. An exception in the supervisor also writes "failed", so the solver never waits forever.
- **Kill test** (pod 3, 22:06 UTC; one TP worker SIGKILLed with 63 requests running):
  - the API server exits on its own 13 s after the kill;
  - old processes were gone in 20 s;
  - the new server was up in 150 s with the same KV pool;
  - all 64 workers resumed;
  - host RAM went 137 -> 14 -> 136 GiB;
  - the 500-handling bug it found is fixed in `1ea2b25d`.
- **Compared with Manh's v8:** his `--supervise` also restarts, but his pinned solver spends a try per in-flight attempt
  on each death and loses some outright, and a second death loses all in flight. We have not kill-tested his
  supervisor. See `kaggle/dsv41_tools/MANH_SAFETY_EDITS.md` for edits that make his notebook safer.

Context stays at 163,840. In B32-M's replica, 1 of 480 attempts hit it, but outputs reach 95k tokens at p90 and 128k
at the max.

## Kaggle commit run: Version 1 (pushed 22:42 UTC, COMPLETE 23:14 UTC, ~32 min)

**B32-M SMOKE OK.**

| Check | Result |
|---|---|
| Inputs | E046 CHECK OK; wheel vllm g8cf0c0664b (v11) |
| Server | MANH SERVER UP after 665 s (+1626 s in the notebook); KV 1,614,903 tokens; model 14.33 GiB/GPU |
| Contract | passes (tool calls, /tokenize 377, token-id completions 64) |
| Supervisor | watching, 0 restarts, stopped by cell 3s |
| Light smoke | 4/4 attempts answered (the test file has no solutions, so 0 correct is expected) |
| Model files | 136/136 present, sizes correct (the sha256 pass is partial in a short smoke, as designed) |
| Host RAM | smoke peak 125.9 GiB; projected at 64 workers 142.3 GiB; cgroup 173.0 GiB; **headroom 30.7 GiB** |

- **Cosmetic, commit-only:** 2 tracebacks at the very end, `'list' object has no attribute 'sleep'` in the settings
  cell's `_smoke_env_sampler`. Our 4b RAM check reused the name `_st`, which that sampler uses for `time`.
  - No effect on the verdict or the scored rerun: 4b and the sampler run in commit runs only.
  - Fixed in the builder (`_cgs`) for future builds; Version 1 is unchanged.
- Log: `scratchpad/kaggle_b32mfix_v1/dsv41-tools-submit-b32m-fix.log` (local).
