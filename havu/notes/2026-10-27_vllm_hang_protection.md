# Does vLLM protect against a hung server? (2026-10-27, for candidate B32)

Two research passes (the fork's installed code read-only on a Runpod box; upstream issues/PRs/docs) plus a scan of our own
server logs. Fork: `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` = upstream main ~2026-09-08 (6c73b08dec)
+ the dsv41 SM89 patches + our overlay. Setup: V1 engine, multiproc executor, TP=4 over PCIe L4, pynccl all-reduce over
shared memory (custom all-reduce off), FULL CUDA graphs, prefix caching, KV CPU OffloadingConnector (E051 patch).

## What vLLM catches by itself
| hang | caught? | how |
|---|---|---|
| A TP worker stops answering (stuck NCCL all-reduce, spinning kernel, stuck stream) | **yes** | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300, `v1/executor/multiproc_executor.py`) raises `TimeoutError("RPC call to execute_model timed out")`, the engine core dies (`EngineDeadError`), the API server's 5-s watchdog loop exits (unless `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1`). Our solver sees the process exit and restarts (E059: tested). Warning lines before it: "No available shared memory broadcast block found in 60 seconds". |
| A worker process dies | yes | `MultiprocWorkerMonitor` on process sentinels |
| An exception on any TP rank | yes, with a KV connector configured | all ranks reply when a connector aggregator is active; without a connector only rank 0 replies (upstream #60928, open) |
| Engine-core Python loop wedged (not waiting on a worker) | **no** | `/health` only reads a flag (`AsyncLLM.check_health`); `/metrics` keeps answering with frozen counters |
| API server stops answering | **no** | its watchdog runs in the same event loop |
| KV-offload load or store that never completes | partly | a blocking flush is inside `execute_model` (300 s); polled loads/stores have no deadline: those requests wait forever while others run |

Not useful here: `VLLM_ENGINE_ITERATION_TIMEOUT_S` (V0 leftover, unused, removed upstream #59781), `--distributed-timeout-seconds`
and torch/NCCL watchdog variables (they do not cover pynccl; vLLM even clears `NCCL_ASYNC_ERROR_HANDLING`), `--enable-fault-tolerance`.

## The relevant open risk for B32
**Async-KV-load scheduler deadlock under KV pressure with OffloadingConnector**: "Running: 0 reqs, Waiting: N reqs", KV
nearly full, `/health` 200 forever. Reproduced upstream on 0.27, 0.28 and 0.30 (#57841, #59497); fixed by **#58947**
(merged 2026-09-28, v0.31.0), not in our fork. It sits in B32's new component and regime.
Our evidence: 30 server logs with KV offload, ~49 h of serving (B32's 11 h replica at KV 0.85-0.99 with 487 preemptions,
heavy-task stress, late-phase replays incl. the 16 GiB capacity-cliff runs): the longest "Running 0 with requests
waiting" streak was 10 s. Judgment: ~1-3% per 12 h run; if it hit, the run would stall from then on (written answers kept).
B32 was submitted with this risk accepted (expected cost ~1 point vs ~+12 expected gain over V6).

## For the next candidate
1. Turn on the solver's progress watchdog S20 (`ARC_STALL_RESTART_S`, live solver; idle = nothing running AND nothing
   waiting). Suggested N = 600 s (lets vLLM's 300 s path act first; under the 900 s client read timeout), or
   `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120-180` with N ~300 s. Upstream evidence of recoverable 60-240 s stalls
   (GLM-5.2 on B300, #41530) argues against going much lower.
2. Cherry-pick #58947 (scheduler deadlock fix; 13 files, +220/−149) into the overlay, with a parity/stress test.
3. Optional: #55700 `--watchdog-config` (stack dumps after 300 s without progress), #54553 (bounded shutdown after a
   fatal error), `PYTHONFAULTHANDLER=1` on the server.
4. Diagnosis on any stall: grep every rank's log for "WorkerProc hit an exception", not only rank 0.

Upstream threads: #41530 (worker hangs incl. DeepSeek-V4/V4.1-Flash TP 2-8; one 4x RTX PRO 6000 PCIe TP4 report: once in
~383 h), #51035, #51593 (fixed, in the fork), #40863 (fixed, in the fork), #49829, #52838, #58445, #52319, #58422.
No runtime hang reports specific to L4/SM89 or NCCL over shared memory were found.
