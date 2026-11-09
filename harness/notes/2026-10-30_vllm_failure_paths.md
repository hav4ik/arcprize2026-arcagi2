# vLLM failure paths in a 12 h Kaggle run (Manh's fork, DSv4.1 E128 2-bit, TP4 on 4x L4)

2026-10-30. This note is code reading only. Nothing was run on Kaggle, on pods or on local GPUs. For each way the run can
fail, it traces where the failure starts, how it travels (TP worker -> EngineCore -> API server -> HTTP client), what the
client sees, whether the engine survives, and what the solvers and supervisors then do.

**Sources**
- **vLLM:** Manh's fork at wheel v11, `8cf0c0664b`, extracted from
  `scratchpad/e062_port/manh_vllm`. Paths below are relative to that tree (`vllm/...`, `csrc/...`).
  - v10 (`f506afa210`) differs from v11 only in model and kernel files: no executor, engine, entrypoint or distributed
    file changed. So the pod 3 kill test, which ran v10, applies to v11 as is.
  - B32-M's wheel `5e1bccd07c` lacks the prefill-memory fixes `f506afa210` and `7ab756940a`.
- **Launcher and supervisor:** Manh's `dsv41_vllm.py` and `qwen38_server.py` (arc2-2026-src as audited, `audit_v12/src`).
- **Manh's notebook:** `kaggle/dsv41_tools/out/manh_v12_363638159/` (his v12 as pulled):
  - cell 2m (env and flags);
  - cell 0b (pinned solver `11bfaaaa`; I extracted it to compare).
- **Our solver:** `kaggle/dsv41_tools/pinned_img/solver_dsv41.py` (`a9983f4c`) with `pinned_img/run_deepseek_tools.py`.
- **Our supervisor:** `kaggle/dsv41_tools/cells/e066_supervisor.py` (cell 2s).
- **Our candidate's settings:** `out/nb_dsv41-tools-submit-b32m-fix-c1/` (not pushed).
- **Evidence:**
  - `experiments/E062_manh_vllm_fork/per_layer/pod3_runs/README.md` (TP3 kill test and the 3.6 h run, with its
    server.log copies in `run3_data/`);
  - `phase_h/README.md` (pod 2 TP2 kill);
  - `kaggle/dsv41_tools/MANH_SAFETY_EDITS.md`.
- **Web stack:** read in the local `.venv`: uvicorn 0.54.0 (the same version as Manh's wheels), starlette 1.7.0 (his is
  1.10.0), openai 3.22.1 (his is 3.39.0).

**Tags:** **[read]** means I read it in code. **[log]** means it was seen in a run log. **[inferred]** means reasoning
that I did not confirm in code or logs.

## The two configurations compared

| | Ours (b32m-fix-c1 build) | Manh v12 as pulled |
|---|---|---|
| Solver | `a9983f4c`: a 500 and `APITimeoutError` count as connection errors | `11bfaaaa`: neither counts |
| `ARC_JOB_TRIES` / `ARC_CONN_TRIES` | 2 / **12** | 2 / 4 (defaults; safety edit #1 not applied) |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | **900** | 300 (default; safety edit #2 not applied) |
| GPU util / seqs / context | 0.90 / 64 / 163,840 | 0.91 / 64 / 163,840 |
| KV CPU offload | SimpleCPUOffloadConnector 10 GiB/rank | the same |
| `--watermark` | 0.1 (his flags) | 0.1 |
| Supervisor | cell 2s: restarts on exit **or 1500 s without generated tokens**; at most 2 restarts; none within 60 min of stop-new | `--supervise 3`: restarts **only on exit**, or on a start not up in time |
| Wheels | vllm-wheels v11 pinned | unpinned (safety edit #4 not applied) |
| Client timeouts | `httpx.Timeout(7200, connect=30, read=900)`, `max_retries=0` | the same |

## Summary table

"In flight" means up to 64 attempts. Costs are per event.

| # | Failure | Origin | Client sees | Engine | Our solver | Manh's solver | Supervisor | Cost in attempts | Detection signal |
|---|---|---|---|---|---|---|---|---|---|
| 1a | CUDA OOM on TP rank 0 | torch allocator at its cap (`VLLM_GPU_DRIVER_RESERVE_MB`), in an eager prefill or mixed step | Within about 1 s: an SSE error event in the stream (`APIError: The server had an error while processing your request.`). For about 5 s after that, new requests get HTTP 500 (`InternalServerError: Error code: 500`). Then the connection is refused. | **Dies.** There is no per-request OOM recovery. | Free re-queue. The attempt reruns from turn 1. | Uses 1 of 2 tries. A retry that lands on the dying server loses the attempt. | Both restart after the API server exits (about 13 s) | All in-flight progress, plus a 3-4 min outage on pods (3-25 min reported) | `Worker failed with error 'CUDA out of memory...` and EngineCore's `fatal error` in server.log. Each rank logs `GPU memory: ... N OOMs` every 60 s. |
| 1b | CUDA OOM on TP rank 1-3 | same | Streams stall with no bytes. Then the same as 1a, once the engine notices. | **Dies, late.** Rank 0 hangs in the collective. Found at the execute timeout (300 s Manh, 900 s ours) or at the 600 s host-AR trap. | as 1a. If the stall exceeds 900 s, streams end with `APITimeoutError` (free re-queue). | as 1a. A stall over 900 s also uses a try. | as 1a | as 1a, plus 5-15 min with no generation | As 1a, but the OOM traceback comes 5-15 min before the fatal error. Also `host all-reduce: unit ... never reduced`. |
| 2 | Host OOM kill of a TP worker | cgroup OOM killer. Sandboxes (`oom_score_adj` 1000) die first, then the largest RSS, which is TP3 (about 43 GiB). | Kill test: in-stream `APIError` 12 s after the kill, 500s for 1 s, then refused | **Dies** in 9-12 s: monitor, then SIGTERM/SIGKILL of the workers, then `RuntimeError: cancelled` | Free re-queue of all in flight (pod 2: 65 re-queues, 0 lost) | Each in flight uses a try. Pod 3: 1 lost at once, 1 lost later. | Restart. If memory pressure persists, the reload can be OOM-killed again. | as 1a | `Worker proc VllmWorker-N died unexpectedly`; `memory.events: oom_kill` |
| 3a | Hang in a collective (inside an RPC) | one rank stuck (GPU, driver, a peer that left) | Streams stall | **Dies.** Manh: at 300 s (execute timeout). Ours: at 600 s if rank 0 waits in a host AR or gather (`__trap`), else 900 s. | Free, both for the 500 and for an `APITimeoutError` at 900 s | One try at 300 s. At 900 s the client timeout and the engine death race, so an attempt can use **both tries**. | Exit: restart. Ours would also fire at 1500 s (later). | 5-15 min with no generation, plus a restart | `RPC call to execute_model timed out.`; trap `printf` lines |
| 3b | Hang outside any RPC | inferred: a dead offload copy thread (see 5b), a stuck scheduler, a stuck API output loop | Streams stall: `APITimeoutError` after 900 s, repeating | Alive, makes no progress. **No vLLM timeout applies.** | Free re-queues every 900 s until ours restarts the server at 1500 s | Uses a try every 900 s: in-flight attempts are lost after 30 min, and new ones the same | **Ours: restart at 1500 s. Manh's: never** (the process is alive). | Ours: about 25 min plus a restart. Manh's: **the rest of the run** | generation counter frozen while running > 0 |
| 4 | A slow step that still makes progress, longer than the execute timeout | inferred: runtime Triton JIT (seen after restart), CPU starvation of the host-AR reducers, an Engram page-fault storm | Stall, then the 1a sequence | **Dies** although the step would have finished | as 1a | as 1a | Restart | as 1a. Ours tolerates 900 s; Manh's only 300 s. | `RPC call to ... timed out.`; `Triton kernel JIT compilation during inference` |
| 5a | KV pool near full | `--watermark 0.1`: at reported usage >= ~0.90 no waiting request is admitted (FCFS head of line) | No bytes until admitted. Waits over 15 min, then `APITimeoutError` after 900 s. | Survives. No error, no timeout, 0 preemptions. | Free re-queue, from turn 1: all of the attempt's turns are lost | Uses a try. Two timeouts lose the attempt (pod 3: `f95b36dd`). | None | The attempt's progress. Throughput drops (running 30-48 of 64). | `Waiting > 0` while `Running < 64`, KV 0.88-0.93 |
| 5b | Offload copy thread dies | `_copy_loop` has no try/except; `copy_blocks` raises on a memcpy error | Affected requests never produce a byte, then `APITimeoutError` | Survives, but **leaks**. Loads never complete; aborted requests that hold loads keep their blocks. KV fills slowly. | Free re-queues; ends as 3b | Uses tries; ends as 3b | Ours only once all generation stops | slow, until a restart | `Exception in thread` in server.log; `KV fetch: ... in progress` growing |
| 6 | Client gives up at 900 s | httpx read timeout (no bytes while queued) | `APITimeoutError: Request timed out.` | Survives. On disconnect the request is aborted and its KV freed **at once** (except an in-flight async load). No wasted server work. | Free re-queue | Uses a try | None | The attempt's earlier turns | the solver's `re-queue`/`retry ... APITimeoutError` lines |
| 7 | Engram / UVA / pinned host memory | Load: Engram prefault (THP), 40 GiB of pinned KV offload, UVA embed. Run: nothing allocates. | Server never comes up, or never comes back after a restart | Load: an OOM kill or a `RuntimeError` at startup. Run: no failure path except process death (2). | Waits for `up`, up to `ARC_SERVER_UP_TIMEOUT_S` | same | A failed start uses up a restart | The rest of the run, if a restart cannot load | Engram `MADV_COLLAPSE failed` (harmless); `Engram ... using pinned memory` (**+22 GiB**); `cudaHostRegister failed` |
| 8 | The restart itself | Supervisor: kill the group, wait until the GPUs are free, relaunch | Refused until up, about 2.5-4 min on pods | New engine; the prefix cache and the CPU tier start empty | Waits for status `up`, then re-queues | Probes `/v1/models` directly | Ours: about 25 s to relaunch. Manh's: plus a 10 s backoff. | in-flight progress; after a restart every attempt re-prefills | old and new engines never coexisted (pod 3 RAM samples) |

## 0. The common death path (used by modes 1, 2, 3a and 4)

Every fatal step error goes the same way. The pod 3 TP3 kill (v10, the same engine code as v11) shows each hop
in `run3_data/dsv41_logs/server/server.log` [log].

**1. Worker -> EngineCore (MultiprocExecutor)**
- Each TP worker loops on `rpc_broadcast_mq.dequeue(indefinite=True)`, which has no timeout
  (`vllm/v1/executor/multiproc_executor.py:1154-1161`).
- An exception in an RPC is logged as `WorkerProc hit an exception.` (`:1186-1194`).
- That rank replies `FAILURE` with `str(e)` only `if output_rank is None or self.rank == output_rank` (`:1183`,
  `:1192`, `:1096-1125`).
- **With a KV connector configured, `execute_model` and `sample_tokens` are sent with `output_rank=None`:**
  - The aggregator branch sets it (`:471-472`). `kv_output_aggregator` exists whenever the scheduler has a connector
    (`vllm/v1/engine/core.py:201-202`). Ours and Manh's both have one (SimpleCPUOffloadConnector).
  - So every rank replies, and the EngineCore reads all 4 response queues.
  - It reads them **in rank order, each against the RPC's deadline** (`multiproc_executor.py:498-550`).
- **Ways the EngineCore notices a failure:**
  - A `FAILURE` reply raises `RuntimeError("Worker failed with error '...'")` (`:523-528`).
  - A deadline miss raises `TimeoutError("RPC call to execute_model timed out.")` (`:507-509`). The deadline is
    `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, counted from the RPC's enqueue (`:414`, `:427`, `:466`).
  - **A worker process dying** is seen by the `MultiprocWorkerMonitor` thread (`:364-398`). It logs `Worker proc ...
    died unexpectedly`, then calls `shutdown()`:
    - it closes the death pipes;
    - it waits `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (5 s), sends SIGTERM, waits 4 s, sends SIGKILL (`:559-608`);
    - it shuts the response queues, so the blocked `dequeue` raises `RuntimeError: cancelled`
      (`shm_broadcast.py:874`).
  - In the kill test [log]: the kill at 22:06:37, SIGTERM at :42, SIGKILL at :46 (the survivors were stuck in GPU or C
    code and ignored SIGTERM), then `RuntimeError: cancelled` and `EngineCore encountered a fatal error` at :46.

**2. EngineCore -> API server**
- The exception leaves `step_with_batch_queue` (`core.py:790`). `log_error_detail` dumps the scheduler output
  (`:579-592`).
- `run_engine_core` logs `EngineCore encountered a fatal error.` and calls `_send_engine_dead()` (`:1630-1636`). That
  pushes `ENGINE_CORE_DEAD` (`:1918-1929`, `:2135-2138`), then shuts the executor down in `finally`.
- **Fault tolerance does not apply:**
  - `run_busy_loop` carries `@fault_tolerant_wrapper`, but it re-raises unless `enable_fault_tolerance`
    (`vllm/v1/fault_tolerance/engine_core_sentinel.py:207-231`).
  - That option is off by default (`vllm/config/parallel.py:486`) and exists for DP scale-down.
  - **vLLM 0.32-dev has no per-request recovery from any step error, OOM included** [read]. The only OOM catches are
    model load and sampler/pooler warmup: `gpu_model_runner.py:5077`, `:5928`, `:6031`.
  - The V2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`, which Manh's launcher sets) has none in `execute_model`.
- On the API side, `process_outputs_socket` raises `EngineDeadError` (`core_client.py:596-598`).
- `AsyncLLM.output_handler` then calls `output_processor.propagate_error(e)` (`async_llm.py:936-944`). Every open
  request's `generate()` re-raises `EngineDeadError`, without an abort (`:815-818`).

**3. API server -> HTTP client**
- **Streams already open:** `completion_stream_generator` catches the error and yields
  `data: {"error": {"message": "The server had an error while processing your request.", "type":
  "InternalServerError", "code": 500}}` and then `[DONE]` (`entrypoints/openai/completion/serving.py:554-558`).
  - The message is `EngineDeadError.CLIENT_MESSAGE` (`v1/engine/exceptions.py:22`;
    `entrypoints/serve/exception_handling/error_response.py:68-72`).
  - The HTTP status is already 200. The openai client raises `APIError(message=...)` on an SSE event with an `error`
    key, so the harness records `TurnError: APIError: The server had an error while processing your request.`
- **New requests:** `add_request` raises `EngineDeadError` at once (`async_llm.py:418-419`). The exception handler
  returns a JSON **HTTP 500** (`exception_handling/handlers/vllm_error.py:28-65`), which the client reports as
  `InternalServerError: Error code: 500 - {...}`.
- **The API server exits:**
  - `watchdog_loop` checks every 5 s (`entrypoints/launchers/launcher.py:315-338`). The 500 handler calls
    `terminate_if_errored` as well.
  - It sets `server.should_exit` unless `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1` (default 0, `envs.py:1216`).
  - On exit the API server force-kills EngineCore and the TP workers [log: `Process manager: force killing remaining
    process EngineCore`].
  - Kill test: `Finished server process` at :50, **13 s after the kill** [log].
- **After the exit:** connections are refused, and the client raises `APIConnectionError: Connection error.`
- The SSE body never contains "EngineDeadError" or "EngineCore encountered an issue". Those strings are only in
  server.log. So Manh's solver's `CONN_ERRORS` (`11bfaaaa` lines 156-160) does not match either 500 form.

**4. Solvers**
- Both turn a turn failure into `TurnError: <exc>`. The harness then runs the session-answer fallbacks before
  recording the error (`run_deepseek_tools.py:1900-1906`).
- **Ours** re-queues for free when any `CONN_ERRORS` substring matches:
  - the 500 text, `APITimeoutError` and `Connection` all match (`solver_dsv41.py:156-167`, `:1119-1124`);
  - it does this up to `ARC_CONN_TRIES` (12) times per attempt.
- **Manh's** sends both 500 forms and `APITimeoutError` to `tries + 1 < JOB_TRIES` (2): one retry, then
  `stop=error`. Only `APIConnectionError` is free.
- **A re-queued attempt restarts from turn 1** (a new trace `_rN`). Only a harvested session answer is kept,
  provisionally (`:1127-1135`).
- **Death detection:**
  - The watchdog calls the server dead when the supervisor status names another start, or says `restarting`,
    `failed` or `stopped`, or when the port refuses connections (`ExternalServer.poll`, `:418-449`).
  - `start_external` then blocks the main loop until the status is `up` again, bounded by stop-new (`:452-484`).
  - Launches are gated on `server_ok` and `poll() is None` (`:1094-1096`).

## 1. CUDA OOM in one TP worker

**Where OOM comes from**
- The fork caps the PyTorch allocator after warmup. `VLLM_GPU_DRIVER_RESERVE_MB=256` makes `_reserve_driver_memory`
  call `torch.cuda.set_per_process_memory_fraction((reserved + free - 256 MiB) / total)` and switch to expandable
  segments (`vllm/v1/worker/gpu_worker.py:1140-1176`).
- So a transient that does not fit under the cap raises `torch.OutOfMemoryError`. The cap leaves 256 MiB for
  non-torch allocations (NCCL, cuBLAS, Triton modules); a failure there is a CUDA error instead. Either way the outcome
  is the same.
- Pod 3 after its restart [log, `r1/server/server.log`]:
  - cap 21.2 GiB with 18.94 GiB reserved at start;
  - 3.4 h later: 19.86 GiB reserved, peak allocated 19.35 GiB, free 1.54-1.59 GiB, 0 retries, 0 OOMs.
  - **Headroom to the cap: about 1.3 GiB.** Reserved grows about 0.1 GiB per 30 min in steps; allocated stays flat
    (pod 2), so this is cache growth, not a leak.
- Decode-only steps replay FULL CUDA graphs, which have their own pool. Eager allocations come from steps with a
  prefill chunk (`--max-num-batched-tokens 2048`) [inferred].
- Manh traced B32-M's OOMs to re-prefill of long requests. The v11 wheel carries the fixes that trim the prefill peak
  (`f506afa210`, `7ab756940a`; see `out/nb_dsv41-tools-submit-b32m-fix/README.md:18`).

**OOM on rank 0:** a `FAILURE` reply on the same step, then the common path above. The client sees the in-stream
`APIError` within about a second, much faster than the 12 s of the kill test, because no worker-termination wait
comes first [inferred from the code order: `_send_engine_dead` runs before `finally: shutdown()`].

**OOM on rank 1, 2 or 3**
- That rank replies `FAILURE`, but the EngineCore first waits on rank 0's queue (`:500-507`).
- Rank 0 is blocked in the next collective, waiting for the rank that left.
- Detection comes at the first of:
  - rank 0's host-AR or host-gather kernel trapping at 600 s, which raises a CUDA error, so `FAILURE`;
  - the execute deadline: 300 s for Manh, 900 s for us;
  - rank 0 completing anyway [inferred: the failed rank goes back to its loop and runs the next queued RPC, so its
    collectives can pair with the others' out of order]. The `FAILURE` read next still kills the engine before any
    output from that step is used (`core.py:790` raises before `update_from_output`).
- Meanwhile no tokens are generated.

**Latent risk without a KV connector** [inferred]
- With `--kv-offload-gib-per-rank 0` there is no aggregator, `output_rank` is 0, and **only rank 0 replies**
  (`:496`, `:1183`).
- A rank 1-3 exception is then only logged. That rank goes on to the next RPC, so the TP collectives fall out of
  step. That is either a hang (caught by the timeout) or calls of equal size paired across different layers, which
  would give garbage tokens with no error.
- Keep the connector on, or patch (see the gaps list).

**What the run pays:** every attempt in flight is restarted from turn 1, plus the outage. There is no in-place
recovery.

## 2. A TP worker killed by the host OOM killer

**Who gets killed**
- vLLM sets no `oom_score_adj` anywhere (grep) [read].
- The tool sandbox writes 1000 for its processes (`out/kit/repo/evals/arc_harness/executor.py:52`), so the cgroup
  OOM killer takes sandbox processes first. Each is capped by `RLIMIT_AS` at 4 GiB, and its death shows up as a tool
  crash in that attempt.
- After them it takes the largest RSS [inferred, kernel badness]. That is **TP3**:
  - Engram 29.7 GiB (pinned; TP0-TP2: 2.8 / 10.8 / 20.7 GiB; pod 3 README:118);
  - 10 GiB pinned KV offload;
  - about 0.5 GiB UVA embed;
  - about 2.4 GiB runtime.
- **Unknown:** whether Kaggle's cgroup has `memory.oom.group=1`. If it does, the whole container dies.

**Path:** a SIGKILL is the same as the kill test (section 0): the engine dies 9-13 s later and the API server exits.

**Solvers and supervisors**
- Pod 2 with our solver (the 500 fix): 65 free re-queues, 0 lost [log, phase_h].
- Pod 3 with Manh's fix-less solver:
  - 64 tries used;
  - `f4c952f2` was lost when its retry 0.3 s later hit the dying server's 500;
  - `f95b36dd` was lost later to an `APITimeoutError`, its second try.
- Both supervisors restart the server.
- The memory pressure that caused the kill is still there. The new engine reloads about 135 GiB of anonymous memory
  and can be killed again while loading. Each such failed start uses a restart [inferred].

**Pinned pages cannot be reclaimed** (anonymous, no swap). Under pressure only page cache can go (checkpoint files,
library text), so expect refaults or thrashing before the kill [inferred].

## 3. Hangs: which timer fires first

| Timer | Where | Scope | Default / ours | What it does |
|---|---|---|---|---|
| Execute RPC deadline | `multiproc_executor.py:406-446`, `:498-509` | EngineCore waiting on a worker reply, measured from enqueue | 300 s (Manh v12) / 900 s (ours) | `TimeoutError("RPC call to <method> timed out.")`, fatal (common path) |
| Host all-reduce GPU timeout | `csrc/host_all_reduce_kernel.cu:458-472` (trap at `:469-471`) | a reader block waiting for a unit's done flag | **600 s, hard-coded** (`host_all_reduce.py:103`, `from_envs` does not pass it) | `printf("host all-reduce: unit %lld of call %u never reduced")`, then `__trap()`. The CUDA context is broken; the next sync raises, and the worker replies `FAILURE`. |
| Host all-gather GPU timeout | `csrc/host_all_gather.cu:98-111` (trap at `:107-109`) | waiting for a peer's flag | 600 s (`host_all_gather.py:37`) | `printf("host all-gather: rank %d never posted call %u")`, then `__trap()` |
| Host AR CPU reducers | `csrc/host_all_reduce.cpp:676-745` | waiting for every rank's input flag | **none**, they spin | exit only through `stop`. During a hang they burn a core each [inferred]. |
| PyNccl collectives | `cuda_communicator.py:463-476`, `:504-521` | NCCL calls through ctypes, also inside CUDA graphs | **none** | spin forever. The c10d `ProcessGroupNCCL` watchdog covers only torch.distributed NCCL calls, which here are fallbacks only (`:466`, `:476`) [inferred]. |
| shm MessageQueue, worker side | `multiproc_executor.py:1158` (`indefinite=True`) | waiting for the next RPC | none, no warning | none |
| shm MessageQueue, broadcast write | `shm_broadcast.py:729-765` | ring full | no timeout; logs every 60 s (`VLLM_RINGBUFFER_WARNING_INTERVAL`) | warning only. With a batch queue of 2 the ring does not fill [inferred]. |
| vLLM thread watchdog | `vllm/utils/watch_dog.py:42-174`, `config/watchdog_config.py` | worker and EngineCore main threads | **off** (empty `dump_dir`) | dumps all Python stacks to a file; never kills |
| API watchdog | `launcher.py:315-338` | the engine's errored flag | 5 s | exits the API server once the engine is dead |
| Client read timeout | `solver_dsv41.py:803` | no byte for 900 s on a stream | 900 s | `APITimeoutError` |
| Our stall supervisor | `e066_supervisor.py:151-158` | `vllm:generation_tokens_total` flat while running + waiting > 0, or `/metrics` unreadable | 1500 s | kill and relaunch |
| Manh's supervisor | `qwen38_server.py:880-960` | process exit only | - | nothing on a hang |

**Hang inside an RPC (3a)**
- **Manh v12:** the 300 s execute deadline fires first. The engine dies at about T+300 s, and every in-flight attempt
  uses one try.
- **Ours, two cases:**
  - Rank 0 waits in a host-AR or host-gather kernel: it traps at 600 s, so `FAILURE` and death at about T+600 s.
  - The stuck rank is rank 0 itself, or it waits in PyNccl: death at T+900 s. That is the moment the clients' 900 s
    read timeouts fire as well. Both outcomes are free re-queues for our solver.
- **Interaction to watch:** suppose Manh applies safety edit #2 (900 s) without #1 or the 500 fix.
  - Each in-flight attempt's client times out (try 1) at about the same moment the engine dies.
  - Its retry then meets the 500s (try 2) and is lost.
  - So **a hang would cost every in-flight attempt.** Edit #2 must go together with #1.
- **Not involved:** the c10d NCCL watchdog (no c10d NCCL ops on the hot path) and the vLLM thread watchdog (off).

**Hang outside any RPC (3b)** [inferred: the code has no such timer]
- No vLLM timer covers the EngineCore loop outside `future.result()`, the scheduler, the connector copy thread or the
  API output loop.
- Our solver's re-queued attempts go straight back into the hung server, because the status is still `up` and the port
  accepts. Each burns 900 s and one of its 12 connection tries, until cell 2s fires at 1500 s.
- Under Manh's supervisor the server is never restarted. His solver finalizes in-flight attempts after 2 x 900 s, and
  every new attempt after that does the same.

## 4. A slow step that still makes progress, longer than the execute timeout

- **What happens:** the deadline covers the time from an RPC's enqueue to its reply.
  - With async scheduling and the batch queue (`core.py:715-800`), the next step's RPC is enqueued while the current
    one runs, so its 300 or 900 s include the rest of the slow step.
  - At the deadline: `TimeoutError` and the common death path, **although the step would have finished**.
  - Nothing retries the step. The worker keeps computing until the shutdown SIGKILL.
- **Candidates for a step over 300 s** [inferred; none was measured]:
  - runtime Triton JIT or autotune of a new shape. After the pod 3 restart, the JIT monitor logged
    `_candidate_flags_kernel`, `_compute_global_topk_indices_and_lens_kernel` and `_mask_candidates_kernel` compiling
    during inference [log, `jit_monitor.py:140`];
  - CPU starvation of the host-AR reducer threads by 64 sandboxes on a small vCPU count;
  - page-fault or compaction storms on host memory.
- **Policy:** 900 s (ours, and safety edit #2) turns these into delays. The cost is a later death for a real deadlock
  (section 3).

## 5. KV pool exhausted, and the CPU offload connector

**Preemption** [read, `vllm/v1/core/sched/scheduler.py`]
- When a *running* request cannot get its next blocks, the victim is `self.running[-1]` under FCFS, i.e. the newest
  (`:1008`).
- `_preempt_request` (`:1881-1925`):
  - recompute mode;
  - frees its blocks, which stay hashed and evictable, so a re-prefill can hit the prefix cache;
  - resets `num_computed_tokens`;
  - **prepends it to `waiting`** (`:1924`).
- A step that preempted admits nothing from `waiting` (`:1120`).
- There is no preemption cap, no starvation guard, no abort of repeatedly preempted requests, and no waiting timeout
  anywhere in the scheduler.

**The watermark** (upstream option, default 0.0, `config/scheduler.py:196`; Manh sets `--watermark=0.1`)
- `watermark_blocks = int(0.1 * num_blocks)` (`kv_cache_manager.py:217`), about 145k tokens of 1.45M.
- It is charged only to WAITING or PREEMPTED requests, and only while something is running
  (`kv_cache_manager.py:544-551`; `scheduler.py:1525-1526`).
- Admission needs `blocks for the full sequence + watermark <= free` (`:557-573`). The full-sequence check comes from
  `scheduler_reserve_full_isl=True` (`config/scheduler.py:190`).
- "Free" counts evictable cached blocks, and usage is `1 - free/total` (`block_pool.py:921-941`).
- Prefix-hit blocks that sit in the free queue count toward the need (`single_type_kv_cache_manager.py:280-290`). So a
  follow-up turn whose earlier turn was freed needs nearly its whole prompt again.
- When the head of the queue fails, the loop `break`s (`scheduler.py:1528-1536`), and **nothing behind it can
  overtake it**.
- **Consequence** [inferred arithmetic]:
  - At reported usage >= 0.90, **no** waiting request is admitted. A 100k-token follow-up needs usage <= about 0.83.
  - Running decodes never pay the watermark, so usage stays near 0.9, and the queue moves only as running turns finish.
  - This matches pod 3's `f95b36dd`: KV 0.89-0.93, 3-6 waiting, **0 preemptions**, running 30-48, and a turn that got
    no byte for 900 s.
  - Our solver's KV gate (`ARC_KV_ADMIT_MAX 0.85`) holds only *new* attempts. Next turns of running attempts always
    join the queue.

**A single request too large**
- There is no runtime "exceeds KV capacity" error. The only check is at startup (`v1/core/kv_cache_planning.py:78-87`).
- A request that could never fit would wait silently forever.
- Here that cannot happen: 163,840 + 145k is far below 1.45M, and the watermark is off when nothing runs.
- A prompt over max-model-len is a 400 (`input_processor.py:605-615`). `max_tokens` is clamped
  (`entrypoints/serve/utils/api_utils.py:248-261`), and our harness's `--context-limit` keeps turns below that anyway.

**SimpleCPUOffloadConnector** (`distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`,
`v1/simple_kv_offload/*`) [read]
- **Host buffer:**
  - The whole buffer is allocated at KV-cache init with `torch.zeros`, then `cudaHostRegister`
    (`worker.py:222-259`, `cuda_mem_ops.py:24-33`). That is 4 x 10 GiB resident from startup.
  - If registration fails, it raises (startup fails, with no fallback). Nothing is allocated at runtime.
- **Store:** eager. GPU blocks hold an extra reference until all ranks report the copy done (`manager.py:1070-1095`).
- **Restore:** always asynchronous. The request sits in WAITING_FOR_REMOTE_KVS until `finished_recving` arrives
  (`scheduler.py:3671-3680`).
- **CPU pool full:** LRU reuse. When nothing is evictable, stores are deferred and boundary hand-offs are dropped
  (`manager.py:758`, `890`, `929`). It **never blocks**. `request_finished` never delays a free (`:1278`).
- **A failed load:**
  - `get_block_ids_with_load_errors` is not overridden, so `kv_load_failure_policy` never applies.
  - The copies run in the daemon thread `_copy_loop` (`copy_backend.py:104-130`), **with no try/except**. A
    `batch memcpy failed` (`cuda_mem_ops.py:269-272`) kills the thread silently [inferred consequence].
  - Every later load then never completes, and those requests wait forever while holding blocks. Aborting such a
    request frees its blocks only when its load reports done (`scheduler.py:3119-3125`), which is never.
  - So KV leaks and usage climbs: mode 5b.
- **Two fixes in the same window are not for this connector:** `16777da347` and `fa3b9d1377` touch only the
  OffloadingConnector, P2P and tiering code.

## 6. Queue wait longer than the client's 900 s read timeout

- **No byte before the first token:**
  - The completion route returns a `StreamingResponse` without awaiting the engine
    (`entrypoints/openai/completion/api_router.py:52-76`). `engine_client.generate()` is a lazy generator
    (`serving.py:210-219`).
  - So headers go out at once. The request enters the engine when the body starts, and the first byte comes with the
    first token.
  - The server's SSE keep-alive is off (`--sse-keep-alive-interval 0`, `entrypoints/common/cli_args.py:164`;
    `serve/utils/sse_keep_alive.py`).
  - So a request waiting in the scheduler sends nothing, and httpx raises `ReadTimeout` after 900 s, which openai
    reports as `APITimeoutError: Request timed out.`
- **Is the request aborted on disconnect?** Yes, at once:
  - uvicorn 0.54 announces ASGI spec_version 2.3 (`uvicorn/protocols/http/httptools_impl.py:228`).
  - With 2.3, Starlette's `StreamingResponse` runs `listen_for_disconnect` next to the body and cancels it on
    `http.disconnect` (`starlette/responses.py:259-282`).
  - The `CancelledError` reaches `AsyncLLM.generate`, which calls `abort` (`async_llm.py:807-811`), then
    `EngineCore.abort_requests` (`core.py:571-576`), then `finish_requests`.
  - That removes waiting requests from the queues and frees their blocks (`scheduler.py:3066-3131`, `:3111-3114`).
    The free waits at most one step under `defer_block_free`, and an in-flight async load waits for its completion.
  - [read in the local .venv; Manh's starlette 1.10 not read]
- **Wasted work** is therefore on the client side:
  - our solver re-queues the attempt from turn 1, losing all its turns (pod 3's `f95b36dd` had 8 turns over 3,212 s);
  - Manh's solver also uses a try.
  - The engine's prefix cache keeps the freed blocks as evictable, so the restarted attempt's first prompt can hit.

## 7. Engram host tables, UVA embed_tokens and pinned host memory

[read, `vllm/models/deepseek_v41/common/engram.py`, `nvidia/model.py`, `model_executor/offloader/uva.py`]

**Engram tables**
- With DP 1 each TP rank owns its tables privately: heads are split over TP (`config/engram.py:85-93`,
  `engram.py:1063-1072`). They are allocated at model `__init__`, before the weights.
- **`use_thp` path** (`engram.py:945-980`):
  - an anonymous `MAP_PRIVATE` mmap with `MADV_HUGEPAGE`;
  - a single-threaded touch of every page;
  - `cudaHostRegister`.
- The GPU reads rows zero-copy through UVA: a Triton gather on a prefetch stream, with indices masked to bounds
  (`engram.py:700-722`, `:1289`, `:2023-2045`).
- **Failure modes:**
  - `MADV_COLLAPSE` failing only logs a warning (`:1219-1230`).
  - An mmap `OSError`, a failed `cudaHostRegister` or an unpinned result falls back to `torch.empty(pin_memory=True)`
    (`:1197-1212`). That goes through PyTorch's caching host allocator, which rounds up to a power of two. **The
    fallback costs about +22 GiB across ranks** [inferred from the logged sizes]. Under fragmentation a restart could
    switch to it and then be OOM-killed.
  - Under heuristic overcommit the mmap itself does not fail. Running out of memory shows up as a **cgroup OOM kill
    during the prefault**, not as an exception [inferred].
  - The prefault time under THP compaction is the restart-time variable. On pod 3, TP3 loaded in 75-210 s, and the
    second table took 4-20 min on the first run.
- At run time nothing is allocated, and registered pages cannot fault. The only failure left is process death
  (mode 2).

**UVA embed_tokens**
- `.pin_memory()` plus a UVA view of the 0.31 GiB vocab shard (`uva.py:74-100`, `:142-149`).
- There is no fallback: a failure raises at startup, and the server never comes up.

**Host AR and gather segments**
- `/dev/shm/vllm_host_ar_<pid>_<uuid>` and `vllm_host_ag_*` are created by rank 0 and **unlinked as soon as all ranks
  have mapped them** (`host_all_reduce.py:160-186`), so a later SIGKILL leaks nothing.
- If setup fails, **host AR is disabled silently** and NCCL is used instead: slower, no error
  (`host_all_reduce.py:182-185`).

**MessageQueue rings** (`psm_*`)
- They are unlinked only in the creator's `__del__` (`shm_broadcast.py:393-397`), so a SIGKILL can leak them
  [inferred: unless Python's resource_tracker survives].
- `check_shm_free_space` refuses to start when /dev/shm is short (`shm_broadcast.py:242-268`).
- About 1.15 GiB nominal per engine.

## 8. Supervisor restart: time, host RAM, GPU memory

**Ours (cell 2s)**
- **Restart:**
  - It polls every 5 s and checks that the server pid is still alive (`e066_supervisor.py:142-150`).
  - It writes `restarting`, sends SIGTERM to the group, and waits up to 45 s **until the group and every nvidia-smi
    compute app are gone**. Then it sends SIGKILL and waits up to 180 s more (`:65-79`, `_sup_stop`).
  - It reruns the launcher into `r<N>` and polls `/v1/models` (`:109-130`).
  - A failed relaunch counts as another death (`:178-181`).
- **Limits:** at most 2 restarts; none within 60 min of stop-new.
- **Kill test** [log]: exit +13 s; group and GPUs empty +20 s; relaunch +25 s; up +174 s (150 s load); all 64 workers
  back +224 s. Pod 2: a 116 s outage.

**Manh's (`--supervise`)**
- **Restart:**
  - It polls `process.poll()` every 5 s (`qwen38_server.py:915-927`).
  - `stop_owned_process`: `killpg` with SIGTERM, 15 s, then SIGKILL. Then `kill_pids(session)`.
  - It backs off 10 s (`:954`), frees the port, waits up to 180 s for no GPU compute apps (`wait_gpu_free`), and then starts.
- **Limits:** a start that fails (for example GPUs not free in 180 s) uses a restart. It skips a restart that could
  not be up before stop-new.
- **Not covered: it has no health or progress check after `up`.**

**Host RAM overlap**
- Neither supervisor relaunches before the old processes are gone. In vLLM itself, a new worker raises at
  `init_device` if the GPU is still held (`gpu_worker.py:557` -> `v1/worker/utils.py:802-811`, a `ValueError` on low free memory).
- Pod 3 [log]: AnonPages 136.7 GiB before the kill, 14.4 GiB 20 s after, a 135.7 GiB peak for the new engine. **The
  two engines never coexisted.**
- Load has no transient copy of the Engram tables: rows are narrowed and copied per rank (`engram.py:615-627`). The
  load peak is the steady state plus short-lived row_map temporaries and page cache.

**GPU memory**
- The driver frees it at process exit. After the restart, pod 3 had 2.04-2.09 GiB free per GPU, as at the first start
  [log].
- A restart also resets the reserved-memory creep (section 1).
- The KV pool came back the same, 1,453,240 tokens.

**What the solver does after a restart**
- The re-queued attempts launch into an empty prefix cache and an empty CPU tier.
- Cost: one re-prefill each, on top of the restarted attempts.

## Gaps and risks, with suggested fixes (small and safe first)

1. **Manh's v12 notebook has none of safety edits #1, #2 and #4.**
   - With his solver, every death costs a try for each in-flight attempt; a second death, or a later queue timeout,
     loses them.
   - **Fix:** `ARC_JOB_TRIES=3`, `ARC_CONN_TRIES=12`, or better, our `a9983f4c` `CONN_ERRORS`; pin vllm-wheels/11.
   - **Apply edit #2 (exec timeout 900) only together with #1** (section 3).
2. **`--watermark 0.1` causes long queue waits.** At about 0.9 usage no waiting request enters, and turns wait more
   than 15 min. This is the probable cause of pod 3's `APITimeoutError`.
   - **Fix:** A/B `--watermark` at 0.02-0.03 (or 0) on a pod. Measure the queue wait and running count against
     preemptions; the watermark exists to prevent preemption thrash.
   - Independently, either raise the client read timeout to 1800 s, or set `--sse-keep-alive-interval 60`. Keep-alive
     comments are skipped by the openai SSE decoder (`openai/_streaming.py:404`) but reset httpx's read timer.
   - With keep-alive, a hung engine is no longer caught by the client, so it needs the stall supervisor.
3. **A transient turn error throws away the whole attempt.** The solver re-queues from turn 1.
   - **Fix:** in the harness, retry the same turn after a short wait for `APITimeoutError`, `APIConnectionError` and
     the 500 text. Keep the transcript, and wait for the supervisor's `up` when it is restarting. This is a medium
     change and needs a pod test.
4. **Manh's supervisor cannot see a hang.**
   - **Fix:** use cell 2s, or add its stall rule (generation counter flat with running > 0 for 15-25 min means
     restart) to `run_supervisor`.
5. **Churn in the dying window** [inferred].
   - For about 1-4 s after `EngineDeadError`, the port is open and the status says `up`. A re-queued attempt can relaunch
     at once and get a 500 again (pod 3 saw one such retry 0.3 s later).
   - With ours that only costs connection tries. The risk is low with 12, but not zero.
   - **Fix:** after N connection errors within 10 s, hold launches until the status changes or 30 s pass.
6. **Execute timeout equal to the client read timeout (900 = 900).** For our solver both are free, so this is
   harmless. Prefer exec 900 with client read 1200+, so the engine's own verdict comes first.
7. **Cheap hang diagnostics.**
   - **Fix:** `--watchdog-config '{"dump_dir": "/kaggle/working/stacks", "timeout": 240}'` dumps every thread's
     Python stack when a worker or EngineCore RPC runs over 240 s. It never kills.
   - Also grep server.log for `never reduced` and `never posted` (trap printfs), `RPC call to`, `Exception in thread`
     and `Host all-reduce is disabled`.
8. **The offload copy thread can die silently.** That gives requests stuck forever in WAITING_FOR_REMOTE_KVS and KV
   leaks that no supervisor sees quickly.
   - **Fix:** a small vLLM patch, a try/except in `_copy_loop` that marks the loads failed or kills the engine. Failing
     loudly beats leaking.
   - Until then, watch `Exception in thread` and the `KV fetch ... in progress` count.
9. **A non-zero rank's failure is noticed late**, because `get_response` reads the queues in rank order. That gives
   5-15 min with no generation.
   - **Fix:** a small patch that polls all response queues. Also keep the KV connector on: without it, rank 1-3
     exceptions are only logged (section 1, latent risk).
10. **GPU OOM headroom.**
    - Torch's cap leaves about 1.3 GiB at 3.4 h, and reserved memory creeps about 0.1 GiB per 30 min. Pod 3
      extrapolated a crossing near 9.6 h by free memory.
    - Any OOM kills the engine.
    - **Fix:** keep util 0.90; watch the per-rank `GPU memory:` line for allocator retries; consider one planned
      restart near hour 6 if reserved memory trends toward the cap. A restart resets it.
11. **Host OOM.**
    - TP3 (about 43 GiB RSS) is the victim after the sandboxes.
    - **Fix:** keep at least 20 GiB of cgroup headroom and log `memory.events` (`oom_kill`) every minute. Try
      `oom_score_adj` < 0 for the vLLM pids; this may fail without `CAP_SYS_RESOURCE`.
    - An Engram fallback to `pin_memory=True` would cost about +22 GiB. Patch it to `torch.empty` + `cudaHostRegister`,
      as `pin_tensor` does.
12. **Restart hygiene.**
    - After each start, assert `Host all-reduce enabled` in the log; a silent NCCL fallback is slower.
    - Before relaunch, delete stale `psm_*` and `vllm_*` files in /dev/shm that no live process maps.
    - Check that /dev/shm has at least 2 GiB free.
13. **Unknowns to check on Kaggle:** cgroup `memory.oom.group`; /dev/shm size; whether Kaggle's container exposes
    `nvidia-smi --query-compute-apps` the same way (both supervisors rely on it); starlette 1.10's disconnect handling
    (read 1.7 only).
