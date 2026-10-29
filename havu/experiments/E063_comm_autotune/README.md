# E063: choose the GPU communication path by measured speed (comm autotune)

User question 2026-10-28 ~14:45 UTC: can the server measure the communication speed between GPUs at runtime and
choose the method from that (P2P, through host memory, NCCL, Manh's host all-reduce, ...)? Built on our branch of
Manh's fork; nothing on Kaggle, B32 and the running replicas untouched.

Why: in E062 Manh's fork ran at 0.51x B32 on Vast 4090 box 58063361 (no P2P, the 4 GPUs on 2 sockets), but +22% on
the L4 pods (one NUMA node). DCP4 adds collectives on every decode step, so the best path depends on the machine, and
Kaggle's (PCIe Gen3, 50 W) can't be measured in advance. The server has to choose for itself.

## Answer
Yes. `VLLM_COMM_AUTOTUNE=1` (off by default) times each lossless candidate once at startup, before profiling and
CUDA-graph capture. It times them at the model's message sizes, takes the slowest rank's median, and keeps the
fastest per size. Every rank gets the same table.
- **On the 4090 box it makes decode +23% faster at 56 streams (256 -> 315 tok/s), +15% at 16, and prefill +43%.**
- Outputs don't change beyond run-to-run noise.

Two limits:
- **The choice is fixed for the run.** CUDA graphs replay whatever was captured, so switching mid-run would mean
  recapturing them.
- **NCCL's own P2P-vs-shared-memory transport can't be switched inside a running process**, but its protocol can.

## Code (nguyen599/vllm `arc/e128-kaggle`, head 786468e, pushed 2026-10-28; `arc/comm-autotune` is identical)
Clone: `scratchpad/e062_port/manh_vllm`. Patches in `patches/`, against `arc/e128-kaggle` e4988d1:
- `0001` = E062 patch 03 (DeepGEMM guard, re-applied here as 5f61bdc; the 4090 agent's local commit was 469f718);
- `0002` = the autotune (33dcc71);
- `0003` = keep NCCL on near-ties within 5% (786468e).

| file | change |
|---|---|
| `vllm/distributed/device_communicators/comm_autotune.py` (new) | `AutotuneTable`, `message_sizes`, `time_candidates` (CUDA events, warmup 5, 20 timed calls, median), `agree` (MAX over the CPU group), `choose` (fastest; NCCL kept unless another is >5% faster), `autotune` (log + optional JSON report) |
| `cuda_communicator.py` | `_autotune()` per group after the communicators exist. **TP:** NCCL, NCCL with `NCCL_PROTO=LL` on a second communicator, custom all-reduce (also on >2 PCIe GPUs if P2P works), host all-reduce. It is skipped if FlashInfer / symm-mem / quick-reduce / AITER are active, so it never bypasses a backend it didn't time. **DCP all-gather:** NCCL, NCCL LL, host gather. `all_reduce` / `all_gather` consult the table first; sizes above it use the default dispatch |
| `triton_sparse.py` (DeepSeek V4.1 DCP) | the query gather follows the DCP table. When it picks host, the gather runs in order on the main stream, not on the async side stream |
| `host_all_reduce.py`, `host_gather.py` | `from_envs(..., force=True)` builds a candidate without its env switch |
| `custom_all_reduce.py`, `parallel_state.py` | `allow_pcie=True`; graph capture also captures the autotune's custom all-reduce |
| `envs.py` | `VLLM_COMM_AUTOTUNE`, `VLLM_COMM_AUTOTUNE_ALLOW_LOSSY` (also time q8 host modes), `VLLM_COMM_AUTOTUNE_REPORT` (JSON prefix) |
| `tests/distributed/test_comm_autotune.py` | buckets, choice, 3-rank gloo agreement, and on GPU: every candidate reduces like NCCL, host gather byte-equal, all ranks the same table. **5/5 pass on the 4090** |

Why the side stream is avoided: host-gather programs spin until their peers write. With every SM spinning, a
concurrent NCCL kernel on another stream (e.g. another rank's TP all-reduce) can be starved, a possible cross-rank
deadlock. The existing `VLLM_DCP_HOST_GATHER` + async query gather combination has the same exposure.

## Inventory (Manh's fork b9ad670a67 + our patches)
| path | collectives | chosen today | lossless | CUDA graph | on SM89 PCIe |
|---|---|---|---|---|---|
| NCCL (pynccl) | TP all-reduce / all-gather / reduce-scatter; DCP query gather (own communicator on a side stream), DCP partials (grouped send/recv) | last in the dispatch chain, always available | yes | yes | transport P2P / SHM / NET chosen by NCCL from topology |
| NCCL with `NCCL_PROTO=LL` | same | not used | yes | yes | **NCCL reads `NCCL_PROTO` at each communicator's init** (verified: an LL communicator in the same process ran 1.19 vs 1.82 ms at 1 MiB). It doesn't affect send/recv |
| NCCL transport envs (`NCCL_P2P_LEVEL/DISABLE`, `NCCL_SHM_DISABLE`) | — | process env | yes | yes | read once per process by NCCL's parameter cache (from NCCL's source, not tested here), so only a launcher-level probe could choose them |
| custom all-reduce (IPC P2P) | TP all-reduce | on unless disabled; off without P2P or on >2 PCIe-only GPUs | yes (fixed order) | yes | off on the 4090 box (no P2P); a candidate where P2P works |
| torch symm-mem, NCCL symm-mem, FlashInfer (fused / PCIe IPC), quick-reduce, AITER | TP all-reduce | capability or env gated | mostly | yes | not on SM89 |
| host all-reduce (Manh, `VLLM_HOST_ALLREDUCE`) | TP all-reduce <= `MAX_MB` | env; first in the chain | modes without 16 (q8) and without `EMULATE`: fp32 rank-order sum, rounded once | yes | needs /dev/shm, one node |
| host gather (Manh, `VLLM_DCP_HOST_GATHER`) | DCP `all_gather` (indexer) | env | yes (bytes) | yes | Triton spin kernel, all programs resident |

The DCP partial exchange (grouped send/recv, the largest DCP cost) has no alternative path, so the autotune leaves it
on NCCL.

## Measurements on Vast 4090 box 58063361 (no P2P "CNS"; GPU0-1 and GPU2-3 per socket, SYS across)
Collective bench (`scripts/comm_bench.py`, `results/bench_4090a_*.json`, `results/bench.log`): 4 ranks, slowest rank's
median ms per call, 4 KiB-16 MiB per rank, five NCCL env settings.
- With the default env, NCCL beats the host all-reduce from 16 KiB up.
- The host gather beats NCCL's all-gather from 256 KiB (2x at 4 MiB).
- `NCCL_PROTO=LL` cuts the all-reduce by ~35% from 1 MiB.
- `NCCL_SHM_DISABLE=1` is 5-10x slower at small sizes.
- `NCCL_P2P_LEVEL=SYS` changes nothing (no P2P possible).

Autotune tables at server start (`results/at.tp_0.json`, `at.dcp_0.json`; E128, hidden 5120, 2048-token chunks).
These were picked before the 5% margin; the margin only changes near-ties:

| group | picks |
|---|---|
| TP all-reduce | NCCL to 480 KiB; **NCCL LL from 640 KiB** (decode at >= ~64 tokens, all prefill chunks; 26.4 vs 30.5 ms at 20 MiB) |
| DCP all-gather | NCCL / NCCL LL near-ties to 32 KiB; **host from 64 KiB** (1.29 vs 1.49 ms at 512 KiB, 8.3 vs 13.6 ms at 4 MiB) |

Server A/B (`scripts/ab_autotune.sh`, `results/{dec56,dec16,pre32,ids}_*.json`):
- Setup: the fork with the L4 replica's options (DCP4, interleave 64, pad 64,64, embed UVA, util 0.92, FULL graphs to
  64, model `model-e128k`), the same box, one arm after the other, warm compile cache.
- Workload: random token prompts, so no prefix hits; greedy, `ignore_eos`.

| workload | default | autotune | change |
|---|---|---|---|
| decode, 56 streams, 2048-token prompts, 512 out: steady gen tok/s (step ms) | 255.9 (219) | **314.9 (178)** | **+23%** |
| decode, 16 streams: steady gen tok/s (step ms) | 232.2 (69) | **266.0 (60)** | **+15%** |
| prefill, 32 x 8192-token prompts, 8 out: wall s | 392.1 | **274.2** | **+43%** throughput |
| server start (warm cache) | 161 s | 170 s | autotune ~5 s |

Correctness (`scripts/tf_autotune.sh`, `results/tf.log`):
- Method: E062's teacher-forced check (24 real tool prompts, 3.4k-41k tokens, 4,912 tokens of B32 continuations).
- **Fork autotune vs fork default comm:** per-token |dlogprob| mean 0.0366, median 0.0020, p99 0.32, NLL -0.0018.
  **B32 vs B32 repeat (noise floor):** 0.0371 / 0.0026 / 0.33 / +0.0002. So the autotune doesn't change outputs
  beyond run-to-run noise.
- Against B32, the autotune arm shifts NLL +0.0022, the default fork +0.0040; both are the fork's own small shift.
- Greedy ids on random-token prompts part after 2-8 tokens. That is expected there: flat distributions, and the
  stack isn't run-to-run deterministic.

Not separated: how much of the gain comes from TP all-reduce over LL vs from the host query / indexer gathers (it
would need an env to switch each part off).

## Next
- An autotune table on an L4 pod after its E062 replica ends (needs the coordinator's OK). On a one-NUMA PHB box,
  custom all-reduce over P2P and Manh's host all-reduce may win where NCCL wins here.
- Then a short decode A/B there.
- Host all-to-all for the DCP partials: the largest remaining DCP cost, and it has no non-NCCL path yet.
- Launcher-level NCCL transport probe (P2P vs SHM per process), mainly for boxes where NCCL's default is wrong.
  Kaggle's default (SHM) is already right per E027.
