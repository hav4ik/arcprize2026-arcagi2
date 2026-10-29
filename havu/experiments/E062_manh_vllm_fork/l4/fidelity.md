# E062 L4 pod: fidelity vs Kaggle and the earlier replica pods

Pod: Runpod `ddj7xv53e5rgre` (`ssh -p 19843 root@213.173.105.17`), rented by the user 2026-10-28 ~07:50 UTC, $2.36/h,
image `runpod/pytorch:1.4.1-cu1300-torch280-ubuntu2404`, 200 GB container disk, no volume. Measured 07:55-08:15 UTC.

| | this pod | earlier replica pods (E055: hegn9880s1tz6m, 157.157.221.29) | Kaggle 4xL4 (B32 smoke, E056 `kaggle_smoke_v2_b32/smoke_env.jsonl`) |
|---|---|---|---|
| GPUs | 4x L4, 23,034 MiB | 4x L4 (21.9-22.0 GiB free at start) | 4x L4, 23,034 MiB |
| power limit / max SM clock | 72 W / 2,040 MHz | 72 W / 2,040 MHz | **50 W cap per the user** for the scored rerun. Our two commit-run smokes logged 72 W (notes 2026-10-26 §345), but the scored rerun can't be observed |
| PCIe | Gen4 x8 max (Gen1 at idle) | not recorded | Gen3 (AGENTS.md E027) |
| GPU topology | GPU0-1 and GPU2-3 PHB, others NODE; 1 NUMA node | not recorded | not recorded |
| CPU | AMD EPYC 9254 (24 cores), `nproc` 48, **cgroup CPU quota 22.95 CPUs** (cfs 2295000/100000; Runpod lists 27 vCPU) | `nproc` 128 (quota not recorded) | g2-standard-48 class (48 vCPU; not measured in the smoke) |
| host RAM | MemTotal 503 GiB, cgroup limit 286 GB (266 GiB) | MemTotal 503 GiB, cgroup 286 GB | MemTotal 188.7 GiB, cgroup 173 GiB |
| /dev/shm | 134 GB | not recorded | 85.5 GiB |
| driver / CUDA | 595.91 / 13.2 (toolkit 13.0 at /usr/local/cuda-13.0) | not recorded | not recorded |

Reading:
- **GPU side:** the same L4 and memory as Kaggle and the old pods, and the KV pool is pinned through `ARC_GPU_UTIL`
  as before.
  - **Power is not matched:** the pods run at 72 W, and Kaggle's scored rerun runs at a 50 W cap (user).
  - **So every pod number is optimistic vs Kaggle.** The replica report's Kaggle-clock correction stretches the pod
    timeline with factors measured on a real Kaggle run (prefill x1.45, decode x0.94); raw pod tok/s are shown next
    to it.
- **PCIe:** Gen4 x8 here vs Kaggle's Gen3 (x16 unknown). Host <-> GPU traffic (KV offload restores, UVA embed /
  Engram reads, DCP all-gathers that go through host memory) can be faster on this pod than on Kaggle. Measured below.
- **CPU is the weakest point of this pod:** a 23-CPU quota. The tool sandboxes, train checks and the harness of 56-80
  concurrent attempts run on the CPU next to vLLM's 6 busy processes. The same-pod B32 window (below) shows whether
  this pod reproduces the old pod's B32 throughput; CPU use is logged.
- **Host memory:** the runs keep the server's footprint within Kaggle's 173 GiB (logged as cgroup `memory.current`).

## NCCL all-reduce (E042 `scripts/nccl_allreduce_bench.py`, torch 2.13 / B32 venv, default transport)

| T (tokens x 5120 bf16) | this pod | Kaggle (E027) | Vast 4xL4 box (E042) |
|---:|---:|---:|---:|
| 64 | 0.162 ms | | 0.349 ms (Simple) |
| 256 | 0.444 ms | | 1.443 ms |
| 2048 | **3.28 ms (6.4 GB/s algbw)** | **~6.4 ms** | 13.9 ms (1.5 GB/s) |
| 8192 | 12.9 ms | | 56.9 ms |

All-reduce on this pod is ~2x faster than Kaggle's. Prefill is ~half communication on these stacks (E042), so this pod
prefills faster than Kaggle. The replica report's Kaggle-clock stretch (prefill x1.45, decode x0.94, measured on an
earlier pod) is applied unchanged; this pod may be somewhat faster than that pod, see the same-pod check.

## Same-pod check: B32 on this pod vs E055's `candBf_x2t` (same settings, workload, kit)

`runs/b32pod_x2t` (started 08:25:20 UTC, stopped by SIGTERM at pod 1.10 h: final write, solver exit 0, server
stopped). Server up after 235 s (old pod 290 s), **KV pool 1,556,850 tokens (identical)**, model loading 16.9 GiB.

| pod h | this pod: gen tokens / attempts done / mean gen tok/s since up | old pod (candBf_x2t) |
|---|---|---|
| 0.5 | 0.56M / 11 / 439 | 0.53M / 10 / 404 |
| 0.75 | 1.02M / 22 / 461 | 0.97M / 21 / 429 |
| 1.0 | 1.50M / 32 / 474 | 1.40M / 30 / 439 |
| 1.07 | 1.65M / 37 / 474 | 1.54M / 34 / 441 |

- **This pod is ~7.5% faster than the old pod** for B32 over the first hour (cumulative to 1.07 h; +3 to +8% by
  window) (KV 0.73 vs 0.72, prefix hits 0.95 vs 0.96,
  0 preemptions in both, 56 running). The faster all-reduce is the likely reason.
- **CPU is not a bottleneck:** 5.4 CPUs used of the 22.95 quota at 56 workers, no throttling after model load.
- So Manh's fork on this pod compares with B32 x ~1.07, not with candBf_x2t's raw numbers.
