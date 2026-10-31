# E081: Kaggle 4×L4 machine survey (host links, NCCL pairs, mount read speed)

User request 2026-10-31. Manh Nguyen found a Kaggle 4×L4 machine where GPUs 2 and 3 have slow links to the host
(H2D 2.6 GB/s, D2H 0.7 GB/s vs 11 GB/s) and every NCCL pair that touches them is 4–8× slower. Question: how common
are such machines, and how fast do our submission's mounted inputs read? Budget: 2 h of GPU (user-authorized).

## Method
- 19 short commit notebooks `chankhavu/e081-l4-probe-01..19` (private), `machine_shape` NvidiaL4, competition attached
  (otherwise the run silently gets 2×T4), internet off, and **exactly the inputs Manh's V12 (LB 76.25) mounts**:
  - model `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1` (136 files, 131.09 GB);
  - `nguyennguyen599/vllm-wheels`, `nguyennguyen599/cu13-install`, `chankhavu/dsv41-tools-kit`,
    `nguyennguyen599/arc2-2026-src`, `nguyennguyen599/unsloth-main` (unpinned, as V12).
- `probe.py` (one run per notebook, ~170 s; writes `probe.json`):
  1. identity: hostname, CPU, GPU uuid/serial/PCI bus/power limit, `nvidia-smi topo -m`;
  2. Manh's `gpucheck.py` verbatim (`gpucheck_manh.py`): pinned 64 MB H2D/D2H per GPU, NCCL all-gather and send/recv
     medians for every GPU pair at 516 KB and 4 MB per rank, alone and two pairs at once;
  3. mounted-input read speed: cold single stream (first ≤2 GiB of the largest file, ≤30 s), cold 8-thread parallel
     over the next ≤16 untouched files (≤1 GiB each, ≤45 s), then a warm re-read (page cache);
  4. `/kaggle/working`, `/tmp`, `/dev/shm` write speed (2 GiB `dd` with fsync). Their read numbers are page cache
     (the container cannot drop caches) and are not meaningful.
- `build.py N OUTDIR` builds notebook N; `launch.py` pushes them with at most 2 running (Kaggle's batch GPU limit)
  and a GPU-minute budget stop; `aggregate.py` writes `summary.md`. Raw results: `runs/probe-NN/probe.json`.
- Cost: ~92 GPU-minutes (each run 2.6–4 min wall, probe ~170 s); probe-20 was skipped to free a batch slot for the
  dsv4-infer session's smoke run. Plus one earlier single check (`chankhavu/arc2-l4-gpucheck`) and ~1 min of an
  interactive session.

## Results (19 batch commits, 2026-10-31 20:05–20:58 UTC)
| | min | median | max |
|---|---|---|---|
| H2D / D2H per GPU (GB/s), all 76 GPUs | 11.5 / 12.1 | 11.5 / 12.3 | 11.5 / 12.3 |
| NCCL all-gather 516 KB, best pair (µs) | 147 | 148 | 151 |
| NCCL all-gather 516 KB, worst pair (µs) | 225 | 228 | 230 |
| model mount, cold single stream (MB/s) | 102 | 169 | 285 |
| model mount, cold 8 parallel (MB/s) | 537 | 884 | 1,034 |
| → whole 131 GB model at that rate (min) | 2.1 | 2.5 | 4.1 |
| vllm-wheels, cold 8 parallel (MB/s) | **56** | 550 | 959 |
| cu13-install, cold 8 parallel (MB/s) | 150 | 540 | 834 |
| `/kaggle/working` write (MB/s) | 311 | 312 | 342 |
| `/dev/shm` write (MB/s) | 2,464 | 2,509 | 2,541 |

- **No bad machine in 19 runs** (17 distinct machines by GPU UUIDs; probe-10/16 and 11/17 landed on the same machine).
  Every GPU had full PCIe gen3 x16 host bandwidth. With 0/19, the rate of Manh-style machines in the batch pool is
  probably below ~15% (rule of three, 95%); it is clearly possible (Manh hit one), but not common.
- **NCCL is uniform across machines and has a fixed shape:** on every machine two pairs are ~228 µs and four are
  ~148 µs at 516 KB (4 MB: ~1,400 vs ~755 µs). The slow pairs were always 0-1 and 2-3 (on some machines one of them
  measured slightly worse); the cross pairs 0-2, 0-3, 1-2, 1-3 are the fast ones. All GPUs hang off the CPU root
  complex (`PHB`), no P2P, power limit 72 W everywhere. Our slowest pair equals the best pair on Manh's bad machine.
- **Mounted model reads vary ~2× between machines** (537–1,034 MB/s with 8 parallel readers; single stream only
  100–285 MB/s). Staging the 131 GB model needs parallel readers: ~2.5 min at the median with 8 threads vs ~13 min
  single-stream. Small-file datasets vary more: one run read `vllm-wheels` at only 56 MB/s (probe-13), others
  240–960 MB/s.
- Local disks are uniform (~312 MB/s write on `/kaggle/working` and `/tmp`; `/dev/shm` ~2.5 GB/s, 86 GB).

## Caveats
- All probes were **batch commits**. Interactive sessions (where Manh likely saw the bad machine) may be drawn from a
  different pool; one interactive session started here but its cell failed, so no interactive data point exists.
- The reads are time-boxed samples (~17 GB of the model per run), not full stagings; cold means "first read in this
  session", which may still hit a Kaggle-side cache if another session read the same blocks shortly before.
