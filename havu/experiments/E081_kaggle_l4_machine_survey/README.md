# E081: Kaggle 4×L4 machine survey (host links, NCCL pairs, mount read speed)

User request 2026-10-31. Manh Nguyen found a Kaggle 4×L4 machine where GPUs 2 and 3 have slow links to the host
(H2D 2.6 GB/s, D2H 0.7 GB/s vs 11 GB/s) and every NCCL pair that touches them is 4–8× slower. Question: how common
are such machines, and how fast do our submission's mounted inputs read? Budget: 2 h of GPU (user-authorized).

## Bottom line (23 batch probes, 20 distinct machines, ~118 GPU-min, 2026-10-31)
- **No machine like Manh's:** every GPU of every probe had full host bandwidth (H2D 11.5, D2H 12.1–12.3 GB/s). With 0/23,
  such machines are probably under ~13% of the batch pool (rule of three, 95%), though they clearly exist.
- **NCCL has one fixed shape on every machine:** pairs 0-1 and 2-3 ~228 µs, cross pairs ~148 µs (all-gather, 516 KB per
  rank); no P2P, 72 W power limit everywhere.
- **Model mounts:** 537–1,034 MB/s with 8 parallel readers (median 867, ~2.5 min for the 131 GB model) but only
  102–285 MB/s single-stream, so staging must read in parallel. Small-file datasets vary more (vllm-wheels 56–959 MB/s).
- **The causal image E128 (`arazzd/...-img-e128-causal`) mounts as fast as V12's text E128 on the same machine** (E081b);
  nothing supports a dataset-region effect, and the zone is not observable from inside a notebook.
- **Not covered:** interactive sessions, which may come from a different pool. Full per-run table: `summary.md`.

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
- `build.py N OUTDIR [b]` builds notebook N (`b`: the E081b dual-model variant); `launch.py` pushes them with at most 2 running (Kaggle's batch GPU limit)
  and a GPU-minute budget stop; `aggregate.py` writes `summary.md` (all 23 runs). Raw results: `runs/probe-NN/` and `runs/probeb-NN/` (`probe.json`).
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

## E081b: does the causal image E128 mount read differently? (4 probes, 2026-10-31 21:14–21:32 UTC)
User hypothesis: each Kaggle model/dataset lives in some region, compute is provisioned near it, and Manh's bad machine
came from the region holding the newer model. Probes `chankhavu/e081-l4-probeb-01..04` attach **both**
`chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1` (V12's text E128, 131.09 GB) and
`arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-causal/PyTorch/default/1` (132.06 GB, 137 files) and read both on the
same machine within the same minute, alternating which is read first (`build.py N OUTDIR b`, `compare_b.py`). They also
query the GCE metadata server for the zone. ~20 GPU-minutes (the rest of the 2 h budget).

| probe | read first | text E128 single / 8-par MB/s | causal image E128 single / 8-par MB/s | host links | NCCL 516 KB best / worst µs |
|---|---|---|---|---|---|
| b-01 | image | 249 / 767 | 147 / 765 | 11.5 / 12.3 GB/s all GPUs | 150 / 229 |
| b-02 | text | 171 / 747 | 255 / 777 | same | 150 / 227 |
| b-03 | image | 246 / 717 | 133 / 706 | same | 150 / 227 |
| b-04 | text | 150 / 957 | 274 / 959 | same | 150 / 228 |

- **The two models read at the same speed on the same machine:** 8-parallel rates differ by at most 1.5% within each
  probe (706–959 MB/s across machines). The only systematic difference is order: whichever model is read first gets the
  lower single-stream number (133–171 vs 246–274 MB/s), a first-read warm-up, not a dataset effect.
- **All 4 machines were healthy** (full host links, the usual NCCL shape), so 23 batch probes in total, no bad machine.
- **The zone is not observable:** the metadata server does not answer inside Kaggle notebooks (zone, machine type and
  instance id all empty), so the region hypothesis can't be tested directly from inside the notebook. Nothing here
  supports it: the newer model's mount is as fast as the old one's, and the machines that mount it look identical.
