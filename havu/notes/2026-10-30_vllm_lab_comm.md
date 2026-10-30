# vLLM lab: tensor-parallel communication over PCIe without P2P

2026-10-30. Hands-on lab, part of E067 (`experiments/E067_vllm_lab/comm/`). Upstream vLLM 0.20.1rc1.dev31, torch 2.11
cu129, NCCL 2.28.9, 2x RTX 3090 on a desktop (Ryzen 9 5900X, one NUMA node, no AVX-512, dual-channel DDR4).

Why: Manh's fork decodes ~1.5-1.7x faster per stream than B32 on 4xL4
(`notes/2026-10-30_vllm_fork_acceleration_map.md`). The main suspect is its communication path: a lossless host
all-reduce (`VLLM_HOST_ALLREDUCE`, MODE 556) and a host all-gather in place of NCCL, on Kaggle's PCIe Gen3 links
without usable P2P. We wanted to see the mechanism first-hand: what NCCL costs per call, where a host all-reduce can
win, and how much decode throughput depends on the comm path.

Model: the weights of `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` are not in the local HF cache, only its tokenizer.
I used `~/.cache/kagglehub/models/yeoyunsianggeremie/aimo-sft-7b/...`, an SFT of it with the same architecture
(Qwen2, 28 layers, hidden 3584, bf16). Communication depends on shapes, not on weight values.

## Summary

- **This box looks like Kaggle at the transport level.** These GeForce cards report no P2P
  (`can_device_access_peer` is False both ways), so NCCL uses SHM: GPU kernels read and write pinned host memory.
  `NCCL_P2P_DISABLE=1` is a no-op here. vLLM's custom all-reduce turns itself off (P2P test fails).
- **NCCL's defaults are the best NCCL setting at every size** (LL for small messages, Simple for large).
  - In CUDA graphs a 2-rank all-reduce costs **13.4 µs at 8 KiB, 63 µs at 256 KiB and 172 µs at 1 MiB**.
    Bus bandwidth levels off at **6.7-6.9 GB/s**, half the 13.4 GB/s a copy engine gets per direction.
  - Forcing `NCCL_PROTO=LL` is 2.4-4.7x slower from 256 KiB, `LL128` 1.7-4x slower at every size, and `Simple` ~2x
    slower up to 16 KiB.
  - With SHM off (`NCCL_SHM_DISABLE=1`) NCCL falls back to TCP sockets: 6-20x slower at decode sizes.
  - Eager `torch.distributed.all_reduce` has a ~30 µs CPU floor, 2.2x the in-graph cost at 8 KiB. Collectives must be
    graph-captured.
- **Host all-reduce prototypes** (all bit-exact against NCCL at 2 ranks):
  - A Python version (cudaMemcpy D2H, CPU sum, H2D) never wins: 33 µs at small sizes with spinning, 95-160 µs with a
    sleeping thread, 2x NCCL at large sizes. It also forces a host sync, so it can't go in a CUDA graph.
  - A zero-copy kernel version (`har_ext.cu`, Manh's design without packing, SIMD or prefetch) **ties NCCL at
    decode sizes**: 13.7 vs 13.4 µs at 8 KiB and 41 vs 46 µs at 128 KiB. **It loses 1.2-2x from 256 KiB.** More CPU
    threads don't help, so the CPU sum isn't the limit.
  - Reducers that sleep 50 µs between polls ("cold") add **~90 µs per call** back to back, and 20-70 µs with compute
    gaps.
- **End to end (vLLM TP2, 7B, CUDA graphs):**
  - TP2 decodes 1.4-1.8x faster than TP1.
  - **Host all-reduce matches NCCL** (85 / 623 / 1,881 vs 86 / 628 / 1,906 tok/s at 1 / 8 / 32 streams), with the
    same greedy tokens.
  - **Cold reducers cost 9-20%.**
  - **The socket fallback costs 4.2x at 32 streams** (455 vs 1,906 tok/s).
  - `step = TP1 step / 2 + 57 x t_allreduce(size)` reproduces these numbers.
- **Why the host all-reduce can win on Kaggle but not here: rank count.**
  - At 2 ranks a ring moves the same bytes per GPU as a host all-reduce (1x up, 1x down) in the same 2 hops.
  - At 4 ranks the ring moves 1.5x each way in 6 dependent hops. The host all-reduce still moves 1x each way (~0.77x
    packed) in 2 hops.
  - So at TP4 its advantage is about 2x in bytes and 3x in hops. Kaggle's NCCL is also ~2x slower than the L4 pods'.
  - It needs hot CPU reducers, one capturable kernel, and a host with enough memory bandwidth.
- **For Kaggle TP4 (~43 layers):**
  - A decode step has ~88 TP all-reduces (~170 collectives with DCP4).
  - NCCL there costs ~0.07 ms per call at 1 stream, 0.14 ms at 24 and ~0.26 ms at 64: **~6, 12 and 23 ms per step**.
  - That is ~15-35% of a 70-110 ms step at 48-64 streams (higher on Kaggle than on the pods). Even free
    communication would give only ~1.2-1.5x.
  - **So the comm path explains part of the ~1.7x, not all of it.**

## 1. Topology and raw link speed

`nvidia-smi topo -m`: GPU0-GPU1 **PHB** (through the CPU's PCIe host bridge). Both are at PCIe Gen4 x8 (Gen1 at idle,
Gen4 under load). P2P isn't available (GeForce). Kaggle 4xL4 (E027): PHB, Gen3 x16; `nvidia-smi` reports P2P OK, but
NCCL P2P there is ~12x slower than SHM.

Pinned host memory, `cudaMemcpyAsync` (`scripts/bw.py`, `results/bw.json`):

| size | H2D GB/s (one GPU) | D2H GB/s (one GPU) | H2D / D2H, both GPUs at once (each) | GPU0 -> GPU1 copy (staged by the driver) |
|---|---|---|---|---|
| 4 KiB | 1.0 (4.2 µs) | 1.0 | 0.4 / 0.4 | 0.3 |
| 64 KiB | 10.2 | 11.6 | 6.6 / 7.0 | 2.8 |
| 1 MiB | 13.1 | 13.3 | 13.1 / 13.3 | 6.1 |
| 64 MiB | 13.5 | 13.6 | 13.3 / 13.4 | 6.3 |

- A copy costs ~4 µs to start.
- Each GPU gets its full link at once, so the host side doesn't limit two concurrent DMAs.
- A device-to-device copy without P2P gets half the link (down, then up).

## 2. NCCL all-reduce, 2 ranks, bf16 (`scripts/ar_bench.py`, `scripts/run_ar_matrix.sh`)

Method:
- In-place calls back to back on one stream. Each call depends on the previous one, as consecutive all-reduces in a
  decode step do.
- Timed eagerly and captured in a CUDA graph (as vLLM decode runs). Slowest rank's median of 5 reps.
- busbw = size / time x 2(n-1)/n (= size / time at n = 2).
- One process per setting, because NCCL reads transport envs once per process.
- `NCCL_DEBUG=INFO` confirms `via SHM/direct/direct` on 2 channels, and `NET/Socket` with `NCCL_SHM_DISABLE=1`.

µs per call (graph-captured unless marked):

| size | default | default, repeat | default, eager | P2P_DISABLE=1 | PROTO=LL | PROTO=LL128 | PROTO=Simple | MIN_NCHANNELS=8 | SHM_DISABLE=1 (sockets) | default busbw GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 KiB | 13.6 | 13.5 | 31.7 | 13.4 | 13.8 | 32.3 | 30.4 | 13.7 | 80 | 0.60 |
| 16 KiB | 17.5 | 15.6 | 31.4 | 15.6 | 17.1 | 37.1 | 32.3 | 15.7 | 147 | 0.94 |
| 32 KiB | 20.9 | 20.0 | 29.8 | 20.2 | 20.2 | 47.5 | 31.0 | 20.2 | 264 | 1.57 |
| 64 KiB | 32.4 | 28.5 | 31.4 | 29.4 | 29.3 | 135.9 | 35.1 | 31.5 | 639 | 2.02 |
| 128 KiB | 49.9 | 46.9 | 56.4 | 47.1 | 48.1 | 110.9 | 44.3 | 49.6 | 916 | 2.63 |
| 256 KiB | 62.7 | 63.2 | 60.6 | 62.3 | 202.7 | 145.0 | 62.2 | 63.0 | 1,054 | 4.18 |
| 512 KiB | 98.3 | 99.7 | 96.4 | 98.4 | 454.7 | 177.6 | 97.6 | 101.4 | 1,155 | 5.33 |
| 1 MiB | 172.0 | 170.5 | 169.4 | 171.1 | 760.1 | 295.5 | 169.8 | 174.5 | 1,369 | 6.10 |
| 4 MiB | 635 | 615 | 621 | 624 | 2,978 | 1,166 | 624 | 635 | 3,171 | 6.61 |
| 16 MiB | 2,421 | 2,423 | 2,436 | 2,413 | 5,702 | 4,331 | 2,448 | 2,555 | 9,611 | 6.93 |
| 64 MiB | 10,033 | 9,616 | 9,858 | 10,044 | 28,140 | 16,790 | 9,685 | 10,117 | 42,107 | 6.69 |

Reading:
- **The default is already the minimum over protocols.**
  - LL (8-byte stores that carry their own flag, so the receiver needs no separate fence) wins below ~128 KiB.
  - Simple (bulk copies plus a flag) wins above.
  - Forcing one of them costs the other range.
  - On E063's two-socket 4090 box, LL was faster from 640 KiB. That box's SHM path is slower, so the crossover is
    machine-specific; this is why E063's autotune measures it.
- **Latency floor ~13.5 µs for 2 hops** (GPU -> host memory -> peer GPU, twice), about 6-7 µs per hop.
  The ring does 2(n-1) hops: 2 at TP2, 6 at TP4.
- **Bandwidth ~6.8 GB/s**, half of what a copy engine gets.
  - With SHM, SM threads load the peer's data from host memory over PCIe, a few warps per channel (est.).
  - More channels don't help (`MIN_NCHANNELS=8`: no change).
  - `NCCL_SHM_USE_CUDA_MEMCPY=1` (copy-engine SHM) didn't finish the sweep within 300 s with graph capture, so it is not
    in the table.
- **Graphs:** eager calls cost ~30 µs at any size up to 64 KiB, which is the CPU's launch cost. vLLM captures decode,
  so the graph column is the relevant one.
- **Noise:** single runs sometimes showed 2x slowdowns at random large sizes (P2P_DISABLE first run: 1.5 ms at 4 MiB,
  19 ms at 64 MiB). Repeats matched the default.
  - `nvidia-smi` logging during the repeat showed the GPUs drop to P5/P8 and PCIe Gen1/2 at times. Comm-only loads
    don't keep the link up.
  - Repeat comm benchmarks, and treat isolated 2x outliers as power-state noise.

## 3. Host all-reduce prototypes (`scripts/host_ar_proto.py`, `scripts/har_ext.cu`)

All paths share one `/dev/shm` segment per path, `mmap`ed by both ranks and `cudaHostRegister`ed (portable + mapped).
- **staged-spin** (Python): D2H `cudaMemcpyAsync` into the rank's slot, spin on the event, raise a flag, spin on the
  peer's flag, sum half the message on the CPU (torch bf16 add = fp32 sum rounded once, 4 threads), flag, spin, H2D.
- **staged-block**: the same, but the thread sleeps on a blocking-sync event and yields while polling ("cold" host).
- **kernel-warm** (`har_ext.cu`, ~250 lines): Manh's design without packing, SIMD, poll-ahead or prefetch.
  - One kernel has writer blocks that store 16 KiB units straight into host memory (zero-copy), fence at system scope
    and post a per-unit flag.
  - Persistent C++ reducer threads (2 per rank) hot-spin on the flags, sum in fp32 in rank order, round once and post
    a done flag.
  - Reader blocks spin on the done flags and copy each unit back.
  - The sequence number lives in device memory, so the kernel can be captured in a CUDA graph.
  - Shape: 12 writer blocks, 12 reader blocks, 256 threads.
- **kernel-cold**: the same, with reducers that sleep 50 µs between polls.
- Gap 200: a 200 µs GPU busy kernel before every call (compute between collectives, during which a cold host falls
  asleep); the gap is subtracted.

µs per call, slowest rank (`results/har_proto.json`, sweep `results/har_sweep_*.json`):

| size | NCCL graph | NCCL eager | staged-spin | staged-block | staged-block, gap 200 | kernel-warm graph (12+12, 16 KiB units) | kernel-warm graph, small shape (4+4, 4 KiB units) | kernel-warm graph, 32+32, 64 KiB units, 6 threads | kernel-cold | kernel-cold, gap 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 KiB | 13.4 | 29.1 | 33.3 | 95.5 | 149.7 | 17.1 | **13.7** | 18.1 | 105.1 | 40.7 |
| 16 KiB | 14.5 | 28.3 | 35.4 | 96.1 | 153.2 | 21.4 | 16.7 | 20.6 | 105.5 | 42.8 |
| 32 KiB | 18.7 | 28.3 | 35.9 | 95.9 | 161.3 | 23.9 | 20.4 | 21.0 | 105.4 | 93.5 |
| 64 KiB | 27.9 | 28.2 | 45.1 | 102.4 | 158.0 | 29.9 | 29.3 | 34.0 | 108.1 | 53.1 |
| 128 KiB | 46.3 | 43.8 | 59.4 | 136.9 | 173.2 | **40.9** | 49.8 | 46.0 | 111.0 | 107.3 |
| 256 KiB | 62.2 | 59.5 | 96.7 | 160.2 | 179.1 | 74.3 | 91.2 | 75.7 | 119.9 | 132.7 |
| 512 KiB | 98.6 | 96.2 | 166.7 | 223.1 | 232.8 | 143.1 | 176.8 | 146.8 | 162.2 | 181.7 |
| 1 MiB | 171 | 172 | 301 | 354 | 360 | 281 | 367 | 303 | 297 | 287 |
| 4 MiB | 620 | 623 | 1,141 | 1,204 | 1,243 | 1,110 | 1,603 | 1,062 | 1,083 | 1,086 |
| 16 MiB | 2,425 | 2,423 | 4,507 | 4,563 | 4,611 | 4,449 | 6,550 | 4,249 | 4,488 | 4,368 |
| 64 MiB | 9,614 | 9,729 | 19,044 | 18,784 | 18,724 | 19,985 | 30,290 | 18,285 | 19,687 | 20,086 |

Every prototype output matched NCCL bit for bit (0 mismatching values at every size). That is expected at 2 ranks,
where each element gets one addition either way. At 4 ranks NCCL's ring rounds to bf16 per hop, so a host all-reduce
differs from it at bf16 noise level (see the acceleration map).

What this shows:
- **The Python staged path loses everywhere.**
  - At small sizes: two copy launches, an event sync and Python flag spins cost ~33 µs, 2.5x NCCL in a graph.
  - At large sizes: D2H, the CPU sum and H2D run one after another (5 + ~9 + 5 ms at 64 MiB), not pipelined.
  - The host sync also means it can't be captured, so a real server would run eager and pay ~30 µs per launch anyway.
- **The zero-copy kernel removes all of that**: one launch, no host sync, graph-capturable.
  - With a small-message shape it ties NCCL's in-graph floor (13.7 vs 13.4 µs).
  - It beats NCCL around 128 KiB (40.9 vs 46.3).
  - At 2 ranks it can't do better. NCCL already uses 2 hops, and both move one message up and one down per GPU.
- **It loses 1.2-2x from 256 KiB**, and 6 reducer threads or 32+32 blocks barely change that (18.3 vs 20.0 ms at
  64 MiB). So the CPU sum isn't the limit. The likely limits:
  - SM-driven zero-copy runs at ~3.3 GB/s each way, against ~13.4 GB/s for a copy engine.
  - The host-memory traffic doubles: GPU writes, CPU reads both slots and writes the sum, GPUs read. On this
    dual-channel desktop that is est. ~27 GB/s at 64 MiB, the same order as NCCL's est. ~28 GB/s of SHM traffic.
  - Manh's packing (1.53 B per value) and AVX-512 loops target exactly these costs. A many-channel Xeon host
    (Kaggle) has more memory bandwidth.
- **Keep-warm matters.**
  - Reducers sleeping 50 µs between polls add ~90 µs per call back to back. With 200 µs gaps they add 20-70 µs: a
    wake-up lands partway through a ~100 µs sleep period (50 µs plus timer slack).
  - A blocking-sync host thread adds 60-120 µs.
  - Manh's reducers spin for ~60 ms of idle before sleeping, and bit 512 keeps the AVX-512 units powered. A core that
    powers them down runs its next sums 2.5-4x slower (Intel; our Zen 3 has no AVX-512, so that half doesn't apply
    here).

## 4. End to end: vLLM TP2 decode (`scripts/e2e_decode.py`, `scripts/e2e_one.sh`)

Method:
- Offline `LLM`, CUDA graphs (FULL_AND_PIECEWISE, default), max_model_len 2048, prefix caching off, util 0.80.
- c random 64-token prompts, greedy, `ignore_eos`. Each run is done at max_tokens 16 and 272; step =
  Δtime / 256, which cancels prefill and start-up. Best of 2.
- Host all-reduce in vLLM: `scripts/har_vllm/sitecustomize.py` (on `PYTHONPATH`, only with `HAR_ENABLE=1`) patches
  `CudaCommunicator.all_reduce` in every worker to route bf16 TP all-reduces through `har_ext.cu`. The installed vLLM
  is untouched.

| config | c = 1: tok/s (step ms) | c = 8 | c = 32 |
|---|---|---|---|
| TP1 (GPU0) | 48 (20.80) | 369 (21.65) | 1,346 (23.77) |
| **TP2, NCCL default (SHM)** | **86 (11.69)** | **628 (12.73)** | **1,906 (16.79)** |
| TP2, `NCCL_P2P_DISABLE=1` | 85 (11.74) | 628 (12.73) | 1,908 (16.77) |
| TP2, `NCCL_PROTO=LL` | 85 (11.71) | 618 (12.94) | 1,761 (18.17) |
| TP2, host all-reduce, warm (12+12, 16 KiB) | 85 (11.74) | 623 (12.84) | 1,881 (17.01) |
| TP2, host all-reduce, warm, small shape for <= 128 KiB, NCCL above | 86 (11.65) | 622 (12.86) | 1,901 (16.83) |
| TP2, host all-reduce, **cold** (50 µs sleeps) | 78 (12.76) | 501 (15.97) | 1,584 (20.20) |
| TP2, `NCCL_SHM_DISABLE=1` (TCP sockets) | 73 (13.62) | 219 (36.59) | 455 (70.31) |

- Greedy tokens are identical across all TP2 runs (host all-reduce included), except the very first TP2 run, which
  built the compile cache and diverged after 3 tokens at c = 1.

**A step model that fits:**
- Qwen2-7B at TP2 does 57 all-reduces per decode step: 2 per layer x 28, plus the vocab-parallel embedding; the logits
  gather comes on top.
- The message is c x 3,584 x 2 B = 7 KiB per sequence.
- Model: step ≈ TP1 step / 2 + 57 x t(size), with t from section 2.

| | c = 1 (7 KiB) | c = 8 (56 KiB) | c = 32 (224 KiB) |
|---|---|---|---|
| NCCL per call (graph) | ~13.4 µs | ~27 µs | ~59 µs |
| 57 calls | 0.8 ms | 1.5 ms | 3.4 ms |
| predicted step (TP1/2 + comm) | 11.2 | 12.4 | 15.3 |
| measured step | 11.7 | 12.7 | 16.8 |
| comm share of the step | 7% | 12% | 20% |
| sockets: predicted / measured | ~14.7 / 13.6 | ~42 / 36.6 | ~70 / 70.3 |
| cold reducers: extra per call (measured step delta / 57) | +19 µs | +57 µs | +60 µs |

The model is good to ~1.5 ms.
- Decode tok/s depends on the comm path only through N x t(size). The weight reads (HBM-bound) set the rest.
- On this box a well-built host all-reduce can't beat NCCL's 7-20% share by much, and a bad path (sockets, sleeping
  reducers) shows up at once.
- `NCCL_PROTO=LL` cost less end to end (+24 µs per call at 224 KiB) than its microbenchmark at 256 KiB suggests;
  not pursued.

## 5. What it implies for Kaggle 4xL4 (TP4, PCIe Gen3, NCCL over SHM)

### Collectives per decode step (~43-layer TP4 DeepSeek V4 / V4.1 Flash)

- **TP all-reduces:** 2 per layer (attention output, MoE output) = 86 at 43 layers (80 at V4.1's 40), plus the
  vocab-parallel embedding. **~84-88 per step.** The acceleration map counts "about 80-84"; B32's profile shows ~84 at
  0.32 ms (pod B).
- **Other collectives:** a logits all-gather, the Engram all-gathers (6.3 ms per step in B32's profile, E042).
- **With DCP4:** ~+76 per step (query all-gather, partial send/recv, indexer gathers): **~160-170** in all.
- **Message:** hidden 5,120 x 2 B = **10 KiB per sequence per all-reduce** (bf16). 64 streams = 640 KiB; a 2,048-token
  prefill chunk = 20 MiB.

### Per-call cost and the per-step budget

NCCL ring over SHM at n ranks ≈ 2(n-1) hops x h + 2(n-1)/n x S / B_link:
- Fitted to E027's two Kaggle points (0.14 ms at 240 KiB, 6.4 ms at 20 MiB): **t ≈ 66 µs + 0.31 µs per KiB**
  (busbw 4.9 GB/s, h ≈ 11 µs).
- On our 2-rank box the same form is 13.5 µs + 0.15 µs per KiB (h ≈ 6.5 µs).

| streams B | message | NCCL per call, Kaggle (fit; measured) | x 88 per step | share of a 34 + 0.61·B ms step (fork fit) | share of a 65 + 0.70·B ms step (B32 fit) |
|---|---|---|---|---|---|
| 1 | 10 KiB | ~0.05-0.07 ms (floor uncertain: LL at 4 ranks) | 4-6 ms | 13-18% | 7-9% |
| 24 | 240 KiB | 0.14 ms (E027, measured) | 12 ms | 25% | 15% |
| 48-64 | 480-640 KiB | 0.21-0.26 ms (fit); pods measured 0.16-0.20 (pod 1) and 0.32 (pod B) | 19-23 ms (fit); 14-28 ms (pods) | ~30% (fit) | ~20% (fit) |
| 80 | 800 KiB | ~0.31 ms (fit) | 27 ms | 33% | 23% |
| prefill chunk | 20 MiB | 6.4 ms (measured) | 0.56 s per chunk | — | — |

The host all-reduce's per-call cost on L4 pods (Q8 modes, from 602df11557): 172 -> 146 µs at 64 tokens with keep-warm.
- MODE 556 matches Q8 on speed (acceleration map), so lossless costs about the same.
- That is ~0.75-0.9x of pod NCCL per call: ~1-5 ms per step at 64 streams, plus whatever the L2 prefetch blocks win
  back by turning the wait into the next GEMM's weight reads.
- Kaggle's NCCL is ~2x slower than pod 1's, so the saving there should be larger: est. 6-17 ms per step at 48-80
  streams, if the host all-reduce runs as fast on Kaggle's host as on the pods (never measured).

**Budget, as a rule of thumb:** to keep TP communication under ~10% of a 70 ms step, it must take ≤ 7 ms, i.e.
≤ ~80 µs per call over 88 calls.
- NCCL on Kaggle is above that beyond a few streams (the fit crosses 80 µs at ~5).
- At 48-80 streams no lossless path in our inventory meets it: the pod host all-reduce is 146 µs at 64 tokens.
- The host all-reduce gets closest, and its L2 prefetch is how the fork puts the remaining wait to use.

### Which path for which size on Kaggle 4xL4

| message | path | why |
|---|---|---|
| decode, 10 KiB-1 MiB (1-100 streams) | **host all-reduce, one graph-captured kernel, small units at <= 64 KiB, hot reducers (spin + AVX-512 keep-warm)** | 2 hops instead of 6, ~0.5-0.67x the bytes per GPU at TP4; ties NCCL even at TP2 here. Cold reducers or a host sync lose 20-100 µs per call |
| prefill, 2-24 MiB | host all-reduce with the large shape (32x32) if the host keeps up, else NCCL default | 1.5x -> 1x bytes per GPU at TP4. At TP2 on a desktop it lost ~2x (host memory, zero-copy rate), so measure on the target host (E063 autotune) |
| any size | **never NCCL P2P on Kaggle** (~12x slower than SHM, E027); never sockets | sockets cost 4.2x decode here; check that the NCCL log says `via SHM` (needs a working /dev/shm) |
| any size | don't force `NCCL_PROTO` | the default already picks LL vs Simple by size; forcing one costs 2-4x on the other side |

### Does communication explain the ~1.7x?

Partly.
- At 48-64 streams the TP all-reduces are ~15-25% of B32's step on pods, probably ~25-35% on Kaggle. Zero-cost
  communication would give ~1.2-1.5x, and a real host all-reduce gets a fraction of that.
- The fork's fitted fixed per-step cost halves (65 -> 33 ms) while the per-stream slope barely moves. That fits a
  per-step cost like communication plus weight reads (the L2 prefetch hides the wait behind the next GEMM's weight
  reads).
- But E062b measured only +9.8% per stream from MODE 556 + row shard (under DCP4).
- So the rest of the 1.7x has to come from the kernel base, prefill speed and the pool size, as the acceleration map
  ranks them.

Measurements still needed, none run here:
- A same-pod A/B of MODE 556 vs NCCL without DCP.
- The same A/B with and without `PREFETCH_BLOCKS`.

## Files

- `experiments/E067_vllm_lab/comm/scripts/`:
  - `bw.py`: copy bandwidth;
  - `ar_bench.py`, `run_ar_matrix.sh`: NCCL matrix;
  - `host_ar_proto.py`, `har_ext.cu`: prototypes; `har_ext.cu` is built with `torch.utils.cpp_extension` into the lab
    scratchpad;
  - `har_vllm/sitecustomize.py`: vLLM hook;
  - `e2e_decode.py`, `e2e_one.sh`: decode bench;
  - `summarize.py`: tables;
  - `burst*.sh`, `lock2.sh`: lock-holding runners. `lock2.sh` waits at most 240 s for gpu1 while holding gpu0, then
    backs off.
- `experiments/E067_vllm_lab/comm/results/`: `bw.json`, `ar_*.json/.log` (with `NCCL_DEBUG=INFO`), `har_proto.json`,
  `har_sweep_*.json`, `e2e_*.json/.log`, `pcie_burst2.csv` (link gen / P-state during burst 2).
- GPU time used: ~26 min under both locks in 5 bursts. That includes ~9 min lost to one hung prototype run; it is
  fixed, and a rank now aborts as soon as its peer fails.
