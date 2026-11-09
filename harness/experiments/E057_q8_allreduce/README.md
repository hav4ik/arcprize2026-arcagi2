# E057: 8-bit (two-shot) all-reduce for TP4 on PCIe L4s (planned)

Status: **planned, not started** (idea from a teammate, 2026-10-27). Not part of candidate B (10-28).

## Why
On Kaggle's 4xL4, vLLM cannot use its custom P2P all-reduce ("not supported on more than two PCIe-only GPUs"); the
symmetric-memory and FlashInfer all-reduces are unavailable on SM 8.9 with 4 ranks. TP all-reduce is NCCL's ring
(`AllReduce_Sum_bf16_RING_LL`) over the SHM transport (data staged through host shared memory; the sum runs on the
GPUs). Forcing P2P (`NCCL_P2P_LEVEL=SYS`) is ~12x slower on Kaggle (E027), so SHM stays.

Measured cost (E027, E042):
- **Decode:** 2 all-reduces per layer, ~84 per step. On a pod at 48 streams ~0.32 ms each, ~27 ms per step (~20%).
  Kaggle's SHM is ~2.4x faster (4.9 vs ~2 GB/s), so ~10% there. Time scales with bytes at these sizes (bf16 T x 5120:
  T 32 0.20 ms, 64 0.44 ms, 128 0.97 ms on pod B), so fewer bytes help directly. The rest of the step is mostly MoE
  expert weight reads (floor ~51 of ~66 ms).
- **Prefill:** a 2,048-token chunk spends ~0.6 s (Kaggle) to ~1.3 s (pod) in all-reduce, about half of prefill time.
  Prefill is where Kaggle is ~1.45x slower than the pods (E055 toolkit).

## Idea
Quantize the all-reduce payload to int8 with per-block scales (e.g. 1 fp16/fp32 scale per 32-128 values) and use a
two-shot scheme, like vLLM's QuickReduce (ROCm only today):
1. quantize the local partial sum; all-to-all of chunks (each rank receives its 1/4 slice from the other 3 ranks);
2. dequantize and sum in fp32 locally; requantize the reduced slice;
3. all-gather the int8 slices; dequantize.
Bytes received per rank: 3/4 B (B = bf16 tensor size) against 1.5 B for the bf16 ring, so 2x less traffic.
A plain int8 all-gather of full tensors saves nothing at 4 ranks (each rank still receives 1.5 B).
Transport: NCCL send/recv / all-gather through pynccl over SHM (no P2P needed), captured in CUDA graphs.

**Not lossless.** int8 with per-block scales has ~0.4% of the block max error per element (FP8 e4m3: ~6% relative per
partial sum, E042), applied twice per layer. It must pass a logprob/NLL drift check against the stack's own
run-to-run noise and a 120-task accuracy A/B, like every non-bit-identical change.

## Expected gain (estimate, to be measured)
- Decode: ~3-7% faster on Kaggle after the quantize/dequantize kernels (~84 x 3-4 small kernels per step).
- Prefill: ~1.3x faster (communication is ~half of prefill).
- Larger on pods (slower SHM), so 4090/pod numbers overstate Kaggle's gain.

## Plan
1. **Microbenchmark on a 4x4090 box (~1 h):** NCCL bf16 all-reduce vs the two-shot int8 at T = 32..2,048 (x hidden
   size), CUDA-graph captured; error vs the fp32 reference. Go if >= 1.6x at both decode and prefill sizes.
2. **vLLM overlay backend** with a kill switch (env), plugged in as an all-reduce backend for the TP group; logprob/NLL
   drift vs old-vs-old noise; tool-replay throughput A/B; 120-task accuracy A/B on 4090s.
3. **L4 replica** (12 h) of the best candidate with it on.
4. Optional, needs the user's approval: a few-minute all-reduce microbenchmark on Kaggle's 4xL4 (as E027 did), since
   Kaggle's SHM bandwidth differs from the pods'.

References: experiments/E027_l4_throughput/README.md (SHM vs P2P on Kaggle), experiments/E042_dsv41_l4_synthetic/README.md
(P3 decode profile, P12b NCCL variants, pod B all-reduce bench), vLLM QuickReduce (ROCm two-shot quantized all-reduce).
