# Offline Muon Python overlay

Upstream source: `theblackcat102/bitsandbytes`, feature/muon commit
`4779028943707ea6aced4669effc9ebfba51981c` (MIT license included in that folder).
The pinned directory name identifies the upstream base, **not an unmodified
copy**: ARC adds per-head routing and the BF16 momentum extension below.

The app's `bitsandbytes_muon_overlay.py` verifies SHA256 hashes, backs up
replaced installed Python files, and overlays these files when the runtime
lacks the ARC capability markers. It does not replace compiled bitsandbytes
CUDA libraries, download anything, or require a new wheel. Copy the complete
`third_party/bitsandbytes_muon` directory with the app in an offline runtime.

## ARC BF16 extension, version 1

`bitsandbytes.optim.muon.MuonBF16` (`--optimizer muonbf16` in `pre_sft.py`):

- One BF16 momentum tensor per matrix, including small matrices. No FP16,
  block scales, quantization maps, or persistent FP32 momentum/master copy.
- FP32 `m = beta * float(old_bf16_m) + float(g)`, then FP32 Nesterov input
  `u = float(g) + beta * m` (or `u = m`). Store `m` and `u` in BF16 once.
- The CUDA implementation fuses this in one Triton pass; FP32 FMA contraction
  is disabled to match the explicit eager oracle. The original 8/32-bit paths
  are unchanged, including their existing rounding.
- Retains existing NS/Gram-NS selection, per-head routing, LR adjustment,
  Nesterov settings, and the app's separate scalar/special AdamW groups.
- BF16 optimizer checkpoints carry `arc_muon_momentum_bits=16`. Loading
  cross-precision optimizer state fails explicitly; model-only loading is
  still allowed. BF16 state is kept BF16 even for FP32 matrix parameters.
- Legacy `percentile_clipping=100` / `block_wise=True` compatibility fields
  allow older base `Optimizer8bit.get_config` implementations. Muon does not
  use these clipping options.

Tests: `tests/test_muon_bf16.py`, `tests/test_bitsandbytes_muon_overlay.py`, and
`tests/test_muon_optimizers.py`. GPU checks compare momentum and NS input
exactly, not just final model loss. See `docs/MUON_BF16_20260907.md` for runtime
measurements, checkpoint size, resume, and the Dion comparison.
