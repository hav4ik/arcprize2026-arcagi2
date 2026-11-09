# Pinned SonicMoE pack

This directory carries a pure-Python wheel built from unmodified SonicMoE
upstream commit `7396f3e604827d8186c2e16e64b28ee33d3defd0`. The build changes only
the package version/identity metadata so an older `0.1.2.post1` release cannot
be selected accidentally. The ARC packed-length autotuning fix is implemented
separately in `diffusion_gemma_sonic_moe.py` and applied at runtime.

Rebuild from an exact clean checkout:

```bash
python build_sonicmoe_pack.py \
  --source /path/to/Dao-AILab/sonic-moe
```

`sonicmoe_runtime.py` validates and extracts the wheel automatically. Fresh
Kaggle commands therefore use `PYTHONPATH=.`; they do not need a SonicMoE
checkout, editable install, or notebook setup cell. `quack-kernels>=0.6.4`
remains an external binary dependency supplied by the offline wheel dataset.
