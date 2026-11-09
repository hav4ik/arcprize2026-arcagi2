# ACE: combine contribution estimates before skipping

Reviewed 2026-10-31. [Xu et al., September 2026, v1](https://arxiv.org/html/2609.05228v1).
Code linked by authors: [xzktx003/moe-prune](https://github.com/xzktx003/moe-prune).

ACE combines offline spectral and router-conditioned response estimates with
runtime gates. It skips only when both estimates consider a slot weak, retaining
the strongest expert. Its headline advantage is over competing compression
methods: on Qwen3-30B-A3B, 50% skipping still loses 6.37 average accuracy points
against the original model. The 1.31–1.41× decode gains use A100, 60% skipping,
short sequences and small batches. Thresholds are workload-matched.

**Our interpretation:** useful motivation to measure contribution rather than
frequency alone. Router-centering arguments rely on softmax translation
invariance, absent from our sqrt-softplus router. Empirical E065 output-response
measurements are a safer starting point. None of these results establishes our
two-point ARC-AGI margin.
