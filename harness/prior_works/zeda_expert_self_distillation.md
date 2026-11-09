# ZEDA: recover dynamic expert skipping through self-distillation

Reviewed 2026-10-31. [Lv et al., v2, 2026-06-08](https://arxiv.org/html/2605.18643v2).
Official code: [TsinghuaC3I/ZEDA](https://github.com/TsinghuaC3I/ZEDA).

ZEDA adds zero-output experts, then applies teacher-generated SFT and on-policy
distillation. Group balancing controls real-versus-zero expert use without
flattening the normal experts' routing distribution.

Across 11 math/code/instruction benchmarks, Qwen3-30B-A3B averages 74.9→74.2
with 51.2% zero slots; GLM-4.7-Flash averages 72.5→71.8 with 53.0%. Reported
throughput improvement is about 20% on one H200, SGLang, 8,192-token sequences,
maximum concurrency 32 and matched token counts. Adaptation used 60k prompts and
30.12/61.37 hours on eight H200s. Long-horizon agentic tasks are not evaluated.

**Our interpretation:** meaningful evidence for recovery after post-training,
but not a cheap, demonstrated E128 recipe. A fixed-schedule student could borrow
the teacher-recovery idea while avoiding zero-expert serving changes; that
variant would be our experiment, not a reproduction of ZEDA.
