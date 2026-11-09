# LExI: static per-layer active-expert allocation

Reviewed 2026-10-31. [Chitty-Venkata et al., 2025, v1](https://arxiv.org/html/2509.02753v1).

LExI profiles each layer's output perturbation at different top-k values using
synthetic standard-normal inputs, then conducts proxy-guided evolutionary search.
It evaluates language and vision-language MoEs through vLLM on H100s. Its
headline accuracy advantage compares against traditional pruning at matched
throughput, not against the unmodified model. Reported language throughput
counts input plus output tokens, unlike our generated-token rate.

**Our interpretation:** the closest deployment match to E074/E076. Our calibration
uses E065 traces and whole-model NLL, not LExI's synthetic proxy. The current
seed14 is a 4/6 schedule derived from the E074 ranking; it is not an exhaustive
optimization over k2/3/4/5/6. Reprofiling and jointly checking schedules could
improve allocation without token-dynamic serving.
