# Separate executed experts from routing normalization

Reviewed 2026-10-31. [Chen and Yao, 2026, v1](https://arxiv.org/html/2609.04575v1),
*Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models*.

The method executes top-k1 but divides weights by top-k2 router mass.
Reported losses versus native routing:

| Model | Execution change | MMLU loss: ordinary / adjusted normalization | GSM8K loss: ordinary / adjusted |
|---|---|---|---|
| Qwen3.6-35B-A3B | 8→4; k2=16 | 4.65 / 0.35 pp | 6.60 / 0.60 pp |
| Qwen3.5-397B-A17B | 10→5; k2=10 | 2.10 / 0.55 pp | 4.20 / 0.80 pp |

MMLU uses 2,000 questions. Non-significance does not establish equivalence.
Perplexity and task accuracy prefer different normalizers. Long-context,
long-form generation and our architecture remain untested.

**Our proposed transfer:** preserve E128's biased selection and raw sqrt-softplus
weights; first compare the current normalizer with native-six reference mass on
reduced-k layers. Protected k6 layers retain their native rule. This is an
unmeasured router change, not a new expert GEMM. See the
[E128 review](moe_active_parameter_pruning.md) for the equation and validation plan.
