# DSL-based systems reporting ARC-AGI-2 results (2025–2026)

Compiled for E015 (`../experiments/E015_dsl_analysis/survey.md`). None of these is ARC Prize-verified.

| System | Date | DSL | How the DSL is used | ARC-AGI-2 result | Notes |
|---|---|---|---|---|---|
| Compositional Neuro-Symbolic Reasoning (Das et al., Stanford), [arXiv 2604.02434](https://arxiv.org/abs/2604.02434) | 2026-04 | 22 atomic "Unit Patterns" over object scene graphs | o4-mini detects patterns as JSON hints; Grok-4 writes the output grid **transductively** with 3–10 self-consistency samples | Public eval pass@2: Grok-4 16.0 → 24.4%; 30.8% ensembled with Berman's ARC Lang solver | Only ARC-AGI-2 ablation of DSL hints against no hints: −6.9 pp without hints (24.4 → 17.5). Failures cluster "beyond the 22-pattern DSL" |
| LLM + genetic search, [arXiv 2609.27242](https://arxiv.org/abs/2609.27242) | 2026-09 | 41 fixed-width ops (27 core + 14 LLM-gated); every mutation stays valid | Qwen3.5-4B (quantized) seeds programs; a GPU GA tests about 1.6M programs per task; built for Kaggle | First 60 public-eval tasks: LLM alone 2 (3.3%); +GA 6 (10.0%); GA without LLM seeds 0 | 20 of the 60 tasks used for development. Authors: 41 ops "express far fewer ideas than plain Python" |
| ARCANA, [arXiv 2607.09059](https://arxiv.org/abs/2607.09059) | 2026-07 | 47 typed primitives | A 48M-parameter CVAE program proposer + executor + "reflective" agent, with TTT | Claims 32.5% "semi-private" under Kaggle limits | **Treat as unverified:** no ARC Prize verification, the semi-private set is not available to authors, and the paper has boilerplate citations |
| icecuber 2020 (baseline) | 2020 | 142 C++ unary transforms | Brute-force depth-4 search | 1.6% semi-private ([arXiv 2505.11831](https://arxiv.org/abs/2505.11831)) | See `icecuber_dsl_search.md` |

**For contrast, ARC-AGI-2 systems that use no DSL:**

| System | Representation | Result |
|---|---|---|
| NVARC | Free Python generators for data, transduction at test time | 24.03% private |
| Pang | Python | 26.0% semi-private |
| Berman | English | 29.4% semi-private |
| Imbue + Gemini 3 Flash | Python | 61.4% public |
| Our Flash + Python tool | Python tool, grid answer | pass@1 ≈ 68% public |

## Takeaway
- **DSL as hints helps a weak LLM a little.** The evidence is one unreplicated paper.
- **DSL as the output language caps out low**, because primitives are missing.
- **No DSL system reaches what frontier or mid-tier LLMs get with Python tools.**
