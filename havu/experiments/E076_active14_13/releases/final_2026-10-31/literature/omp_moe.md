# OMP-MoE: reconstruction and marginal layer budgets

Reviewed 2026-10-31. [Li et al., v2, September 2026](https://arxiv.org/html/2609.31631v2).

The main method selects stored experts using residual reconstruction, then
allocates a global budget across layers with routing stability. A separate
adaptive extension predicts contribution energy to skip runtime expert work.
The 50% compression headline primarily describes stored-expert pruning; it is
not automatically a 50% active-compute reduction. The arXiv record calls the work
in progress; the abstract and HTML differ on code-release availability, and no
implementation was verified in this review.

**Our interpretation:** use marginal whole-schedule error and routing stability
as additional signals. Avoid treating independently measured layer losses as
exactly additive. A reproduction is lower priority than our existing static
schedule and normalization experiments.
