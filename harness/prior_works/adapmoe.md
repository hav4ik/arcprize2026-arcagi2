# AdapMoE: sensitivity-aware gating plus expert offloading

Reviewed 2026-10-31. [Zhong et al., ICCAD 2024](https://arxiv.org/html/2408.10284v1),
[official implementation](https://github.com/PKU-SEC-Lab/AdapMoE).

AdapMoE combines sensitivity-based token/layer gating with expert prefetching and
cache allocation. It reports 25% fewer active experts and 1.35× speedup without
accuracy degradation in its evaluated setup. The system targets expert-loading
bottlenecks; the repository's accuracy scripts focus on Mixtral and warn that
different evaluation harnesses give different results.

**Our interpretation:** historical support for heterogeneous expert needs, but
its speedup cannot be attributed solely to reduced top-k or transferred to our
resident two-bit routed-expert weights. E128's long tool transcripts need their
own prefix-cache and wall-time measurements.
