# Non-uniform expert counts per layer (2026-10-26, offline, no box)

Input: the tool workload's per-expert scores from the HF dataset `chankhavu/dsv41-flash1017-expert-importance`
(`experts/tool/scores.json`, downloaded to `data/dsv4_prune/importance/`). Script: `alloc.py`.

Proxy: per layer, the share of the layer's score kept by its top-E experts. Objective: total lost share over the 40
layers, each layer normalised to its own total. A greedy allocation (floor 64) is optimal for this proxy.

| score | same memory as uniform E128: equivalent uniform | same loss as uniform E128: experts/layer | last 4 layers |
|---|---|---|---|
| contribution_dec + 9 x tool-block (the E128 rule) | E135 | 121.6 (-5% expert memory) | 64 each (floor) |
| selections_gen_target + 9 x tool-block selections | E131 | 125.4 (-2%) | 143-157 |

- **The two proxies disagree** (per-layer E correlation 0.33), mostly on layers 36-39.
  - There one expert holds 55-80% of the norm-based score: a massive-activation expert, so a norm-based share says
    the other 383 barely matter.
  - The selection-count score says those layers need more experts than average.
- They agree that layers 0, 1, 13, 15 and 19 want more experts, and that layers 25 and 28-31 want fewer.
- **Effect size:** at equal memory, about +3 to +7 experts per layer of uniform-equivalent quality on this proxy, or
  2-5% less expert memory (0.3-0.6 GB per GPU on L4, about +0.2-0.4M KV tokens) at equal loss. That is smaller than
  the E112 -> E128 step.
- **Speed:** unchanged at equal total experts. Decode reads about the same expert bytes per step; any speed-up comes
  only from fewer total experts.
- **Validation needs a box:** dev NLL and tool-block NLL with runtime masks (the masks need only the full model), then
  GPTQ and an eval. Serving needs a per-layer `n_routed_experts` patch in the vLLM fork and the exporter.
