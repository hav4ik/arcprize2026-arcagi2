---
license: mit
tags: [deepseek, moe, pruning, reap, expert-importance, engram, arc-agi-2]
---

# DeepSeek-V4.1-Flash-1017: routed-expert and Engram-row importance (ARC-AGI-2 workloads)

Importance statistics for every routed expert (40 backbone layers x 384 experts) and every Engram row of
`deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), measured on ARC-AGI-2
reasoning traffic, plus ready keep sets, Engram masks and the results of the keep sets we evaluated. Made in
experiment E043 (repo `hav4ik/arcprize2026-arcagi2`, `havu/experiments/E043_dsv41_real_pruning`, README has the full
story). Use the scores to prune at any level and combine with any quantizer.

## Two workloads

| workload | traffic | expert statistics | Engram census |
|---|---|---|---|
| `notool` (**use this for no-tool deployments**) | no-tool harness (official ARC Prize prompt + task summary, effort low + minimal header, one turn, answer = grid in the text) | 20.0M replayed tokens, 898 rollouts | 109.2M tokens |
| `tool` | tool harness (persistent Python tool, s20x60 + minimal header) | 10.0M tokens | 251.8M tokens |

Calibration data never includes the 120 ARC-AGI-2 evaluation tasks; no-tool screens used a family-disjoint dev split of
the training rollouts. Statistics are from the full native model (all 384 experts, MXFP4 as released).

## Experts

`experts/<workload>/scores.json` (per layer, arrays over the 384 experts) and `scores.csv` (one row per layer x expert):

| field | meaning |
|---|---|
| `selections` | times the expert was in a token's top-6 (all positions) |
| `selections_gen_target` | same, positions whose target (next token) is model-generated |
| `selection_share` | selections / all selections of the layer (sums to 1) |
| `router_weight_sum`, `router_weight_mean` | final top-k weights handed to the expert (after renormalization and route scale) |
| `contribution` | sum over positions of the L2 norm of the expert's *native weighted* output (routing weight applied inside the expert, as the model does) |
| `contribution_dec` | `contribution`, but on layers 20-39 (the CED decoder tower) only positions with a generated target; layers 0-19 all positions. **The best single rule we found.** |
| `reap` | conditional mean amplitude `contribution / selections` (plain REAP; keeps strong-but-rare experts; much worse here) |
| `amp_sq_sum`, `amp_max` | second moment and max of the native weighted output norm |
| `block_selections`, `block_contribution` | the same, restricted to positions whose target is in the generated answer block (`notool`: `</think>` to EOS = the grid) or tool-call block (`tool`) |
| `recommended_score` | `contribution_dec + (W-1) x block_contribution`, W = 10 (`notool`) or 30 (`tool`) |
| `rank_recommended`, `rank_contribution_dec` | rank within the layer (0 = most important; ties by lower id) |

Keep the top-E experts **per layer** by a score (a uniform budget per layer; ranking raw scores across layers is not
meaningful). Example:

```python
import json, numpy as np
s = json.load(open("experts/notool/scores.json"))
E = 112
keep = np.zeros((40, 384), bool)
for L in s["layers"]:
    keep[L["layer"], np.argsort(L["rank_recommended"])[:E]] = True   # rank 0..E-1
```

Raw sufficient statistics (to recompute any rule): `experts/<workload>/reap_stats.npz` with arrays
`N, A, Q, W, MX` of shape `[40 layers, 2 input roles, 2 target roles, 384]` (role 0 = prompt/template token, 1 =
model-generated; target = the next token), `pos_count [2, 2]`; and `<block>_stats.npz` with `N, A [40, 384]`.

**Pruning semantics these scores assume:** pruned experts are removed from selection (selection bias -1e9) and the
native top-6 refills from the kept experts with renormalized weights (router weights, attention, shared expert
unchanged). Physically exported checkpoints keep the router rows of the kept experts (see `scripts/`).

### Keep sets (`experts/keep_sets/<workload>/keep_*.npy`, bool `[40, 384]`; `experts/keep_sets/index.json` lists rule, size and kept ids)

Names: `<score>[_dec]_e<E>` (`_dec` = decoder layers on generated targets), `..._sp<k>` = protect the top-k block
experts, `..._w<W>` = block weight W (the recommended score with W). Evaluated (no-tool harness, 120 eval tasks /
167 pairs, self-hosted):

| keep set (notool) | weights | dev NLL vs full | answer written correctly (dev probe) | ARC pass@1 | ARC pass@2 |
|---|---|---:|---:|---:|---:|
| full model (384) | MXFP4 | 0 | 99.5% | 64.7% | 76.8% |
| `contrib_dec_sp0_w10_e128` | MXFP4 | +8.3% | | | |
| `contrib_dec_sp0_w10_e128` | 2-bit GPTQ (uint2, w13 g128 / w2 g64) | +12.1% | 97.4% | running | running |
| `contrib_dec_sp0_w10_e128` | 2-bit RTN | +22.0% | | | |
| `contrib_dec_sp0_w10_e96` | MXFP4 | +12.7% | | | |
| `contrib_dec_e96` | MXFP4 | +11.1% | 95.3% | | |
| `contrib_dec_sp0_w10_e72` | MXFP4 | +22.2% | 99.5% | **57.5%** | **66.9%** (task-weighted) |
| `contrib_dec_sp0_w10_e64` | MXFP4 | +27.0% | 95.8% | 38.9% | |
| `contrib_dec_sp0_w20_e64` | MXFP4 | +34.1% | 100% | 32.3% | |

All rows with Engram masks of 48-56.6 GiB (free). Plain pruning without the answer-block weight loses ~1 in 5 answers
to unparsable output even when handed correct reasoning; W = 10 fixes it. Tool workload: E72 w30 + `--rescue-code`
29.3% pass@1 (tool harness), E64 w30 18.0%. `results/` has the per-sequence NLL, probe and eval statistics.

## Engram rows

- `engram/<workload>/counts.npz`: lookup counts per row, `uint32`, keys `l{0,1}_r{0,1}` (Engram layer 0 = backbone
  layer 1, Engram layer 1 = backbone layer 14; r0 = prompt-token lookups, r1 = generated-token lookups), `ntok`.
  384,006,168 and 384,016,682 rows (264 bytes each as released: 256 FP8 E4M3 + 8 UE8M0).
- `engram/geometry.npz`: the n-gram hash geometry (token map, multipliers, primes, column offsets) needed to map
  token n-grams to row ids; `scripts/engram_census.py` has the hash (`hash_ids`) and the census / select / coverage
  commands.
- `engram/<workload>/masks/engram_top<B>gib.npz`: top-frequency keep masks for a B GiB budget, keys `layer0`,
  `layer1` = `np.packbits(bool_mask, bitorder="little")`; `engram/masks_index.json` has rows kept per layer and the
  calibration coverage. A dropped row reads as a zero row (before Wkv).
- No-tool hit rates (share of lookups landing on kept rows; dev split / full-model traces on the 120 eval tasks):
  16 GiB 91.1 / 88.0%, 32 GiB 93.7 / 91.3%, 48 GiB 95.3 / 93.4%, 56.6 GiB (30%) ~96 / ~94%, 64 GiB 96.6 / 95.1%,
  all observed (73.2 GiB) 97.5 / 96.3%. Dev NLL cost: +0.05-0.10% for 48-73 GiB (within noise). Misses are almost
  all (99.8%) n-grams of the model's own reasoning text.

## Scripts (`scripts/`)

`reap_aggregate.py` (statistics, `select`), `special_stats.py` (block statistics, weighted / protected selection),
`engram_census.py` (hash, census, masks, coverage), `compare_routing.py`, `e043_obs.py` + `apply_e043_patches.py`
(runtime expert / Engram masks and the observer in the vLLM DeepSeek-V4.1 fork), `e043_humming.py` +
`apply_e043_humming.py` (real-weight low-bit experts through Humming, per-sublayer bits, precomputed GPTQ tensors),
`gptq_experts.py` (per-expert GPTQ with held-out check).
