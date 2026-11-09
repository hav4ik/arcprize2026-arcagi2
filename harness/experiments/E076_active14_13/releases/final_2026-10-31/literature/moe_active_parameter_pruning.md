# Active-parameter pruning for E128 (review, 2026-10-31)

The target is the causal image E128 checkpoint in E066, already physically expert
pruned and GPTQ-quantized to two bits. Transfer from other MoE models is a hypothesis;
none of the papers below establishes preserved ARC accuracy for this checkpoint.

## Final E076 evidence and implications

Both modalities are now complete on all 120 Kaggle ARC-AGI-2 tasks, two draws.
Text+tools is the primary acceptance modality: 14.02B task-weighted pass@1/pass@2
is 69.24%/79.44% versus E128 72.29%/81.81%. The -3.06pp pass@1 point estimate
exceeds the two-point loss target; the paired 95% interval [-8.68,+2.78]pp does
not establish noninferiority. Matching elapsed minutes 5-155 yields +2.61%
generation throughput, but 2.74% more output tokens and 1.13% longer total runtime.
The current schedule is a poor tradeoff for this text harness. The 14.58B text
reference was explicitly skipped. Deeper pruning remains unmeasured; the
recommendation is to establish useful end-to-end speed savings before investing
in more aggressive schedules or recovery. No new pruning job was launched.

If this research is resumed, an actionable direction is **per-layer top-k plus separately calibrated
output scaling**, followed by recovery distillation if a lower budget needs it.
Token-dependent counts are a later serving project. These are proposed methods,
not implemented or measured improvements to E076.

The completed image+tools measurement is on 120 Kaggle ARC-AGI-2 tasks, two draws:
task-weighted mean-draw pass@1 is 71.67% for 14.02B versus 72.43% for E128 and
70.83% for 14.58B; pass@2 is 83.89% versus 83.47%/81.81%. Candidate full-run
generation throughput gains are 9.46%/2.95%, but wall time improves only 4.39%
versus E128 and is essentially tied with 14.58B. These are observations of a
previously frozen candidate, not selection data for future pruning. See the
[final reproduction record](../experiments/E076_active14_13/releases/final_2026-10-31/README.md).

**Data boundary, latest user clarification:** future calibration, recovery training,
and pruning-setting selection mix original ARC tasks from the 1,000-task training
set with NVARC training-parent rollouts. This supersedes the earlier NVARC-only
restriction. Use E065 `arc2_train` and `nvarc_b0`/`nvarc_b1` image/text configurations;
exclude `nvarc_full`, `arc2_evaluation6`, and all 120 Kaggle evaluation parents or their
derivatives. Read evaluation IDs only for exclusion checks; never fit against
their outcomes. Historical E074/E076 pools are mixed E065 training rollouts
with zero evaluation-parent overlap and comply with this source boundary. The
[policy and guard](../experiments/E076_active14_13/configs/training_data_policy.json)
validate both sources. Record source counts and weights, report diagnostics by
source as well as jointly, and keep each original and its synthetic descendants
in one parent-family split. Prior exposure limits generalization claims, not
eligibility for calibration. No mixture ratio is fixed yet.

## Methods and applicability

| Method | Evidence / mechanism | E128 application |
|---|---|---|
| Uniform reduced top-k | Simple baseline that actually executes fewer experts | Test before searching a layer schedule; preserve native router semantics |
| [LExI](lexi_active_experts.md) | Per-layer perturbation profiling and evolutionary search | Closest static serving match; original-training and NVARC activations can supplement the synthetic proxy |
| [Reference-mass normalization](moe_reference_mass_normalization.md) | Separate executed count from the routing-weight denominator | First additional variable to test with the existing schedule; transfer remains unmeasured |
| [ASET](aset_expert_skipping.md) | Static schedules and confidence/entropy-driven token activation | Supports heterogeneous budgets; gate semantics and dispatch need adaptation |
| [ACE](ace_expert_skipping.md) | Conservative combination of contribution estimates | Useful profiling idea; its softmax assumptions do not directly hold here |
| [ZEDA](zeda_expert_self_distillation.md) | Zero experts and two-stage self-distillation | Stronger recovery evidence, but training and dynamic serving costs are material |
| [Matryoshka MoE](matryoshka_moe.md) | Layer-wise expert-count variation during training | Motivation for a separate SFT/recovery experiment, not a proven short-SFT recipe |
| [OMP-MoE](omp_moe.md) | Reconstruction-based stored pruning plus an adaptive extension | Borrow marginal error/stability measurements; distinguish stored and executed experts |
| [AdapMoE](adapmoe.md) | Sensitivity-based gating, prefetching and offload caching | Historical evidence; its offload speedups do not directly transfer |
| NAEE expert skipping | Dynamic inference skipping alongside permanent expert pruning | Useful precedent, but selecting fewer experts and deleting stored experts are different axes |
| ShortGPT layer removal | Ranks blocks by hidden-state change (Block Influence) and removes low-influence blocks | More intrusive here: CED boundary, shared KV/index producers, Engram and delayed mHC state must remain coherent |

Primary sources:
- [LExI, 2025](https://arxiv.org/html/2509.02753v1), sections 3–4. Its reported
  inference benefits motivate the design, not a prediction for our hardware.
- [Faster MoE LLM Inference for Extremely Large Models, ACL 2026](https://aclanthology.org/2026.findings-acl.2140/)
  and [paper](https://aclanthology.org/2026.findings-acl.2140.pdf), section 4.
- [ACE, 2026](https://arxiv.org/html/2609.05228v1), sections 3.1–3.2 and appendix.
- [Not All Experts are Equal, ACL 2024](https://aclanthology.org/2024.acl-long.334/).
- [ShortGPT, 2024](https://arxiv.org/abs/2403.03853).

The arXiv record for [Ada-K (2410.10456)](https://arxiv.org/abs/2410.10456)
is marked withdrawn. Do not cite its abstract's speedup as established evidence.

New source versions checked: normalization v1 (2026-09-04), ZEDA v2
(2026-06-08), Matryoshka v2 (2026-10-13), OMP-MoE v2 (2026-09-29). Per-method
notes distinguish reported results from our proposed adaptations. Benchmarks
called ARC-C/ARC-E in this literature are AI2 question answering, not ARC-AGI.

## Proposed E128 normalization experiment

This is our adaptation, not an implementation or measured result. Let
`s_i = sqrt(softplus(logit_i))`; let `T_q` select q experts using `s_i + bias_i`,
with the existing text/image selection bias. Evaluate only the experts in `T_k`:

`routed_output = 1.5 * sum(i in T_k, s_i * expert_i(x)) / sum(j in T_q, s_j)`.

Current reduced-k serving uses `q=k`. The first alternative is `q=6` on reduced
layers, keeping protected six-expert layers at their native `k=q=6` rule.
Computing reference gate scores does not execute the extra expert networks.
The shared expert is unchanged. This tests whether preserving the native
reference mass helps; it does not assume six is the optimal denominator.

A second low-dimensional control is a fitted routed-output scalar on reduced
layers. Estimate it using original-training and NVARC teacher activations, then
validate on separate training-parent families. Larger reference sets can follow only if the first screen is
useful. Do not copy a softmax formula literally: correction biases select experts
but must not enter their mixture weights, and sqrt-softplus scores do not sum to
one over the full pool.

Screen output norm, direction and reconstruction error alongside generated-token
and tool-block NLL. Then run generated text+tools diagnostics on held-out training
families, including both originals and NVARC, checking correctness, output length, tools, cap stops and missing
answers. NLL is a screening signal, not the final setting-selection criterion.
Keep image/text pools paired and family-disjoint. Freeze the resulting design
before running the 120-task Kaggle evaluation; always retain both references.

## Serving consequences and priorities

These judgments are based on our
[read-only Manh fork audit](../experiments/E076_active14_13/analysis/manh_g22df_audit/audit.json),
not on a newly validated port.

| Change | Expected serving work |
|---|---|
| Static per-layer k | Parameterize fixed-six router guards, loops, output shapes and warmups; initialize per-layer workspaces before CUDA graph capture. Existing expert GEMM should remain usable. |
| Static k plus different normalization | Add reference-mass/scalar arithmetic in the router; keep expert dispatch shape fixed. Additional GPU parity/performance checks are required. |
| Token-dependent k or zero experts | Ensure skipped token-expert pairs are actually omitted before expensive work. Zero weights alone may save no compute. Packing, dispatch and CUDA graphs need validation and possibly custom kernels; a new GEMM is not automatically necessary. |
| Entire transformer-layer removal | Higher structural risk from shared KV/index producers, Engram, encoder/decoder transition and delayed mHC state. Retain as a separate investigation. |

For future SFT transfer, preserve the ordinary SFT checkpoint as the teacher and
baseline. Re-profile its original-training and NVARC activation responses. If recovery is needed, first
try a fixed-schedule student so the serving path remains simple; a full ZEDA
conversion is a different architecture. A sampled-k training experiment must
hold every protected layer at six, including during training. Verify routed
expert updates and re-quantized export quality rather than assuming adapters
updated the intended tensors. Do not change the teammate's current SFT run.

The next architecture screen should examine supported k3/4/5/6 choices and
joint interactions rather than treating today's 4/6 ranking as optimal. Lower
counts need their own evidence. The current 12 protected layers remain protected
unless the user changes that constraint; expand the set when new sensitivity
evidence requires it.

## What approximate 12B means here

Using the same rounded 16B anchor and exact 35,389,440 parameters per routed
slot, 128 total slots give approximately 12.036B active parameters, mean k=3.2.
That is **46.7% fewer routed slots**, despite only a roughly 25% total-active
parameter reduction. If 12 layers remain at six, the other 28 must average two
to reach 128 slots. This explains why normalization and recovery may matter much
more at 12B than at the current mean k=4.6. The target is soft: accept a larger
model rather than compromise a sensitive layer. Future candidate eval order
remains text+tools first, then images+tools.

## Why the model matters

The [official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json)
has hidden size 5120, expert width 2304, 40 layers, six routed experts and one shared
expert active per layer. Source code in the pinned fork confirms normalized
sqrt-softplus gates, selection-only correction biases and separate image biases.
Shared and routed experts must not be counted interchangeably.

The existing generic router kernel has a compile-time TOP_K loop with up to eight
slots. E128 k=1..6 needs an allowlist extension and correct layer initialization,
not a newly authored router kernel. GPU tests pass in E074; uniform k=5 full-model Humming and CUDA-graph
checks also passed on RTX PRO 6000. A mixed k=4/k=6 full-model screen also passed. Distinguish a new JIT specialization from a
new kernel implementation.

The earlier E066/E062 per-layer expert-count work kept top-k=6, changed weight
shapes and incurred Humming packing overhead. A layer-specific top-k schedule
leaves the E128 weight shapes fixed. Both are static at initialization, despite
the earlier label "dynamic count".

Whole-layer removal has additional structural dependencies: KV source layers
2/8/14/20, index sources 2/8/14/20/24/28/32/36, candidate source 20, Engram at
1/14, the encoder/decoder transition at 20, and delayed mHC mixing/final collapse.
Merely shortening `num_hidden_layers` or deleting modules is not a valid port.

## Measurement that would answer the user's question

Keep teacher, checkpoint, harness, modality, inference backend and sampling fixed.
Select the architecture using training-family data only. Measure the joint
schedule because single-layer losses are not additive. Use real ARC solving on
the 120 Kaggle task versions after selection, with task-level paired uncertainty.
Report parameter deltas, tokens per attempt and actual wall time separately:
faster token generation can be cancelled by longer or less reliable reasoning.

For the first narrow experiment, reducing mean top-k from six to five removes
1.416B routed parameters. With only 4/5/6 allowed, preserving a layer at six means
assigning another layer four. Keeping only one on average is much more aggressive
and cannot preserve any layer at six unless other layers use zero routed experts.
