# Active-parameter pruning for E128 (review, 2026-10-31)

The target is the causal image E128 checkpoint in E066, already physically expert
pruned and GPTQ-quantized to two bits. Transfer from other MoE models is a hypothesis;
none of the papers below establishes preserved ARC accuracy for this checkpoint.

## Methods and applicability

| Method | Evidence / mechanism | E128 application |
|---|---|---|
| Uniform reduced top-k | Simple baseline that actually executes fewer experts | Test before searching a layer schedule; preserve native router semantics |
| LExI | Per-layer top-k perturbation profiling, followed by proxy-guided evolutionary search; estimates perturbations using synthetic Gaussian inputs | Most direct static architecture match. Use actual E065 training activations / held-out NLL as additional evidence rather than trusting synthetic scores alone |
| Static schedules and ASET | Studies uniform/layer-specific skipping on fine-grained MoEs; ASET adjusts activation from confidence and entropy | Supports testing different layer budgets. Native sqrt-softplus routing and image selection biases require a careful port |
| ACE | Combines offline spectral and router-conditioned expert contribution proxies with runtime gates; only skips slots deemed weak by both, retains the strongest | Potential second stage. Its router-centering rationale assumes softmax translation invariance, which does not directly apply to DeepSeek's sqrt-softplus scores |
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
