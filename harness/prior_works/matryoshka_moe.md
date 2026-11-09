# Matryoshka MoE: train for several active-expert budgets

Reviewed 2026-10-31. [Wang et al., v2, 2026-10-13](https://arxiv.org/html/2509.26520v2),
arXiv record marked EMNLP 2026.

Layer-wise sampling of expert counts during training makes one checkpoint usable
at several inference budgets. The main experiment uses a 20B, 96-expert model:
80B tokens of continued pretraining after a 1T-token base checkpoint. This is
substantial training, not evidence that a short SFT round alone produces the
same elasticity. Its ARC-C benchmark is AI2's science-question dataset, not
ARC-AGI grid puzzles.

**SFT implication for us:** once an ordinary SFT control exists, a separate
recovery run could sample only supported per-layer counts and distill from the
unpruned SFT teacher. Protected layers must always use six, including training.
Check routed-expert gradients and exported weight changes; sample-count training
is ineffective if the intended expert tensors never update. Recalibrate the SFT
checkpoint rather than assuming the old E128 schedule preserves its accuracy.
