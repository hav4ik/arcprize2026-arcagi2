# E026: Trace strategy analysis. Does the task summary simplify DeepSeek's thinking, and how does Qwen differ? ($0)

This is a read-only analysis of existing eval traces; no API calls. Reports:
- `deepseek_strategy.md`: DeepSeek Flash with no tools (E001d), with tools and no summary (S80 = sub80s40, S96), and with the summary (V1, V3). It combines regex features over 1,527 attempts (`ds_features.jsonl`, `scripts/ds_*.py`) with a structured reading of 31 traces (`ds_annotations/`).
- `qwen_strategy.md`: Qwen3.8 (E012, K3, x96p, x96pc8) against DeepSeek (V3, sub80s40) on the same pairs. Regex statistics over all attempts, plus a close reading of 37 pairs.

## Answers
- **Without the summary, DeepSeek always computes the same statistics itself.**
  - With no tools it does so by hand, about 33% of tokens, and hand slips cause most failures.
  - With tools it does so in cheap Python: about 1.6k tokens for shapes, bboxes, components and diffs.
- **The summary simplifies the thinking only a little.** Tokens fall 9% [−6.1k, −1.1k]. Copied grid rows go down, but cell-coordinate reasoning in prose (about 9.4k tokens per attempt, the real cost) is unchanged. The saved tokens move into hypothesis prose.
  - Its framing misled the model in 6 of 10 read traces (background vs separator colour, 8-connected multi-colour blobs, object-level views suggesting wrong rules). It helped in 5 of 10. No summary number was wrong.
- **Qwen vs DeepSeek (matched harness):** train-pass rates are equal (77% vs 78%), but P(correct | train_pass) is 72% vs 84%.
  - Qwen reaches code slowly: first turn 8–26k vs 4.3k tokens; first `solve` at about 38k vs about 20k. This is driven by the xhigh tier.
  - Qwen trusts hand counts over code output, and fits training pairs with coincidental rules where DeepSeek uses relational ones.
- **For SFT:**
  - Imitate fast-to-code and code-as-truth.
  - Keep only rollouts whose answer comes from a train-passing program.
  - Down-weight long prose turns.
  - Train with the summary only if Qwen also gets it at inference.
