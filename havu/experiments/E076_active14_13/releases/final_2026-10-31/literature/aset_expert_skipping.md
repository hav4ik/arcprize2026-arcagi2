# ASET and static skipping in fine-grained MoEs

Reviewed 2026-10-31. [Yang et al., ACL Findings 2026](https://aclanthology.org/2026.findings-acl.2140/),
[full paper](https://aclanthology.org/2026.findings-acl.2140.pdf).

Static layer schedules vary substantially between models; a useful DeepSeek-V2
allocation can fail on DeepSeek-V3. ASET selects token-dependent counts using
confidence mass and entropy, bounded by native activation. The dynamic study
uses OLMoE. The paper reports static throughput gains of 10–78%, including at
least 10% on DeepSeek-V3 without measured loss. Efficiency experiments use
controlled input/output lengths; these are not long tool-use measurements.

**Our interpretation:** supports preserving sensitive layers and investigating
token-dependent budgets later. Entropy is a proxy, not proof of reasoning
difficulty. Our sqrt-softplus scores and selection biases require a separate
definition of confidence; copying a softmax threshold is unjustified. Actual
dispatch must omit work, and speed needs measurement with our long transcripts.
