# Review: Manh's vLLM fork github.com/nguyen599/vllm (read 2026-10-28, not run)

Claim (teammate): it reaches ~550 tok/s with DeepSeek-V4.1 E128 on 4xL4. B32's stack (our fork, yhfgyyf
vllm-deepseek-v4-sm89 0.28.1rc1 + 16-file overlay) averaged 389 tok/s in the 11 h local replica.

**What it is.** Upstream vllm-project/vllm main (last merge 5d35ce3934, 2026-10-27; so newer than v0.31 and including
the OffloadingConnector deadlock fix #58947) plus the teammate's commits. Not a GitHub fork. The DeepSeek part is one
commit, `cded323df` (2026-10-27 18:48 UTC, +4,708 / -123 in 37 files), "[DeepSeek V4.1] SM89 Triton sparse MLA with
DCP, E128 GPTQ experts":
- a Triton sparse-MLA attention backend for SM89 (`models/deepseek_v41/nvidia/triton_sparse.py`,
  `nvidia/ops/triton_sparse_mla.py`), selected automatically on capability 8.9;
- **decode context parallelism (DCP)**: each rank keeps an interleaved shard of the compressed KV caches (the SWA window
  stays replicated, `sparse_swa.py`: `dcp_sharded=False`), attends its shard plus a disjoint slice of the window, and
  merges partial outputs by log-sum-exp; queries and partials cross ranks as **fp8 e4m3 rows with an fp32 scale**; the
  query all-gather overlaps the indexer on a side stream (`VLLM_DCP_ASYNC_QUERY_GATHER`, default on); optional
  pinned-host gather for PCIe boxes (`VLLM_DCP_HOST_GATHER`);
- a DCP indexer (per-rank candidate blocks, Triton top-k merge, Triton MQA logits where DeepGEMM is missing);
- a loader for the E128 2-bit GPTQ routed experts (`expert_gptq.py`, Humming MoE method);
- UVA offload of `embed_tokens` to host memory, Engram row maps for pruned checkpoints, a batched Humming MXFP8 linear,
  warmed-up DSv4 top-k router variants. Unit tests for each kernel.
- Earlier L4 work on the same branch: q8 host all-reduce (llama.cpp-style int8 + fp16 scale per 16), small-M Marlin
  padding, KV-offload CPU LRU refresh fix, watermark-gate fix for requests holding KV, hybrid prefix caching for
  multi-turn tool calls, Qwen/Flash-Next experiments (branches `checkpoint/fp4-local-*`: HiSparse, DFlash2, host reducer).

**Why 550 is plausible.** We are KV-bound. Our fork keeps the compressed KV whole on every TP rank; DCP over the 4
ranks stores each compressed block once, so the same memory holds several times more context (minus the replicated
SWA window), i.e. more concurrent streams, fewer preemptions, less host offload traffic. Moving `embed_tokens` to host
memory frees more KV. Upstream main also brings the newer runner/scheduler.

**What the commit itself says is not done:** "Work in progress"; "The fp8 DCP transport is checked against a full
launch on the Triton CPU interpreter; GPU validation is still pending."

**Before it can carry a submission** (kaggle/SUBMISSION_TESTING_PROTOCOL.md):
1. Numerics: the fp8 query/partial transport and the new attention kernel change the attention outputs. Gate on a
   kernel-level comparison against our fork plus an accuracy A/B on the 120 eval tasks (E128, same harness), not on
   throughput alone (the stack is not run-to-run deterministic; AGENTS.md 2026-10-26 findings).
2. Weights: `expert_gptq.py` must load our exact E128 checkpoint (chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit)
   bit-exactly; check against our fork's expert outputs.
3. Feature parity with B32: KV offload connector (upstream has it, with the deadlock fix), prefix-cache retention
   interval 512, the #156 indexer skip or its equivalent, `/tokenize` + token-id `/v1/completions` for `--token-exact`,
   deepseek_v41 tokenizer mode and parsers, `--language-model-only`, Engram CPU offload, FULL CUDA graphs <= 64.
4. Packaging: an offline SM89 / py3.12 / CUDA 13 wheel set for the Kaggle kit.
5. A 6-12 h 4xL4 replica on the 240-task workload at Kaggle's KV settings, then a Kaggle smoke.

**Questions for Manh:** under what conditions was 550 measured (GPU type and power, KV pool / util, real multi-turn tool
traffic or synthetic decode, context lengths, concurrency, prefix caching, DCP on/off, fp8 transport on)? Has the DCP
path been validated on GPU since the commit, and has accuracy been compared with our fork? Does he have Flash-Next
numbers on 4xL4 from the `checkpoint/fp4-local-*` work (relevant to E060)?
