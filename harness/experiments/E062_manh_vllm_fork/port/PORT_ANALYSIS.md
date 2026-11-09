# E062 port analysis: B32's improvements on top of nguyen599/vllm

CPU-only code reading, 2026-10-28 ~08:00-09:30 UTC. Nothing was run.

**Sources.**
- `M:` = nguyen599/vllm main at **b9ad670a6713** (2026-10-28 06:52 UTC). That is upstream main 08a236fa43 + 52 commits,
  shallow clone in scratchpad `e062_port/manh_vllm`. Manh's diff against upstream comes from `gh api compare`
  (scratchpad `e062_port/compare.json`).
- `O:` = B32's 16-file overlay, `experiments/E056_candidate_b/kit/vllm_overlay_rw.tgz` (1a3a6679).
- `F:` = the yhfgyyf vision11 fork that the overlay sits on.

**Context.** E053 (`experiments/E053_vllm_upgrade/README.md`) found that upstream could not run V4.1 on SM89, for lack
of three pieces: sparse-MLA attention, an indexer without DeepGEMM, and an o_proj without DeepGEMM. Manh's fork
supplies all three:
- `M:vllm/models/deepseek_v41/nvidia/triton_sparse.py`, selected on capability 8.9 (`nvidia/model.py:268`);
- `M:.../kernels/attention/dsa/triton_mqa_logits.py` + `dcp_indexer_topk.py`;
- Humming MXFP8 bmm for `wo_a` (`M:vllm/models/deepseek_v4/nvidia/ops/o_proj.py`).

So the question is no longer "port our fork's SM89 path onto upstream". It is which of B32's ~25 pieces his tree already
has. **Most of B32's overlay is already there or obsolete.** Upstream absorbed it, and Manh ported our E128 loading
formats.

## Verdict table

Value is for 4xL4 / Kaggle. Effort counts code plus a test.

| # | B32 item | In Manh's fork? | Port? | Effort | Value | Notes |
|---|---|---|---|---|---|---|
| 1 | TP4 | yes | config | 0 | - | Adds DCP: `--decode-context-parallel-size 4` |
| 2 | Humming dense MXFP8 (`--linear-backend humming`). Ours: fork whitelist shim + `O:kernels/linear/__init__.py` (+2) | **yes**, native upstream `--linear-backend humming`. Manh also routes o_proj `wo_a` through `HummingMxfp8BmmLinearKernel` (DeepGEMM is SM90+) | drop our files, keep the flag | 0 | same as B32, plus o_proj | Our dense-Humming gain (E042 P12/R7: decode +4-8%, prefill +11-13%) carries over |
| 3 | 2-bit GPTQ experts through Humming. Ours: `O:e043_humming.py`, `O:quant_config.py` hook, `O:quantization/humming.py` + `humming_utils.py` shims for humming 0.1.12; env `DSV41_EXPERT_HUMMING*`, `E043_GPTQ_DIR` | **yes**, `M:vllm/models/deepseek_v41/expert_gptq.py`. It reads exactly our `gptq/layer_LL.safetensors` (`e{E}.w{1,2,3}.{q,s,z}`). It infers the per-sublayer schema from the shapes (w13 uint2 g128, w2 uint2 g64, zero points). It is wired in `quant_config.py:151` and `nvidia/model.py:1857`, and activates whenever the model dir has `gptq/` | none: drop all DSV41_/E043_ env | 0 | - | Runs on humming-kernels **0.1.18** (ours: 0.1.12). Speed on E128 shapes unmeasured; see item 4 |
| 4 | **Humming MoE padding.** Ours: hardcoded `pad_n_to_multiple=256, pad_k_to_multiple=128` (`O:humming_utils.py:907-908`) | **knob present.** `VLLM_HUMMING_MOE_PAD_MULTIPLE` defaults to "256,128" (`M:envs.py:1997`, `M:utils/humming/moe.py:406`). Manh's own comment: '"64,64" for a 576 intermediate shard' | **env `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64`.** Reverse port to B32 = `R1_b32_overlay_humming_moe_pad.diff` | 0 (env) / 0.5 d (R1 + test) | **HIGH** (est.) | See below. E128's TP4 expert shard is N 1152 (w13) and K 576 (w2), padded to 1280 and 640: ~11% zero weights. By shapes that is **~1.3 GiB per GPU** (12.8 vs 11.5 GiB of experts), against 2.29 GiB of KV per GPU in B32 (+55% KV if freed). E042 saw the same: Humming experts cost +1.2 GiB vs Marlin on E64 (`E042 README` P12b hmoe). It also streams ~10% fewer expert bytes per step. Needs a parity check and an A/B |
| 5 | MXFP4 -> BF16 expert decode (`e043_humming.dequant_expert_weights`) | not needed | N/A | - | - | E128 ships GPTQ experts only |
| 6 | Engram compact table + `row_map` (E043C, `O:common/engram.py` +47/-6) | **yes**, `text_config.engram_compact_kept` + `row_map` with the same semantics (`M:common/engram.py:228, 630, 1086-1118`) | none | 0 | - | Manh ported our scheme |
| 7 | **Engram row-chunk loader** (`layers.{1,14}.engram.embed.weight.rows_<start>`; `O:nvidia/model.py` `_e043_engram_row_chunks`) | **no** | **`01_engram_row_chunks.diff`** | 0.5 d | **required for Kaggle** | The HF repo has whole Engram tensors and loads as is (tonight's 4090/L4 runs are fine). The **Kaggle Model** copy (`chankhavu/dsv41-flash-1017-arc-tool-e128-2bit`) stores row chunks; the fork would not load it |
| 8 | Router fast path for (128, 6) (`O:nvidia/router.py`) | **yes**, `dsv4_topk` accepts 128 experts, top-k 6 (`M:fused_moe/router/dsv4_topk.py:31`) | none | 0 | - | |
| 9 | `synth_init.py`, `dummy_loader.py`, route stats | N/A | drop | - | - | Test tooling only |
| 10 | Indexer: fork PR #154 (Q predecode) + #156 (grid-stride context skip) in `O:dsv41_indexer.py` | **equivalent built in.** Different indexer: fp8 K with Triton MQA logits. The paged kernel's grid strides over each row's own length and leaves columns past it unwritten: the #156 idea (`M:triton_mqa_logits.py:183-191`). Also `VLLM_TRITON_SPARSE_INDEXER_LOGITS` (score only candidate blocks) and a DCP indexer | N/A | 0 | - | #154 is MXFP4-specific. Upstream's fp8 indexer K is larger per token than the fork's MXFP4 |
| 11 | Attention kernel (fork's custom FlashInfer `mla.deepseek_v41`, mixed FP8/FP4 KV) | **replaced** by Triton sparse MLA on `fp8_ds_mla` records, with DCP4 sharding of the compressed caches | N/A | - | core of the fork | Commit cded323: "GPU validation is still pending" |
| 12 | KV offload: 4 E051 files (ring skip, dense MLA host store, SWA checkpoint policy, private memory, LRU) | **superseded** by the upstream connector: #57145 offloads only prefix-cacheable groups; #57652 + a shared mmap region give one host copy for TP-replicated MLA; #58947 deadlock fix; DCP-aware canonical mapping. Plus Manh's b1d9dfe391 (refresh the CPU LRU on GPU hits). SWA is replayed (#56227), not stored | config translation | 0 | **MED-HIGH** | B32 stored 4 private copies of replicated KV (32 GiB = 8 GiB unique). Upstream stores distinct DCP shards, or one shared copy without DCP, so the same 32 GiB holds up to 4x more unique KV. Keep `offload_prompt_only: false`: **upstream default is true** (`M:v1/kv_offload/base.py:754`), which would drop decoded tokens, i.e. most of a tool loop's next prefix. `swa_store_policy` / `shared_region` are ignored (`extra_config.get`). On CUDA the pool is always a /dev/shm mmap; Kaggle has 85.5 GiB free there |
| 13 | `--prefix-cache-retention-interval 512` | flag exists, but it applies only to SWA/Mamba groups (`M:config/cache.py:275-281`). With bounded replay (default on) SWA is not prefix-cached | drop | 0 | - | No-op for V4.1 here |
| 14 | #56227 bounded replay, #58132/#59894 decoder replay | **yes**, upstream defaults (`swa_bounded_replay=True`) | none | 0 | + | Removes B32's "640-token hit" full re-prefills. Each hit recomputes its last 128 tokens |
| 15 | #58678 Engram `wkv` split (E054: NO-GO on our fork) | **yes**, upstream | none | 0 | +~10% KV | E054: not bit-identical under Humming; -2.4% decode at B=48 on our fork. Here it is simply the baseline |
| 16 | FULL CUDA graphs, capture <= 64, `--max-num-seqs 64` | config | raise `max_cudagraph_capture_size` to >= max-num-seqs (80-96 for Manh's batch 80) | 0 | needed for B >= 64 | DCP's async query gather works under FULL graphs; it is skipped only in piecewise graphs (`M:triton_sparse.py:197`) |
| 17 | fp8 KV (`--kv-cache-dtype fp8`) | yes; Triton backend accepts fp8/fp8_e4m3/fp8_ds_mla | none | 0 | - | |
| 18 | Engram CPU offload | yes, upstream `engram_config.cpu_offload` | keep flag | 0 | - | |
| 19 | Server API the solver needs: `--language-model-only`; deepseek_v41 tokenizer / reasoning / tool parsers; `cache_salt`; `/tokenize` with messages + tools + `chat_template_kwargs`; token-id `/v1/completions` with `return_token_ids`; `/v1/models`; metrics (`generation_tokens`, `prefix_cache_hits/queries`, `kv_cache_usage_perc`, `num_requests_running/waiting`, `num_preemptions`, `external_prefix_cache_hits`) | **all present** (`M:engine/arg_utils.py`, `tokenizers/registry.py:53`, `v1/metrics/loggers.py`) | none | 0 | - | Integer `reasoning_effort` 50 is accepted (`M:tokenizers/deepseek_v41.py:92-105`). Check `/tokenize` parity against B32's server byte for byte |
| 20 | **Previous-turn thinking on re-render.** The harness sends `reasoning_content` | **no.** Upstream reads only `reasoning` on input (`M:renderers/chat_utils/chat_message.py:619`); `reasoning_content` is not in the request TypedDict | **`02b_harness_send_reasoning.diff`** (harness, preferred) or `02_reasoning_content_input.diff` (server) | 0.1 d | low, correctness | Normal token-exact turns are unaffected (prefix check compares two server renders). Only resync / re-render turns (~0.1% of turns in B32) would lose all earlier thinking |
| 21 | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | yes | none | 0 | - | |
| 22 | Solver: KV gate 0.85 + queue hold, multi-test 0.5, token-exact, auto-repair, S20-S22, train check | unchanged code works | retune `ARC_WORKERS` (56 -> 80?) and the gate once the pool size is known | 0 | - | The KV gate reads `kv_cache_usage_perc` of the (DCP) pool. Manh's 10d0079a76 (no watermark gate for requests that hold KV) changes behaviour under pressure |
| 23 | GPU util 0.945 (0.97 OOMed within ~1 h on our fork) | Manh's b9ad670a67: `VLLM_GPU_DRIVER_RESERVE_MB` + pre-warming each Humming kernel range at startup targets exactly "first prefill of a new size OOMs after the KV cache took the device" | try util 0.955-0.96 with a reserve | env | MED (+0.2-0.4 GiB KV/GPU) | Our 0.97 OOM cause is unverified; a reverse port onto our fork is medium (worker changes) |
| 24 | E057 8-bit all-reduce (planned for B32) | Manh: q8 host all-reduce (`VLLM_HOST_ALLREDUCE*`, int8 + fp16 scale per 16, llama.cpp-style) + L2 weight prefetch during all-reduce (c5d9fd8cc6) | opt-in A/B | env | decode +3-7% (E057 est.) | **Lossy:** accuracy A/B before use. Has csrc, so it cannot be overlaid onto our Kaggle kit |
| 25 | `embed_tokens` on GPU | Manh: UVA offload of `embed_tokens` | `--cpu-offload-gb 1 --cpu-offload-params embed_tokens` (explicit only, `M:offloader/uva.py:81-98`) | 0 | +~0.33 GiB KV/GPU | 129,280 x 5,120 bf16 / TP4 |
| 26 | Kaggle packaging (offline wheel set: torch 2.13, FlashInfer 0.6.18 fork, humming 0.1.12) | fork needs **torch 2.14.0, flashinfer 0.7.2.post1, humming-kernels 0.1.18, tilelang 0.1.12, cutlass-dsl 4.8.0, tokenspeed-mla 2.0.3** and its own SM89 wheel (csrc changes: host all-reduce, Marlin, cutlass sm89 fp8 tiles) | new kit + dataset version | 1-2 d | required | Rebuild the `chankhavu/dsv41-sm89-fork-py312-cu130` equivalent; re-check the CUDA 13 shim and Triton warmup startup time (B32 server up in 220-300 s) |

## Other things to know before tuning

- **DCP prefix-cache granularity.**
  - With several KV groups, the scheduler block is the LCM of the groups' effective blocks: `block_size x dcp` for the
    sharded ones (`M:v1/core/kv_cache_utils.py:828-875`). At `--block-size 128` and DCP4 that is 512 tokens.
  - Prefix hits then round down to 512 tokens instead of 128. That is ~200 more re-prefilled tokens per turn on
    average, on top of the 128-token SWA replay.
  - Try `--block-size 64` (or 32) if the kernels allow it, and compare the prefix-hit rate.
- **DCP interleave.** It must cover whole candidate blocks: `cp_kv_cache_interleave_size % candidate_block_size == 0`
  (`M:layers/sparse_attn_indexer.py:510`; `dcp_indexer_topk.py:275`). It must also be >= compress ratio, because
  triton_sparse uses `interleave // compress_ratio`, and block size must be divisible by it
  (`M:config/parallel.py:452`).
- **Numerics that change against B32.** Each needs the accuracy A/B, not only throughput:
  - attention kernel;
  - fp8 query/partial DCP transport;
  - fp8 indexer K;
  - humming 0.1.18;
  - the #58678 split;
  - bounded replay;
  - optional q8 all-reduce.

## Manh's claims: ~700 tok/s at batch 80, ~4M KV tokens

**4M KV tokens: plausible as a GPU-only DCP4 pool.**
- Under DCP the reported "GPU KV cache size" counts sharded groups at `block_size x dcp` (`M:kv_cache_utils.py:741-743`).
  Every compressed token is stored once across the 4 ranks instead of on each of them; only the SWA window (128 tokens
  per layer per request) stays replicated.
- B32's 2.29 GiB/GPU pool holds 1.557M tokens (1.54 KB/token/GPU). x4 minus the SWA overhead gives ~4-6M. The fp8
  indexer K is larger than the fork's MXFP4, which pulls this down.
- Freed memory pushes it back up:
  - #58678: +0.22 GiB;
  - embed UVA: +0.33 GiB;
  - Humming pads 64,64: ~+1.3 GiB, if he set them.
- **So 4M does not need host RAM.** With host offload on top, the effective budget is larger still. Note the counting
  for comparison: B32's own 4090 twin (1.59M GPU + 16 GiB host) held in-flight context of up to 3.33M without
  reaching its 4.5-4.8M cliff (E059 B16).

**~700 tok/s at batch 80: depends entirely on the hardware, which the claim doesn't state.**
- **On 4x RTX 4090 it would be parity with B32.** B32's own stack already makes ~600-870 gen tok/s at 56 workers on
  4x4090 at 300 W. That was real tool traffic, the Kaggle-sized 1.59M pool and 16 GiB offload (E059 B16 twin: 627
  before the kill, 741 after). It would not be an improvement.
- **On 4x L4 at 72 W (Kaggle) it would be a large jump over B32's 389 tok/s.** Our L4 decode fit (E053 section 4) is
  step = 58 ms + 0.9 ms x B at 32K. At B = 80 that is 130 ms, i.e. ~615 decode tok/s and ~480 end to end (x0.78).
- **700 end to end needs a ~89 ms step.** That means roughly halving the per-stream slope. DCP does cut each rank's
  compressed-KV reads 4x.
- **But on Kaggle's PCIe Gen3 (NCCL over SHM ~4.9 GB/s, P2P 12x slower) DCP's own traffic is large.**
  - Per compressed layer and token, each rank receives ~25 KB of fp8 queries (3 peers x 16 heads x 512) and sends
    ~25 KB of partials.
  - At B = 80 x ~38 layers that is ~150 MB per rank per step: **~30 ms**, only the query half overlapped.
- **Realistic range on Kaggle: ~430-550 tok/s.** Most of the gain would come from capacity: no preemption bursts
  (B32: 487), fewer host restores and more streams. More than that needs the L4 measurement.

**The claim doesn't say:**
- hardware, power cap and PCIe;
- workload (synthetic decode vs multi-turn tool traffic with prefill and prefix hits), context lengths;
- whether prefix caching, offload and FULL graphs were on;
- whether q8 all-reduce, fp8 DCP transport or Humming pads were set;
- steady state or peak;
- any accuracy validation. The DCP commit says GPU validation is pending.

## Patches written (`port/`, against M = b9ad670a67 unless noted)

| file | what | applies |
|---|---|---|
| `01_engram_row_chunks.diff` | `nvidia/model.py`: load `<layer>.engram.embed.{weight,scale}.rows_<start>` chunks into the rank's (compact) table, check exact coverage, report the params as loaded. Maps only the module prefix, so both the text-only and the VL-wrapper load paths work | `patch -p1` dry-run OK |
| `02_reasoning_content_input.diff` | server: `parse_chat_message_content` falls back to `reasoning_content` for assistant turns; declares `reasoning_content` on `CustomChatCompletionMessageParam` so request validation keeps it | dry-run OK; unverified on GPU (pydantic union choice) |
| `02b_harness_send_reasoning.diff` | harness (`evals/run_deepseek_tools.py`; same hunk applies to `kaggle/dsv41_tools/pinned_candb/run_deepseek_tools.py`): the token-exact `render()` adds `reasoning` next to `reasoning_content` | dry-run OK on both copies |
| `R1_b32_overlay_humming_moe_pad.diff` | **reverse port, B32 overlay** `humming_utils.py`: env `DSV41_HUMMING_MOE_PAD="N,K"` (default 256,128 unchanged) | against `O:` 1a3a6679; humming 0.1.12 support for 64,64 unknown |

Priority and measurement plan: `READY.md`.
