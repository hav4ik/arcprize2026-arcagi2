# E062 port: what to try on top of Manh's fork, in priority order

For the 4090 / L4 agents, **after** the baseline (Manh's fork as is, B32 harness and solver).
- Background and line references: `PORT_ANALYSIS.md`.
- Patches are against nguyen599/vllm b9ad670a67 unless noted. Apply with `patch -p1` from the source root, or into
  site-packages for the Python-only files.
- They are third-party-adjacent code written without a GPU: dry-run first.

## Launch-flag translation, B32 -> Manh's fork (not a patch)

**Drop:**
- env `DSV41_INDEXER_Q_PREDECODE`, `DSV41_EXPERT_HUMMING`, `DSV41_EXPERT_HUMMING_W2`, `E043_GPTQ_DIR`: GPTQ experts are
  auto-detected from `<model>/gptq/`;
- `--prefix-cache-retention-interval` (no-op under bounded replay);
- `swa_store_policy` and `shared_region` in the offload JSON: they are ignored.

**Keep:**
- `--tensor-parallel-size 4 --kv-cache-dtype fp8 --enable-prefix-caching --language-model-only`;
- `--tokenizer-mode / --reasoning-parser / --tool-call-parser deepseek_v41`, `--enable-auto-tool-choice`,
  `--enable-prompt-tokens-details`, `--linear-backend humming`;
- `--engram-config '{"cpu_offload":true}'`, `--max-num-batched-tokens 2048`, `--max-model-len 163840`;
- `VLLM_USE_V2_MODEL_RUNNER=1` (bounded replay needs it).

**Add or change:**
- `--decode-context-parallel-size 4` and `--cp-kv-cache-interleave-size <N>`. N must be a multiple of the candidate
  block size and >= the largest compress ratio, and the block size divisible by N. Read Manh's tests and asserts, and
  start from block 128 / interleave 128.
- `--max-num-seqs 80-96` with `--compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":96}'`.
- Offload:

  ```
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":
  {"cpu_bytes_to_use":34359738368,"offload_prompt_only":false,"eviction_policy":"lru"}}'
  ```

  `offload_prompt_only:false` matters: upstream defaults to true. The pool is a /dev/shm mmap, so check
  `df -h /dev/shm`.

**Never** set `NCCL_P2P_LEVEL=SYS` on L4/Kaggle.

## Items

| prio | item | how | measure | go if |
|---|---|---|---|---|
| 1 | **Humming MoE pads 64,64** (est. ~1.3 GiB/GPU of zero padding on E128's 1152/576 shards) | env `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64` | "Model loading took" GiB/GPU and "GPU KV cache size" vs default; greedy ids on 5 fixed prompts (incl. a 30-60K tool transcript); decode step ms at B = 48 / 80 (E054 replay method) | weights -0.8 to -1.4 GiB/GPU; ids identical, or first divergence >= 200 tokens with abs(dlogprob) < 1e-3; decode >= -2% |
| 2 | **Prior thinking on re-render** | `02b_harness_send_reasoning.diff` on the harness copy you run. The server-side alternative is `02_reasoning_content_input.diff` | `/tokenize` + `/detokenize` of a 2-turn transcript with `reasoning_content`: the thinking text must appear. Compare ids with B32's server on the same messages | thinking present; ids equal to B32's render, or a documented diff |
| 3 | Offload translation (above) | config | host-restored tok/s, host hit share, anon+shmem peak (Kaggle cgroup 173 GiB), preemptions | no deadlock or stall in >= 2 h under KV pressure; memory within budget |
| 4 | Higher util with a driver reserve | `VLLM_GPU_DRIVER_RESERVE_MB=512` (start), util 0.955 then 0.96 | KV tokens; 2 h without OOM under mixed prefill sizes | no OOM; KV +5% or more |
| 5 | `embed_tokens` UVA | `--cpu-offload-gb 1 --cpu-offload-params embed_tokens` | KV tokens (+~0.33 GiB/GPU), decode step | decode >= -1% |
| 6 | DCP prefix granularity | `--block-size 64` (and interleave to match) vs 128 | prefix-hit rate and recomputed prompt tok/s on the tool replay | hits up, gen tok/s not down |
| 7 | q8 host all-reduce (lossy) | `VLLM_HOST_ALLREDUCE=...`: read `M:vllm/envs.py` for the mode values | decode step; then the 120-task accuracy A/B | throughput +3% or more and pass@1 within the noise band |
| 8 | **Engram row chunks (Kaggle Model layout)** | `01_engram_row_chunks.diff` | load the Kaggle Model copy, or split the HF Engram tensors into `rows_<start>` chunks of <= 5 GB in a scratch dir; compare the loaded table bytes (hash per rank) with the HF-layout load | identical tables; startup OK. **Required before any Kaggle build on this fork** |
| R1 | **Reverse port to B32's stack:** Humming MoE pads on our fork | `R1_b32_overlay_humming_moe_pad.diff` on the overlay's `humming_utils.py`, env `DSV41_HUMMING_MOE_PAD=64,64` | as in item 1, on the B32 venv | as in item 1. If humming 0.1.12 accepts it, this is the cheapest KV gain for a B32.x: est. 1.55M -> ~2.3M Kaggle KV tokens |

**Every item changes numerics or capacity.** A throughput win alone is not a GO: finish with the 120-task accuracy A/B
(E052 pattern) and an L4 soak (`kaggle/SUBMISSION_TESTING_PROTOCOL.md`).
