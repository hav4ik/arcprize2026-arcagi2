# E060: self-hosted Qwen3.8-Flash-Next on SM89 (4x RTX 4090), serving + accuracy arms

Status: running (2026-10-27). Sub-agent run; box: Vast 4x RTX 4090 57983124 (`ssh -p 62174 root@159.112.204.189`),
user-approved for this work.

Why: the 3rd-place ARC-AGI-2 team (81 on the Kaggle LB, 4xL4, 12 h) hinted that it uses Qwen3.8-Flash-Next without
fine-tuning. E048 measured 59.8 pass@1 / 72.9 pass@2 through Alibaba's API (provider sampling, our DeepSeek-tuned
harness with budget notes, 2 attempts, text only). Questions: how well does open-weight, 4-bit Flash-Next do on
our tool harness when run properly, and how fast does it serve on L4-class GPUs (SM89)?

**Every throughput number here is from RTX 4090s.** They are the same generation as Kaggle's L4 (SM89), but have
~3.4x the memory bandwidth (1008 vs 300 GB/s) and 300 W vs 72 W, so decode throughput here overstates L4.
`scripts/bench_flashnext.py` is the microbenchmark to repeat on an L4 box for the conversion factor.

## Setup

### Checkpoint
`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` @ `c6af9fee7ad2cb8ef44b534a99b8c1b81fb0f6da` (129.0 GB, 88 files).
It holds Intel's AutoRound target tensors unchanged (W4A16 routed experts, INT4 sym g128, `quant_method: auto-round`),
everything else BF16, and the vision tower. The 51.2B-parameter n-gram/PLE table is RadixArk's FP8 E4M3 version
with one global scale (`ple_embedding_dtype: float8_e4m3fn`). There is also a compact INT4-g32 MTP draft in
`runtime/mtp-int4-g32`.

Tensor bytes by category, from the safetensors headers: routed experts 58.45 GiB, PLE table 47.68, linear attention
(GDN) 3.89, hyper-connections 1.19, embed_tokens 1.18, lm_head 1.18, self-attention (QSA) 1.11, vision 0.84,
shared experts 0.44, router 0.12, other PLE 0.06, QSA indexer 0.04. Total 116.18 GiB.

Other candidates (HF API sizes): wtdcode AWQ 180.8 GB and Intel AutoRound 181.2 GB, both with a 102.4 GB BF16 PLE
shard. Qwen FP8 is 185.6 GB and BF16 is 360 GB.

### Disk
The box has a 204 GB overlay with 38 GB free. To make room, the user approved deleting the E128 DeepSeek model copy
`/workspace/dsv41/model-e128t` (= `/root/dsv41/model-e128t`; 121 files, 131,090,911,045 B; list in
`logs/e128t_manifest.tsv`). Before deleting, these were checked: no open files or mappings, no GPU processes, and
no symlinks into the directory anywhere under /. To restore it (~14 min), run
`bash /root/dsv41/dl_e128_http.sh` with HF_TOKEN sent over SSH stdin. It re-downloads
`chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit@9bce98c5` (the E059 B16 setup chain,
`experiments/E059_restart_under_load/b16_4090/setup_chain.sh` step 5).

### Software
Upstream vLLM 0.32.0, the exact build preinstalled in the box image
(`/usr/local/lib/python3.12/dist-packages`, torch 2.13.0+cu130, FlashInfer 0.7.0.post1). It is run from a new venv,
`/root/e060/svenv`, that holds its own copy of the `vllm` package and adds the image's dist-packages through a
`.pth` file, so the image's environment is never modified. Its `qwen4_exp` model supports the FP8 PLE table, keeps
the PLE in pinned host memory with UVA lookups (`--engram-config '{"cpu_offload": true}'`), and loads the auto-round
INT4 experts through the Marlin WNA16 MoE backend. The wtdcode backport fork ships only as Docker images and is not
needed.

Harness: `evals/run_deepseek_tools.py` through `scripts/run_local_tools.py`. That wrapper runs the tool code in
`arc_harness.executor.LocalExecutor`, as the Kaggle solver does (IPython-style output, 4 GiB per session, no
Docker), and lives in `/root/e060/hvenv` with numpy 2.2.6, scipy 1.15.3, scikit-image 0.25.2 and the other
toolenv packages.

Server command (`scripts/serve_flashnext.sh`; settings by env; started with setsid, stopped by its process group):
```
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
/root/e060/svenv/bin/python -m vllm.entrypoints.cli.main serve /root/e060/model --served-model-name flashnext \
  --host 127.0.0.1 --port 18321 --tensor-parallel-size 4 --enable-expert-parallel \
  --engram-config '{"cpu_offload": true}' --max-model-len 196608 --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 --max-num-batched-tokens 2048 --enable-prefix-caching --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --enable-prompt-tokens-details \
  [--language-model-only | --limit-mm-per-prompt '{"image":40,"video":0}' \
   --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1048576}}']
```
Notes:
- `--enable-expert-parallel` keeps each expert whole: 128 of the 512 experts per GPU, on the Marlin WNA16 MoE backend.
- QSA and GDN run on Triton kernels on SM89; vision attention runs on FlashAttention 2.
- Custom all-reduce is disabled automatically (more than two PCIe GPUs), so all-reduce goes through NCCL.
- Prefix caching runs in mamba "align" mode, and the attention block size is raised to match the mamba page size: 784 tokens with BF16 KV, 1568 with FP8 KV.
- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` is required at util >= 0.95. QSA's paged-prefill indexer allocates its
  logits workspace (default 512 MiB) outside the profiled memory, and at util 0.96 that OOMed the engine at 16
  streams.

## Phase 1 results

### Memory and KV capacity, per GPU, 4x RTX 4090 (24.0 GiB each)
Measured with `--load-format dummy` (same shapes as the checkpoint) before the disk was free. Real-weight numbers
follow in the next section.

| setting | weights | CUDA graphs | KV memory | KV pool (tokens) |
|---|---|---|---|---|
| vision, BF16 KV, util 0.92, max-num-batched-tokens 8192 | 19.02 GiB | 0.82 GiB | -0.77 GiB | does not start |
| text-only, BF16 KV, util 0.95, mnbt 2048 | 18.24 | 0.82 | 2.16 | ~169k (196,608 needs 2.48 GiB) |
| text-only, FP8 KV, util 0.96, mnbt 2048 | 18.24 | 0.53 + 0.21 | 2.64 | 380,108 (OOM at 16 streams: QSA workspace) |
| text-only, FP8 KV, util 0.95, mnbt 2048 | 18.24 | 0.53 + 0.21 | 2.41 | **346,612** |
| vision, FP8 KV, util 0.95, mnbt 2048, max_pixels 1,048,576 | 18.46 | 0.53 + 0.21 | 2.17 | **310,203** |

- KV costs about 13.5 KB per token per GPU in BF16. The model has 12 full-attention (QSA) layers with 2 KV heads,
  and TP4 duplicates each head on 2 GPUs. QSA has no DCP support.
- The 36 GDN layers also take mamba state blocks from the same pool.
- Kaggle L4 projection: an L4 has 22.49 GiB (1.5 GiB less than a 4090). At Kaggle's util 0.938 the budget is
  ~21.1 GiB. The text-only weights plus graphs take ~19.0 GiB, which leaves ~1-1.5 GiB of KV (~150-200k FP8 tokens)
  before activations. By comparison, the E128 pool on Kaggle is 1.55M tokens.

### Decode and prefill, dummy weights (kernel speed only; 4090 overstates L4)
`bench_flashnext.py` on the FP8-KV text-only server at util 0.95, without MTP. Decode holds C streams of real ARC
prompts with ignore_eos and reports generation tok/s from the server counter over steady windows.

| streams | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| gen tok/s (total) | 103 | 516 | 771 | 674 (744 -> 663 as contexts grow) |
| per stream | 103 | 65 | 48 | 21 |

Prefill: 4.4k tok/s (14k and 55k-token prompts, no cache).

### Images (arm C)
With `--image-cell 0 --image-align qwen` (E058; small grids are upscaled k x k to reach 64 tiles) and max_pixels
1,048,576, the server charges exactly tiles + 2 tokens per image for every size from 1x1 to 30x30. That matches
Alibaba's counts.

Extra prompt tokens per attempt over the 167 eval pairs: mean 2.4k, median 2.2k, p90 3.9k, max 8.1k. Each turn
re-reads them from the prefix cache.

The vision tower costs 0.22 GiB of GPU memory per GPU, which is -10% of the KV pool on the 4090 and relatively more
on an L4.

### Earlier-turn thinking: a vLLM 0.32 request-field bug in our harness path
On the rendered prompt (`scripts/check_render.py`: POST /tokenize with messages, then /detokenize), vLLM 0.32
**ignores `reasoning_content`** on input assistant messages and reads only `reasoning`. With the harness's
`reasoning_content`, every earlier turn rendered as an empty `<think>\n\n</think>`, whatever `preserve_thinking`
was set to. With `reasoning`, the thinking is kept; Flash-Next's chat template keeps it by default
(`preserve_thinking` undefined or true).

`scripts/run_local_tools.py` therefore converts `reasoning_content` to `reasoning` per request (env
`E060_REASONING_FIELD`, default `reasoning`). All E060 arms keep earlier thinking. E048 also sent `reasoning`
(OpenRouter path); whether Alibaba used it is unknown.

The B32 fork (0.28.1rc1) keeps both fields (the coordinator checked it on hegn), so B32 is not affected.

The template's default reasoning effort is "xhigh"; it also accepts medium and low through
chat_template_kwargs.reasoning_effort. All E060 arms use the default.

### Expert selection counts (for a later prune)
`patches/e060_expert_counts.py` and `patches/model_runner_e060.diff` (applied only to the svenv copy by
`patches/apply_expert_counts.py`) add a per-step counter. With `E060_EXPERT_COUNTS=<dir>` set, TP rank 0 adds every
step's real rows from vLLM's RoutedExpertsCapturer (the routing-replay buffer: logical expert ids, graph-safe,
padding rows excluded) into a [48 x 512] counter. It saves cumulative snapshots `counts_<t>.npy` and an `index.jsonl`
every 300 s. Per-arm counts are the difference between snapshots.

Validation on dummy weights: the per-layer sums equal tokens x 10 exactly. Overhead: 513 vs 516 tok/s at 8 streams.

## Rebuild on a fresh box (or after a restart)
Box requirements: at least 4x 24 GB SM89 GPUs (4x4090 here; L4 for the real Kaggle numbers), at least 140 GB free
disk for the 129 GB checkpoint, and at least ~120 GB host RAM. The FP8 PLE pins 47.7 GiB, and the tool executors
use up to 4 GiB each. Image `vastai/vllm_v0.32.0-cuda-13.0` was used.

1. From the workstation: `bash experiments/E060_flashnext_selfhost/scripts/push_to_box.sh HOST PORT`. This copies
   evals/ (without results), the 120 eval tasks, notes/prompts, check_image_tokens.py, scripts/, patches/ and the
   arm configs into `/root/e060/repo`.
2. On the box: `bash /root/e060/repo/scripts/setup_box.sh`. It builds svenv (a copy of the image's vLLM 0.32.0, or
   PyPI `vllm==0.32.0` on other images; the second path is untested) and applies the expert-count hook. It also
   builds hvenv (openai 3.40.0, httpx, numpy 2.2.6, scipy 1.15.3, scikit-image 0.25.2, networkx 3.4.2, pandas
   2.3.3, sympy 1.14.0, matplotlib, pillow, tokenizers) and dry-runs the harness wrapper (120 tasks, 167 jobs).
3. Model: send HF_TOKEN over SSH stdin to `scripts/dl_model.sh` (see its header). It pins
   `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE@c6af9fee` and finishes with `DL_DONE` in `/root/e060/runs/dl_model.log`.
4. Server: `PORT=18321 UTIL=0.95 MNBT=2048 MAX_SEQS=32 KV_DTYPE=fp8 EXTRA="--language-model-only"
   E060_EXPERT_COUNTS=/root/e060/runs/expert_counts_<arm> setsid nohup bash repo/scripts/serve_flashnext.sh > runs/serve_<name>.log 2>&1 < /dev/null &`.
   - For arm C, replace EXTRA with
     `--limit-mm-per-prompt {"image":40,"video":0} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":1048576}}`.
   - For MTP, add `SPEC='{"method":"mtp","model":"/root/e060/model/runtime/mtp-int4-g32","num_speculative_tokens":3}'`
     (not tried yet).
   - Do not put "sglang" in any command line. Stop the server only by its own process group
     (`kill -TERM -- -<pgid>`).
5. Checks: `hvenv/bin/python repo/scripts/check_render.py` should show the earlier thinking kept when it is sent as
   `reasoning`. The image-token check is the inline script described under Images.
6. Runs: `bash repo/scripts/run_arm.sh ARM TAG WORKERS [harness args]`. ARM is A, B or C (`exp/cfg_ARM.txt` plus
   `exp/extra_ARM.json`). It starts `vllm_monitor.py` beside the harness, which logs KV usage, prefix hits,
   preemptions and gen tok/s every 15 s. Smoke: `run_arm.sh B smokeB 8 --tasks $(cat repo/exp/smoke_tasks.txt) --attempts 1`.
   Full arms: `run_arm.sh A armA 12 --attempts 1` (likewise B and C).
7. Analysis, after copying `runs/results/*` back:
   - `scripts/score_e060.py RUN.jsonl --vs E048.jsonl+E048_retry.jsonl`;
   - `scripts/reasoning_stats.py --tokenizer tokenizer.json --trace-root /root/e060/runs/results=<local dir> RUN.jsonl`.

   On E048 these two scripts reproduce E048's figures: pass@1 59.8% on 165 pairs, pass@2 72.5% ranked over all
   167 pairs, 35.0k output tokens per attempt, 9.5 turns, 89% of output in thinking, reasoning per turn 2.0k / 28.5k
   / 119k chars (median / p90 / max).

Arm configs (`cfg_*.txt` and `extra_*.json` in this folder):

| arm | harness | sampling (extra-body) |
|---|---|---|
| A (E048 settings) | lean2sub, answer tool, session final mode, lenient parse, 64k per turn, 80k total, budget notes 0.01 / warn 0.75 / soft 40k, train check | none, so the server default from generation_config.json applies: T 1.0, top-p 0.95, top-k 20 |
| B | same, but no budget notes or soft budget, 128k total, 64k per turn | T 0.7, top-p 0.95, top-k 20, preserve_thinking true |
| C | B + `--grid-format text+image --image-cell 0 --image-align qwen` | as B |

All arms send earlier thinking as `reasoning` (see above). The reasoning effort is the template default, xhigh.

## Log
- 2026-10-27 20:27 UTC: started on box 57983124. Only 38 GB of disk was free, and E128 deletion needed the
  user's approval. The `rm` itself was blocked by the Claude Code permission classifier and left to the user.
- 20:45-21:30: dummy-weight serving tests (above). Then the vision server, the image-token check and the
  expert-count hook.
- ~21:40: the instance was found **stopped** (intended_status stopped; not by this agent). No E060 process was
  running at the time. Waiting for the user's decision: restart 57983124 or use another box.
