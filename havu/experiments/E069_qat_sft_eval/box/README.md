# E068 box: QAT + SFT E128 on Vast 58398827 (2x RTX PRO 6000 Blackwell Server, sm120)

Box notes for the QAT + SFT E128 eval (`chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch`), served with B32's compact Engram
table and checked with B32's frozen tool harness. Box set up 2026-10-30 12:10-12:52 UTC. No API spend.

## Box
- Vast 58398827: 2x RTX PRO 6000 Blackwell Server Edition (96 GB), driver 580.173.02, image vastai/vllm:v0.32.0-cuda-13.0.
- `ssh -p 61847 root@38.20.137.73`.
- **cgroup limits: 362 GiB RAM (`memory.max` 389,323,685,888 B), ~30.7 CPUs.** `free` and `nproc` show the host's
  755 GB / 128 CPUs.
- Disk: 396 GB overlay, ~5.4 GB/s direct writes, so models go on disk rather than `/dev/shm` (94 GB).
- The image's template vLLM (Qwen3.5-9B) was stopped by `scripts/free_gpus.sh`: `supervisorctl stop vllm`, then its two
  orphaned workers killed by PID.

## Layout on the box (`/root/e068`)
| path | what |
|---|---|
| `venv/` | E043 fork venv from `E043/locks/venv_vllm.freeze` (vllm `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`, flashinfer vision2, torch 2.13.0+cu130, humming-kernels 0.1.12) + E043 patches `apply_e043_patches.py`, `apply_compact_engram.py`, `apply_e043_humming.py` (`scripts/setup_env.sh`, 18 min, mostly downloads) |
| `model-e128t/` | B32's model, HF `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` (E043 HF layout, 131,090,911,045 B = REPLICATE.md's revision 9bce98c5; 19 min at ~120 MB/s) |
| `model-qat025/` | `chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch`, 52 files, 59,297,514,191 B (9 min) |
| `model-qat025-eng/` | the serving dir: symlinks to the QAT files + B32's 3 Engram files, plus a merged index (`scripts/build_qat_dir.py`) |
| `runs/<name>/` | `server.log`, `cmd.txt`, `env.txt`, `server.pid` (session id) |
| `scripts/` | copies of this folder's scripts |

Disk after both downloads: 212 GB used, 185 GB free.

## Serving
`scripts/serve_e128.sh <run>` (env `MODEL`, `PORT` 18000, `GPUS` 0,1, `TP` 2, `SEQS` 64, `MBT` 2048, `UTIL` 0.90,
`MAXLEN` 163840, `RETAIN` 512). Stop with `scripts/stop.sh <run> 0,1`.

```
MODEL=/root/e068/model-qat025-eng bash /root/e068/scripts/serve_e128.sh qat025_tp2   # running now, port 18000
bash /root/e068/scripts/serve_e128.sh e128_tp2                                       # B32's model (stopped)
```
Command, as written to `runs/qat025_tp2/cmd.txt`:
```
vllm serve /root/e068/model-qat025-eng --served-model-name dsv41 --host 127.0.0.1 --port 18000 --trust-remote-code
  --tensor-parallel-size 2 --distributed-executor-backend mp --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840
  --max-num-seqs 64 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --enable-prefix-caching
  --prefix-cache-retention-interval 512 --engram-config {"cpu_offload":true} --tokenizer-mode deepseek_v41
  --language-model-only --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details --compilation-config {"cudagraph_mode":"FULL","max_cudagraph_capture_size":64}
```
- Env: `VLLM_USE_V2_MODEL_RUNNER=1`, `FLASHINFER_CUDA_ARCH_LIST=TORCH_CUDA_ARCH_LIST=12.0`, `TRITON_PTXAS_BLACKWELL_PATH`,
  the Humming GPTQ env (`DSV41_EXPERT_HUMMING` uint2/g128/zero point, `DSV41_EXPERT_HUMMING_W2` g64,
  `E043_GPTQ_DIR=$MODEL/gptq`), and `E043_CTRL` pointing to a null control file.
  - The log must show `[E043] ctrl loaded: experts=None engram=None observe=None`.
- Startup: about 7 min the first time (weights 51-80 s, then TileLang/JIT and graph capture), 2 min with warm caches.
- Memory: 30.24 GiB weights per GPU; 53.61 GiB KV per GPU, i.e. **37,360,260 KV tokens** (228x at 163,840).
  `nvidia-smi` shows ~88.4 GB per GPU in use.
- The Engram log line "offloaded to pinned host memory: 192001740 rows x 256, 47.21 GiB per rank" prints the
  uncompacted `part_num_embeddings`. The compact patch allocates only the kept slice (~16-17 GiB per layer and rank, 64 GiB
  in total).
- Reach it from the workstation through `ssh -N -L 18068:127.0.0.1:18000 -p 61847 root@38.20.137.73`.

## How this stack differs from B32's Kaggle stack
| | B32 (Kaggle 4xL4) | here |
|---|---|---|
| vLLM code | fork wheel + 16-file overlay (kit 12 + #156 indexer skip + 4 KV-offload files) | fork wheel + E043 patches (as E066's B32 Engram A/B) |
| `DSV41_INDEXER_Q_PREDECODE=1` (#154) | on | absent (the wheel has no such gate; the kit overlay adds it) |
| #156 indexer decode skip | on | absent |
| `--linear-backend humming` (dense MXFP8 via Humming) | on | off (needs the kit overlay; without it, it silently falls back) |
| KV CPU offload 32 GiB, util 0.945 | on | off, util 0.90 |
| TP | 4 (L4, sm89) | 2 (sm120) |
| retention 512, max-num-seqs 64, MBT 2048, FULL graphs <= 64, fp8 KV, block 128, 163840, Humming 2-bit experts, Engram cpu_offload, `--language-model-only` | same | same |
| tool code | LocalExecutor in the solver | Docker sandbox `http://172.18.0.2:6000` |

Accuracy impact:
- #156 was measured bit-identical (logits, 74 shapes; E054). #154 and KV offload were parity-checked (E056).
- The dense backend (Humming vs the default MXFP8 kernel) and TP2-vs-TP4 reductions change floating-point
  rounding only, not the model.
- E066 ran B32's model on this same E043 stack (TP4, 4x RTX PRO 6000): pass@1 69.9 / pass@2 81.4 (Kaggle key).
  E046 run 7 (kit overlay stack) got 70.0 on attempt 0. Nothing points to a stack effect, but E066's README notes that
  E052's 76.5 was not reproduced (cause open).

## QAT serving dir (`scripts/build_qat_dir.py`)
- The QAT repo ships no Engram table. Its index expects Kaggle row-chunk files (`model-engram-000NN-of-00016`,
  `layers.{1,14}.engram.embed.weight.rows_<start>`, 7 chunks of 19,527,154 rows per layer), which the repo does not
  include.
- `build_qat_dir.py` symlinks every QAT file except the index. It symlinks B32's `model-engram-00001..00003.safetensors`
  at the **top level** and writes an index made of the QAT weight_map minus its 18 Engram table entries, plus B32's 6.
  - Top level, because this wheel's `DefaultModelLoader` globs only `<model>/*.safetensors` and then requires every
    index file to be in that list. The first try, with an `engram/` subdir, failed: "Weight files referenced in index but
    missing".
- The QAT model keeps its own Engram projections (`layers.{1,14}.engram.{q_weight,k_weight,wkv.weight,wkv.scale}`
  in `model-00002-of-00002`). They equal B32's anyway.
- Checks it runs:
  - Engram config keys equal in both repos (whole `config.json` byte-identical, tokenizer too).
  - B32 `embed.weight`/`embed.scale` rows == `sum(engram_compact_kept)` (136,114,884 / 124,186,164).
  - The QAT chunk starts tile those rows exactly (last chunks 18,951,960 / 7,023,240).
  - `row_map` length == `engram_num_embeddings`.
  - The merged index has the same 1,260 tensor names as B32's.
- Our `apply_compact_engram.py` can also load row-chunk keys (`embed.{weight,scale}.rows_<start>`), but the
  whole-tensor HF files were used.

### B32's Engram part (to reuse)
| file | tensors |
|---|---|
| `model-engram-00001.safetensors` (35,934,329,640 B) | `layers.1.engram.embed.weight` F8_E4M3 [136114884, 256]; `layers.1.engram.embed.scale` F8_E8M0 [136114884, 8] |
| `model-engram-00002.safetensors` (34,321,172,344 B) | `layers.1.engram.embed_tokens.row_map` I32 [384006168]; `layers.14.engram.embed.weight` F8_E4M3 [124186164, 256]; `layers.14.engram.embed.scale` F8_E8M0 [124186164, 8] |
| `model-engram-00003.safetensors` (1,536,066,880 B) | `layers.14.engram.embed_tokens.row_map` I32 [384016682] |

The Engram-related `text_config` keys are `engram_layer_ids` [1, 14], `engram_num_embeddings` [384006168, 384016682],
`engram_max_ngram_size` 4, `engram_vocab_size` 16000000, `engram_n_heads` 8, `engram_head_dim` 256,
`engram_pad_token_id` 2, `engram_compressed_vocab_size` 99092 and `engram_compact_kept` (2 x 24 per-hash-column counts).
The top level has `e043_pruning.engram_compact` true. There are no ngram/ple keys besides `engram_max_ngram_size`.

### What the QAT checkpoint changes (tensor sha256 vs B32, `scripts/cmp_models.py`, `cmp_tensors.py`)
- **Only the 240 shared-expert tensors differ:** `layers.N.ffn.shared_experts.{w1,w2,w3}.{weight,scale}`, all 40 layers.
- **Every routed-expert tensor in `gptq/` is byte-identical to B32's:** 40/40 layers, 1,152/1,152 tensors each. Only
  the safetensors header metadata differs.
- Router, attention, norms, embeddings and Engram projections are identical too: 1,014 of 1,254 compared tensors.
- So this export behaves as "B32 + SFT'd shared experts". If the SFT was meant to train the routed experts, their
  update did not reach the 2-bit export.

## Harness (from the workstation)
`run_b32_harness.sh <tag> [args]`: B32's frozen harness.
- Module set: the V5 kit's `repo/evals` snapshot (`kaggle/dsv41_tools/kaggle_kit_v5/dsv41_kit.tgz`) with
  `pinned_candb/{run_deepseek_tools.py, train_check.py, auto_repair.py, auto_repair_names.json}` copied over it, in
  `b32kit/` (hashes 882c817f / 8fe36175 / 2292369c / 571f47eb, checked at every start). This is the module set B32's
  notebook runs.
- Flags: B32's solver flags (multi-test 0.5, auto-repair, token-exact, context 163840, 40 turns), with
  `cache_salt` = tag.
- Uses the Docker sandbox (`http://172.18.0.2:6000`; never restart it) and writes `results/<tag>.jsonl`.
- `--token-exact` uses the server's `/tokenize` and `/v1/completions`, so no local tokenizer is needed.

120-task run (main session; not started here):
```
bash experiments/E069_qat_sft_eval/box/run_b32_harness.sh <tag> --attempts 2 --workers <N> \
  --challenges data/arc-agi_evaluation_challenges.json
```
Score with the Kaggle key (`data/arc-agi_evaluation_solutions.json`), which differs from the GitHub copy on 4a21e3da,
f560132c and b6f77b65.

## Smokes
- B32 model (`e128_tp2`), stack check, `e068_smoke_b32`: stopped by hand after 7 token-exact tool turns over ~5 min
  (scope change: QAT first). 25 requests, all 200, no errors.
  - Decode ~145 gen tok/s at 2 streams (~73 per stream); prefix-cache hit rate 64% early in the run.
- **QAT (`qat025_tp2`), `e068_smoke_qat025`: 2/2 tasks correct**, 0 errors, both `submitted` with `train_pass`.
  `results/e068_smoke_qat025.jsonl`, log in `logs/`.

  | task | test pairs | turns / tool calls | output tokens | time |
  |---|---|---|---|---|
  | 1ae2feb7 (one multi-test rollout) | 3/3 correct | 8 / 8 | 11,141 | 168 s |
  | 136b0064 | 1/1 correct | 7 / 7 | 17,849 | 246 s |

  - token_exact: 2 breaks avoided per attempt, 0 rerenders, resyncs or parse fallbacks.
  - Server: ~146 gen tok/s at 2 streams (~73 per stream); prefix-cache hit rate 88% by the end.
- Decode bench (`scripts/bench_decode.py --n 48 --max-tokens 1536`, short contexts, QAT server): 1,193 tok/s
  aggregate (server log 1,250-1,280), 24.9 tok/s per stream, KV usage 0.5%.
  - Real tool traffic at long contexts will be slower per stream.
  - With 37M KV tokens, concurrency is capped by `--max-num-seqs 64`, not by KV, so 64 harness workers fill the server.

## Result (2026-10-30, main session)
Full run `e069_qat025_full`: 120 Kaggle eval tasks (172 test inputs), 2 attempts, B32 frozen harness (all B32 flags incl.
multi-test 0.5, auto-repair, rescue-code, token-exact, context limit 163840), 64 workers, untimed; 240 rollouts, 344
attempt rows, **0 errors**, 5,774 tool calls, 2 h 54 min. Scored locally with the Kaggle answer key
(`data/arc-agi_evaluation_solutions.json`; the harness itself printed `correct=unscored`).

| Model (same E043 sm120 stack) | pass@1 | pass@2 |
|---|---|---|
| **QAT/SFT 025ep** (`chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch` + B32 tool Engram), TP2 | **71.3%** | **80.0%** (96.00/120) |
| B32 E128 (E066 reference, 4x RTX PRO 6000 TP4) | 69.9% | 81.4% |

A tie (within run-to-run noise, which reached ~7 points across setups). Expected: the export's routed experts are
byte-identical to B32's; only the 240 shared-expert tensors differ. Output tokens per rollout averaged 39.8k.
Box Vast 58398827 stopped automatically at 15:56 UTC after the run (user: the box is only for this benchmark).
