# ARC Prize 2026 · ARC-AGI-2: a pruned DeepSeek-V4.1-Flash agent on 4×L4

Our team's code for the [ARC Prize 2026 ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
Kaggle competition.

Our best submission scored **76.25 on the public leaderboard**. It runs DeepSeek-V4.1-Flash-1017, a ~510 GB, 748B-parameter
mixture-of-experts model, as a Python-tool agent inside one Kaggle notebook on 4× L4 GPUs (96 GB of VRAM in total):

- **Pruning.** Each MoE layer keeps 128 of its 384 routed experts, quantized to 2 bits with GPTQ. A trimmed 64 GiB
  Engram table lives in host RAM, for 131 GB in total. Every statistic comes from the model's own tool-use rollouts on
  ARC training tasks, with tool-call tokens up-weighted.
- **Harness.** A plain tool-use loop: the model writes Python in a persistent session, tests it on the training pairs,
  and submits an expression evaluated in that session. A host-side check re-runs the program on the training inputs
  and picks the two answers we submit.
- **Inference.** Our vLLM fork adds sparse-attention kernels for the L4, runs all-reduces through host memory, and
  keeps 64 agent sessions decoding at once.

Before that, we built on the ARC Prize 2025 winners: NVARC's Qwen3-4B with test-time training, plus a confidence
router that sends uncertain tasks to a second model.

## Results

| System | Public LB |
|---|---|
| NVARC Qwen3-4B, as released | 28.47 |
| + 4,000 steps on our fixed synthetic data, inference in vLLM | 35.97 |
| + LLaDA-8B on ~40% of tasks via the confidence router | 36.94 |
| Qwen3-4B + Qwen3.8-27B with chain-of-thought, same router | 50.14 |
| Qwen3.8-27B tool agent with image input, pass@1 | 42.92 |
| **Pruned DeepSeek-V4.1-Flash tool agent** | **76.25** |

The fixed NVARC training data (a recoloring bug merged colors in about 95% of samples) is at
[livobereshka/nvarc-fix](https://huggingface.co/datasets/livobereshka/nvarc-fix).

## Repository layout

```
harness/   agent harness, evaluations, experiments and Kaggle notebook builders
infer/     the Kaggle inference pipeline: the supervised vLLM server, the 4B cascade and the router
```

### `harness/`

| Path | What |
|---|---|
| `evals/` | The tool harness: `run_deepseek_tools.py`, the `arc_harness/` package (prompts, grids, executor, selection), `train_check.py`, `auto_repair.py` |
| `kaggle/dsv41_tools/` | The DeepSeek-V4.1 tool solver for Kaggle 4×L4: `solver_dsv41.py`, the kit and notebook builders. Start at `README.md`; end-to-end replication in `REPLICATE.md` |
| `kaggle/` | Earlier Qwen3.8-27B submissions and the wheel-source notebooks; every submission follows `SUBMISSION_TESTING_PROTOCOL.md` |
| `experiments/` | Experiments E001–E081, indexed in `experiments.tsv` and summarized in `results_summary.md`. Pruning: `E043_dsv41_real_pruning`, `E066_dsv41_image_pruning`. Evaluation of the quantization-aware SFT: `E069_qat_sft_eval` |
| `reports/` | The experiments report, `arc_agi2_experiments_report.md` |
| `notes/` | Dated findings; the DeepSeek-V4.1 Kaggle track starts at `2026-10-30_dsv4_infer_handoff.md` |
| `prior_works/` | Notes on related work |
| `data_tools/` | Eval reports, trace viewers, the contamination check and a secret scan |

### `infer/`

| Path | What |
|---|---|
| `qwen38_pipeline.py` | The submission writer for the Qwen3.8, DeepSeek-V4-Flash and DeepSeek-V4.1-Flash variants. `--qwen38-variant dsv4-vllm --dsv41-harness havu` runs the harness solver against our supervised vLLM-fork server, which is how the best submission ran |
| `dsv41_havu.py`, `qwen38_server.py` | Run the harness solver as an external client of the server; the server restarts under a supervisor, and the solver follows its status file through restarts |
| `havu/` | A snapshot of the harness files that the pipeline runs (the B32-M solver plus our edits), kept at the harness paths; see its `README.md` |
| `dsv41_vllm.py` | Our own REPL tool loop with image input for DeepSeek-V4.1 on the vLLM fork |
| `dsv41_repl.py`, `dsv41_prep.py` | The same model on llama-server (GGUF) |
| `submit.py`, `cascade_4b2llada.py`, `cascade_4b2qwen38.py`, `confidence_routing.py`, `arc_decoder.py` | The bottom-up line: NVARC Qwen3-4B with test-time training, and the confidence router to LLaDA-8B or Qwen3.8-27B |
| `patches/` | vLLM patches for the Kaggle runtime; see its `README.md` |
| `third_party/` | Vendored code and wheels: DeepSeek encoders, Sonic MoE, FlashAttention 3 for sm89, a bitsandbytes Muon optimizer |
| `vllm_delta_mhar_plugin/`, `vllm_diffusion_gemma_plugin/` | vLLM plugins |

## Data and models

- **Competition data:** `arc-prize-2026-arc-agi-2` on Kaggle. The 120 public evaluation tasks, and everything derived
  from them (including NVARC's `nvarc_full` and `arc2_evaluation6`), are used for reporting only. Calibration, pruning
  and training use the 1,000 training tasks and NVARC puzzles derived from training parents.
- **Pruned model:** [chankhavu/dsv41-flash-1017-arc-tool-e128-2bit](https://www.kaggle.com/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit)
  on Kaggle.

## Writeup

[DeepSeek-V4.1-Flash on Kaggle 4×L4](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/writeups/deepseek-v4-1-flash)
on the competition's writeups page.

## License

[Apache License 2.0](LICENSE).
