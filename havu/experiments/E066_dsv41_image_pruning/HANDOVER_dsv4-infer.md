# E066 handover to dsv4-infer: the two image-harness candidates (2026-10-29)

Goal (user): a Kaggle submission at 0 UTC today with one of these models in the image harness (E058). Your side:
serving on Manh's fork (vision, per-layer counts, row-chunk Engram) and the image harness in the Kaggle kit. Both
models use the same GPTQ recipe as the shipped tool E128 (B32 / B32-M), with FP8 Engram (MXFP4 Engram later; see the
end). The user has not approved Kaggle Model copies yet ("Don't upload to kaggle yet"); ask before making any.

## 1. The two models (private HF, account chankhavu)

| | Candidate 1 (fallback) | E72-152 (recommended) |
|---|---|---|
| HF repo | `DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit` | `DeepSeek-V4.1-Flash-1017-arc-tool-img-E72-152-avg120-2bit` |
| status | on HF, verified (names + sizes), 137 files, 132,061,520,719 bytes | export + upload by ~18:30 UTC (box 1 chain) |
| sha256 manifest (your format: root / files / total_bytes / entries) | `results/c1_gptq_sha256_manifest.json` | `results/e72c152_gptq_sha256_manifest.json` after the upload |
| routed experts | 128 per layer, uniform (5,120) | per layer, 4,800 in all (avg 120), floor 72, cap 152, step 16 |
| config | `text_config.n_routed_experts` = 128 | `n_routed_experts` = 152 (max) + `n_routed_experts_per_layer` (40 ints, below) |
| gptq/layer_LL.safetensors | e0..e127 | e0..e{E_l-1}; compact id = rank among the kept original ids (`e043_manifest.json`) |
| router | gate.weight / bias / bias_vl sliced to the kept rows | the same, per layer |
| Kaggle KV, your measurement (dummy weights, vision on, util 0.938) | 2.93M | 3.38M |
| serving patch on Manh's fork | patch 01 (Engram row chunks), as for E128 | your `port/04c_rowchunk_plus_per_layer_model.diff` (row chunks + per-layer) |

Per-layer counts (E72-152), layers 0..39:
`72,72,72,152,152,152,152,72,152,136,152,152,152,152,104,152,152,152,104,152,120,104,120,152,88,72,72,72,72,104,104,72,88,104,88,152,152,152,152,152`

Both:
- `e043_manifest.json` at the root: original kept ids per layer, the Engram mask hashes.
- Vision tower kept (`vision.*`, `aligner.*`, `image_*`; 259 + 4 + 3 tensors). DSpark (`mtp.*`) dropped, as in the shipped E128. Serve without `--language-model-only`.
- Engram: the 64 GiB tool table as the same 16 row-chunk files as the Kaggle E128 Model / HF E112 (mask sha checked against `e043tool_top64gib.npz`). FP8, unchanged.
- Every safetensors file <= 5 GB (largest 4.999 GB).
- Experts: Humming uint2 GPTQ. `text_config.e043_lowbit` is identical to the shipped E128's except its name:
  `DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'`,
  `DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'`, `E043_GPTQ_DIR=<model_dir>/gptq`. Tensor shapes as you checked:
  w1/w3 q [2304,320], s [2304,40], z [144,40]; w2 q [5120,144], s [5120,36], z [320,36].
- GPTQ: `gptq_experts.py`, percdamp 1.0, batch 4, 1.4M calibration tokens (seed 61, MoE inputs of 10% of
  positions), replayed under each candidate's own mask. The calibration is half E065 image-harness training rollouts,
  half text twin; the shipped E128 used text tool rollouts only.

## 2. Quality so far (never the 120 eval tasks for choices; dev = 64 held-out E065 training rollouts)

Dev NLL per generated token (full model: image 0.7522, text 0.7190):

| | image | text | tool-call tokens (img / txt) |
|---|---|---|---|
| Candidate 1, before quantization | 0.8166 (+8.6%) | 0.7766 (+8.0%) | |
| Candidate 1, 2-bit, runtime masks | 0.84815 (+12.8%) | 0.80680 (+12.2%) | 0.1385 / 0.1497 |
| Candidate 1, 2-bit, its export (serving parity) | 0.84825 | 0.80641 | 0.1385 / 0.1494 |
| E72-152 before quantization, paired vs C1 | -0.0115 | -0.0094 | |
| E72-152, 2-bit, runtime masks | pending (box 1, ~18:05) | pending | |

Free-generation probe (first turn, 25 image + 25 text dev prompts, same server): E72-152 49/50 valid tool calls,
49/49 compile, 0 undefined names; C1 50/50, 50/50, 1. The full model gets 48/50.

ARC eval (120 tasks x 2, E058 image harness and its text twin, RTX PRO 6000 TP4), running now:
- E72-152 on box 1: `e066_e72_img`, then `e066_e72_txt`.
- C1 on box 4 (Vast 58284661): `e066_c1x_img`, then `e066_c1x_txt`; served from its HF export, sha-checked.
- Image results ~21:00-21:30 UTC, text ~22:30-23:00.
- Scoring: `scripts/compare_evals.py --paired` (ARC Prize partial credit; reproduces E058 81.1 / 90.7, E046 77.6 / 88.6,
  shipped E128 text 73.7 / 84.6).
- Discard `e066_c1_img` / `e066_c1_txt`: that run was unmasked (README "Invalid").

## 3. Harness the numbers are for (`scripts/eval_arc_img.sh`)

E046 best with images: `--prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000
--max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 --budget-warn 0.75 --effort-header minimal
--max-turns 40 --turns-note 5 --fallback-solve --multi-test --multi-budget 0.5 --train-check --rescue-code`,
`--extra-body {"chat_template_kwargs": {"thinking": true, "reasoning_effort": 50}, temperature 1.0, top_p 1.0,
cache_salt}`.
- Image arm: `--grid-format text+image --image-cell 0`, chat path (token ids carry no pixels, so no `--token-exact`).
  The server needs `--limit-mm-per-prompt '{"image": 64}'`.
- Text arm: `--token-exact --context-limit <max-model-len>`.
- Context: the chat path asks for up to 64K output tokens every turn. Our eval servers run max-model-len 262,144;
  your `--context-limit` on the chat path covers Kaggle's smaller limit. E066 full_ref at 131,072 without it lost
  19 of 120 tasks to 400s.
- Effort 50 (= encoding.py "low") is the prefix the calibration rollouts were tokenized with.

## 4. Open items on your side

1. Real-weight serve of both exports on Manh's fork (vision, `--limit-mm-per-prompt`). For E72-152, the per-layer +
   row-chunk patch, then a short image-harness ARC slice for throughput, prefix hits and plausibility. Pod access to
   the private HF repos is the user's call (token vs Kaggle copy).
2. Kaggle kit: the image-harness switch (HARNESS hard-codes `--token-exact`), the current `arc_harness/grid_image.py`,
   image limits, the context guard on the chat path. You are building these as `build_img.py` on B32-M.
3. Kaggle Model copies when the user approves: plain files like the E128 Kaggle Model, `e043_manifest.json` at the
   root, my sha256 manifests for the smoke check. I can upload from box 1 (~30 min each + ~16 min processing).
4. Per-layer memory overhead: Humming's repack holds ~0.2 GiB/GPU beyond the expert bytes for non-uniform layouts.
   A fix there would give E72-152 ~+0.2M KV with no export change.

## 5. Later (not tonight)

- **MXFP4 Engram is free on quality:** C1 export, all four dev NLL changes < 0.0005. Format errors: MXFP4 best-of-two
  11.8% rel L2, MXFP4 OCP rule 12.0%, NVFP4 9.5%, MXFP6 1.2%. Shipping it needs PR 61530 ported to the fork plus a
  packed export. It saves ~30 GiB of host RAM. Files: `scripts/fp4_engram_chunked.py`, `fp4_pilot.sh`,
  `engram_format_error.py`.

## Files (`experiments/E066_dsv41_image_pruning/`)

- `scripts/export_candidate.sh`: E043 `export_pruned.py --no-mtp --no-routed-experts [--per-layer]` +
  `make_lowbit_export.py`.
- E043 `check_lowbit_export.py`: byte-level content check, per-layer aware.
- `scripts/upload_candidate.sh` + `scripts/cards/`: HF upload and model cards.
- `scripts/sha_manifest.py`: the sha256 manifests.
- `scripts/serve_eval.sh`: eval servers. `c1x` serves the export; `e72q` uses runtime masks and checks the server's
  last `ctrl loaded` line.
- `README.md`: everything above in more detail. The decision page is https://claude.ai/artifact/BvMjvSSVkVrDRohJNCpSAU.
