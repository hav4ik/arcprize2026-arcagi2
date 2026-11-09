# Project Maya (mw00/project-maya): single-PC DeepSeek-V4.1 / GLM inference engine

- Repo: https://github.com/mw00/project-maya (MIT; reviewed at HEAD 6afa525, v1.1.28 of 2026-10-25). C++/CUDA/HIP engine on
  ggml + Strata; experts spread over VRAM / RAM / SSD; OpenAI- and Anthropic-compatible API. **One conversation at a
  time** (README; issues #332, #409), so not usable as our server (also rejected in E042 README:1524).
- No ARC content. Claims are community single-stream reports (DeepSeek decode 21.5 tok/s on 2x4090, 51.8 on 4xMI100).
- Quants (HF cards, MIT): Maya-S (experts IQ2_XXS gate/up, IQ3_XXS down, IQ2_S/IQ3_S on first/last 4 layers; 92.6%
  top-1 agreement, +3.2% NLL); Maya-XS (all 384 experts IQ1_S; 87.5%, +8.7% NLL).

## Ideas relevant to us (review of 2026-10-26)

| Idea | Evidence in Maya | For us |
|---|---|---|
| **SWA state checkpoints at turn boundaries** (prompt end, one before, a re-render "anchor"; ~3 MB per conversation) instead of caching all SWA history | v1.1.25, #356/#394; `ds4_model.cu:1143-1150, 6142-6236` (agent traffic: 170/176 long prompts reused nothing -> near-full reuse) | The proper fix for our cache churn (`kaggle/dsv41_tools/next_candidate_fixes.md` S3b, S3); 2-3 days in the fork's KV manager |
| **Numeric reasoning effort** (encoder accepts 1-100; "low" = 50) | `serve/ds41_encoding.py:445-478` | We send 50 + the minimal header; probe 25 / 10 on E128 for tokens vs pass@1 |
| **Exact long-prompt skip**: layers 21-39 only need the last (40-20)*127+128 = 2,668 prompt tokens | `ds4_model.cu:11985-12074`, `docs/SETTINGS.md:441-453` | Exact version of our rejected CED prefill; ~31-40% less prefill work on 7-14k prompts, est. +3-6% tok/s; 3-5 days |
| **GPTQ recipe**: routing-weighted per-expert Hessians, layer-wide prior (16,384 tokens), damping 0.1, sequential calibration | `tools/maya_quant/gptq.py:9-14, 170-182` | Could narrow E128's 2-bit NLL gap; check held-out error first; needs an 8x box |
| Copy-engine all-reduce for prefill chunks only | #344 | We measured `NCCL_SHM_USE_CUDA_MEMCPY` (E042:756-772); low priority |

Already covered or not applicable: DSpark drafter (KV cost, low tool-mode acceptance), forced thinking budget (= our
`--cap-nudge`), thinking off, packed FP4 KV (native in our fork), expert skipping, programmatic dependent launch
(sm_90), tree verification (proposal), IQ-format GGUFs (no batched vLLM kernel). Licenses MIT except the GLM DFlash2
drafter (CC BY-NC-ND). Follow-ups are items I1-I4 in `kaggle/dsv41_tools/next_candidate_fixes.md`.
