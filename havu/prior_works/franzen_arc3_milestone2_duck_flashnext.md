# Franzen: ARC-AGI-3 Milestone 2 solution (Duck harness + Qwen3.8-Flash-Next)

Source: Kaggle notebook `dfranzen/arc-agi-3-milestone-2-solution` (public, read 2026-10-27 with `kaggle kernels pull`;
not run), write-up `kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2`, repo `github.com/da-fr/arc-agi-3-solution`.
Context: ARC-AGI-2 discussion 750546 (2026-10-27) claims many leading ARC-AGI-2 teams now use Qwen3.8-Flash-Next with
the Duck harness, and that Flash-Next reaches "~75% with one pass, 80% with two" (the poster's rough modeling, not a
measurement). As of 2026-10-27 there is no public evidence for that claim (one team mentions using "Qwen" without saying whether it is Qwen3.8-27B or Flash-Next).
Update (user, 2026-10-27): the 3rd-place ARC-AGI-2 team (81 on the Kaggle LB) hinted that they use Flash-Next. E048's 59.8 / 72.9 on our DeepSeek-tuned harness (API, provider-default sampling, budget notes from 1% of an 80k budget, 2 attempts, text only) is therefore likely an underestimate of the model.

**What it is.** An ARC-AGI-3 (interactive games) agent: Tufa Labs' Duck harness (`github.com/Tufalabs/duck-harness`
@ 7652836) plus a 16-file patch (5.3k-line `tool_agent.py`, Python tool sandbox, persistent functions kept across a
game's levels, vision context with frame-diff images and animations, no-op/stale-state guards, a priority scheduler
over games, context drain).

**Hardware: one RTX PRO 6000 (96 GB, Blackwell SM120)**, `machine_shape: NvidiaRtxPro6000` on the ARC-AGI-3
competition, not 4xL4.

**Serving.** "Pennyroyal" SGLang fork v2.5.3 for SM120 (TORCH_CUDA_ARCH_LIST 12.0; patches from mratsim and
gabrielolympie), offline wheelhouse installed with a bundled uv into its own venv.
- Target `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (INT4), TP1, `--quantization auto-round`, fp8 e4m3 KV, page 64,
  mem fraction 0.96, context 136k, **max 10 running requests**, CUDA graphs bs 1-10, chunked prefill 8192,
  max prefill 16384, schedule `lpm`, Mamba/GDN state cache 60 slots with radix strategy `extra_buffer` (track every
  64 tokens), flashinfer linear-attention backends, PLE embeddings offloaded, `--watchdog-timeout 1800`.
- Speculative decoding: NEXTN (MTP) with 3 steps / 4 draft tokens, albucino's INT4 g32 MTP draft
  (compressed-tensors, fp8 draft KV), accept thresholds 1.0, **FR-Spec** (draft vocab limited to a 64k hot-token map).
- Model files pre-read into page cache with parallel 32 MB reads while the server starts; the harness starts before
  the server is ready (one warm-up action) so Kaggle does not cancel an idle run.
- Sampling T 0.7, top-p 0.95, top-k 20, thinking on (`preserve_thinking`), 12k max output per turn, tool output cut
  to 3k tokens in the middle, history drained by 58k tokens near the 128k window.

**Relevance to our ARC-AGI-2 stack (4xL4, DeepSeek-V4.1 E128).**
- Model: E048 measured full-precision Flash-Next (Alibaba API) on our best tool harness at 59.8 pass@1 / 72.9 pass@2
  vs DeepSeek-1017 74.3 / 82.4; E128 2-bit scores 72-76 task pass@1 on the same eval (E050, E052). Switching would
  need a harness that adds ~15 points for Flash-Next; the Duck harness is a game agent and has no ARC-AGI-2 port here.
- Speculative decoding pays at 10 streams on one big GPU; on 4xL4 at ~56 streams it did not (E041 DFlash: +1-3% at
  c8-c24, half the KV; E042/E043 DSpark: the full drafter leaves no KV on 4xL4, acceptance 1.4-2.1 tokens at N=1-2 on
  tool traffic).
- Already in our stack: fp8 KV, prefix caching with retention, staged model files, server watchdog.
- Open check: whether ARC-AGI-2 notebooks can select the RTX PRO 6000 (Kaggle UI, no quota needed).
