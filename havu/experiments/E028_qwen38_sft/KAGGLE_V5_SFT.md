# Kaggle notebook v5_sft: the E028 SFT model in the v4 submission machinery

Built 2026-10-17. Notebook `kaggle/nb_v5_sft/` (new private slug `chankhavu/arc2-qwen38-e028-sft-submit`), built by
`kaggle/build_notebook.py --variant v5_sft` from `kaggle_v5_sft_settings.json` (this folder). The machinery is
v3/v4's: mode switch, local commit gate, deadlines, emergency writer, guard, repair cell. What changes is the model
and the agent interface the model was trained on.

## What is different from v4

| | v4 | v5_sft |
|---|---|---|
| Model | `mikedan7/qwen3-8-27b-fp8-official` (block FP8) | `chankhavu/qwen3-8-27b-e028-sft/Transformers/fp8/1`: E028 LoRA-SFT, ModelOpt static FP8, `quant_method: modelopt` |
| Tokenizer | the model's own | `--tokenizer` = the official FP8 mirror (still attached), because vLLM 0.23 / transformers 4.57 cannot load the SFT checkpoint's `Qwen3_5Tokenizer` config. The chat template is md5-identical and token parity was verified. |
| vLLM flags | TP4, fp8 KV, TRITON_ATTN, prefix caching, `--no-async-scheduling` | the same, plus `--language-model-only` and `--tokenizer`. Never set `NCCL_P2P_LEVEL` and leave `ARC_FP8_CONFIGS` unset (those configs are for block-FP8 shapes). |
| Agent | program mode, `lean2progroles`, think cap, gated retries | the E025 training interface: submit mode, `lean2roles`, `tool_desc=e025`, executor `style=ipython`, task summary, helpers, xhigh, notes 0.01 / warn 0.75, `max_turns` 40, tool timeout 30 s, `solve_fallback`. There is no think cap. |
| Second attempt | gated retries + free attempt_2 | free attempt_2 only (`ARC_ATTEMPT2=off`): one model attempt per pair |
| Checks | | `INTERNET: OFF` cell; `MODEL IDENTITY` before the server starts (fails with `WRONG MODEL` if `e028_quant_info.json` is missing or the path is not the SFT slug); `SERVER IDENTITY` after startup (quantization and KV lines, `/v1/models`); `ARC_STRICT_FEATURES=1`, so the commit gate fails if the self-test switched off any requested agent option |

New solver settings (all opt-in; defaults reproduce v4):

| Setting | v5_sft value | What it does |
|---|---|---|
| `ARC_MODEL_SOURCE` | `chankhavu/qwen3-8-27b-e028-sft` | serve this attached model (`/kaggle/input/models/<owner>/<slug>/**/config.json`), never silently another one |
| `ARC_TOKENIZER` | `official` | vLLM `--tokenizer` (SGLang `--tokenizer-path`): `official` = the mirror's dir, or an owner/slug or path |
| `ARC_MODEL_REQUIRE` | `e028_quant_info.json` | a file that must exist in the model dir, else the run fails |
| `ARC_TOOLS_DESC` | `e025` | `AgentConfig.tool_desc` |
| `ARC_EXEC_STYLE` | `ipython` | `LocalExecutor(style=...)`: the sandbox text format of the training rollouts (self-tested; dropped if broken) |
| `ARC_TOOLS_SOLVE_FALLBACK` (alias `ARC_TOOLS_FALLBACK_SOLVE`) | 1 | `AgentConfig.solve_fallback` |
| `ARC_TOOLS_MAX_TURNS` | 40 | `AgentConfig.max_turns` (default 60) |
| `ARC_TOOLS_SOFT_FRAC` | 0.5 | `budget_soft` = 0.5 × the attempt's hard cap, the training ratio (40k/80k) |
| `ARC_AGENT_EXTRA` | unset | JSON of further AgentConfig fields (unknown fields are logged and dropped), e.g. a per-turn cap if the harness adds one |
| `ARC_STRICT_FEATURES` | 1 | the commit gate fails if a requested option was dropped |
| `ARC_LOCAL_WALL_MIN` | 95 | local (commit) runs only: an emergency exit at T0 + 95 min |

Sampling is temperature 1.0, top_p 0.95, top_k 20, as in every earlier variant (`Client._stream`).

## Budget and deadlines (rerun: 240 tasks, 259 pairs, 4×L4)

- **Window:** force at 11:00 = 39,600 s from T0. The first token comes ~15 min after T0 (install, preloading ~30 GB,
  server start; v4 runs took 9–15 min), so the generation window is ~38,700 s (10.75 h).
- **Throughput:** v4b measured ~224 gen tok/s on Kaggle at c24. The chat-completions path on L4 managed only 132
  tok/s (36% cache hits), but agent_qwen appends generated text verbatim, so v4b-like ~86% hits are expected.
  Planning value: **200 tok/s** (`ARC_RATE_PRIOR=200`); the solver measures the real rate after 15 min.
- **Token supply = rate × 38,700 s:**

  | rate | supply | per pair |
  |---|---|---|
  | 180 tok/s | 7.0M | 26.9k |
  | 200 tok/s | 7.7M | 29.9k |
  | 224 tok/s | 8.7M | 33.5k |
  | 250 tok/s | 9.7M | 37.4k |

- **Demand at the training budget:** dev56 so far (20 SFT attempts): mean 53.9k output tokens, p50 57k, p90 80k
  (6 of 20 at the cap). 259 × 54k = 14.0M tokens. That needs 361 tok/s over the window, or 19.4 h at 200 tok/s,
  so **it does not fit.**
- **So the cap is the solver's adaptive share,** not 80k: each first attempt gets min(80k, capacity left / pairs
  left), with capacity = measured rate × time to force minus what in-flight attempts may still use. At 200 tok/s
  this is **~30k**, with a soft target of 15k and the "submit now" warning at 22.5k. At the hard cap the harness
  forces a `submit_answer` turn (FORCE_THINK). If there is still no answer: the session's `answer`, then
  `solve(test_input)` (`solve_fallback`), then salvage. The 80k ceiling binds only above ~535 tok/s.
- **Pacing:**
  - 24 concurrent attempts at ~8.3 tok/s each (plus tool time) means a 30k attempt takes ~60–65 min.
  - 259 / 24 ≈ 10.8 waves × ~60 min ≈ 10.8 h, so the queue drains close to the force time.
  - The ≤ 24 attempts still running at 11:00 are forced to answer; the hard stop is 11:30.
- **KV:** the SFT checkpoint gets 1.54M KV tokens (E027, vLLM 0.23, fp8 KV). With a ~30k output cap, p90 final
  context is about 45–55k (prompt 3–8k, tool outputs, output), and 24 × 55k = 1.32M fits, so concurrency stays
  at v4b's 24.
- **Deadlines are unchanged from v4:** force 11:00, hard stop 11:30, emergency writer 11:42, guard 11:45 (Kaggle
  limit 12 h).
  - Moving the force time to 11:15 would add only ~2% of supply.
  - Throughput (E027) is the real lever.

**Accuracy risk from the cap.** In dev56, SFT attempts that were correct within C output tokens were 3 of 20 at
32k, 7 at 40k, 11 at 64k and 12 at 80k. This is a lower bound, since a capped attempt is forced to answer and
might still be right. The base model in the same harness had 9/46 at 32k vs 20/46 at 80k. At Kaggle's ~30k
shares the SFT model is far from its 80k operating point. **Compare SFT and base at a ~30k cap on dev56 before
using a submission slot.**

## Commit run (local self-scored set, `chankhavu/arc2-local-testcases`, all 8 tasks / 13 pairs)

- **Local time box:** force at max(T0 + 55 min, server ready + 20 min), but no later than T0 + 80 min. Hard stop
  5 min after the force. The solver exits by T0 + 95 min whatever happens (`ARC_LOCAL_WALL_MIN`).
- **GPU time:**
  - typical: startup ~15–20 min + generation until T0 + 55 + hard stop + summary ≈ **60–65 min**;
  - worst case (a server start retry): < 100 min. Under the user's 2-hour approval either way.
- **Gate (last cell):** answered > 0, tool calls > 0, the tool agent is in use, and no requested option was
  dropped.

## Commands (not run by this sub-agent; the push was refused by its permission classifier)

```bash
cd /home/vu/Workspace/ARC-AGI-2
export KAGGLE_API_TOKEN="$(sed -n 's/^kaggle_api_token:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
.venv/bin/python kaggle/build_notebook.py --variant v5_sft --env "$(cat experiments/E028_qwen38_sft/kaggle_v5_sft_settings.json)"
.venv/bin/python kaggle/tests/check_notebook.py --variant v5_sft --extract-to /tmp/nb_v5_check   # NOTEBOOK CHECK OK
# commit run (version 1 of the new kernel; ~1 GPU-h)
.venv/bin/kaggle kernels push -p kaggle/nb_v5_sft
.venv/bin/kaggle kernels status chankhavu/arc2-qwen38-e028-sft-submit
.venv/bin/kaggle kernels output chankhavu/arc2-qwen38-e028-sft-submit -p kaggle/runs/v5-sft-v1
grep -E "INTERNET|MODEL IDENTITY|SERVER IDENTITY|v3 self-test|concurrency|COMMIT GATE|SUMMARY" kaggle/runs/v5-sft-v1/*.log
# competition submission: uses a daily slot, needs the user's explicit OK
.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-2 -k chankhavu/arc2-qwen38-e028-sft-submit -v 1 \
    -f submission.json -m "v5_sft: E028 SFT FP8, e025 interface, adaptive cap ~30k, conc 24"
.venv/bin/python kaggle/monitor_submissions.py --once
```

## Risks

1. **The cap is far below training's budget.** The notes show e.g. "aim 15k (hard limit 30k)" where training showed
   40k/80k. The model mostly ignores notes and runs to the cap, so most attempts end in the forced submit or a
   fallback. See the accuracy risk above.
2. **Throughput on Kaggle is unmeasured for this checkpoint.** ModelOpt per-tensor W8A8 may run differently from
   the official block-FP8 Marlin path. The rate is measured after 15 min and the shares adapt. If it is slower,
   caps shrink further.
3. **Snapshot drift.** The notebook embeds `evals/arc_harness` at build time. Rebuild and re-check right before
   any push. Embedded md5s: agent_qwen `5624b4df`, executor `2fe64a4e`.
4. **Model handle casing.** `Transformers` in `model_sources` follows the model URL. The mount path is lower case;
   the solver globs, so it doesn't care.
5. **Budget units.** E025 budgets counted DeepSeek tokens; here they are Qwen tokens.
