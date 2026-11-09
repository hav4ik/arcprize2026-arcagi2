# 2026-10-13: Qwen3.8-27B tool harness (E019–E021) and the Vast pool

Summary of the 2026-10-12/13 session. Numbers are Kaggle pass@k (pair-weighted, 167 public-eval test pairs) unless marked task-weighted. Details:
`experiments/E019_qwen_program_answers/README.md`, `experiments/E020_qwen_harness_80/README.md`, `experiments/E021_kaggle_v3/README.md`,
and the report (`reports/arc_agi2_experiments_report.html`, Table 1 column "Kaggle 4×L4 / 12 h").

## Kaggle budget

The Kaggle rerun gets about 44k generated tokens per test pair: ~295 tok/s end to end on 4×L4 × ~11 h usable ÷ 259 pairs.

| Config (1 attempt/pair) | pass@1 | Tokens/attempt | Fits? |
|---|---|---|---|
| K0 = submitted v2 (medium, 48k/24k, `submit_answer`) | 26.9% | 36.7k | yes |
| K1 = K0 + 8k per-turn think cap | 27.5% (tie) | 34.9k | yes |
| K2nc = program answers + task summary + helpers + roles prompt + rich feedback | 35.3% (+8.4 [+0.6, +16.1] vs K0) | 34.6k | yes |
| K2 = K2nc + think cap (2 samples) | 32.0% | 34.8k | yes |
| **K3 = K2 at xhigh, still 48k/24k (2 samples), the v4 config** | **37.4%** (+5.4 [+0.6, +10.2] vs K2) | 38.1k | yes |
| x96pc8 = K3 recipe at 96k/48k + think cap | 55.7% | 56.1k | no, 1.3× |
| no tools, xhigh, 240k (E008) | 35.9% | 73.6k | no, 1.7× |

- **Gated 2nd attempt.** Simulated with the real 2nd sample as the retry. It retries only pairs without a train-passing program, held to a 44k mean per pair.
  - K3: 38.3% pass@2 at 44k; 43.1% at a 64k mean.
  - K2: 37.1% at 44k.
  - The submitted v2/v3 config: ~28%.
  - The free attempt_2 (other grids from the same attempt) adds +0 to +1.2.
- **Train check reliability.** The train check is a good ranker only when the model has budget. P(correct | train-passing program):
  - medium-48k, E019: 41%;
  - K2: 62%;
  - K3 and x96pc8: 72%.
- **Think cap.** A tie at medium-48k (K1 vs K0, K2 vs K2nc), but +7.2 [+0.6, +14.3] at xhigh-96k.
- **Throughput is the main Kaggle lever.** A 48k → 96k cap is worth ~+19 pp pass@1, and it needs ~1.3× today's Kaggle throughput.
  - Candidates: MTP / speculative decoding, and 2 model copies (TP2 × 2) with per-attempt pinning. Neither is tested yet.
  - Risk the other way (E021): at ~250 tok/s K3 would get no retries and lower first-attempt caps, and could fall to ~28%.

## Unconstrained ceiling

- **Single attempt.** x96pc8 reaches 55.7% pass@1, +21.0 [+12.3, +29.9] vs no-tool xhigh-240k at fewer tokens.
- **Pooling.** x96pc8 + x96p + 2 no-tool xhigh-240k samples, voting with train-pass tiebreaks, gives 69.5% pass@2 (70.7% task-weighted) at ~261k tokens/pair. The oracle is 71.3%, or 74.3% with every logged candidate. The limit is coverage, not selection.
- **80% was not reached.**
- **Weaker candidates hurt the vote.** Medium-48k tool attempts outvote correct answers, so pool only strong modes.
- **No gain, so far:**
  - the test-audit / lint review turn: +0.8 on 122 pairs, and it changed 1 answer in 91 reviews;
  - the worked example: 8 pairs only, inconclusive;
  - cross-attempt memory, augmentation and adaptive k were never run.

## Serving on the Vast pool

1. **One model copy per GPU beats one copy over 4 GPUs.** 1× RTX PRO 6000 gave ~600–630 tok/s on tool traffic, against ~300–480 per GPU with TP4. That is about 2× tok/s per dollar.
2. **KV capacity, not request slots, sets concurrency.**
   - Size it as KV tokens ÷ p90 attempt context: ~10–12 per box for xhigh-96k attempts, ~20 for 48k attempts.
   - Early-run measurements mislead, because contexts keep growing.
   - Overload shows as a falling prefix-cache hit rate (down to 4%) and retractions.
3. **SGLang Model Gateway (smg, sglang-router 0.3.2).**
   - The manual policy with `x-smg-routing-key` keeps a conversation on one worker.
   - `min_load` counts in-flight HTTP requests, so conversations waiting on a tool call look idle and new keys pile onto that box.
   - `x-smg-target-worker` is ignored under the manual policy.
   - Fix: a harness-side per-box slot scheduler (`router/slots.json`, `--worker-slots`). Each attempt holds one slot and talks to its box directly.
4. **Vast quirks.**
   - Caddy auth breaks gateway health checks (they carry no key), so every box goes through an SSH tunnel.
   - Tunnels to boxes on the same host hung a few times. A keeper restarts the ssh PID, and client retries make a hang cost minutes.
   - The token is the instance's `jupyter_token`.
   - Register boxes by IP plus port, because two boxes can share an IP. IPs can change during startup.
   - Boxes in CN can get stuck pulling images.
5. **Sandbox.**
   - Session DELETEs lacked the `X-Session-ID` header, so finished kernels lived until the 30-min reaper (~14 GB). Fixed in `evals/sandbox_client.py`.
   - Cap each Python session at ~2 GB (`--session-mem-mb`).
   - Use a request timeout.

## Incidents and rules

- **v2's emergency writer kills by name.** It runs `pkill -9 -f '[s]glang'`. A CPU test of it on this host killed every eval process and the gateway. Name-based kills now run only on Kaggle (`/kaggle` present), and off Kaggle the solver kills only its own descendants.
- **Concurrent edits broke a run.** Two agents edited `evals/arc_harness/agent_qwen.py` at once, and a run imported a half-edited file. Rule: no shared-harness edits while runs import it. Use an atomic replace, or send a diff to the coordinator.

## Kaggle submissions

- **Scores so far (public LB):**
  - v2 (57088016) **11.94**: concurrency 35, reproduced by the load probe as a KV/prefix-cache collapse (10.4% on 48 pairs).
  - v3 (57121897) **18**: the same agent at concurrency 29, plus free attempt_2 and the gated retry. The +6 is mostly the concurrency cut, as predicted (estimate ~18–23).
  - v4b (57157836, submitted 2026-10-13 22:39 UTC): **19.44**.
  - v4b resubmitted unchanged (57197296, 2026-10-14 23:01 UTC): **26.0**.
  - **The same notebook moves by ~6.6 points between runs** (sd ~4.5–5 per submission). The external no-tool notebook moved 21 → 13. Compare configs only by averages over several submissions, or by paired 48-pair probes; v4b's expected LB is ~22–23.
- **v2** (submission 57088016, 2026-10-11) was still pending after 20+ h, because of a Kaggle compute shortage.
- **v3** = v2 agent + free attempt_2 + gated retry + concurrency 29 + a 2 GB session cap. It passed the 4×L4 commit run on 2026-10-12, with expected gain ≈ +1–2 pp.
- **v4** = K3 + think-call rescue + gated retry. It is built in `kaggle/nb_v4/` and passes the CPU and local checks. Before submitting, it needs a 4×L4 commit run, and `ARC_RATE_PRIOR` should be considered at ~280.

Cost: about $65–70 of Vast box time (estimate from list prices); no API spend.
