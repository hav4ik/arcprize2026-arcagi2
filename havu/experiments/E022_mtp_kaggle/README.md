# E022: MTP (speculative decoding) for the Kaggle tool agent: planned, deferred

Status: **planned, deferred**. Decided 2026-10-13: keep the Kaggle GPU quota for submissions this week. At decision time 7 GPU-hours were left until Friday 2026-10-16 EOD, and the next weekly reset is the earliest slot.

## Why test it again

On 2026-10-13, MTP with 2 draft tokens on v4b hurt badly: pass@1 14.6% vs 39.6% without MTP on the same 48 pairs (`experiments/E021_kaggle_v3/load_probe.md`).

The cause was KV memory, not draft quality:
- Draft acceptance was 69%: position 1 at 0.73, position 2 at 0.58, mean acceptance length 2.39.
- The draft head cut vLLM's KV capacity from 1.40M to 1.17M tokens (−16.5%).
- At concurrency 24, peak KV reached 97%. The prefix-cache hit rate fell to 2–10% between minutes 50 and 90, and throughput dropped from 224 to 181 tok/s.

At a concurrency that keeps KV under ~70%, MTP may still pay off. The attempts are decode-bound at ~10 tok/s each, and 2.39 tokens per step could give up to ~1.5–2× per-stream speed. In throughput this would be the cheapest step toward the ~1.3× needed for one xhigh-96k attempt per pair (x96pc8, 55.7% pass@1 on the pool).

## Plan

1. **Arm A:** v4b settings + `ARC_MTP=2` + `ARC_CONCURRENCY=18–20`, sized as 1.17M KV ÷ ~55k p90 context ≈ 20.
2. **Arm B (optional):** the same with `ARC_MTP=1`, which costs less KV.
3. **Probe design:** a throughput probe is enough, with no accuracy claim. Run the real scheduler on ~24 public-eval pairs and force at server-ready + 60 min. Measure:
   - aggregate and per-attempt tok/s;
   - peak KV and prefix-cache hit rate over time;
   - assigned budgets.
4. **Compare** with v4b (no MTP, concurrency 24) on the same time window: 224 tok/s average, 202 tok/s in minutes 60–130, 10.4 tok/s per attempt.
5. **Adopt MTP only if** aggregate tok/s is ≥ +15% over v4b, KV stays under 75%, and cache hits stay above 90%.

## Cost

| Option | Kaggle GPU-hours |
|---|---|
| One 60-min throughput probe: ~10 min server start + 60 min + 15 min hard stop | ~1.3 |
| Arms A and B in parallel (2 GPU sessions) | ~2.6 |
| Full accuracy probe like v4b: 48 pairs, 120 min | ~2.3 per arm |

Queue time doesn't count toward the quota. On 2026-10-13 the mtp2 probe waited ~1.5 h before starting.

**Cheaper alternative, no Kaggle quota:** run the same probe on a Vast.ai 4×L4 box (same GPU, SM89) if one is available, or on 4× L40S (same SM89 architecture, more bandwidth; results transfer only roughly). This costs ~$1–3/h. Use the same vLLM 0.23 wheel, FP8 model and `kaggle/src/solver.py` in local mode with `ARC_LOCAL_SOURCE=eval`.

## Budget note (week ending Friday 2026-10-16)

- Left as of 2026-10-13 evening: ~7 GPU-hours.
- Each submission candidate needs either:
  - a ~0.3 h commit run; or
  - a full load probe (~2.3 h), which can then be submitted directly, as v4b was.
- So reserve the week's quota for submissions. Run E022 after the weekly reset, or on Vast.
