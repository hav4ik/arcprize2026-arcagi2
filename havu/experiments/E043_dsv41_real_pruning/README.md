# E043: real-weight pruning of DeepSeek-V4.1-Flash-1017 (Engram rows + MoE experts)

**Goal (user, 2026-10-23):** implement the plan pack's Engram-first pruning on the rented 8x RTX PRO 6000 box:
frequency-prune Engram rows, then REAP expert statistics with all 384 experts, E64 (primary) / E72 candidates.
Calibration data: `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036` (tool-use rollouts, E031/E036/E038).
Quality: the 120 ARC-AGI-2 eval tasks with the same harness, full vs pruned on the same server.
The plan pack (unzipped upload) is in the session scratchpad: `arc_deepseek_v41_plan/` (START_HERE.md,
REAP_IMPLEMENTATION_CONTRACT.md, TASKS_ENGRAM_FIRST.md, ...). Deployment counterpart: E042 (session dsv4-infer).

## Setup

- **Box:** Vast 57370044, 8x RTX PRO 6000 Blackwell Server (96 GB, SM120), 1.5 TB RAM, 256 vCPU, 1 TB disk
  (RAID1, only ~120 MB/s sustained writes), driver 580.105.08, $13.73/h.
- **Checkpoint:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, 510.31 GB, downloaded into `/dev/shm`
  (RAM; the disk was too slow) at ~400 MB/s. Model load from shm takes ~42 s.
- **Runtime:** vLLM fork `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` + flashinfer vision2 (E042's
  `requirements.box.freeze`), venv `/workspace/dsv4/venv`. MXFP4 experts run on Marlin with native MXFP8
  activations and the router weight applied before the fc2 quantization (`router_weight_before_fc2_quant`).
- **E043 hooks** (`patches/apply_e043_patches.py`, `patches/e043_obs.py`), controlled by a JSON file `$E043_CTRL`
  re-read at each eager forward:
  - expert masks: pruned experts get selection bias -1e9 in the native top-6 kernel, so top-6 refills from kept
    ids and weights renormalize over the selected set, exactly as in a physically pruned model;
  - Engram row masks: dropped row ids are rewritten to -1, which the fork's owner-sharded lookup reads as a zero
    row before Wkv (the plan's declared miss rule);
  - REAP observer on `MarlinExperts.moe_sum`: per token and slot, ids, router weight and the L2 norm of the native
    weighted expert output (`reap_native_amp`), all-reduced over the EP group (needs `--enable-expert-parallel`).
- **Serving:** `scripts/serve.sh` (TP4 or TP8 + EP, fp8 KV, prefix caching, FULL_DECODE_ONLY graphs, Engram CPU
  offload, `--reasoning-parser/--tool-call-parser deepseek_v41`). Two TP4+EP4 servers: A (GPUs 0-3, port 18000)
  for ARC evals, B (GPUs 4-7, port 18001) for replay/NLL with its own control file.

## Data and checks

- `scripts/tokenize_rollouts.py`: rebuilds each rollout with the model's `encoding/encoding.py` (thinking mode,
  tools on the system message so earlier reasoning is kept, numeric effort 50), cut after the last assistant
  message; roles mark model-generated tokens.
  - 10,001 rollouts, 251.8M consumed tokens (106.5M prompt/tool, 145.3M generated).
  - Per-turn check against the API usage (`verify_tokenization.py`): prompt tokens equal the rebuilt prefix on
    **all 78,992 turns**; completion tokens equal the span on 87% of turns and differ by +1 on 13% (the sampled
    stream had one more token than the re-tokenized text, e.g. trimmed whitespace; the next turn's prompt still
    matches), plus a few length-truncated turns.
  - The server's chat template (`/tokenize`, tools + effort 50) gives identical ids on 12/12 sampled rollouts.
- Held-out set: 334 eval-task traces from E030 `s20x60_hdrmin` (same harness, 120 eval tasks), 12.4M tokens.
- Engram hash parity: the fork's GPU hash ids equal the CPU census hashes (`engram_census.py`, a numpy port of the
  official `inference/engram.py`) on 2,188 x 24 ids in both layers.

## Engram row census (report: `reports/engram_census.html`)

- 6.04B lookups per Engram layer touch 223.6M distinct rows (58.2% of 384M); all observed rows = 110 GiB of the
  188.8 GiB table. 22% of touched rows are hit once, 61% at most 4 times; rows hit > 64K times carry 39% of
  lookups. 4-gram banks are 95% touched (saturated by collisions), 3-gram 67%, 2-gram 13%.
- Global top-frequency masks (`engram_census.py select`), held-out eval-trace hit rates:

| kept GiB | calib coverage | eval hit | eval generated | eval 2/3/4-gram | tokens with all 48 hit |
|---:|---:|---:|---:|---|---:|
| 8 | 85.6% | 81.2% | 71.7% | 95.9 / 81.6 / 66.0% | 62.3% |
| 16 | 89.3% | 84.8% | 77.2% | 96.9 / 85.2 / 72.3% | 65.4% |
| 32 | 93.3% | 88.9% | 83.3% | 97.6 / 88.6 / 80.5% | 68.3% |
| 64 | 97.6% | 94.0% | 91.1% | 98.3 / 92.6 / 91.2% | 72.2% |
| 96 | 99.5% | 97.3% | 95.9% | 98.9 / 96.0 / 97.0% | 77.3% |
| 110 (all observed) | 100% | 98.4% | 97.7% | 99.1 / 97.3 / 98.9% | 84.3% |

Kaggle 4xL4 host RAM leaves ~122 GiB for the pinned table plus a 2.9 GiB int32 map (E042), so all observed rows fit.

**Rows the eval uses but training never touched** (`engram_unseen.py`, all 2,004 Flash-1017 eval-task traces of the
6 lean2sub runs, 82.6M tokens): 11.6% of the 164.6M distinct rows the eval touches per layer are absent from the
census, but they carry only 1.7% of eval lookups (two-thirds of them are hit once). By token source
(`engram_unseen_kinds.py`, 21M tokens): task prompt (system + grids) ~0% unseen; tool outputs 0.85% (7% of all unseen
lookups: new digit runs from printed grids, ASCII renderings); model-generated text 2.4% (3-gram heads 4.1%; 93% of
unseen lookups: novel phrasing and code identifiers such as `ates with vertical`, `_obj(m):`, ` grid.tol`). The
census covers the task format completely; the gap is the open-ended tail of the model's own reasoning text.
Hit rates of the masks on the 82.6M-token set: 64 GiB 93.6%, all observed 98.3% (same as on the 12.4M-token set).

### Data sufficiency (held-out eval hit rate, census on half vs all of the calibration rollouts)

| budget | half data | all data |
|---|---:|---:|
| 16 GiB | 84.4% | 84.8% |
| 32 GiB | 88.4% | 88.9% |
| 64 GiB | 93.3% | 94.0% |
| all observed rows | 96.4% (89 GiB) | 98.4% (110 GiB) |

The frequency ranking has converged for a fixed budget (+0.5 pp from doubling the data); more data mostly adds
rare rows, which only matters near the ~122 GiB cap.

### Engram masks: teacher-forced NLL on held-out eval traces

`scripts/nll_candidates.sh` on server B (runtime masks), 32 E030 `s20x60_hdrmin` eval-task traces (<= 40k tokens),
514,113 model-generated target tokens, NLL in nats/token. Two runs of the full model differ by 0.0004
(0.6671 vs 0.6667: request batching changes the reduction order), so that is the noise floor.

| Engram | of table | NLL gen | vs full (mean of 2 runs) | top-1 agreement |
|---|---:|---:|---:|---:|
| full (2 runs) | 100% | 0.66705 / 0.66666 | - | 77.95% / 77.96% |
| all observed, 110 GiB | 58% | 0.66724 | +0.04% | 77.91% |
| 96 GiB | 51% | 0.66740 | +0.06% | 77.90% |
| 64 GiB | 34% | 0.66760 | +0.09% | 77.94% |
| 32 GiB | 17% | 0.66887 | +0.28% | 77.87% |
| 16 GiB | 8.5% | 0.67194 | +0.74% | 77.83% |

**Frozen Engram mask for REAP: `engram_top64gib`** (within noise of the full table, ~46 GiB less Kaggle RAM than
keeping every observed row). 110 GiB is the fallback.

## REAP statistics

- Replay (`scripts/replay_reap.py`): calibration rollouts sent one at a time as token prompts (`max_tokens=1`,
  unique `cache_salt`) to server B with the observer on and the frozen Engram mask; `reap_aggregate.py` checks that
  the recorded `input_ids` equal the request streams and that selections = 6 x positions in every layer.
- Pilot: 41 rollouts, 1.02M tokens, 169 s (6.0k tok/s). All checks pass; the least covered layer saw 367/384 experts.
- At E64 the scoring rules disagree sharply (pilot): plain REAP (conditional mean amplitude) keeps experts that are
  strong when chosen but rarely chosen, covering only 32% of the original routing selections (14% in the worst
  layer); frequency keeps 72% (55%), contribution total (sum of amplitudes) 70% (53%), router-weight mass 72%.
  Keep-set overlap REAP/frequency is 33%. Candidates are compared by NLL rather than assuming REAP.

- Main run: 377 more rollouts, 9.02M tokens (1,501 s). Pilot + main = 418 rollouts, **10.04M tokens**
  (4.28M prompt/tool inputs, 5.76M generated-token inputs); the sparsest layer saw 375/384 experts. Keep sets are
  stable: E64 sets from the 1M pilot and the 10M total overlap 96% for every rule.

### E64 scoring rules (one-shot), teacher-forced NLL on held-out eval traces

32 eval-task traces (a different 32-trace sample than the Engram table above: 419,452 generated target tokens),
all with the frozen 64 GiB Engram mask unless noted.

| model | rule | kept routing share (min layer) | NLL gen | top-1 |
|---|---|---:|---:|---:|
| full (384 experts, full Engram) | - | 100% | 0.6640 | 78.0% |
| 384 experts + Engram 64 GiB | - | 100% | 0.6646 | 78.0% |
| E64 | contribution total, decoder layers on generation targets | 66% | **0.857** | 73.6% |
| E64 | contribution total (sum of native amplitudes) | 70% (54%) | 0.871 | 73.3% |
| E64 | sqrt(freq) x mean amplitude | 64% | 0.873 | 73.3% |
| E64 | plain REAP (conditional mean amplitude) | 32% (14%) | 0.922 | 72.1% |
| E64 | router-weight mass | 72% (55%) | 1.006 | 70.5% |
| E64 | frequency | 72% (56%) | 1.064 | 69.4% |

Expert pruning to E64 costs ~+0.19 nats/token (+29%) with the best one-shot rule, while the Engram mask is free.

### Iterative pruning (E042's suggestion) and the E48 / E64 / E72 candidates

`scripts/iterative_prune.sh`: 384 -> 192 -> 128 -> 96 -> 72 -> 64 -> 48 with the same rule, 1.5M fresh calibration
tokens observed under each stage's mask (+ Engram 64 GiB), the next set chosen inside the current one. It is worse
than one-shot at every size, so the candidates are **one-shot "contribution total, decoder layers on generation
targets"** keep sets (`reap/keep_all/keep_contrib_dec_e{48,64,72}.npy`).

| experts | one-shot NLL | top-1 | iterative NLL | iterative top-1 |
|---:|---:|---:|---:|---:|
| 384 (full) | 0.664 | 78.0% | - | - |
| 72 | 0.822 | 74.3% | 0.879 | 73.1% |
| 64 | 0.857 | 73.6% | 0.903 | 72.6% |
| 48 | 0.962 | 71.5% | 1.037 | 69.9% |

### Physical export and parity

`scripts/export_pruned.py` -> `export/e64_oneshot_contribdec_eng64` (129.6 GB: backbone ~58 GB + compact Engram
~71 GB in `model-engram-*` files shared by E48/E64/E72), loaded on the fork with `patches/apply_compact_engram.py`
(TP4: **15.17 GiB weights per GPU**, KV 17.3M tokens on the RTX box). Teacher-forced NLL of the export on the same 32
traces: 0.85694 vs 0.85723 with runtime masks on the full checkpoint (diff 0.0003 < run-to-run noise 0.0004),
identical top-1 (73.63%).

### Free generation breaks below ~E96 (found by the first ARC evals)

- The first pruned ARC runs (E64 physical export on A, E48 runtime masks on B) solved **0/11 and 0/5**: every attempt
  ended after one turn without a valid tool call. Stopped after ~15 min.
- `scripts/decode_consistency.py`: greedy decode agrees with the teacher-forced argmax except for a few near-ties
  (full model too, fp8 KV numerics), so the decode path is fine.
- **Artifact:** switching runtime masks on a server with prefix caching reuses KV computed under the previous mask
  (shared system/tool-schema prefix) and corrupts generation. `gen_probe.py` therefore uses a per-config
  `cache_salt`. (The stopped E48 eval on B was contaminated this way; the E64 export on A was not.)
- Clean first-turn probes (1 simple tool prompt + 3 ARC prompts greedy + 2 T=1 samples each; valid = parsable
  `run_python`/`submit_answer` call):

| model (+ Engram 64 GiB) | valid tool calls | note |
|---|---:|---|
| full / Engram 64 GiB only | 9/10, 4/4 | |
| E192 / E128 / E96 (contrib_dec) | 9/10, 8/10, 10/10 | |
| E72 / E64 / E48 (contrib_dec) | 0/10, 0/10, 3/10 | reasoning fine, DSML block never closed |
| E72 protect 16 / 32 tool-block experts | 8/10, 6/10 | |
| E72 ranked by tool-block contribution only | 9/10 | |
| E64 / E48 protect 32 | 2/10, 0/10 | |

- Failure mode (E72, greedy, special tokens visible): the first `<｜DSML｜ invoke name="run_python">` is correct, but after
  `</｜DSML｜ invoke>` the model opens another invoke (`run_p_python`) instead of `</｜DSML｜ calls>` and runs to
  max tokens. The structural tokens are rare (tool-call blocks are ~6% of generated positions), so all-token NLL and
  contribution totals underweight them.
- Fix under test: expert statistics on tool-call-block targets (`scripts/special_stats.py`, 616k positions from the
  10M-token records) used to protect experts or to up-weight those positions (w = 10, 30) in the contribution score;
  screened with tool-block NLL (`nll_eval.py` now reports `nll_blk` / `top1_blk`).

### Tool-block screen (teacher-forced, 32 eval traces; `nll_blk`/`top1_blk` = targets inside tool-call blocks)

| model (+ Engram 64 GiB) | NLL gen | top-1 | NLL tool block | top-1 tool block | first-turn probe |
|---|---:|---:|---:|---:|---:|
| full | 0.664 | 78.0% | 0.128 | 95.4% | 9/10 |
| E72 contribution (decoder on gen) | 0.822 | 74.4% | 0.305 | 91.0% | 0/10 |
| E72 protect 16 tool-block experts | 0.824 | 74.4% | 0.253 | 92.4% | 8/10 |
| E72 tool-block weight w=10 | 0.878 | 73.2% | 0.205 | 93.5% | 2/10 |
| **E72 w=30** | 0.946 | 71.9% | 0.184 | 94.0% | **10/10** |
| E72 tool-block only | 1.069 | 69.5% | 0.185 | 93.9% | 9/10 |
| **E64 w=30** | 1.013 | 70.6% | 0.199 | 93.5% | **10/10** |
| E64 tool-block only | 1.127 | 68.5% | 0.193 | 93.6% | - |
| E48 w=30 | 1.153 | 68.5% | 0.235 | 92.7% | 0/10 |
| E48 tool-block only | 1.289 | 65.9% | 0.229 | 92.9% | - |

w = weight of tool-call-block positions in the contribution score (`special_stats.py select --weight`).

### Harness safety net `--rescue-code` (evals/run_deepseek_tools.py, opt-in)

A non-final turn without a valid tool call but with code (garbled DSML `code` parameter, JSON `"code"` field,
```python block, or plain code lines) becomes a `run_python` call; near-miss tool names (`run_ppython`) map to the real
tools; a turn with no call, no code and no answer grid gets a short reminder to use the tools (max 3 per attempt).
Evals on runtime masks send a per-run `cache_salt` (`eval_arc.sh` `SALT`).

### What the failures are (Sonnet trace review of the early pruned evals)

Not format-only. E64 shows semantic damage beyond the DSML syntax: hallucinated module names
(`from scipy import nddfficiency`, `scscpy`), ungrammatical reasoning, and code that would not parse even when
extracted. 57% of its tool calls return an error (full model: 1.7%), and garbled-word rates run ~80/1k words vs
~10/1k. E72 is much milder: once it is calling tools, its analysis and code look like the full model's (terser), with
a 19% tool-error rate. Its main problem was first-turn format failures, which the rescue fixes.

### Do the pruned models still see the puzzle's idea? (Sonnet review of the w=30 + rescue evals, partial)

Labels: A = main rule recognized, B = partial, C = wrong or none, D = garbled/no reasoning; judged against the full
model's trace and the task file; partial runs (E72 41 rows, E64 35 rows); counts are a reviewer's calls.

| run | outcome | read | A | B | C | D |
|---|---|---:|---:|---:|---:|---:|
| E72 w30 | correct | 12 | 11 | 1 | 0 | 0 |
| E72 w30 | incorrect | 14 | 7 | 6 | 1 | 0 |
| E64 w30 | correct | 9 | 9 | 0 | 0 | 0 |
| E64 w30 | incorrect | 18 | 2 | 10 | 3 | 3 |

E72's losses are mostly execution: half of its wrong answers state the right rule and miss by a few cells
(e.g. 4/784 cells on 7ed72f31, 3/676 on cbebaa4b), sometimes after degenerate stretches ("Need Need Need").
E64 loses the idea more often, with garbled reasoning ("Weod=w?? Need."), typo loops and turn-limit stops.

## ARC-AGI-2 eval (120 tasks / 167 pairs, 1 attempt, s20x60 + minimal header, self-hosted)

| model | pairs correct | pass@1 pairs | pass@1 task-weighted | notes |
|---|---:|---:|---:|---|
| API Flash-1017 (E030 s20x60_hdrmin), attempt 0 / 1 | 112 / 115 | 67.1% / 68.9% | 68.6% / 71.3% | reference |
| full 384 experts + full Engram (server A) | 110 | 65.9% | 67.4% | `e043_full_a1` |
| E64 + Engram 64 GiB (physical export) | 0 of 11 | | | stopped: no valid tool calls (see above) |
| E48 + Engram 64 GiB (runtime masks) | 0 of 5 | | | stopped; also prefix-cache contaminated |
| E72 w30 + Engram 64 GiB + `--rescue-code` (runtime masks) | 49 | 29.3% | 28.6% | complete |
| E64 w30 + Engram 64 GiB + `--rescue-code` | 30 | 18.0% | 17.5% | complete (44 output_budget, 17 max_turns) |
| E48 w30 + Engram 64 GiB + `--rescue-code` | | | | running (long degenerate turns) |
| 384 experts + Engram 64 GiB (control) | | | | running |

## Artifacts (private HF, account chankhavu)

- `chankhavu/dsv41-flash1017-e043-pruning-artifacts` (dataset): Engram counts/masks/geometry, REAP stats and keep
  sets, token streams (calibration + eval traces), raw observer records (20 GB), NLL results, code snapshot.
- `chankhavu/DeepSeek-V4.1-Flash-1017-arc-engram64`: shared compact Engram files + `engram_index.json`.
- `chankhavu/DeepSeek-V4.1-Flash-1017-arc-E64` (uploading), `-arc-E48`, `-arc-E72`: backbone checkpoints.

## Status / next

1. NLL of Engram masks vs full table on the eval traces (server B), ARC baseline of the full model (server A).
2. Freeze the Engram mask, REAP replay pilot (~1M tokens) with it, then E64/E72 keep sets, NLL + ARC evals.
3. Export the chosen E64 (HF format, sliced router) + compact Engram (values/scales + int32 map) for E042.

## Incidents

- The image's auto-started SGLang service held GPU memory; stopped with `supervisorctl stop sglang model-ui`.
- The first download went to the RAID1 disk (~120 MB/s flush) and stalled; switched to `/dev/shm`. A watchdog that
  judged progress by file size killed healthy xet downloads (xet buffers in RAM); fixed to watch NIC bytes.
- `pkill -f` patterns matched my own shells twice (remote and local); the local AGENTS.md rule forbids killing by
  name on this host: use PIDs only.
- Observer bug, fixed before any statistics: non-local rows of the EP `moe_sum` input are uninitialized; masking by
  multiplication let NaN through (`where` now). Amplitudes are stored in fp32 (fp16 overflowed in deep layers).
- The full-workspace GitHub sync is blocked by the secret scan on `.SECRETS.yaml.swp` (an open vim swap file);
  E043 was pushed with only its own files.
