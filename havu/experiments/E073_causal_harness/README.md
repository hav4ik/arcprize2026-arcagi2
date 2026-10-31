# E073: causal E128 harness against Manh V12 on 4×L4

Started 2026-10-31 UTC, on the user's allocated Runpod `ccn5ns0b8emb3o` (`yawning_salmon_cobra`, SSH port 29668,
157.157.221.29). This experiment owns that pod's workloads; E072's runner is stopped there. Other pods are untouched.
User objective: improve on Manh V12 using
`chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`.

## Baseline and success criteria

The gold standard is Manh V12, original notebook SHA256 `c47be5a819ad…`: **80.83% task-weighted pass@2 on the
240-task Kaggle-version workload** (120 evaluation originals plus 120 transformed copies), completed at 10.17 pod hours
with no failures. On the 120 originals its pass@2 is **81.81%**. See E062 `per_layer/phase_h/manh_v12/README.md`.

E073 first measures a new **same-pod** V12 control on the **120 Kaggle evaluation originals**, two independent attempts
per task. Its harness, settings, model, and server cell remain V12's; the only functional file edit is the mandated bounded
model lookup. Other bounded notebook lookups follow `input_paths.py`. Pin-check failures abort the experiment in rerun mode
instead of silently falling back to the kit's v3 harness. Frozen files and hashes are in `frozen/receipt.json`.

Selection uses the existing E072 stratified development/holdout split (60/60, seed 2026). It never chooses prompts, policies,
or retries using test-answer correctness on the holdout. A successful accuracy validation needs the paired holdout 95% task
bootstrap CI above zero; otherwise label it inconclusive/loss. A full 240-task candidate replica at V12's deadline is still
required before claiming an improvement in deadline score or recommending a Kaggle submission. No submission is automatic.

## Hypotheses and first batch

The current all-image replica has completed its first attempts at 71.25% task-weighted pass@1 versus V12's 73.12%, and trails
its score curve. Adding images to every attempt is therefore not the default hypothesis. E066's offline text/image mix suggested
complementary errors, while re-tokenizing image transcripts breaks prefixes on roughly 29% of turns. This batch tests both
complementarity and exact token history, without the unproven replay CUDA graphs or watermark-reload-only serving changes.

| Phase | Work | Purpose |
|---|---|---|
| Gold | 120 originals × 2 text attempts, original E128 and wheel g8cf0c0664b | Same-host V12 control, accuracy and throughput |
| Development | 60 dev tasks × 3 fresh draws: text, text, token-exact text+image, causal E128 | Share the first text draw to compare two-answer policies with less sampling noise |
| Selection | TT = draws 0/1; TI = draws 0/2; retry = TT plus image only when neither text attempt train-passes | Select by dev pass@2, subject to output-token cost ≤1.15× gold; preserve holdout if none improves |
| Holdout | 60 untouched tasks with the one selected policy, causal E128 | Independent paired accuracy validation; conditional retries are actually scheduled here |

All draws use the unchanged sub80s40 + minimal-header prompt, 40 turns, multi-test budget 0.5, train checks, auto-repair,
context limit 163840, V12 admission gate and worker count. Image turns use the existing PR #3 implementation through
`/inference/v1/generate`; text turns keep V12's exact token path. No prior answer or thinking is shown to another attempt.
All scoring selects **at most two distinct grids**, training-pass first and then attempt order, including the retry arm.
It never reports an oracle over three guesses as pass@2.

The checkpoint is pinned to **209c810a747c043c2aca474843de8e357f920ef6**, with 121 non-Engram file sizes and LFS hashes
checked against HF metadata before launch. `stage_causal.py` verifies all 137 staged files by SHA256, retaining the 16 existing
Engram tables. After the control finishes, only the re-downloadable old non-Engram weight files are removed to fit the new
checkpoint on the 200 GB disk. Its metadata/small files and all control results remain. Tokens travel over SSH stdin only.

The causal runs use Manh's wheel **g22df163d2d**, already correctness-tested for PR #3. This is a recorded stack difference
from gold; the causal TT control separates policy effects within the new stack/model. Server flags otherwise match V12,
plus 32 image slots and `--disable-chunked-mm-input`. The model configuration must not enable bidirectional image attention.

## Operation and bounds

`build.py` creates frozen notebooks and workloads. `test_policy.py` checks independent per-attempt formats, conditional retry
eligibility, multi-test scoring, two-answer ranking, pins and baseline equivalence. Existing `test_v12_img_te.py` exercises the
token-exact image stream and guarded force-deadline behavior. Notebook install/launch and memory monitoring use E072's tested
runner, sampler and reaper; no indefinite filler queue is started.

`run_batch.py` is a local detached controller. It waits for `gold`, verifies its completion, stages the causal checkpoint,
runs development, selects using dev only, then optionally runs the selected holdout policy. It synchronizes small results every
five minutes and archives executed notebooks and traces after each job. Errors or incomplete arms block advancement rather
than becoming successful comparisons. Outputs: `runs/controller.log`, `runs/{gold,dev,holdout}/`, `runs/selection.json`, and
`runs/holdout.json`; `runs/blocked.txt` names a failed prerequisite.

Expected first batch roughly **12 hours / $28.32** of the existing $2.36/hour pod; controller bound **14 hours / $33.04**,
with per-job bounds 6.5/4.5/4 hours. GPU time is already rented; no API evaluation calls or new instances are authorized by this
controller. It never stops/deletes the pod. At the batch limit it stops only its own notebook run. The pod remains billed while
idle after completion, per the user's instruction that rented pods must not be stopped.

Beyond the first batch: if accuracy is promising, qualify the exact selected notebook on all 240 tasks under the 12-hour
deadline, calibrate pod speed against V12 at matched load, and measure healthy prefix hits, CPU restores, preemptions,
allocator retries, cap minus reserved, host RAM, and final answer coverage. A higher accuracy point estimate alone is insufficient.
