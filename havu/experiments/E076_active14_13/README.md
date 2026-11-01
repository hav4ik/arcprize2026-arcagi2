# E076: E065-calibrated active-expert budget near 14B

**Final publication package:** [code, metadata and SFT transfer instructions](releases/final_2026-10-31/README.md),
[downloadable archive](releases/E076_active14_final_code_metadata_2026-10-31.tar.gz).
Hugging Face publication: [verified private revision d92fb468](https://huggingface.co/chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview/commit/d92fb46867667f5fa0092b4bd4708190012db74f).
This version includes final text and image results, exact schedules, all five
evaluated arms, calibration provenance, frozen harness and serving code, and
integrity/rescoring tools. The earlier image-only handoff archive remains unchanged.

**Final tools-harness evaluation, completed 2026-10-31 18:57 PDT:** all three
required new arms are complete. Each reported score below uses all **120 Kaggle
ARC-AGI-2 evaluation tasks, two draws**, task-weighted. The 14.58B text reference
was explicitly skipped. Current causal E128 weights are identical across active
expert schedules; the ~16B reference is k6, not the original unpruned model.

| Modality | E128 ~16B pass@1 / pass@2 | E074 ~14.58B pass@1 / pass@2 | E076 ~14.02B pass@1 / pass@2 |
|---|---:|---:|---:|
| Text + tools (primary) | 72.29% / 81.81% | Skipped | 69.24% / 79.44% |
| Image + tools | 72.43% / 83.47% | 70.83% / 81.81% | 71.67% / 83.89% |

The final 14.02B text pass@1 difference is **-3.06 percentage points**, outside
the user's at-most-two-point loss target on this run; pass@2 is -2.36pp.
The paired-task bootstrap 95% interval for text pass@1 is [-8.68,+2.78]pp:
this run neither establishes noninferiority nor precisely estimates the true
loss. Image pass@1 changes -0.76pp versus E128 / +0.83pp versus E074;
image pass@2 changes +0.42pp / +2.08pp. Image retention does not establish text
retention. The -1.60pp image result belongs to **E074 14.58B versus E128**.

Text candidate/E128 generated tokens mean/p50/p90: 42,380/31,946/94,318 versus
41,251/31,454/86,629. Tool calls total/mean: 4,408/18.37 versus 4,535/18.90;
after-auto-repair tool errors/rejections: 544/4,408 (12.34%) versus
461/4,535 (10.17%). Cap stops: 12 versus 10; attempts missing any test answer:
11 versus 8. Multi-test pair records are deduplicated by task/attempt.
Server/harness failed requests are zero in both text arms; no preemptions.
Last candidate cache sample at 18:57:13 PDT: 92.28% cumulative / 95.09% recent
prefix hits. This sample preceded the final request completion.

Text generation throughput over matching elapsed minutes 5-155 is
1,089 versus 1,061 tok/s (**+2.61%**). Full-run output throughput is
1,060 versus 1,044 tok/s (+1.59%); candidate generated 2.74% more tokens and
took 159.87 versus 158.08 minutes (**1.13% longer**). Image matching minutes
5-150 remains +7.97% versus E128 / +2.89% versus E074. These are real-harness
rates with changing batches, not isolated-kernel measurements.

Only `arc-e076-report.timer` was stopped and verified inactive after completion.
The GPU rental remains running. No smaller-model pruning job was launched during
final reporting. Future approximate 12-13B work remains governed by the mixed
training-only data policy and mandatory protection of sensitive k6 layers.
[Final snapshot, task IDs, source hashes and paired comparisons](analysis/completed_tools_report.json),
[final usage tables](analysis/summary_tables.md). Dated partial snapshots below
are retained as history and superseded by this final comparison.

**Primary acceptance modality, user clarification 2026-10-31:** prioritize
**text-only + tools** when judging retained performance. The target remains at
most two percentage points of task-weighted pass@1 loss versus causal E128 k6.
Completed image results provide secondary evidence; they do not establish text
retention. Report exact matched subsets while text is incomplete. Keep both
modalities in training-only sensitivity checks and preserve protected k6 layers.

**Early feasibility evidence:** [training-only metric audit](analysis/early_feasibility_signals_20261031.json)
records the evidence available before full14B ARC evaluation. On paired E065
select24, seed14 generated-token NLL rose 1.06% image / 1.21% text versus k6;
tool-block NLL rose 3.75% / 4.10%. Historical uniform4 (~13.17B, including reductions
of now-protected layers) rose 3.09% / 3.21% generated and 6.98% / 6.95% tool NLL.
Seed14's per-rollout generated-NLL p90 increase was 2.33% image / 2.67% text,
versus uniform4's 5.21% / 6.52%. These are relative loss changes, not accuracy
percentage points or an independently validated prediction rule. The twelve-task
original-training diagnostic had no seed14 cap stops or missing answers in
either modality. Small/familiar tasks and one draw limit that evidence. The
candidate's independent audit32 was not completed; no universal NLL threshold
certifies the two-point accuracy target. No new calibration jobs were launched
for this retrospective analysis; finish the current14B eval first.

**Latest queue decision, 2026-10-31 PDT:** the user skipped the 14.58B text-only
reference and clarified that the current **14.02B evaluations must finish first**,
before building the next approximate 12-13B candidate. Only the waiting
`e076_reference1458` supervisor was stopped; `e076` and its current text eval
remain running unchanged. A persistent skip guard prevents the reference queue
from launching if restarted. Its receipt is `results/dual_baseline_extension.json`
(`state=skipped`). No smaller-model calibration or GPU job was launched.

[Current queue policy](configs/eval_queue_policy.json) supersedes earlier queued
14.58B text references and four-arm completion rules below. The three required
new arms are `preview14_img`, `baseline_txt`, and `preview14_txt`; all are complete.
Images retain both E128 and 14.58B references; text compares with E128 only.
The 30-minute report timer was stopped after these three required arms completed
at approximately 18:57 PDT. The skipped reference adds no waiting time.
The GPU rental remains running.
Subsequent 12-13B work uses mixed original-training/NVARC training-parent data,
keeps sensitive layers at six, treats size as a soft target, and evaluates text first.

**E128 identity and historical results, checked 2026-10-31:** all current E074/E076
arms use the E066 **causal image-recalibrated E128 (C1f)** checkpoint at
`209c810a747c043c2aca474843de8e357f920ef6`, including text-only evaluation.
The 14.58B and 14.02B variants retain its weights and 128 stored experts per layer;
only active top-k changes. This is distinct from E043's original tool E128 and
E066's earlier image-calibrated C1. [Historical scores rescored from raw predictions](analysis/historical_e128_scores_20261031.json)
include source hashes, own-version 120-task scores, and the 114 identical-task
subset. Older E066 notes claiming 115 identical tasks are superseded; six whole
task versions differ, including d8e07eb2's training pairs. Harness differences
remain even on the 114 subset, so these are historical references, not isolated
model-effect measurements.

**E128 text+tools control has completed:** all 120 Kaggle tasks x2, task-weighted
pass@1/pass@2 **72.29%/81.81%**, zero failed requests; eval time 9,484.54 seconds.
The 14.02B text candidate has also completed; the user skipped 14.58B text.
Historical tables below retain their explicitly dated partial snapshots.

**Training/calibration boundary, latest user clarification 2026-10-31:** use a
mixture of **original ARC tasks from the 1,000-task training set and NVARC
training-parent rollouts** for training, calibration, training diagnostics and
pruning-setting selection. This supersedes the earlier NVARC-only restriction.
E065 `arc2_train`, `nvarc_b0` and `nvarc_b1` image/text batches are permitted.
The 120 Kaggle evaluation tasks and their derivatives are evaluation-only;
exclude `nvarc_full` and `arc2_evaluation6`. Evaluation IDs may be read for
contamination checks, never their outcomes for fitting or setting selection.
[Policy](configs/training_data_policy.json),
[historical provenance audit](analysis/training_data_boundary_audit_20261031.json)
(its NVARC-only policy interpretation is superseded by this clarification).
The eight frozen E074/E076 modality/pool indices have zero evaluation-parent
overlap, but include original ARC training rollouts: E074 12/32 per modality;
E076 screen 8/16, select 6/24, audit 9/32. These mixed pools comply with the
revised source boundary; no NVARC-only rebuild is required. Record source
weights and report original-training and NVARC diagnostics separately and
jointly; no fixed mixture ratio has been prescribed. Keep originals and their
synthetic descendants in the same parent-family split. Prior model exposure
limits generalization claims, but does not disqualify training originals.
The local `arm.py` validates both allowed sources before server startup;
original-task diagnostics are allowed after checking the training registry.
All six E076 pool/modality indices and the 12-task original diagnostic pass;
eight boundary tests pass. [Revised validation receipt](analysis/training_data_boundary_clarification_20261031.json).
This guard was not deployed to the running GPU queue, which continues its frozen
evaluations unchanged. The historical SFT handoff archive is preserved; future
SFT calibration still needs checkpoint-specific profiling and provenance.

**Literature revisit:** [updated review](../../prior_works/moe_active_parameter_pruning.md)
and eight per-method notes distinguish published evidence from our adaptations.
Priorities after the current evals: mixed original-training/NVARC re-profiling; joint top-k and
reference-mass/scalar normalization checks; recovery distillation if needed.
Token-dependent routing is a later serving project. This review launches no
training/calibration jobs and changes no eval settings.

**Interim text check, 2026-10-31 18:52 PDT:** candidate236/240 attempts,
with four remaining. On117 matched Kaggle tasks x2 (234 logical attempts/model),
14.02B task-weighted pass@1/pass@2 is70.16%/80.20% versus E12873.29%/82.62%:
pass@1 delta-3.13pp. This remains a completion-biased subset, not a final
acceptance decision. Post-repair tool errors12.11%vs9.77%; caps12vs10,
missing-answer attempts11vs8. Equal-window minutes5-150 generation throughput
1,114vs1,084tok/s (+2.78%). Current serving has zero failed requests/preemptions;
prefix hits92.24% cumulative/96.51% recent. No new jobs launched.
[Snapshot](analysis/scheduled_update_20261031_1852.json).

**Scheduled update, 2026-10-31 18:41 PDT:** candidate text+tools is at
226/240 attempts (94.2%), with 14 still running. On the same **112 Kaggle tasks
with both draws complete** (224 logical attempts/model), task-weighted
pass@1/pass@2 is **71.95%/81.99% for 14.02B versus 74.63%/84.23% for E128 k6**.
The partial pass@1 difference is **-2.68 percentage points**, currently outside
the user's at-most-two-point loss target. This is still completion-biased:
the eight tasks lacking both candidate draws are excluded, so the full-set
acceptance decision remains pending. Earlier positive partial differences did
not establish retention. Images remain secondary to text acceptance.
Matched generated tokens mean/p50/p90 are 38,684/30,366/80,000 versus
38,212/29,024/78,896; tool calls total/mean are 3,951/17.64 versus 4,019/17.94.
After-auto-repair tool errors/rejections are 484/3,951 (12.25%) versus
395/4,019 (9.83%); cap stops are 11 versus 8 and attempts missing any answer
are 10 versus 6. Multi-test records are deduplicated by task/attempt.
Server/harness failed requests remain zero. Matching elapsed minutes 5-140
give 1,152 versus 1,125 generation tokens/s (+2.43%); unfinished-attempt
tokens and evolving batch sizes are included, so this is not a kernel benchmark.
The 14.58B text reference is **skipped**, not queued. Candidate cache at
18:40:02 PDT is 92.02% cumulative / 92.54% recent prefix hits, 1.95% KV use,
14 running / 0 waiting requests, zero preemptions and zero server-error log lines.
ETA remains approximately **19:00 PDT**, allowing until **19:30 PDT** for
long attempts. Full 120-task image results are unchanged: pass@1/pass@2
14.02B 71.67%/83.89%, E128 72.43%/83.47%, 14.58B 70.83%/81.81%; matching
minutes 5-150 throughput +7.97%/+2.89% versus the two references.
Two of three required new arms are complete; the timer and GPU rental continue.
No eval settings changed and no new GPU/pruning jobs launched.
[Frozen snapshot and provenance](analysis/scheduled_update_20261031_1841.json).

**Scheduled update, 2026-10-31 18:11 PDT:** candidate text+tools is at
180/240 attempts (75%). On the same **86 Kaggle tasks with both draws finished**
(172 logical attempts/model), task-weighted pass@1/pass@2 is
**76.45%/84.88% for 14.02B versus 74.71%/82.56% for E128 k6**.
This is a completion-biased subset: faster/easier attempts finish first, so it
does not establish full-set retention or the two-point acceptance target.
Matched generated tokens mean/p50/p90 are 35,242/27,535/80,000 versus
35,016/24,550/77,951; tool calls total/mean are 2,899/16.85 versus 2,885/16.77.
After-auto-repair tool errors/rejections are 367/2,899 (12.66%) versus
307/2,885 (10.64%); cap stops are 8 versus 6 and attempts missing any test
answer are 7 versus 5. Multi-test records are deduplicated by task/attempt.
These are model/tool outcomes, distinct from zero failed evaluation requests.
Matching elapsed minutes 5-110 give 1,194 versus 1,169 generation tokens/s
(+2.17%); this includes unfinished-attempt tokens and changing batches.
The 14.58B text reference is **skipped by user**, not queued.
At 18:10:59 PDT the candidate has 91.55% cumulative / 92.13% recent prefix
hits, 4.50% KV use, 60 running / 0 waiting requests, no preemptions and no
server-error log lines. Remaining eval ETA is **19:00-20:00 PDT**, provisional
because long attempts remain. All 120-task image comparisons are complete:
14.02B pass@1/pass@2 71.67%/83.89%, E128 72.43%/83.47%, and 14.58B
70.83%/81.81%; matching minutes 5-150 throughput is +7.97%/+2.89% versus
the two references. Images are secondary to text acceptance. Two of three
required new arms are complete; the report timer and GPU rental continue.
No eval settings were changed and no new GPU/pruning job was launched.
[Frozen snapshot and source hashes](analysis/scheduled_update_20261031_1811.json).

**Scheduled update, 2026-10-31 17:42 PDT:** candidate text+tools is at
126/240 attempts (52.5%). On the same 55 Kaggle tasks with both draws finished
(110 logical attempts/model), task-weighted pass@1/pass@2 is
79.55%/89.09% versus E128 75.91%/83.64%. Completed-subset bias remains;
the unfinished long candidate attempts are excluded. Candidate output-token
mean/p50/p90 is 29,789/25,134/65,109 versus 30,223/24,071/64,520; tool calls
average 15.22 (1,674 total) versus 15.65 (1,722); post-repair tool
errors/rejections are 199/1674 (11.89%) versus 162/1722 (9.41%). Cap stops
are 5 versus 4, and attempts missing any answer are 4 versus 3.
The 14.58B text reference remains queued. Equal elapsed minutes5-80 generation
throughput is 1199.33 versus 1174.84 tok/s (+2.08%); no text comparison
with 14.58B is available yet. These rates include unfinished-attempt tokens.

Candidate cache at 17:41:56 PDT: 91.29% cumulative / 90.88% recent-five-minute
hits, KV 4.96%, 64 running, zero waiting/preemptions/server-error log lines
or failed eval requests. The completed120-task image comparison remains
14.02B71.67%/83.89%, E12872.43%/83.47%, 14.58B70.83%/81.81%; equal-window
minutes5-150 throughput gains remain +7.97%/+2.89%. Provisional ETAs remain
candidate text19:00-20:00, all text22:00-23:00 PDT. No settings or queues changed.
[Snapshot and provenance](analysis/scheduled_update_20261031_1742.json).
The local-only `scripts/capture_tools_snapshot.py` now reproduces snapshot
statistics/tables from synced logs; it never controls GPU jobs or the timer.

**Scheduled update, 2026-10-31 17:12 PDT:** candidate text+tools is at
73/240 attempts. On the same 33 Kaggle tasks with both draws finished
(66 logical attempts/model), task-weighted pass@1/pass@2 is
86.36%/92.42% versus E128 85.61%/89.39%. This is still
completion-biased, not a full-set accuracy estimate. Candidate output-token
mean/p50/p90 is 22,341/19,174/44,756 versus 24,341/20,224/46,250; tool calls
average 12.55 (828 total) versus 13.18 (870); post-repair tool
errors/rejections are 89/828 (10.75%) versus 71/870 (8.16%). The candidate
has one turn-cap stop and one missing-answer attempt; E128 has zero of each
on this subset. The 14.58B text reference remains queued. Equal elapsed
minutes5-50 generation throughput is 1230.52 versus 1195.46 tok/s
(+2.93%); no text comparison with 14.58B is available yet.

Candidate cache at 17:11:23 PDT: 90.57% cumulative / 93.97% recent-five-minute
hits, KV 4.99%, 64 running, zero waiting/preemptions/server-error log lines
or failed eval requests. The completed120-task image comparison remains
14.02B71.67%/83.89%, E12872.43%/83.47%, 14.58B70.83%/81.81%; equal-window
minutes5-150 throughput gains remain +7.97%/+2.89%. Provisional ETAs remain
candidate text19:00-20:00, all text22:00-23:00 PDT. No settings or queues changed.
[Snapshot and provenance](analysis/scheduled_update_20261031_1712.json).

**Scheduled update, 2026-10-31 16:42 PDT:** candidate text+tools has 27/240
attempts finished. On the same ten Kaggle tasks with both draws complete
(20 logical attempts/model), candidate and E128 both score task-weighted
pass@1/pass@2 100%/100%. Strong completed-subset bias applies; this small,
quickly finished subset cannot establish retained accuracy. Candidate output
mean/p50/p90 is 9,832/9,208/17,974 versus E128 11,604/9,322/22,425; tool calls
8.25 (165 total) versus 9.75 (195); post-repair tool errors/rejections 17/165
(10.30%) versus 10/195 (5.13%). Both have zero caps/missing-answer attempts on
that subset. The 14.58B text reference is queued, so its matched values remain
unavailable. Candidate/E128 generation rates over equal elapsed minutes5-20 are
1275.21/1231.69 tok/s (+3.53%); early windows include unfinished attempts and
are not final full-run throughput.

E128 text is complete120 tasks x2: pass@1/pass@2 72.29%/81.81%, mean output
41,251 tokens, 18.90 tool calls, 10.17% post-repair tool errors, ten cap stops,
eight attempts missing any answer, zero failed requests. Candidate cache at
16:42:10 PDT: 87.66% cumulative / 89.77% recent-five-minute hits, KV3.54%,
62 running, zero waiting/preemptions/error log lines. Completed120-task image
results remain 14.02B71.67%/83.89%, E12872.43%/83.47%, 14.58B70.83%/81.81%;
minutes5-150 throughput gains remain +7.97%/+2.89%. Provisional ETAs: candidate
text19:00-20:00, all text22:00-23:00 PDT. No eval settings or queues changed.
[Snapshot and provenance](analysis/scheduled_update_20261031_1642.json).

**Scheduled update, 2026-10-31 16:12 PDT:** E128 text+tools is at 234/240
attempts (97.5%); six remain. The same 116 Kaggle tasks with both draws complete
(232 attempts) have task-weighted pass@1/pass@2 73.78%/83.05%. This remains a
completion-biased subset: the final long attempts are excluded. Output-token
mean/p50/p90 is 39,166/29,904/80,000; tool calls average 18.349 (4,257 total),
with 418 errors/rejections after auto-repair (9.82%), ten cap stops (five output
budgets, five turn limits), and eight attempts missing at least one answer.
These are attempt outcomes, not infrastructure failures: zero failed eval
requests, server-error log lines, or preemptions. Cache at 16:11:33 PDT is
92.58% cumulative / 95.50% over five minutes, KV 0.85%, five running and zero
waiting. Both pruned text arms remain queued; no matched text comparison yet.
Completed image+tools remains 120 Kaggle tasks x2 for all three models:
14.02B pass@1/pass@2 71.67%/83.89%, E128 72.43%/83.47%, 14.58B 70.83%/81.81%.
Equal elapsed minutes 5-150 throughput remains 1083 versus 1003/1053 gen tok/s
(+7.97%/+2.89%). ETAs remain provisional: E128 text 16:15-17:00, candidate
19:00-20:00, all arms 22:00-23:00 PDT. The last few attempts can dominate the
remaining time; no settings or queues were changed.
[Snapshot and provenance](analysis/scheduled_update_20261031_1612.json).

**Scheduled update, 2026-10-31 15:45 PDT:** E128 text+tools has 199/240
attempts complete (82.9%). The 96 Kaggle tasks with both draws complete
(192 attempts) score task-weighted pass@1/pass@2 76.56%/84.38%; this is a
completion-biased subset, not a full-set estimate. On those same attempts,
output-token mean/p50/p90 is 34,716/24,550/76,373; tool calls average 16.781
(3,222 total), with 337 post-repair errors/rejections (10.46%), six cap stops,
and five attempts missing at least one answer. Failed eval requests and
server-error log lines remain zero. Latest cache sample (15:43:51 PDT) shows
91.88% cumulative / 93.97% recent-five-minute prefix hits, KV 4.19%, 42 running,
zero waiting and zero preemptions. Both pruned text arms remain queued.
The completed 120-task image comparison is unchanged: candidate
pass@1/pass@2 71.67%/83.89% versus E128 72.43%/83.47% and 14.58B
70.83%/81.81%; equal elapsed minutes 5-150 generation throughput is
1083 versus 1003/1053 tok/s (+7.97%/+2.89%). Provisional ETAs remain
E128 text 16:15-17:00, candidate 19:00-20:00, all text arms 22:00-23:00 PDT.
About 126 minutes of baseline text have elapsed; its long tail is unfinished.
[Snapshot and provenance](analysis/scheduled_update_20261031_1545.json).

**Scheduled update, 2026-10-31 15:11 PDT:** E128 text+tools reached 144/240
attempts (60%), with 65 Kaggle tasks having both draws complete (130 attempts).
Task-weighted pass@1/pass@2 on that completion-biased subset is 79.62%/84.62%;
output-token mean/p50/p90 is 28,866/23,832/61,693; tool calls average 16.262
(2,114 total), with 233 post-repair errors/rejections (11.02%). The included
attempts have three cap stops (two turn limits, one output budget) and three
attempts missing at least one answer. Neither indicates an infrastructure error:
failed eval requests, server-error log lines and preemptions remain zero.
Prefix hits are 91.74% cumulative / 92.54% over five minutes, KV 4.90%, 64
running, zero waiting. Candidate and 14.58B text arms remain queued. Full
120-task image accuracy and matched minutes 5-150 throughput are unchanged
(candidate +7.97% versus E128, +2.89% versus 14.58B). Provisional ETAs remain
E128 text 16:15-17:00, candidate 19:00-20:00, all arms 22:00-23:00 PDT.
[Exact snapshot and provenance](analysis/scheduled_update_20261031_1511.json).

**Scheduled update, 2026-10-31 14:41 PDT:** E128 text+tools reached 90/240
attempts, with 41 Kaggle tasks having both draws complete (82 attempts).
Task-weighted pass@1/pass@2 on that completion-biased subset is 82.32%/86.59%;
output-token mean/p50/p90 is 24,835/22,256/52,371; tool calls average 12.976
(1,064 total), with 104 post-repair errors/rejections (9.77%). All 82 included
attempts submitted, with zero cap stops or missing answers. Both pruned text
arms remain queued; there is no matched text candidate comparison yet.
Text health: zero failed requests, server-error log lines or preemptions;
prefix hits 90.94% cumulative / 92.35% over five minutes, KV 4.75%, 64 running,
zero waiting. The full 120-task image comparison is unchanged: candidate
pass@1/pass@2 71.67%/83.89% versus E128 72.43%/83.47% and 14.58B
70.83%/81.81%; matched minutes 5-150 throughput 1083 versus 1003/1053 tok/s
(+7.97%/+2.89%). Provisional ETAs remain E128 text 16:15-17:00, candidate
19:00-20:00, all three arms 22:00-23:00 PDT. The observed text progress is
consistent with this range, but the long tail remains unfinished.
[Exact snapshot and provenance](analysis/scheduled_update_20261031_1441.json).

**Scheduled update, 2026-10-31 14:11 PDT:** E128 text+tools reached 46/240
attempts, with 20 Kaggle tasks having both draws complete (40 attempts).
Task-weighted pass@1/pass@2 on those same tasks is 83.75%/87.50%; output-token
mean/p50/p90 is 16,454/16,895/29,169; tool calls average 10.725 (429 total), with
26 errors/rejections after auto-repair (6.06%). All 40 attempts submitted, with
zero cap stops or missing answers. Completed-subset bias remains substantial;
the candidate and 14.58B text arms are queued and cannot yet be compared.
Text health: zero failed requests, server-error log lines or preemptions;
prefix hits 88.67% cumulative / 91.59% over five minutes, KV 3.82%, 63 running,
zero waiting. The completed 120-task image comparison is unchanged: 14.02B
pass@1/pass@2 71.67%/83.89% versus E128 72.43%/83.47% and 14.58B
70.83%/81.81%; matched elapsed minutes 5-150 throughput 1083 versus 1003/1053
tok/s (+7.97%/+2.89%). Provisional text ETAs remain 16:15-17:00 for E128,
19:00-20:00 for the candidate, and 22:00-23:00 PDT for all three arms.
[Exact snapshot and provenance](analysis/scheduled_update_20261031_1411.json).

**Scheduled update, 2026-10-31 13:55 PDT:** E128 text+tools has 17/240
attempts complete; seven Kaggle tasks have both draws (14 attempts). On that
completion-biased subset, task-weighted pass@1/pass@2 is 96.43%/100%; mean output
8,737 tokens, 7.50 tool calls, 4.76% post-repair tool errors/rejections, zero cap
stops or missing answers. The 14.02B and 14.58B text arms are queued, so there is
no matched text comparison yet. Current text cache hits are 86.16% cumulative /
90.36% over five minutes, KV 3.05%, 64 running, zero waiting/preemptions and zero
failed eval requests. Provisional ETA: E128 text 16:15-17:00, candidate 19:00-20:00,
all three text arms 22:00-23:00 PDT, inferred from the image runtimes and subject
to the text long tail.

Completed image accuracy remains 71.67%/83.89% for 14.02B versus E128
72.43%/83.47% and 14.58B 70.83%/81.81%, on all 120 Kaggle tasks x2.
Using the same elapsed minutes 5-150 in each image run, generation rates are
1083.09 versus 1003.18/1052.72 tok/s (+7.97%/+2.89%); full-run end-to-end rates
remain +9.46%/+2.95%. Missing-answer attempts counted across **any** test input
are 7/240 for the candidate versus 9/240 and 4/240; the earlier pair-0-only
candidate count was six. The image server logged one output-handler traceback
during deliberate shutdown after all attempts finished, with no failed eval
requests. Exact snapshot: [scheduled report](analysis/scheduled_update_20261031_1355.json).

**Final summary tables, refreshed 2026-10-31 18:59 PDT:** [image+tools and text+tools](analysis/summary_tables.md)
include task-weighted pass@1/pass@2, generated-token mean/p50/p90, tool-call
mean/total and tool-error rates for both references and the candidate. Exact
task subsets, input hashes and definitions are in [the snapshot metadata](analysis/summary_tables.json).
The completed image table supersedes the partial 13:25 usage audit below;
the text comparison now covers all 120 matched tasks with both draws complete. The 14.58B text reference is skipped.

**Reproduction and SFT handoff:** [methodology and transfer instructions](handoff/README.md),
[machine-readable metadata](handoff/metadata.json), and
[validated archive](E076_active14_SFT_handoff_2026-10-31.tar.gz).
The archive contains the schedules, exact calibration inputs/provenance, frozen
harness, raw image results, runtime metadata and SFT config-overlay helper.
All 44 frozen assets and 26 prepared-data hashes passed verification; secret
scans found zero hits. [Validation receipt](analysis/handoff_validation.json).
The existing measurements apply to E128; an SFT checkpoint needs fresh calibration
and matched evaluation before claiming the same accuracy.

The private HF model also now has an exact copy of its default schedule at
[`schedules/active_experts.json`](https://huggingface.co/chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview/blob/main/schedules/active_experts.json),
commit `4d5a79aec0cf5a7deca465b6156b34048e5f174c`.
The root copy and model weights are unchanged; [verification receipt](upload/schedule_copy_receipt.json).

**Full image+tools result, 13:36 PDT:** all 120 Kaggle evaluation tasks x2
attempts are complete. Task-weighted mean-draw pass@1 / pass@2: 14.02B
71.67% / 83.89%; E128 72.43% / 83.47%; 14.58B 70.83% / 81.81%. Candidate
pass@1 deltas are -0.76 pp versus E128 (95% paired task-bootstrap interval
[-6.04,+4.38]) and +0.83 pp versus 14.58B ([-3.75,+5.42]). The point estimate
meets the original two-point target, but this sample does not establish
non-inferiority. Full output throughput is 1016.47 versus 928.65/987.31 tok/s
(+9.46%/+2.95%). Eval wall time is 9613.59 versus 10054.51/9597.79 seconds:
4.39% shorter than E128, 0.16% longer than 14.58B. Extra generated tokens erase
the throughput gain over 14.58B in total runtime. No server/harness errors.
The controller immediately moved to `baseline_txt`; candidate and 14.58B text
arms follow. Frozen full image summary: `analysis/completed_image_tools_report.json`.

**Future ~12B evaluation order:** once a candidate is ready, prioritize
**text+tools first**, then text+images+tools, always against both E128 and 14.58B.
The current 14B full text queue retains priority. Reuse completed references only
with matched data, harness and serving settings. This order is recorded in
`configs/exploratory12_constraints.json`; no new 12B job is launched here.

**Future ~12B constraint:** the user explicitly treats 12B as approximate. A layer
that needs six routed experts MUST remain at six; hitting a parameter target may
never override that protection. Conservatively initialize protected layers from
the current seed14 k6 set and expand it using E065 image/text tool evidence. This
initial set is a precaution, not a claim that every member has been proven to
require six. Reduce only supported layers and accept a final size above 12B when
needed. `configs/exploratory12_constraints.json` records the hard protection and
soft size target. This records future exploration policy; it launches no GPU
work and does not alter the frozen 14B tools evaluations.

**Token/tool audit, 13:25 PDT:** 228/240 candidate image attempts finished; the
matched subset has 113 Kaggle evaluation tasks with both draws (226 logical
attempts per model). For E128 / 14.58B / 14.02B respectively: mean output tokens
36,100 / 37,033 / 37,797; mean tool calls 16.89 / 17.17 / 17.30; tool error/rejection
rates 12.29% / 13.14% / 13.07%; missing-answer attempt rates 3.98% / 1.77% / 2.65%;
budget, turn or context cap stops 5.31% / 2.65% / 3.54%. Server/harness errors are
zero in all three. Tool errors are counted after auto-repair and include rejected
or malformed calls and skipped truncated calls; they are not infrastructure
failures. Format rescues are 2 / 2 / 1, while auto-repair was applied in 82 / 100 /
98 attempts. Training-check pass rates are 74.78% / 76.99% / 74.34% (not eval
accuracy). Candidate output-token increases are +4.70% versus E128 and +2.06%
versus 14.58B; tool-call increases are +2.41% and +0.72%. Extra output can consume
the small throughput advantage over 14.58B. Hard unfinished attempts remain
excluded; these are not final full-set usage estimates. Pair records from
multi-test tasks were deduplicated, since tool counters repeat across pairs.
Exact snapshot: `analysis/usage_tools_20261031_2025.json`.

**Scheduled update, 13:12 PDT:** image+tools has 212/240 attempts finished. On the
same 103 Kaggle evaluation tasks with both draws complete, mean-draw task-weighted
pass@1 is 75.49% (14.02B), 75.32% (E128), and 74.03% (14.58B); pass@2 is 86.41%,
87.38%, and 84.47%. Candidate pass@1 deltas are +0.16 pp versus E128 and +1.46 pp
versus 14.58B, with descriptive task-bootstrap intervals [-5.58,+5.99] and
[-3.40,+6.31] pp. The completion-biased subset still cannot establish the <=2 pp
full-set margin. Minutes 5-135 throughput is 1125.03 versus 1026.99/1097.95 gen
tok/s (+9.55%/+2.47%). No eval errors or preemptions; cumulative prefix hits 91.1%
(recent engine window 92.7%). Estimated image finish is another 30-45 minutes;
all three text arms remain queued. Exact snapshot:
`analysis/intermediate_tools_20261031_2012.json`.

**Reporting cadence, 2026-10-31:** the user requested an update in this conversation
every 30 minutes. Local timer `arc-e076-report.timer` invokes
`scripts/scheduled_report.py`, which queues/resumes the existing Codex thread;
`monitor/report_schedule.json` records the destination and bound. The timer stops
after queuing the final update when all four new eval arms are complete, or at the
original batch deadline plus two hours. It requires this computer and the Codex
daemon to remain running. It does not alter GPU jobs or rental state. The
separate 90-second data-sync watcher and Luna health monitor remain active.

At 12:40 PDT (19:40 UTC), 158/240 image task attempts were complete. On the same
74 Kaggle evaluation tasks with both draws finished, task-weighted mean-draw
pass@1 was 79.05% (14.02B), 77.14% (E128), and 78.72% (14.58B); pass@2 was
89.19%, 87.84%, and 89.19%, respectively. Completed-subset bias remains and the
paired descriptive intervals cross the two-point margin; no full-set conclusion.
Minutes 5-100 engine throughput was 1142.01, 1024.77, and 1110.09 generated tok/s,
respectively: +11.44% versus E128 and +2.88% versus 14.58B. No eval errors.
Snapshot: `analysis/intermediate_tools_20261031_1940.json`.

**Comparison policy, 2026-10-31:** at the user's request, every candidate is now
compared against two references: E128 k6 (~16B active) and E074 `mean5_profiled`
(~14.58B active). Accuracy, generation throughput, and full-run wall time must
show both reference values and both deltas, on matching modalities and task sets.
The original <=2 pp loss target versus E128 remains unchanged. Image references
are already complete on all 120 Kaggle evaluation tasks x2. The missing 14.58B
text+tools reference is appended after the existing three-arm tools queue; it
adds approximately three hours and preserves the original 48-hour deadline.
The active image eval is not interrupted. `configs/dual_baseline_manifest.json`
pins the second reference schedule; `queue_reference.py` waits for the existing
queue before launching `batch.py --reference-only`. The local watcher remains
active across that handoff.

**Priority update, 2026-10-31 17:50 UTC:** the user requested full evaluations and
tools harness first. Remaining calibration screens and joint refinement are
deferred after preserving the current completed layer arm. The frozen, uploaded
`seed14` schedule is evaluated immediately, without using ARC results to choose
its architecture. Queue: `preview14_img` (120 Kaggle tasks x2, images+tools),
`baseline_txt` (k6, text+tools), `preview14_txt` (seed14, text+tools).

The completed E074 `baseline_arc` supplies the image k6 reference. All 44 frozen
harness assets match, and the new image arm requires equality of serving/eval
commands except port, base URL, output path and per-run cache salt. This is a
reused control, not a fresh replicate. It saves approximately three GPU-box hours.
Both modalities retain the same tool harness, auto-repair, budgets and sampling;
text uses token-exact history. Expected first image result is about three hours,
and all three new tool runs about eight to nine hours, subject to observed runtime.
`configs/tools_first_manifest.json` records the frozen schedule, challenge hash,
HF revision, execution order and comparison scope. The original 48-hour deadline
remains in effect. `--tools-first` completes this evaluation block and leaves the
remaining architecture search deferred for review of the tool results.

Execution confirmed at 17:55 UTC: `preview14_img` passed both reference-command
checks and started all 240 task attempts, with successful model requests and
generation underway. Controller PID 482949 owns this queue; the persistent local
sync/report watcher and Luna monitor remain active.

Intermediate snapshot, 18:46 UTC: 63/240 task attempts finished, spanning 38
Kaggle evaluation tasks; 25 tasks have both draws complete. On those same 25
tasks, seed14 mean-draw task-weighted pass@1 is 91.00% versus k6's 91.33%
(delta -0.33 pp); pass@2 is 96.00% versus 98.00%. The descriptive task-bootstrap
interval for pass@1 delta is [-11.67, +9.67] pp. This subset is selected by
candidate completion time, strongly favors faster tasks, and cannot establish
full-set accuracy or the two-point margin. No eval errors were recorded. Exact
task IDs and alternate partial views are saved in
`analysis/intermediate_tools_20261031_1846.json`. Architecture and run settings
remain unchanged; text tools are queued after the image run.

Throughput snapshot, 18:48 UTC: over minutes 5-50 of each run (first five minutes
excluded), duration-weighted engine generation throughput is 1,187.44 tok/s for
seed14 versus 1,048.32 tok/s for k6, +13.27%, on the same four RTX PRO 6000s and
64-worker image+tools harness. These rates include tokens from unfinished
attempts. The full first-50-minute window gives +14.56%; minutes 20-50 give
+12.21%, while minutes 40-50 give only +4.48%. Task/context mix evolves differently
between runs, so this is a provisional workload measurement, not an isolated
kernel speedup or a final wall-time reduction. The completed E074 14.58B run had
+6.32% end-to-end output tok/s and 4.54% less eval wall time; its own first-50-minute
gain was +10.86%, illustrating why the current partial gain may shrink.
Raw window summaries: `analysis/throughput_intermediate_20261031_1848.json`.

**Scope update, 2026-10-31 17:43 UTC:** the user dropped the 13B target. Only the
184-slot (~14.018B) target remains in the running queue. The earlier uniform-k4
13B diagnostic results are retained as historical measurements; no further 13B
schedule search, audit or full ARC runs are queued. The shared k3/k4/k5 layer
profiles remain useful for allocating the 14B budget. Final full evaluation now
has four arms: baseline and selected 14B, each with images+tools and text+tools.
The controller was updated at a completed-arm boundary and reuses all finished
work. Its original 48-hour deadline is preserved across the restart.

The original two-target plan below is retained for provenance; this scope update
supersedes references to continuing 13B work.

Started 2026-10-31. User authorizes both targets and explicitly requires the latest
E065 data family for calibration. Accuracy target remains at most 2 percentage
points task-weighted pass@1 loss. The two targets are 184 active routed slots
across 40 layers (mean 4.6, approximately 14.018B decode-active parameters) and
160 slots (mean 4, approximately 13.169B). Counts use the rounded advertised 16B
anchor; the routed parameter reductions are exact. All 128 stored experts remain
in every layer. This does not imply weight-memory savings or proportional speedup.

## Frozen model, data, and controls

- Causal image E128 at HF revision `209c810a747c043c2aca474843de8e357f920ef6`, with
  the same compatible B32 Engram table as E074. Never modify the source checkpoint.
- Calibration uses E065 image/tool traces and paired E065 text/tool traces only.
  Their parent IDs are checked against ARC training IDs and against all 120 Kaggle
  evaluation IDs. The audit pool also excludes every E074 calibration family.
- Three disjoint family pools: screen16, select24, audit32. Both modalities use
  matching rollout keys. Screen traces are at most 30k tokens; select and audit
  include longer traces up to 60k. Exact keys, counts, and lengths are in
  `data/manifest.json`. Data comes from E066's tokenized E065 development split.
- `frozen/` pins the same harness as E074, with SHA256 checks at batch start.
  Serving uses the isolated E043 vision11 fork on the 4x RTX PRO 6000 Vast box.
  Causal image attention, prefix caching, context limits and cache metrics remain
  enabled. Text tools use token-exact history; images use the chat image path.
  This is not the newer Manh V12 token-exact-image stack. Auto-repair stays enabled
  in all arms; the unrelated E072 result does not alter these comparisons.

## Bounded first batch

1. Baseline NLL on all three pools and short image/text tool diagnostics on 12
   ARC training originals. Initial 14B/13B schedules receive the same diagnostics.
2. All 40 layers individually at k=3,4,5 versus k=6 on screen16, both modalities.
   This measures 120 perturbations, with tool-block loss recorded separately.
3. Exact-budget dynamic programming builds generated-loss, tool-aware and
   uncertainty-penalized schedules. Complete schedules are measured on select24.
   Up to two rounds of three budget-preserving slot swaps per target refine the
   joint loss; no assumption that single-layer effects add exactly.
4. Freeze one candidate per budget before independent audit32. Generated short
   diagnostics remain training-only and do not select the architecture.
5. Fresh matched baseline, 14B and 13B on 120 Kaggle evaluation tasks x2 attempts,
   separately for image+tools and text+tools: six full arms. Evaluation answers
   stay local. Report mean single-draw task-weighted pass@1, pass@2, task-paired
   intervals, errors, tokens, cache health and wall time. No tuning on ARC scores.

Selection minimizes equal-modality relative generated NLL + 0.25 relative
tool-block NLL + 0.25 worst-modality generated NLL, measured on select24. The
independent audit is reported regardless of outcome. E074's results warn that
small NLL changes do not establish preserved ARC accuracy.

This batch is bounded to 48 box-hours (at $5.649705/h, at most about $271.19 of
batch compute; the running rental is still billed outside batch work). Expected
duration is approximately 24-36 hours, to be revised from observed arm timings.
No new rental, Kaggle submission, upload, stop or destroy is performed. Other
Runpod harness jobs are left intact. One retry per failed arm preserves previous
logs; a repeat failure stops the batch instead of consuming the whole budget.

## Recovery work

The first batch measures untrained schedules. If the two-point target is missed,
use teacher recovery on E065 training traces, keeping architecture selection and
audit families separated. Candidate approaches are layer-output reconstruction,
router/scale calibration, and quantization-aware expert adapters. Before any
training claim, demonstrate nonzero routed-expert updates and verify their
exported tensors: E069's prior SFT had unchanged routed experts. Recovery training
is authorized by the user's experiment request but has not started here.

## Operation

`supervisorctl status e076` on Vast 58489667. Service does not auto-restart;
`results/batch_status.json` is written atomically at phase changes. `results/` and
`configs/` are periodically synced locally by `scripts/monitor.py`. Cleanup uses
captured process identities. The supervisor never manages the rental itself.

Scripts: `prepare.py` (data/provenance), `arm.py` (owned engine and one arm),
`search.py` (paired losses and exact-slot allocation), `batch.py` (bounded queue),
`diagnostic.py` (training-only generated checks), `report.py` (local scoring).

## Private preview upload (2026-10-31)

At the user's request, the early leading 14.018B `seed14` candidate is uploaded as
[chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview](https://huggingface.co/chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview),
private, revision `87059808afa6cf3f3fe928ad80749de6f244f5bd`. This is an explicitly
preliminary schedule, not the final selection from the ongoing search.

The repo contains all 46 safetensors files (132,053,122,744 bytes), including the
compatible Engram tables; weights were copied server-side from the pinned source
models without modification. All 134 files were verified: source LFS SHA256/git
blob equality for copied files, downloaded SHA256 checks for the new metadata and
instructions, and a complete index exactly matching the served checkpoint.
Receipt: `upload/receipt.json`; reproducible packaging/publishing scripts:
`scripts/package_preview.py` and `scripts/publish_preview.py`. The latter reads
the full-access credential from SECRETS.yaml at runtime and never stores it.

The default `active_experts.json` is the 184-slot seed14 schedule. The separately
evaluated E074 mean-five (200-slot, 14.584B) schedule and its full Kaggle120 image
results are included as an alternative. These results must not be attributed to
seed14. The model card reports the 24-family E065 NLL and 12-task training-only
diagnostics and states that seed14 full ARC accuracy is pending.

The model requires the per-layer active-top-k patch. The tested E043 patch is
included; Manh g22df163d2d still requires its router/model port and GPU validation.
An unpatched server may ignore the metadata field and run the k6 baseline.
