# Prior works

An index of public methods and datasets relevant to ARC-AGI-2. Each method gets its own summary file in this folder. Longer notes from the first research pass are in `../notes/research_2026-09-30.md`.

## Methods

| Method | Year | Type | Headline | Links | Summary |
|---|---|---|---|---|---|
| NVARC | 2025, ARC Prize 1st | Qwen3-4B fully fine-tuned on 3.2M synthetic + augmented puzzles; per-puzzle LoRA at test time; DFS sampling + augmentation-based rescoring | 27.6% semi-private (Kaggle, 4×L4, 12h) | [repo](https://github.com/1ytic/NVARC), [writeup](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc) | `../notes/research_2026-09-30.md` |
| Imbue: ARC-AGI-2 evolution | 2026-02 | Per-task evolutionary search over Python `transform()` programs; LLM mutation from cell-level diffs; fitness = train correctness + LLM-judged generalization + simplicity | Public eval pass@2: Gemini 3 Flash 34→61.4% ($2.42/task); Kimi K2.5 12.1→34.0%; Gemini 3.1 Pro 88.1→95.1%. Code is AGPL-3.0 | [blog](https://imbue.com/blog/2026-02-27-arc-agi-2-evolution), [code](https://github.com/imbue-ai/darwinian_evolver) | [imbue_arc_agi_2_evolution.md](imbue_arc_agi_2_evolution.md) |
| Noemon agentic solver | 2026-04 | Prose rules only, no code execution. Reasoner → leave-one-out Validator (cell diffs) → refine ×3, plus extra solver calls; stops early on duplicate grids, otherwise a judge picks. Instruction set is per task only. MIT | 92.5% public eval (Gemini 3.1 Pro; leaderboard mislabels it "Public Train"); $3.9/task list price. Base model alone: 88.1% at ~$1/task | [repo](https://github.com/noemon/noemon-arc-agi) | [noemon_agentic_solver.md](noemon_agentic_solver.md) |
| Berman: natural-language instruction evolution | 2025-09 | Evolves natural-language instructions; scored by an LLM applying them to the train inputs (cell-match fraction). Released code: ≤35 instructions/task, stops at the first perfect one. No license | 29.4% semi-private (Grok 4), $30.4/task ($3,648 total) | [repo](https://github.com/jerber/arc-lang-public), [post](https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again) | [berman_nl_instruction_evolution.md](berman_nl_instruction_evolution.md) |
| Pang: library evolution | 2025-09 | Grok-4 writes Python `transform` programs; a global library of each task's top program (538 seeded from ARC-AGI-2 training) is shown as hints; ≤10 calls/task. No license | 26% semi-private, $3.97/task ($476 total). Our check: the library rarely helps on eval tasks, and there is no ablation | [repo](https://github.com/epang080516/arc_agi), [post](https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary) | [pang_library_evolution.md](pang_library_evolution.md) |
| Greenblatt: draw more samples | 2024-06 | ~8k GPT-4o Python programs per task, filtered on train pairs, then voting; diff-based revision. MIT | 43% ARC-AGI-1 semi-private (~$40k total). Revision 37→50%; ~+3% per doubling of samples | [repo](https://github.com/rgreenblatt/arc_draw_more_samples_pub) | [greenblatt_draw_more_samples.md](greenblatt_draw_more_samples.md) |
| ARM (Accelerated/Abstraction Reasoning Model) | 2026-06 | Not a harness: a ~10.8M-parameter TRM variant with Perceiver latents and per-puzzle embeddings; votes over 1,000 augmentations. No license | 40.3% pass@2 on the ARC-AGI-1 **eval** set (leaderboard mislabels it "Public Train"); training protocol unverifiable | [repo](https://github.com/MarcelxMensch/arm-arc-agi) | [arm_abstraction_reasoning_model.md](arm_abstraction_reasoning_model.md) |
| ARC-AGI-3 agent harnesses (14 entries) | 2026 | Game agents; the top ones verify code world models by exact replay of observations with diffs | Tycho 100, Retrodict 99.86, baseline1 98.97 (ARC-AGI-3 public demo) | [leaderboard](https://arcprize.org/leaderboard/community) | [arc_agi_3_agent_harnesses_survey.md](arc_agi_3_agent_harnesses_survey.md) |
| Hodel ARC-DSL + RE-ARC | 2023–24 | Functional Python DSL (160 primitives) with solvers for the 400 ARC-1 training tasks; RE-ARC = 400 generators verified by DSL solvers | ARC-1 only. Our check: the solvers fit 0/120 ARC-AGI-2 eval tasks | [dsl](https://github.com/michaelhodel/arc-dsl), [re-arc](https://github.com/michaelhodel/re-arc) | [hodel_arc_dsl_re_arc.md](hodel_arc_dsl_re_arc.md) |
| icecuber DSL search | 2020 Kaggle 1st | Brute-force search, depth ≤4, over 142 C++ grid transforms | 20.6% ARC-1 private; **1.6% ARC-AGI-2 semi-private** | [repo](https://github.com/top-quarks/ARC-solution) | [icecuber_dsl_search.md](icecuber_dsl_search.md) |
| BARC (Li, Ellis et al.) | 2024 | Python seeds plus a helper library ("library, not DSL"), remixed into 200k–400k problems; induction + transduction ensemble | 56.75% ARC-1 public eval (61.9% with Akyürek TTT) | [paper](https://arxiv.org/abs/2411.02272), [repo](https://github.com/xu3kev/BARC) | [barc_induction_transduction.md](barc_induction_transduction.md) |
| DSL systems on ARC-AGI-2 (3 papers) | 2026 | DSL hints for Grok-4; LLM + GA over a 41-op DSL; ARCANA | 24.4% public (hints); 10% on 60 tasks (GA); ARCANA's 32.5% unverified | see file | [dsl_systems_arc_agi_2.md](dsl_systems_arc_agi_2.md) |
| The ARChitects | 2024 1st / 2025 2nd | LLM, test-time training, Product-of-Experts candidate scoring | — | [2024](https://github.com/da-fr/arc-prize-2024), [PoE paper](https://github.com/da-fr/Product-of-Experts-ARC-Paper) | — |
| Tiny Recursive Models (TRM) | 2025 paper award | 7M-parameter recursive model, per-puzzle embeddings | ~8% ARC-AGI-2 | [repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | — |
| SOAR | 2025 paper award | Self-improving evolutionary program synthesis with hindsight relabelling | — | [data](https://huggingface.co/datasets/julien31/soar_arc_train_5M) | — |
| Poetiq | 2025 | Refinement harness on frontier LLMs | — | [repo](https://github.com/poetiq-ai/poetiq-arc-agi-solver) | — |
| Project Maya (mw00) | 2026-10 | Not ARC: single-PC C++/CUDA engine for DeepSeek-V4.1 / GLM (experts over VRAM/RAM/SSD, one conversation at a time); low-bit IQ quants | Ideas for our V4.1 serving: SWA checkpoints at turn boundaries, numeric reasoning effort, exact long-prompt skip, routing-weighted GPTQ | [repo](https://github.com/mw00/project-maya) | [project_maya.md](project_maya.md) |
| Franzen ARC-AGI-3 Milestone 2 (Duck harness + Qwen3.8-Flash-Next) | 2026-10 | ARC-AGI-3 game agent (Tufa Labs' Duck harness + patch); INT4 AutoRound Flash-Next with an INT4 MTP drafter + FR-Spec on an SGLang SM120 fork, 10 streams on one RTX PRO 6000 | ARC-AGI-3 Milestone 2; an ARC-AGI-2 thread claims leading teams use this model + harness (no public evidence) | [notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), [repo](https://github.com/da-fr/arc-agi-3-solution) | [franzen_arc3_milestone2_duck_flashnext.md](franzen_arc3_milestone2_duck_flashnext.md) |
| Tufa Labs Duck harness (+ Franzen's changes) | 2026-07 / 2026-09 | ARC-AGI-3 coding agent: one `python` tool in a fresh restricted subprocess per call, game state preloaded (segmentation, history), `action()` called from code; rolling history, world-model notes, grid images. Franzen adds retained functions, animations, diff images, UNDO, no-op guards, a hysteresis context drain, and a cache-aware priority scheduler | ARC-AGI-3 June milestone winner (1.21); Franzen public LB 27.89. Our check: its general machinery is mostly already in our harness. Hazard data says do not cut long attempts; only small ideas transfer (`--helpers` A/B, object diff in `check`, middle truncation) | [repo](https://github.com/Tufalabs/duck-harness) @ `7652836`, [fork](https://github.com/da-fr/arc-agi-3-solution) @ `10882e3`, [blog](https://tufalabs.ai/research/duck-harness/) | [duck_harness.md](duck_harness.md) |
| ARC Prize benchmarking harness | — | Plain single-prompt eval used for the leaderboard | DeepSeek V4 Flash 0731 max: 61.4% semi-private | [repo](https://github.com/arcprize/arc-agi-benchmarking) | Prompts in `../notes/prompts/` |
| MoE active-parameter pruning | 2024–26; revisited Oct 31 | Static budgets, normalization, adaptive skipping and recovery | E076 final: text pass@1 -3.06pp with small speed gain; images -0.76pp. Future calibration mixes ARC training originals and NVARC training parents; protect k6 layers | [E076](../experiments/E076_active14_13/README.md) | [review](moe_active_parameter_pruning.md) |
| Reference-mass normalization | 2026-09 | Execute fewer experts while separately choosing the normalization mass | Cheap candidate variable; Qwen evidence does not establish E128 transfer | [paper](https://arxiv.org/abs/2609.04575) | [summary](moe_reference_mass_normalization.md) |
| ZEDA | 2026-06 v2 | Zero experts, SFT and on-policy self-distillation | Recovery evidence; substantial training and dynamic serving work | [paper](https://arxiv.org/abs/2605.18643) | [summary](zeda_expert_self_distillation.md) |
| Matryoshka MoE | 2026-10 v2 | Train with varying per-layer expert counts | Relevant to a separate SFT/recovery study; evidence uses continued pretraining | [paper](https://arxiv.org/abs/2509.26520) | [summary](matryoshka_moe.md) |
| LExI | 2025-09 | Static per-layer active-expert allocation | Closest serving match to E074/E076 | [paper](https://arxiv.org/abs/2509.02753) | [summary](lexi_active_experts.md) |
| ASET | ACL Findings 2026 | Confidence/entropy-dependent expert counts | Heterogeneous budgets; careful router and dispatch port needed | [paper](https://aclanthology.org/2026.findings-acl.2140/) | [summary](aset_expert_skipping.md) |
| ACE | 2026-09 | Conservative contribution-based expert skipping | Useful profiling signals; aggressive compression still loses accuracy | [paper](https://arxiv.org/abs/2609.05228) | [summary](ace_expert_skipping.md) |
| OMP-MoE | 2026-09 v2 | Reconstruction-based stored pruning and adaptive extension | Distinguish stored-expert results from runtime activation cuts | [paper](https://arxiv.org/abs/2609.31631) | [summary](omp_moe.md) |
| AdapMoE | ICCAD 2024 | Sensitivity-aware gating and offload management | Offloading-focused evidence; not a resident-weight speed prediction | [paper](https://arxiv.org/abs/2408.10284) | [summary](adapmoe.md) |

## Datasets

| Dataset | Where | Size | Origin | Local copy / notes |
|---|---|---|---|---|
| ARC-AGI-2 (Kaggle 2026) | Kaggle `arc-prize-2026-arc-agi-2` | 1000 train / 120 eval / 240 test | Human-made | `../data/` — eval solutions differ from GitHub on 4 tasks (`b6f77b65`, `4a21e3da`, `f560132c`, `faa9f03d`) |
| NVARC synthetic | Kaggle `sorokin/nvarc-synthetic-puzzles` | 103k puzzles | LLM-written generators (gpt-oss-120b) | `../data/nvarc/` — the `nvarc_full` subset (55.9k) is seeded from the 120 eval tasks |
| NVARC augmented | Kaggle `sorokin/nvarc-augmented-puzzles` | 3.2M samples | Augmented versions of the above plus real tasks | `../data/nvarc/` |
| Nemotron-RL-ARC-AGI-v1 | HF `nvidia/Nemotron-RL-ARC-AGI-v1` | 10k train + 514 val | Sampled from ARC-AGI-2 + NVARC + neoneye | `../data/nemotron_rl/` (val only). Val = ARC-AGI-1 eval (400) + ARC-AGI-2 eval (120), some transformed |
| Nemotron-SFT-ARC-AGI-v1 | HF `nvidia/Nemotron-SFT-ARC-AGI-v1` | ~18 GB | LLM reasoning traces | Not downloaded |
| ARC-GEN | GitHub `google/ARC-GEN` | 900 generators | Human-written | Covers 891/1000 v2 train tasks, 0 eval tasks |
| RE-ARC | GitHub `michaelhodel/re-arc` | 400 generators | Human-written (Hodel DSL; verified by DSL solvers) | ARC-1 training only (391/400 in v2 train, 0 in eval). See `hodel_arc_dsl_re_arc.md` |
| ARC-TGI | GitHub `Omega-Reasoning/ARC-Task-Generators-Inventory` | 461 generators | Human-validated | 11 target v2 eval tasks: exclude those |
| BARC heavy | HF `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` | 200k | LLM-generated | Program + description per task |
| neoneye collection | GitHub `neoneye/arc-dataset-collection` | 20 subsets | Mixed | Its ARC-AGI-2 folder contains eval tasks |
| ARC Prize raw model outputs | HF `arcprize/arc_agi_v2_public_eval` | 106 MB | Model attempts on the public eval | Useful for error analysis only |

## Public scores on ARC-AGI-2

Numbers come from the [ARC Prize leaderboard](https://arcprize.org/leaderboard), checked 2026-10-01. Public eval is pass@2 over all test pairs.

| Model | Public eval | Semi-private | $/task |
|---|---|---|---|
| Claude Fable 5.1 xhigh | 99.2 | 90.0 | 3.12 |
| GPT-6 Astra xhigh | 97.9 | 93.3 | 0.83 |
| Claude Opus 5.5 high | 93.5 | 93.3 | 0.41 |
| Gemini 3.8 Flash high | 92.5 | 89.2 | 0.40 |
| GLM-5.3-Flash max | 66.8 | 65.8 | 0.09 |
| DeepSeek V4 Flash 0731 max | 63.9 | 61.4 | 0.04 |
| Kimi K3 max | 61.7 | 60.4 | 1.59 |
| DeepSeek V4 Pro 0813 max | 59.4 | 61.3 | 0.60 |
| NVARC 2025 (Kaggle) | — | 27.6 | 0.20 |
| DeepSeek V3.2 | 3.9 | 4.0 | — |

No public scores exist on NVIDIA's Nemotron-RL ARC-AGI validation split.

The ARC Prize community leaderboard snapshot (22 entries) is in `arcprize_community_leaderboard_2026-10-01.json`. Its "cost" field is usually the total for the run, not per task. Two entries are labelled "Public Train" but were actually evaluated on eval sets.

## Harnesses ported to DeepSeek V4.1-Flash (estimates)

Our baseline is the plain ARC Prize prompt at 2 calls per test pair: 78.1% pass@2 on the public eval, ~$0.07–0.09/task. The estimates below assume $0.035/call and come from the per-method summaries; none has been measured.

| Harness | Hypothesis form | Calls/task | Est. $/task | Notes |
|---|---|---|---|---|
| Plain prompt, 2 attempts (E001d) | Direct grid | ~2.8 | 0.07–0.09 | Measured |
| Pang-style: 10 code samples + train check | Python | ≤10 | ≤0.35 | Library likely irrelevant; test the sampling and verification alone first |
| Imbue evolution | Python | ~9–10× base | 0.2–0.4 | AGPL: reimplement |
| Noemon | Prose + leave-one-out validator | 17–34 | 0.6–1.2 | MIT; prompts portable |
| Berman natural-language evolution | Prose + LLM executor | 27–147 | 1–5 | No license |
| Greenblatt | Python, thousands of samples | ~8k | ~280 | Not viable; keep the filter-and-vote idea |

**Common core of every strong harness:**
1. Write the hypothesis as code or prose.
2. Check it against all training pairs.
3. Return cell-level diffs.
4. Refine.
5. Choose the final 2 answers from the verified candidates.

They differ mainly in the hypothesis form and the budget. The ARC-AGI-3 survey's evidence: verified executable models beat text-only ones, and fixed multi-agent role teams don't pay off.

## DSLs (E015)
The survey is `../experiments/E015_dsl_analysis/survey.md`.
- **At inference:** DSL search and DSL-output systems score ≤10% on ARC-AGI-2, apart from unverified claims.
- **Where DSLs pay off:** data generation, with Python generators plus a helper library as in BARC, NVARC and ARC-GEN, and possibly an optional helper library in the sandbox.
- **Recommendation:** keep Python as the hypothesis language.
