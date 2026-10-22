# ARC-AGI-2 SFT Puzzle Datasets, 3-Level Verification Pipeline & Trace Generation Catalog

**CNS Data Root**: `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/`  
**Format**: Identical to `/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data` (`data/training/<task_id>.json`, `data/training.txt`, `data/evaluation/<task_id>.json`, `data/evaluation.txt`, plus `data/training.tar.gz`, `data/evaluation.tar.gz`, and `full_tasks_arc_agi2.jsonl` where every task is `{"train": [{"input": [[...]], "output": [[...]]}], "test": [{"input": [[...]], "output": [[...]]}]}`).  
**Zero-Leakage & Verification Policy**:
* All `120` official `ARC-AGI-2` public evaluation tasks are strictly blacklisted from SFT candidate selection (`leakage_audit_zero_arc_eval: True`).
* All third-party unverified LLM-generated reasoning traces have been **removed**; we retain **only human-generated traces & descriptions** (`H-ARC`, `ARCTraj`, `ARC-AGI-2 Human Testing`, and `BARC 160 Human Seeds`) under `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/human_traces/` and generate our own **3-Level Verified** multi-turn Python-REPL trajectories under `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/`.  
**Companion Docs**:
* Repository Summary & All Experiments: [`README.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/README.md)
* Pilot 2 Controlled Hypotheses (`h1`–`h8`): [`pilot_2/PILOT_2_EXPERIMENTS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_2/PILOT_2_EXPERIMENTS.md)
* Pilot 1 Experimental Findings: [`pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/ARC_AGI2_EXPERIMENTS_FINDINGS.md)

---

## 1. CNS Directory Layout (`/cns/lu-d/home/arazdai/kaggle/arc_sft_data/`)

Every puzzle dataset (`26` datasets, **`200,052` standardized tasks** across `198,772` training-eligible records) is stored under `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/datasets/<dataset_name>/`, alongside our active SFT preparation and trace-generation pipeline in `sft_prep_pipeline/`:

```text
/cns/lu-d/home/arazdai/kaggle/arc_sft_data/
├── DATASETS_SUMMARY.json                      # Exact task & file counts across all 26 standardized datasets
├── datasets/
│   └── <dataset_name>/
│       ├── data/
│       │   ├── training/
│       │   │   └── <task_id>.json             # {"train": [{"input": ..., "output": ...}], "test": [{"input": ..., "output": ...}]}
│       │   ├── training.txt                   # Full list of training task_ids (one per line)
│       │   ├── training.tar.gz                # Single-archive bundle of 100% of training/<task_id>.json files
│       │   ├── evaluation/
│       │   │   └── <task_id>.json             # {"train": [{"input": ..., "output": ...}], "test": [{"input": ..., "output": ...}]}
│       │   ├── evaluation.txt                 # Full list of evaluation task_ids (one per line)
│       │   └── evaluation.tar.gz              # Single-archive bundle of 100% of evaluation/<task_id>.json files
│       └── full_tasks_arc_agi2.jsonl          # All tasks in 1-line-per-task ARC-AGI-2 JSONL format
├── sft_prep_pipeline/                         # Deduped, 3-level verified SFT candidate pipeline & generated traces
│   ├── canonical_manifest.jsonl               # 115,385 deduplicated non-official puzzles across 2,914 families (267 MB)
│   ├── duplicate_and_exclusion_report.json    # Cross-dataset dedup, eval blacklist audit & Level-3 pair recovery stats
│   ├── family_split_assignments.json          # Family-level train/val split assignments (zero generator leakage)
│   ├── puzzle_annotations.jsonl               # Structural & concept-tag annotations (23 MB)
│   ├── autorater_prompt_v1.{json,txt}         # Rubric for two-operation compositional frontier scoring
│   ├── screening_1k_queue.jsonl               # 1,000 stratified puzzles for empirical difficulty calibration
│   ├── screening_1k_runs_sft_screen1k/        # Qwen3.8-27B-FP8 2-attempt screening outputs on the 1,000-puzzle queue
│   ├── screening_1k_empirical_report.json     # Empirical 3-level pass rates & verified trajectories per 1M tokens
│   ├── verified_success_manifest.jsonl        # 1,175 3-level verified positive trajectories from 1k screening (47.8 MB)
│   ├── failed_trajectories_for_repair.jsonl   # 825 failed/overfit trajectories saved for repair training (95.7 MB)
│   ├── recommended_next_generation_queue.jsonl# 4,000 prioritized follow-up tasks (50% 1/2 frontier, 30% 0/2, 20% 2/2)
│   ├── top_10k_deepseek_candidates.jsonl      # Top 10,000 highest-learning-signal SFT candidates with Level-3 pairs (202.8 MB)
│   ├── top_10k_selection_report.json          # Breakdown of Top 10K by tier, source, family & concept tags
│   ├── top_10k_task_ids.json                  # Ordered task ID list for the Top 10,000 candidates
│   └── deepseek_v41_traces_top10k_ds41_fp_top10k/
│       ├── verified_sft_trajectories_merged.jsonl          # 1,122 3-level verified DeepSeek-V4.1-Flash-1017 trajectories (156.5 MB)
│       ├── verified_sft_trajectories_shard_000..015of016.jsonl
│       ├── failed_or_overfit_trajectories_shard_000..015of016.jsonl
│       └── summary_shard_000..015of016.json
├── human_traces/                              # Human-generated reasoning traces & descriptions ONLY
│   ├── H-ARC_human_traces.tar.gz              # >1,700 humans' action traces + natural-language solution descriptions
│   ├── ARCTraj.csv                            # ~10,000 human step-by-step subgoal/object-editing trajectories
│   ├── arc_agi_2_human_testing_attempts.csv   # Official ARC Prize human pass-rate & solve-time logs per pair
│   └── barc_160_human_seeds.jsonl             # 160 human-written seed descriptions + Python programs from BARC
├── hodel_dsl_and_rearc/                       # Michael Hodel's 400 DSL solvers + 400 RE-ARC procedural generators/verifiers
│   ├── arc_dsl_dsl.py, arc_dsl_solvers.py, arc_dsl_arc_types.py, arc_dsl_constants.py
│   └── rearc_generators.py, rearc_verifiers.py, rearc_dsl.py
├── raw_preserved/                             # Original multi-pair zip archives (30-pair NVARC, 250-pair ARC-GEN-100K)
│   ├── nvarc-synthetic-puzzles.zip            # Used to recover 146,532 extra Level-3 verification pairs
│   └── the-arc-gen-100k-dataset.zip           # Used to recover 28,115 extra Level-3 verification pairs
├── papers_and_writeups/                       # Winning papers & data loaders (NVARC, ARChitects, MindsAI, Barbadillo)
└── repos/                                     # Winning solution codebases & procedural generator repos (tar.gz)
```

---

## 2. Master Table of All 26 Standardized Puzzle Datasets on CNS

### 2.1 Competition-Winning Synthetic & Difficulty-Stratified Datasets (`189,766` Tasks)

| Dataset Directory (`datasets/<name>/data`) | Origin / Winning Team | Total Unique Tasks | `training` Split Tasks | `evaluation` Split Tasks | Description & Role in SFT Pipeline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`NVARC-Synthetic-103k`** | 1st Place ARC Prize 2025 (`sorokin/nvarc-synthetic-puzzles`) | **103,223** | 47,337 (`nvarc_training`) | 55,886 (`nvarc_full`) | NVARC's 4-stage SDG puzzles (H-ARC/BARC human summaries $\to$ Instruct-SkillMix compositional mixing $\to$ unit-tested Python input generators $\to$ 20-program output consistency voting). Original 30-pair lists preserved in `raw_preserved/nvarc-synthetic-puzzles.zip` to supply up to **25 withheld Level-3 verification pairs per puzzle**. |
| **`NVARC-TRM-Curated`** | 1st Place NVARC (`cpmpml/arc-prize-trm-*`) | **2,913** | 2,793 | 120 | The exact **2,793 highest-signal NVARC synthetic puzzles** curated by Jean-François Puget & Ivan Sorokin for training TRM (100% deduplicated against `NVARC-Synthetic-103k`; `120` official eval tasks excluded). |
| **`Nemotron-RL-ARC-AGI-10k`** | NVIDIA Official Post-Competition Release (`nvidia/Nemotron-RL-ARC-AGI-v1`) | **10,514** | 10,000 | 514 | Sampled across ARC-AGI-2 + NVARC + `neoneye` (with official eval IDs blacklisted) and **graded by empirical pass-rate** across 9 frontier models (`12% Easy`, `35.9% Medium`, `52.1% Hard`). Stripped of all LLM prompts into pure ARC-AGI-2 JSON tasks + `difficulty` metadata. |
| **`ARC-GEN-100K`** | 2nd Place The ARChitects (`arcgen100k/the-arc-gen-100k-dataset`) | **19,777** | 19,378 | 399 | 100,000 procedural examples across 400 generators from Google Code Golf 2025, grouped into 5-pair (`4 train + 1 test`) tasks plus **up to 10 withheld Level-3 verification pairs** recovered from `the-arc-gen-100k-dataset.zip`. |
| **`MindsAI-Synthetic-27Gen`** | 3rd Place MindsAI & Tufa Labs (`jcole75/mindsai-tufa-extra-datasets`) | **53,239** | 52,969 | 270 | MindsAI's **27 custom ARC-AGI-2 compositional generators** (~2,000 tasks per generator), parsed from T5 text prompts into clean `{"train": [...], "test": [...]}` ARC-AGI-2 JSON format (`10` tasks/generator = `270` in `evaluation/`). |
| **`ARC-1.5`** | 3rd Place MindsAI / Michael Hodel (`jcole75/arc-1-5-json`) | **100** | 100 | 100 | Michael Hodel's curated harder split of ARC-1 (`challenges.json` + `solutions.json` merged into complete ARC-AGI-2 JSON tasks). |

---

### 2.2 The 20 Community & Academic ARC-Format Datasets (`10,286` Tasks)

| Dataset Directory (`datasets/<name>/data`) | Source / Paper | Total Unique Tasks | Description & Core Reasoning Priors Tested |
| :--- | :--- | :---: | :--- |
| **`ARC-AGI-2`** | Official ARC Prize 2025/2026 | **1,120** | 1,000 training + 120 evaluation tasks (`evaluation/` strictly blacklisted from SFT). |
| **`ARC-AGI-1`** | Chollet (2019) | **827** | Original 405 training + 422 evaluation tasks (+ variants). |
| **`ARC-Heavy`** | BARC Curated Subset (`barc0`) | **1,187** | 1,187 curated synthetic tasks from BARC's `200k_HEAVY` distribution converted to pure ARC-AGI-2 grids (no LLM text/code). |
| **`arc-dataset-diva`** | DIVA Visual Analogies | **1,200** | Structured visual analogy and geometric transformation tasks in ARC format. |
| **`arc-dataset-tama`** | Tama Procedural ARC | **951** | Ray-casting, cellular symmetry, and bounding-box interaction puzzles. |
| **`RE-ARC`** | Michael Hodel (2024) | **799** | Reverse-engineered procedural tasks across all 400 ARC-1 training concepts. |
| **`arc_1d`** | Reasoning Gym / Community 1D | **738** | 1D cellular and sequence transformation tasks in ARC format. |
| **`1D-ARC`** | Xu et al. (2023) | **615** | 1D object-tracking tasks (move, fill, denoise, flip, mirror, scale, pattern copy on $1\times N$ arrays). |
| **`IPARC`** | Interactive / Parallel Procedural ARC | **600** | Morphological graph/grid operations (dilation, erosion, hit-or-miss, skeletonization) designed around parallel cellular rules. |
| **`ARC_synthetic_extend`** | Extended Synthetic ARC | **419** | Procedural extensions of core ARC training tasks. |
| **`ConceptARC`** | Moskvichev et al. (TMLR 2023) | **179** | Human-designed tasks systematically organized into **16 core concept groups** (Above/Below, Center, Clean Up, Complete Shape, Copy, Count, Extend To Boundary, Extract Objects, Filled/Not Filled, Horizontal/Vertical, Inside/Outside, Move To Boundary, Order, Same/Different, Top/Bottom 2D/3D). |
| **`Mini-ARC`** | Kim et al. (2022) | **149** | Compact $5\times 5$ human-designed tasks testing pure conceptual transformations without large-grid coordinate overhead. |
| **`PQA`** | Pattern Question Answering | **70** | Rule-completion and pattern-selection tasks across 10 structural rules. |
| **`synth_riddles`** | Synthetic Visual Riddles | **52** | Riddle-like symbolic grid puzzles requiring hidden invariant discovery. |
| **`Sequence_ARC`** | Sequential Pattern ARC | **25** | Sequence continuation problems represented as 2D grids. |
| **`arc-community`** | Community ARC Contributions | **23** | Human-created puzzles submitted via the community ARC editor. |
| **`dbigham`** | David Bigham Diagnostic Pack | **21** | Hand-crafted diagnostic ARC tasks. |
| **`Sort-of-ARC`** | Assouel et al. (2022) | **20** | $20\times 20$ grids with $3\times 3$ objects testing relational predicates (color, shape, position). |
| **`nosound`** | Nosound Puzzle Pack | **9** | Hand-crafted hard multi-step puzzles. |
| **`arc-dataset-bad`** | Adversarial Edge-Case Pack | **2** | Edge-case diagnostic tasks. |

---

## 3. Deduplication, 3-Level Verification & Empirical Screening Pipeline (`sft_prep_pipeline/`)

### 3.1 Why 3-Level Verification Is Mandatory for SFT Trace Generation
In standard ARC tasks (`3–4` train pairs + `1` test pair), a synthesized `def solve(grid):` program can accidentally pass both `train` and `test` via shortcut heuristics. By recovering the full generator pools from `raw_preserved/nvarc-synthetic-puzzles.zip` (`30` pairs/task) and `raw_preserved/the-arc-gen-100k-dataset.zip` (`250` pairs/task), every candidate trajectory is gated by **3-Level Verification**:
1. **Level 1 (`train` demonstrations)**: Program executes inside the sandbox during the multi-turn loop and must achieve `100%` exact match on all `train` pairs.
2. **Level 2 (`test` held-out pair)**: Program is evaluated post-hoc on the unseen `test` pair(s).
3. **Level 3 (`extra_verification_pairs`)**: Program is evaluated on up to **25 withheld NVARC generator pairs** (`146,532` verified extra pairs recovered across `6,000` NVARC tasks) or **10 withheld ARC-GEN pairs** (`28,115` extra pairs across `2,884` ARC-GEN tasks).

### 3.2 Deduplication & Zero-Leakage Audit (`duplicate_and_exclusion_report.json`)
* **Total records scanned**: `198,772` tasks across all datasets.
* **Official Evaluation Blacklist**: All `120` `ARC-AGI-2` evaluation tasks (including their `120` exact copies inside `NVARC-TRM-Curated/evaluation`) are permanently excluded (`leakage_audit_zero_arc_eval: True`).
* **Exact Grid-Hash Deduplication**: `2,793 / 2,793` training tasks in `NVARC-TRM-Curated` matched tasks in `NVARC-Synthetic-103k` and were merged with priority metadata, leaving **`115,385` unique eligible non-official tasks across `2,914` generator/concept families** (`canonical_manifest.jsonl`).

### 3.3 1,000-Puzzle Empirical Screening Results (`screening_1k_empirical_report.json`)
We ran `Qwen3.8-27B-FP8` (`2` attempts/puzzle = `2,000` trajectories) across a stratified `1,000`-puzzle screening queue (`screening_1k_queue.jsonl`):
* **Overall Trajectory Yield**: **`1,175` 3-level verified positive trajectories** (`verified_success_manifest.jsonl`) and **`825` failed/overfit trajectories** saved for code-repair training (`failed_trajectories_for_repair.jsonl`).
* **Puzzle Pass Distribution (`3-Level Verified`)**: `536` puzzles solved `2/2` (`53.6%`), `103` solved `1/2` (`10.3%` frontier), `361` solved `0/2` (`36.1%` hard/overfit).

| Source Group | Puzzles Screened | Total Attempts | Level 1 (`train`) Pass | Level 1+2 (`train+test`) Pass | **Level 1+2+3 (Full 3-Level) Pass** | `2/2` Bucket | `1/2` Frontier | `0/2` Hard/Overfit | Verified Trajectories / `1M` Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nvarc`** | `300` | `600` | `444` (`74.0%`) | `364` (`60.7%`) | **`225` (`37.5%`)** | `94` | `37` | `169` | `15.7` |
| **`mindsai`** | `200` | `400` | `231` (`57.8%`) | `130` (`32.5%`) | **`130` (`32.5%`)** | `57` | `16` | `127` | `10.1` |
| **`arc_gen`** | `150` | `300` | `284` (`94.7%`) | `275` (`91.7%`) | **`237` (`79.0%`)** | `113` | `11` | `26` | `103.1` |
| **`nemotron_rl`** | `150` | `300` | `260` (`86.7%`) | `239` (`79.7%`) | **`239` (`79.7%`)** | `110` | `19` | `21` | `40.9` |
| **`arc_heavy_and_community`** | `100` | `200` | `180` (`90.0%`) | `168` (`84.0%`) | **`168` (`84.0%`)** | `79` | `10` | `11` | `85.1` |
| **`rearc_concept_mini_1d_morph`** | `100` | `200` | `183` (`91.5%`) | `176` (`88.0%`) | **`176` (`88.0%`)** | `83` | `10` | `7` | `76.7` |
| **Total (`1,000` Puzzles)** | **`1,000`** | **`2,000`** | **`1,582` (`79.1%`)** | **`1,352` (`67.6%`)** | **`1,175` (`58.8%`)** | **`536`** | **`103`** | **`361`** | **`29.7`** |

> [!IMPORTANT]
> **Critical Finding from Level-3 Verification on NVARC**: On `NVARC`, `364 / 600` attempts passed both `train` (Level 1) and the single `test` pair (Level 2), **but `139` of those `364` programs (`38.2%`) failed on the 25 withheld Level-3 generator pairs (`364 -> 225`)!** Without Level-3 verification, nearly 40% of NVARC SFT trajectories would train the model on brittle shortcut code.

---

## 4. Top 10,000 Highest-Learning-Signal Candidate Queue & DeepSeek-V4.1 Trace Generation

### 4.1 Top 10,000 Candidate Selection (`top_10k_deepseek_candidates.jsonl`)
Using the empirical 1K screening results + structural autorater scoring (`autorater_prompt_v1.txt`), we selected the **Top 10,000 SFT candidates** (`top_10k_selection_report.json`) across **`2,548` unique generator/concept families** (`5,428` tasks with Level-3 extra check pairs):
* **Difficulty Tier Allocation**:
  * **Tier 1 — Frontier `1/2` & Two-Op Compositional Generalization (`5,687` tasks, `56.9%`)**: Empirical `1/2` inconsistent frontier puzzles (`103`), family siblings of `1/2` frontier puzzles (`860`), mixed `0/2`+`2/2` boundary families (`376`), and two-interacting-operation compositional tasks (`4,348`).
  * **Tier 2 — Hard `0/2` Unsolved & Level-3 Anti-Overfit Targets (`2,246` tasks, `22.5%`)**: Empirical `0/2` unsolved (`256`), demo-overfit puzzles that failed Level-3 extra pairs (`90`), and family siblings of `0/2` and Level-3 overfit puzzles (`1,845` + `55` multi-step targets).
  * **Tier 3 — Verified `2/2` Clean Concept Anchors (`2,067` tasks, `20.7%`)**: Empirical `2/2` high-clarity anchors (`382`), family siblings (`1,053`), and single-op concept anchors (`632`).
* **Source Breakdown (`10,000` Total)**:
  * `NVARC`: **`3,798`** (`38.0%`) · `ARC-GEN-100K`: **`1,630`** (`16.3%`) · `Nemotron-RL`: **`1,600`** (`16.0%`) · `ARC-Heavy & Community`: **`1,305`** (`13.1%`) · `RE-ARC / ConceptARC / Mini-ARC / 1D / IPARC`: **`1,042`** (`10.4%`) · `MindsAI-27Gen`: **`625`** (`6.3%`).
* **Compositional Complexity**: **`7,377` tasks (`73.8%`)** require **two interacting operations** (e.g., `object_detection` [`4,991`], `multi_object_interaction` [`4,967`], `output_size_inference` [`3,397`], `color_mapping` [`3,085`]), while **`2,623` tasks (`26.2%`)** provide clean single-operation primitives.

### 4.2 DeepSeek-V4.1-Flash-1017 Full-Precision Trace Generation (`deepseek_v41_traces_top10k_ds41_fp_top10k/`)
Launched via [`launch_deepseek_v41_traces_xm.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm/launch_deepseek_v41_traces_xm.py) and [`deepseek_v41_tpu_vllm_traces.py`](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm/deepseek_v41_tpu_vllm_traces.py) across `16` TPU v7x slices (`ghostfish=2x2x1`, full-precision `BF16` dequantization with IEEE-754 `FP32` MXU accumulation, `reasoning_effort=100`, `max_turns=5`, `16k` token budget):
* **Progress Across 16 Shards**: **`9,005 / 10,000` candidate puzzles processed** (`11.09M` generated tokens at `~268 tok/s/slice`).
* **Merged 3-Level Verified Output**:
  * `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/deepseek_v41_traces_top10k_ds41_fp_top10k/verified_sft_trajectories_merged.jsonl` (**`1,122` 3-level verified trajectories**, `156.5 MB`).
  * Combined with the **`1,175` 3-level verified Qwen3.8-27B trajectories** in `verified_success_manifest.jsonl` (`47.8 MB`), we currently have **`2,297` strictly 3-level verified multi-turn Python-REPL SFT trajectories** plus **`2,497` failed/overfit trajectories** with exact failing coordinate diffs ready for repair-loop SFT.

---

## 5. Human-Generated Reasoning Traces (`/cns/lu-d/home/arazdai/kaggle/arc_sft_data/human_traces/`)

1. **`H-ARC_human_traces.tar.gz`** (`Le-Gris/h-arc` — *LeGris et al., 2024/2025*):
   * Action traces (grid edits, selections, copies, floods) and **natural-language solution descriptions from $>1,700$ human participants** across ARC-1 training and evaluation tasks. Used by 1st-place NVARC as their primary seed source for Stage 1 puzzle summaries.
2. **`ARCTraj.csv`** (`SejinKimm/ARCTraj` — 225 MB):
   * ~10,000 step-by-step human action trajectories recording object-level subgoal decomposition on ARC tasks.
3. **`arc_agi_2_human_testing_attempts.csv`** (`arcprize/arc_agi_2_human_testing`):
   * Official ARC Prize 2025 human calibration data recording human pass rates, attempt counts, and solve times per ARC-AGI-2 task pair.
4. **`barc_160_human_seeds.jsonl`** (`xu3kev/BARC/seeds/`):
   * 163 human-annotated seed files containing human-written `# concepts:`, `# description:` (explicit English explanation of what the input shows and how to transform it), and reference Python code.
