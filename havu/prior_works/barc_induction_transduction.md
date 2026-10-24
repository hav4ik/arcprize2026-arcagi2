# BARC: Combining Induction and Transduction (Li, Ellis et al.)

- **Paper:** *Combining Induction and Transduction for Abstract Reasoning*, arXiv 2411.02272 (Nov 2024; ARC Prize 2024 paper award).
- **Code and data:**
  - https://github.com/xu3kev/BARC (licence unclear);
  - HF `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` and the related sets.

Source tag: [P] = paper.

## TL;DR
- **Seeds:** 100, later 160, hand-written Python seeds. Each has:
  - an NL description and a list of concepts;
  - `transform_grid`;
  - `generate_input`.

  All seeds share a **`common.py` helper library**: `flood_fill`, `draw_line`, `find_connected_components`, `detect_objects`, `blit_sprite`, `crop`, `bounding_box`, `detect_{translational,mirror,rotational}_symmetry`, `random_sprite`, `collision`/`contact` and others. [P]
- **"From domain-specific languages to domain-specific libraries":** BARC deliberately does not restrict programs to a DSL. "We still allow arbitrary Python code, which helps cover the long tail." [P]
- **Data:** LLMs remix the seeds into ARC-Heavy (200k problems) and ARC-Potpourri (400k, including 100k RE-ARC).
- **Results** (ARC-1 public eval, pass@2):

  | Model | Score |
  |---|---|
  | Induction (Llama-3.1-8B, 20k samples, vote) | 38.0% |
  | Transduction with TTT and reranking | 43.0% |
  | Ensemble | 56.75% |
  | Ensemble, with Akyürek's better TTT | 61.9% |

## Findings [P]
- **Induction and transduction solve different tasks.**
  - Induction is better at counting, precise computation and composing several concepts.
  - Transduction is better at fuzzy perceptual concepts.
  - The ensemble tries induction first and falls back to transduction.
- **Scaling:** more seeds saturate quickly, but more synthetic data keeps improving the model.
- **False positives:** about 9% of train-fitting programs fail the test pair. About half of those would be removed by majority vote.
- **Why input generators are code:** an LLM alone could not generate complex, precisely-correct ARC inputs.

## Caveats for ARC-AGI-2
- The data is ARC-1-flavoured.
- NVARC tested BARC data and dropped it. Their paper (`nvarc_2025.pdf`, Results) gives one reason: "in last year's competition, we found that BARC was not useful for the 2D transformer we developed", so they "used real puzzles with our synthetic puzzles only". No ablation is shown for their 2025 models. NVARC did reuse BARC's 160 human-written seed descriptions, together with H-ARC descriptions, as seeds for its own 716-puzzle generation.

## Reusable for us
- **Template for our generator pipeline and sandbox helpers:** Python, a shared helper library, and seeds with NL descriptions plus input generators.
- **Contamination:** dedup against eval before use.
