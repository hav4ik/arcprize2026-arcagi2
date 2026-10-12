# icecuber: DSL brute-force search (ARC 2020 Kaggle winner)

- **Author:** Johan S. Wind. Kaggle Abstraction and Reasoning Challenge, 2020.
- **Code:** https://github.com/top-quarks/ARC-solution, with `ARC-solution_documentation.pdf`. C++17 with no dependencies. The licence file is in the repo.

Source tags: [D] = documentation, [P] = papers citing it.

## TL;DR
- **Search space:** 142 unary image transforms built from 42 functions, chained up to depth 4. Duplicate intermediate images are removed. A greedy "stacker" overlays candidates to fit the train pairs. [D]
- **Scores:**

  | Set | Score |
  |---|---|
  | ARC-1 private | 20.6% (LB 0.794 top-3 error) |
  | ARC-1 public eval | 39% |
  | ARC-1 semi-private | 17% |
  | **ARC-AGI-2 semi-private** | **1.6%** (arXiv 2505.11831, Table 1) |

- **Effect on benchmark design:** pooling all 2020 entries, all DSL search, solved 49% of the ARC-1 private set (ARC Prize 2024 report). ARC-AGI-2 was explicitly designed to be "less brute-forcible".

## Method [D]
- **Key transforms:**
  - `Cut` splits the non-background pixels into corner-connected groups;
  - `filterCol`, `colShape`;
  - `composeGrowing` overlays images, with the sparsest on top;
  - `compress` crops to the content;
  - `rigid` covers the 8 dihedral transforms;
  - `pickMax` selects by a property.
- **Where the transforms came from:** hand-solving 100 training and **100 evaluation** tasks, so the DSL saw the eval set.
- **Ensemble:** depth 3, depth 3 with diagonal-flip augmentation, then depth 4 until the 9 h / 16 GB budget runs out.
  - Candidates are ranked by train pairs solved, then depth, then number of stacked images.
  - Colour remapping as preprocessing "worked surprisingly well".
- **Cost of depth:** depth 3 is about 20× cheaper than depth 4. Depth 2 is about 15× cheaper again and still gets about 80% of depth 3's solves.

## Reusable for us
- **Possible Kaggle add-on:** a CPU-only fallback that could run alongside the GPU model at almost no GPU cost. On ARC-AGI-2, expect about 1–2% at best.
- **Check first:** whether it solves any pair the Qwen tool harness misses. Low priority.
