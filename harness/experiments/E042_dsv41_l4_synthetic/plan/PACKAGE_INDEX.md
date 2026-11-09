# Archive contents

## Current inference plan — start here

| File | Purpose |
|---|---|
| [START_HERE.md](START_HERE.md) | Entry point, current instruction, precedence, roles and status |
| [LOCAL_AGENT_PROMPT.md](LOCAL_AGENT_PROMPT.md) | Complete coordinating-agent task |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Full synthetic-first 4×L4 experiment plan |
| [AGENT_WORK_ORDERS.md](AGENT_WORK_ORDERS.md) | Source-backed implementation tasks and correctness gates |
| [experiment_matrix.json](experiment_matrix.json) | Adaptive cases, dependencies and spending caps |
| [synthetic_contract.json](synthetic_contract.json) | Shape, format, initialization, Engram and runtime requirements |
| [software_lock.json](software_lock.json) | Source pins, published artifacts and environment fields to resolve |
| [SOURCE_INDEX.md](SOURCE_INDEX.md) | Primary sources and provenance |

## Supplied helpers and metadata

| File | Purpose |
|---|---|
| [tools/prepare_synthetic_metadata.py](tools/prepare_synthetic_metadata.py) | Offline E64/E72 metadata; optional bounded tokenizer/encoding downloads |
| [tools/probe_environment.py](tools/probe_environment.py) | Read-only host/GPU/environment inventory |
| [tools/estimate_budget.py](tools/estimate_budget.py) | Explicit native memory/cost arithmetic |
| [metadata/source_config.json](metadata/source_config.json) | Exact pinned original configuration |
| [metadata/SOURCE_MODEL_LICENSE](metadata/SOURCE_MODEL_LICENSE) | Original model-repository license |

These helpers do not implement the GPU loader, router patch, compact native Engram adapter or integrated benchmark runner.

## Recording and verification

| File | Purpose |
|---|---|
| [result_record.template.json](result_record.template.json) | Per-run provenance, settings, memory, correctness, traffic and metrics |
| [decision_report.template.md](decision_report.template.md) | Session outcome and proceed/fix/stop decision |
| [handoff_contract.template.json](handoff_contract.template.json) | Interface for replacing synthetic contents with real pruned weights |
| [verification/BUDGET_NOTES.md](verification/BUDGET_NOTES.md) | E64/E72 memory assumptions and calculated scenarios |
| [verification/budget_scenarios.json](verification/budget_scenarios.json) | Exact estimator arguments and outputs |
| [verification/metadata_preparation.json](verification/metadata_preparation.json) | Offline metadata derivation record |
| [verification/validation.json](verification/validation.json) | Prior CPU helper/specification checks; no GPU result |
| [verification/packaging_validation.json](verification/packaging_validation.json) | This consolidation's file/link/manifest checks |
| [MANIFEST.sha256](MANIFEST.sha256) | Checksums for every other file in this archive |

## Complete original reference pack

The entire supplied pack is preserved under **reference_original_plan/**, excluding disposable Python bytecode. Its internal links and original filenames remain intact. Root plans supersede its older initial inference sequence.

Key references for inference agents:

- [TASKS_SERVING.md](reference_original_plan/TASKS_SERVING.md): detailed native serving source map, build/launch references and implementation checks.
- [KV_MEMORY_NOTES.md](reference_original_plan/KV_MEMORY_NOTES.md): exact cache/state/shared-buffer derivations.
- [COMMUNITY_PORTS.md](reference_original_plan/COMMUNITY_PORTS.md): audited ports and reuse boundaries.
- [DS4_REUSE.md](reference_original_plan/DS4_REUSE.md): CPU hashing/row-I/O reuse and engine limitations.
- [TASKS_ENGRAM_FIRST.md](reference_original_plan/TASKS_ENGRAM_FIRST.md): Engram semantics, mask validation and pruning handoff.
- [tools/compact_engram_reference.py](reference_original_plan/tools/compact_engram_reference.py): CPU compact-row reference.
- [CED_CALIBRATION.md](reference_original_plan/CED_CALIBRATION.md): later CED/calibration semantics; CED stays off initially.
- [MASTER_PLAN.md](reference_original_plan/MASTER_PLAN.md): full original architecture/pruning context.
- [REAP_IMPLEMENTATION_CONTRACT.md](reference_original_plan/REAP_IMPLEMENTATION_CONTRACT.md), [REAP_HARDWARE.md](reference_original_plan/REAP_HARDWARE.md), and [TASKS_PRUNING.md](reference_original_plan/TASKS_PRUNING.md): separate pruning/calibration workstream; these do not change the first synthetic 4×L4 rental.

The original README, START_HERE, SOURCE_INDEX, remaining CPU helpers and their verification records are also included. The archive contains plans, references, metadata and CPU utilities; it contains no model weights or vendored GPU runtime.
