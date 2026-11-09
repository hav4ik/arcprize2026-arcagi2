# DeepSeek-V4.1-Flash-1017: synthetic-first experiments on 4×L4

Research prepared 20 October 2026; consolidated local-agent handoff packaged 21 October 2026. E64/top-6 is primary and E72 is the alternative in the supplied ARC-AGI-2 pruning plan.

**Decision:** rent four L4s together for the first GPU session. Start with a full-depth synthetic E64 model initialized in native checkpoint-compatible formats and processed through the standard runtime finalizers. Download no DeepSeek checkpoint shards. Budget 128 GB host RAM and 150–200 GB SSD; 192–256 GB RAM is optional comfort for a larger synthetic host Engram scenario. The first session has a four-hour core and a two-hour conditional extension: 16–24 allocated GPU-hours.

Random weights remove the large checkpoint download and disk footprint. They do not remove the approximately 60.1 GiB of E64 GPU weights, initialization/repacking peaks, cache, workspaces, or the chosen in-memory Engram allocation. Stock `--load-format dummy` does not correctly prepare this model; the custom path in the work orders is a prerequisite.

## Read and execute in this order

Start with **[START_HERE.md](START_HERE.md)**. Give **[LOCAL_AGENT_PROMPT.md](LOCAL_AGENT_PROMPT.md)** to the coordinating local agent. **[PACKAGE_INDEX.md](PACKAGE_INDEX.md)** maps the complete archive.

1. **EXPERIMENT_PLAN.md** — complete decision sequence, hardware, controls, workloads, time caps and real-weight transition.
2. **AGENT_WORK_ORDERS.md** — source-backed implementation tasks and GPU correctness gates.
3. **software_lock.json**, **synthetic_contract.json**, **experiment_matrix.json** — machine-readable planning specifications. These are not vLLM launch configs or executable benchmark programs.
4. **tools/prepare_synthetic_metadata.py** — supplied CPU-only preparation helper. It verifies the pinned source config and produces explicitly synthetic E64/E72 metadata. Optional downloads are allowlisted tokenizer/encoding files, capped at 256 MiB total. It does not create weights or implement an engine loader.
5. **tools/probe_environment.py**, **tools/estimate_budget.py** — supplied read-only inventory and arithmetic helpers from the original agent pack.
6. **result_record.template.json**, **decision_report.template.md**, **handoff_contract.template.json** — outputs to fill after actual experiments.
7. **SOURCE_INDEX.md**, **verification/** — provenance and CPU validation, separate from GPU results.

## CPU-only quick start

From this package root, using a Python 3 environment:

```bash
python3 tools/prepare_synthetic_metadata.py --output fixtures/metadata-offline
```

This works without network access, a GPU or model weights. For an engine/tokenizer exercise, use a fresh destination and explicitly request the small asset allowlist:

```bash
python3 tools/prepare_synthetic_metadata.py --output fixtures/metadata-serving --download-metadata
```

Inside a vLLM checkout, follow that repository's environment instructions and use its `uv`/`.venv` Python. No full-model serve command is supplied as working: the custom synthetic loader and compact Engram constructor hook must first exist and pass their gates.

## Status and scope

Included and CPU-validated: plan/specifications, pinned source config and license, metadata preparation helper, budget arithmetic and inventory helper. The GPU initializer, compact native Engram adapter, E64/E72 router patch and integrated benchmark instrumentation are implementation work orders. No paid instance, GPU kernel test, full-model inference, pruning, accuracy test or throughput benchmark was run while preparing this package.

This package supersedes the earlier real-weight-first experiment addendum and its 32-GPU-hour initial allocation. It does not change the pruning order: validate/freeze the Engram mask, collect fresh REAP statistics with that behavior, then export and evaluate E64/E72. The complete original pruning/serving agent pack is included under **reference_original_plan/**. Its relative links remain intact. Root START_HERE.md and EXPERIMENT_PLAN.md control the initial inference rental; older real-weight commands and pruning-machine instructions are later reference material.

