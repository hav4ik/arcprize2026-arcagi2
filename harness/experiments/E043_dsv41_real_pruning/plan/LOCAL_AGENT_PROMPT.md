# Prompt for the local coordinating agent

Copy the following into the local agent that will work on the rented machine. Give it the extracted packet and the existing SSH connection details/configuration. Keep large downloads and model execution on the remote node.

---

Implement the DeepSeek-V4.1-Flash-1017 pruning project in this packet. My current Vast.ai machine has **eight RTX 6000 GPUs and 1 TB disk**; this supersedes the earlier four-GPU/2 TB description. The intended SKU is RTX PRO 6000 Blackwell 96 GB; verify actual GPU names, compute capability, allocated RAM, disk quota/free space, driver and topology before assuming the hardware matches. Use my existing SSH configuration; ask only for connection details that are genuinely missing. Do all independent local/CPU preparation while resolving a connection blocker.

Read RENTED_NODE_START.md and DISK_1TB_PLAN.md first, then START_HERE.md, REAP_IMPLEMENTATION_CONTRACT.md, CED_CALIBRATION.md, TASKS_ENGRAM_FIRST.md and TASKS_PRUNING.md. Use TASKS_SERVING.md for the pinned native runtime and later four-L4 deployment work; its L4-specific architecture flags do not apply to the current Blackwell node. RENTED_NODE_START.md supplies the SM120 overrides and the current execution order.

Act on the work orders and produce implementation, not another general research plan. Work in an isolated project directory, preserve existing files and source weights, and keep commits/patches and run records. The supplied CPU helpers and plan documents are the initial assets; the loader, native collector, compaction adapter and exporter remain implementation work.

Start with environment inventory and the supplied CPU contracts. Verify the existing container's interpreter and paired vLLM/FlashInfer stack before changing packages. Run small CUDA, four/eight-rank communication and native operator checks before committing to a full checkpoint download. Preserve the original packed expert representation; never materialize the full teacher in BF16. Keep one canonical source checkpoint, convert/repack into RAM/VRAM during loading, and export one candidate at a time within the 1 TB allocation. Before a large write, usable free space must cover the worst-case additional live bytes plus the 150 GB planning reserve. Avoid a full converted backbone copy and a duplicate full Engram sidecar. Measure actual existing disk usage and all temporary coexistence.

Use a short TP4/EP baseline on four of the already rented GPUs, then validate one TP8/EP resident model as the intended main collector. Both use all 384 original experts. Two independent TP4 workers are an optional fallback/comparison only after correctness and aggregate host-memory checks, sharing immutable source files; the full CPU Engram table may otherwise be duplicated. A one-card packed reference is an optional debugging path inside this rental. Count all eight reserved GPUs in cost even when fewer are active. Do not rent additional machines or change the paid allocation without asking me.

Run these workstreams in parallel where practical: source/data manifests and exact tokenization; CPU Engram address census and compact row export; corrected native reference/observer; SM120 resident runtime and distributed observer integration. Use compatible tool-free ARC traces already available to seed a small pilot. If the corpus is missing, finish synthetic/reference and environment work, then identify the exact trace/tokenizer/harness inputs needed. Do not launch an unnecessary billion-token data-generation job.

Preserve these research decisions:

- First frequency-prune Engram rows while preserving original hashes, canonical token mapping and original-to-compact addressing. Missing retained-table rows use the exact declared zero-before-Wkv behavior. Validate and freeze the mask before collecting final MoE statistics.
- Collect fresh expert statistics using all 384 original routed experts with the frozen Engram mask. Preserve native gate placement and activation quantization. Follow the audited conditional normalization, score naming and distributed reduction rules. Do not reuse the upstream collector's altered forward or multiply already gate-weighted contributions by the gate again.
- Replay the necessary causal history through both architectural halves. Prove per-layer/phase counts. Encoder prompt/continuation and decoder generation positions have distinct scoring masks; the final prompt token predicts the first generated token. Engram frequency counts semantic consumed inputs once, not every physical warmup/replay.
- Start calibration eager, with CED decoder-prefill skipping, DSpark and prefix caching off. Prove full-position observer visibility; disabling a serving flag alone is insufficient. Tile the reference indexer before long histories, and measure native-runtime workspaces separately.
- E64, top-6 unchanged, is the primary candidate; E72 is the larger comparison. Preserve shared experts, rerouting/refill, native selected-set normalization, RMS/mHC behavior and attention topology. REAP is the baseline; HOPE, merging and recovery are optional later experiments driven by results.
- The final target remains four L4 GPUs. RTX measurements establish calibration behavior only; never report them as L4 throughput or memory validation.

After Engram validation and observer parity, run a 0.25–1M processed-token pilot, record scored positions separately from processed history, and decide whether more data is warranted. Save resumable sufficient statistics, checksums, immutable shard IDs, commands actually executed, per-rank memory peaks, elapsed time and billed whole-node time. Keep original/full-table, frozen-mask and pruned-model results distinct.

Continue routine implementation, reversible repairs and small correctness experiments autonomously. If the available RAM is below the recommended 512 GB, evaluate a bounded placement using the documented path rather than assuming the node must be replaced. Ask only when an actual missing input/access issue or additional paid resource prevents progress. Do not infer or silently choose a quality-loss tolerance: build the measurements and report the candidate quality/memory frontier.

First deliver `runs/bootstrap/`, `locks/rented_node.json`, the source/data inventory, implemented code/patches, and a concise report of what passed, what failed and the next executable step. Then proceed through the Engram-first integration sequence. A successful import, loaded model, or fluent short completion is not completion of the pruning project.

---
