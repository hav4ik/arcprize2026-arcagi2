# Start on the rented Vast.ai eight-GPU node

Updated **23 October 2026**, including the user's correction: the active rental is **8 × RTX 6000 with 1 TB disk**. This supersedes the earlier four-card/2 TB rental description. The expected SKU from the discussion is **RTX PRO 6000 Blackwell, 96 GB per GPU, SM120**; verify that on the node. The actual RAM allocation, driver, image and SSH endpoint have not been supplied or tested here.

The current job is **Engram pruning and REAP calibration on this rented node**. The final inference target is a separate **4 × L4, SM89** environment. This addendum takes precedence over older rental-selection advice; it does not change the pruning math, native computation or validation gates.

## 1. Start immediately with the existing rental

1. Read this file, [LOCAL_AGENT_PROMPT.md](LOCAL_AGENT_PROMPT.md), [START_HERE.md](START_HERE.md), [REAP_IMPLEMENTATION_CONTRACT.md](REAP_IMPLEMENTATION_CONTRACT.md), and [CED_CALIBRATION.md](CED_CALIBRATION.md).
2. Use **one TP8/EP resident model** as the intended main implementation. Establish a short TP4 baseline on four of the already rented cards, then verify TP8 outputs, observation counts and memory before scaling. TP8 has not been validated by this research. The corrected one-card packed streamer remains a reference/debugging option inside the rental, not a turnkey distributed collector.
3. Keep the original 384 routed experts per backbone layer while evaluating Engram masks and collecting teacher statistics. The E64/E72 models are outputs of pruning, not the scoring teacher.
4. Use existing compatible tool-free traces for a small pilot. Implement independent environment, data and CPU census work in parallel. A billion-token generation job is not a prerequisite.
5. Keep model weights and large intermediates on the rented node. Use the local machine to orchestrate over SSH and retain code, manifests, compact statistics and reports. Follow [DISK_1TB_PLAN.md](DISK_1TB_PLAN.md): one canonical source checkpoint, runtime conversion, bounded scratch and one physical candidate export at a time.

The packet supplies plans and CPU reference helpers, **not a completed model loader, REAP collector, exporter or GPU-validated pruning engine**. Paths described as outputs in task documents must be created by the implementing agent.

## 2. Locate the working directory and inspect the node

Extract the ZIP to an isolated directory on the rented node, for example:

```bash
mkdir -p /workspace/deepseek-pruning
unzip deepseek_v41_arc_agi2_agent_pack.zip -d /workspace/deepseek-pruning
cd /workspace/deepseek-pruning/arc_deepseek_v41_plan
mkdir -p runs/bootstrap locks

python3 tools/probe_environment.py --paths . /tmp /dev/shm \
  > runs/bootstrap/environment.json
nvidia-smi topo -m > runs/bootstrap/topology.txt
nvcc --version > runs/bootstrap/nvcc.txt
```

These paths are examples. Discover the real storage mount and use it consistently. `probe_environment.py` is supplied and read-only; the shell commands write only their named reports. It does not execute a CUDA matrix operation or benchmark communication.

Record in `locks/rented_node.json`:

| Item | Expected or required action |
|---|---|
| GPU inventory | Eight Blackwell RTX PRO 6000 cards, each about 96 GB; compute capability `(12, 0)` |
| Driver/toolkit | Host supports the CUDA 13 stack; record exact driver and toolkit versions and execute the native operators |
| Host memory | Record allocated/effective cgroup limit and current usage, not only physical host RAM; separately check pinned-allocation behavior and relevant process limits |
| Storage | Record Vast's 1 TB quota, actual free bytes on the chosen mount, and local-versus-network storage; apply DISK_1TB_PLAN.md before each large write |
| Shared memory | Inspect `/dev/shm`; do not assume generic Docker flags were honored by Vast |
| GPU topology | Record PCIe/NUMA topology; prove both four-rank baseline and eight-rank communication before assuming P2P works |
| Runtime provenance | Container tag/digest if available; executable/import paths; package versions; source commits and patches |

**512 GB allocated RAM is the recommended full-Engram configuration, not a confirmed property of this rental.** Original Engram payload is about 203 GB. If the allocation is smaller, use a documented bounded lookup/streaming design or validate a compact-table path; do not attempt an unbounded full-table allocation or silently replace missing rows with a different rule. Continue independent CPU work while resolving that placement.

Filesystem tools can report the backing filesystem's capacity instead of a container quota. Check the actual Vast allocation too. Avoid destructive disk tests and never benchmark writes into source weights.

## 3. Use the pinned container's environment

Recommended image:

```text
ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
```

Published digest observed during the 22 October lookup:

```text
sha256:af58a59d32d65fbed1785f265bc9b9969a1010b8e1c6a588087adf796021911e
```

Record the image actually used. A different installed image is an environment to inspect and reconcile; do not claim the expected digest is installed without evidence. The expected published stack is Python 3.12, CUDA 13.0, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.16.1, vLLM Vision11, and paired FlashInfer Vision2. Exact wheel versions and hashes are in [TASKS_SERVING.md](TASKS_SERVING.md).

In Vast, **Interactive shell server, SSH** replaces the image's entrypoint. Packages remain installed, but entrypoint-only activation and environment setup may not run. Locate the existing interpreter containing the pinned stack, inspect its imports, and use that environment. Do not create an empty environment and assume the installed packages followed it. Creating a separate environment is appropriate for an intentional source build; preserve the baseline environment and lock both.

After activating that interpreter, set these in each relevant shell or a project-local setup file:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0
export VLLM_USE_V2_MODEL_RUNNER=1
unset VLLM_MARLIN_INPUT_DTYPE

export TRITON_PTXAS_BLACKWELL_PATH="$(python -c \
'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas-blackwell")')"
"$TRITON_PTXAS_BLACKWELL_PATH" --version
```

The pinned recipe expects the bundled `ptxas-blackwell` to report CUDA 13.1; toolkit/PyTorch stay cu130. Do not overwrite the selected FlashInfer with upstream latest. Do not disable native activation-quantization checks to force a generic kernel to run.

**SM120 override:** the `FLASHINFER_CUDA_ARCH_LIST=8.9` and unset-Blackwell-assembler instructions in TASKS_SERVING.md apply to the later L4 target, not this rented node. Likewise, that document's `uv venv` installation recipe is an alternative build path, not an instruction to discard a functioning container environment.

The minimum CUDA 13.x compatibility driver family is R580. Because this stack JIT-compiles native operators and uses additional assembler tooling, a driver-family check alone is not a pass. Installing a toolkit inside the container does not upgrade the host driver.

## 4. Complete short checks before downloading the full model

Run the supplied CPU contracts:

```bash
python3 tools/reap_stats_reference.py --self-test
python3 tools/ced_phase_masks_reference.py --self-test
python3 tools/compact_engram_reference.py
```

Then implement and execute a small device-check script that:

1. Imports the expected `torch`, `vllm`, and `flashinfer` from the intended environment and records their paths and versions.
2. Allocates and multiplies small BF16 matrices on each of the eight GPUs, synchronizes, and checks finite results.
3. Runs four-rank and eight-rank NCCL sums with known expected results and bounded timeouts; records topology/transport failures rather than hanging indefinitely.
4. Runs the existing small native V4.1 attention tests and a representative native quantized expert operation on SM120. A generic matmul does not prove these kernels work.

Use the source map and tests in TASKS_SERVING.md, adapted to SM120. Report CPU, basic CUDA, distributed communication, native operators and real-weight parity as separate results. The new device-check script is an implementation task; it is not included as a prevalidated command in this ZIP.

If a version conflict or missing build dependency is repairable within the rented container, fix it and document the change. Request user input only for a concrete missing credential, inaccessible required data, or an additional paid resource that cannot be avoided.

## 5. Weights and disk layout

Pin `deepseek-ai/DeepSeek-V4.1-Flash-1017` to revision:

```text
1bae0e1088993b9b82de08f59de5be5a34e42170
```

Inventory the config, index and tensor geometry first. Maintain one canonical source checkpoint location. Avoid downloading both a complete Hugging Face cache and a second copied source tree. Source weights are immutable. The 1 TB allocation is feasible only with controlled peak disk usage; the detailed implementation contract is [DISK_1TB_PLAN.md](DISK_1TB_PLAN.md).

Proposed simultaneously live payloads, in decimal GB:

| Payload | Approximate amount |
|---|---:|
| Original source checkpoint | 510 GB |
| Standalone compact Engram and maps, if also copied into the candidate | 45–50 GB allowance |
| One E64/E72 export including its compact host table | 100–120 GB allowance, format dependent |
| Pilot traces, census arrays, statistics and logs | 20–50 GB allowance |
| Software, writable container state and build/download caches | 40–70 GB allowance |
| Free space to retain after each large write | 150 GB planning floor |

The proposed live payload ranges total about 715–800 GB before the free-space floor; actual source/file sizes and existing disk usage must replace these estimates. Do not also serialize a full converted 300–320 GB backbone or duplicate the 203 GB original Engram table by default. Repack into RAM/VRAM during loading and read source table extents directly. Do selective extent reads and avoid physical exports of every candidate. Account for temporary coexistence during atomic writes. Store compact sufficient statistics, not token-by-expert hidden-state dumps. Before an operation, require usable free bytes to cover its worst-case additional live bytes **plus** the 150 GB floor.

The packed backbone has a provisional aggregate payload near 320 GB before calibration buffers and topology-dependent replication. Eight cards provide more residency/workspace headroom, but this is not a measured TP8 peak. Preserve packed expert bytes and native quantization boundaries. Expanding the entire 552B backbone to BF16 would require roughly 1.1 TB and is not this implementation.

## 6. Correct eight-card collection

Establish the original teacher's short native **TP4/EP baseline on four of the rented GPUs**, then validate **TP8/EP on all eight**, keeping all 384 original experts. Compare native outputs, phase-specific counts, statistics and memory at each topology. Start eager, with DSpark off, prefix caching off, and CED decoder-prefill skipping off. Use supported loader options to exclude unneeded draft/vision state; record what is actually loaded. CPU Engram remains the baseline placement. Verify per-rank memory rather than treating eight devices as one VRAM pool. Pruned E64/E72 router support and physical L4 tests are separate deployment tasks; neither should delay establishing the original teacher and observer parity on this node.

If TP8 exposes an unresolved kernel/collective issue or performs poorly, compare two independent TP4/EP workers on disjoint corpus shards after correctness is proven. They must share immutable source files without another checkpoint copy and have explicit GPU ownership. Account for possibly duplicated pinned CPU Engram, rank-local buffers and disk bandwidth before launching both; eight available GPUs do not guarantee sufficient host RAM for two model processes. A single correct TP4 worker is an interim fallback, with the entire eight-GPU node still counted in cost. No alternate rental is required just to try these topologies.

**Turning off CED is necessary but not by itself proof of complete replay.** Verify that all required positions execute all 40 backbone layers and reach the observer. The standard serving entrypoint may select only last-position outputs or skip observer visibility. Instrument actual selected-expert contributions before top-k summation and prove token counts against the corrected reference.

Apply REAP_IMPLEMENTATION_CONTRACT.md exactly: preserve native gate placement inside quantization-sensitive expert computation, distinguish native-amplitude scores from reference REAP, use the appropriate conditional denominators, keep routing biases separate from mixture weights, reduce TP partial vectors before taking norms, and count EP observations once. Do not run the existing single-GPU LibertAI script unchanged under `torchrun`.

The 8B-prefill/16B-decode design requires phase-aware scoring, not halving the teacher or assigning a blanket 2× weight. Encoder layers 0–19 and decoder layers 20–39 have different scored-position masks. The last prompt input predicts the first generated token. Count Engram on each eligible consumed encoder input once; physical replay/warmup must not multiply frequency. See CED_CALIBRATION.md for exact intervals and boundary tests.

For the reference whole-prefix path, tile the indexer's query axis while preserving all causal keys and candidate rules before attempting 32K/64K histories. Arbitrary multi-token continuation at nonzero offsets is not an already supported substitute. For the native runtime, measure its actual workspace separately. Small persistent KV does not imply small prefill temporaries.

## 7. Experiment order and first report

| Stage | Work to complete | Evidence before advancing |
|---|---|---|
| Bootstrap | Inventory, interpreter lock, CPU contracts, small GPU/native/communication checks | `runs/bootstrap/` and `locks/rented_node.json` with actual results |
| Source/data | P00/P01, exact tokenization, root-disjoint calibration/selection/audit partitions | Source/tensor/data manifests and small compatible traces |
| Engram prototype | E01–E03, original-ID frequency census, compact row exporter; P02a forward work in parallel | Hash/address parity, compact mapping checks, coverage curves |
| Engram validation | E04 using original full table and candidate masks | Fixed-prefix checks, complete held-out rollouts, chosen frozen mask/checksum |
| Final teacher setup | P02b with all original experts and frozen Engram behavior | Observer-on/off parity, all-40-layer counts, correct reductions and measured peaks |
| Calibration pilot | Initially 0.25–1M processed full-history tokens | Scored/processed counts, score stability, throughput, per-rank peaks, billed node time |
| Pruned candidates | E64/top6 primary, E72 comparison; larger retained counts only as diagnostics if needed | Correct rerouting/normalization, full-rollout quality, export parity |
| Deployment handoff | Selective native-format export and frozen Engram adapter | Weight/map checksums and later tests on the actual four-L4 environment |

Do not begin final REAP statistics collection before the Engram mask is frozen. Short debugging passes using the full table are allowed and must be labeled separately. If the mask or native forward changes, invalidate incompatible statistics instead of merging them.

The first report should say what ran, exact pins, what passed or failed, current storage/RAM/VRAM use, the available trace corpus, and the next implementation step. Measure time using the whole rented eight-GPU node even when a reference experiment uses one or four cards. Increase beyond the pilot only when measured coverage or quality uncertainty warrants it. No quality-loss tolerance was specified; report measured quality/cost tradeoffs without inventing an acceptance threshold.

## Sources and status

- [Published GHCR package and digest](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pkgs/container/vllm-deepseek-v4-sm89)
- [Fork setup and validated SM120 recipe](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/main/README_EN.md)
- [Vast template settings and SSH entrypoint behavior](https://docs.vast.ai/guides/templates/template-settings)
- [Vast allocated resources, storage, shared memory and SSH environment behavior](https://docs.vast.ai/guides/instances/docker-environment)
- [CUDA 13.0 release/driver compatibility notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)
- The packet's pinned primary sources and source-derived memory calculations are in SOURCE_INDEX.md, REAP_HARDWARE.md and TASKS_SERVING.md.

Public runtime/template details were inspected on 22 October. This 23 October update incorporates the user's rental and packages instructions; it does not claim that the rental was accessed or tested from this conversation.
