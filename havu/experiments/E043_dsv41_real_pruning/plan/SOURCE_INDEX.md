# Primary source index

Observed 19 October 2026. Source revisions and repository claims can change. Links to papers and model cards establish architecture or reported results; they do not establish measured performance of the proposed 4× L4 deployment. Pinned source inspection and published wheel tests are different evidence.

## Model and architecture

| Source | Used for |
|---|---|
| [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | Exact model release and recommended encoding/sampling; model revision 1bae0e1088993b9b82de08f59de5be5a34e42170 recorded by the serving audit |
| [Exact configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) | Layer topology, expert counts, dimensions, attention and Engram source geometry |
| [Technical report](https://arxiv.org/html/2609.19969v1) | CED, CSA2, cache reuse, sparse indexing, bounded replay, mHC and optional DSpark |
| [Reference model code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py) | Native computation order, routing semantics, cache ownership and reference limitations |
| [Exact Engram helper](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py) | Address geometry, tokenizer compression and hash implementation |
| [Reference quantization kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py) | Quantization boundaries and expert activation details |
| [Reference inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/README.md) | Native reference setup and checkpoint conversion |

## Serving

| Source | Used for |
|---|---|
| [Official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) | Current V4.1/1017 support, teacher-node launch examples, CED/Engram/DSpark and reasoning-effort configuration |
| [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx) | Preview image requirement and documented high-end hardware paths |
| [SM89 vLLM fork](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) | Primary Ada deployment base; audited source 3f3bbaea916707c851ca027b87b97fb6605998b8 |
| [Vision11 release under retained Vision9 tag](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/tag/v0.28.1rc1-vision9-sm89-sm120-cu130) | Wheel pairing, hashes and limited latest-run SM89 validation; wheel Python source 86d35745c8797c9d2524877fb235084a6a5f5a7d |
| [FlashInfer native V4.1 branch](https://github.com/yhfgyyf/flashinfer/tree/deepseek-v4.1-native-sm89-sm120) | Mixed sparse attention; audited source d4eef3c44c70594ff18b9126381ae6a978694e3f |
| [Native attention tests](https://github.com/yhfgyyf/flashinfer/blob/d4eef3c44c70594ff18b9126381ae6a978694e3f/tests/attention/test_deepseek_v41_mixed_sparse.py) | Existing operator parity and graph-state tests to run on an actual L4 |
| [Native router](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/router.py) | Supported-shape whitelist and required pruning adaptation |
| [Native attention/cache ownership](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/attention.py) | Full per-rank KV and indexer replication |
| [Marlin FP4 repacking](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py) | Packed expert values and expanded BF16 scales |
| [V4.1 quantization config](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/quant_config.py) | wo_a BF16 expansion and dense quantization behavior |
| [L20 issue 110](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/110) | Actual SM89 deployment evidence and shape/SM-count failure mode; not an L4 benchmark |
| [vLLM backport](https://github.com/wtdcode/vllm-backport) | Alternative older-GPU packaging and V4.1 integration |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Current supported architectures and format changes; not an Ada solution |

## Engram offload

| Source | Used for |
|---|---|
| [Native fork Engram](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/common/engram.py) | Full-table pinned host offload and TP sharding |
| [Native Engram configuration](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/config/engram.py) | CPU offload defaults; compact row remapping remains a required adaptation |
| [Original Engram paper](https://arxiv.org/html/2601.07372v1) | Deterministic addressing, contextual gating and reasoning effects; trained allocation results do not establish post-hoc ARC row-pruning tolerance |
| [LibertAI two-Spark recipe](https://github.com/Libertai/dsv41-flash-vllm-2x-spark) | Practical exact disk-Engram recipe and observed caveats |
| [Underlying disk/prestage code](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/a2664a7e4ac7bc823ef29c8d7725abfe4d286f8a/patch/cudagraph-prestage) | DiskEngramTable, CPU reads, dequantized staging and graph-boundary integration |
| [Closed experimental vLLM disk-offload PR](https://github.com/vllm-project/vllm/pull/56757) | Alternative implementation/profiling evidence; not a merged stock feature |
| [Follow-up issue](https://github.com/vllm-project/vllm/issues/62238) | Current upstream design discussion, not a deployed capability |

## Pruning and data

| Source | Used for |
|---|---|
| [REAP paper v3](https://arxiv.org/html/2510.13999v3) | Conditional saliency, local substitution assumptions and the normalization ambiguity examined in REAP_IMPLEMENTATION_CONTRACT.md |
| [Cerebras REAP, pinned](https://github.com/CerebrasResearch/reap/tree/1970473c51ca3caeb98c10392f15b3a08a672974) | Source-audited score definition and adaptation limits |
| [Canonical pruning metrics](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/src/reap/pruning_metrics.py) and [observer](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/src/reap/observer.py) | Conditional denominator, generic softmax/all-expert path, misleading pair-frequency and variance fields |
| [Native V4.1 REAP, pinned](https://github.com/Libertai/deepseek-v41-flash-reap/tree/210ccebac92172fb58655c7b95f6a6776f57a641) | Source-audited packed streamer, native-forward shortcut, duplicate-batch counting, single-GPU scope, exporter and verification gaps |
| [HOPE implementation, pinned](https://github.com/awslabs/HOPE/tree/f335c52d6dd2e97a3d2611c08b950386d43fb529) and [paper](https://arxiv.org/abs/2609.18916) | Actual co-selection normalization, second moments, SLSQP behavior and evaluated pruning range |
| [RAZOR implementation](https://github.com/nick7nlp/Razor) and [paper](https://arxiv.org/abs/2609.30465) | Replacement-aware expert scores and task-dependent quality results |
| [REAM](https://github.com/SamsungSAILMontreal/ream) | Expert-alignment and merging alternative |
| [Router distillation research](https://arxiv.org/abs/2603.02217) | Motivation for separately gated router recovery |
| [NVARC authors' repository](https://github.com/1ytic/NVARC) | Synthetic/augmented puzzle dataset IDs and construction artifacts; solver rollouts must be generated through the intended harness |
| [ARC AGI 2 canonical repository](https://github.com/arcprize/ARC-AGI-2) | Canonical task provenance and evaluation roots |

## ds4 components for the pruning prototype

The additional source audit pins `antirez/ds4` at `fc80bd695da76ee14bcb86820bf900a5cdfda806` (7 October 2026). Its bundled model recipes predate 1017. Reuse the listed components with the exact 1017 metadata; do not treat its older converted artifacts or backend measurements as 1017 validation. See DS4_REUSE.md for the integration work orders.

| Source | Used for |
|---|---|
| [Engram CPU API](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_engram.h) and [implementation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_engram.c) | Standalone rolling hashes, original 264-byte row reads, bounded deduplication and reference dequantization |
| [Metadata generation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/deepseek41_metadata.py) | Source tokenizer normalization, original primes and multipliers without constructing a GGUF |
| [V4.1 converter](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/deepseek41_quantize.py) | Bounded native Engram row packing and expert enumeration; compact selection is new work |
| [Core model implementation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c) | Full-shape validation, sequential imatrix replay, observer integration points, V4.1 single-GPU-per-rank restriction and F32 cache allocation |
| [CUDA implementation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu) | Selected-expert streaming and router shape restrictions |
| [Model support documentation](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/MODELS.md) | V4.1-specific formats and concurrency limitations; distinguish from V4's Ada multi-GPU claims |
| [Engram CPU tests](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/tests/test_engram.c) | Existing hashing/history, bounded row I/O and value-rounding tests; executed successfully on CPU in this audit |

## Hardware prices and competition

| Source | Used for |
|---|---|
| [NVIDIA compute capabilities](https://developer.nvidia.com/cuda-gpus) | L4 is SM89 |
| [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | 96 GB GDDR7, 1,597 GB/s per-card memory bandwidth |
| [NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) | Up to 288 GB HBM3e, 8 TB/s and 1.8 TB/s bidirectional NVLink; actual rented SKU/topology matters |
| [SM89/SM120 fork hardware report](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) | Publisher-reported full V4.1 operation on four RTX PRO 6000 96 GB cards with TP4/EP and CPU Engram; not a REAP benchmark |
| [CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) | Standard driver baseline |
| [CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) | Conditional alternative; must be checked against actual host |
| [Runpod pricing](https://www.runpod.io/pricing) | Example L4, RTX PRO 6000, B300 and H200 rental rates; page updated 8 October 2026 and retrieved 19 October |
| [Lambda pricing](https://lambda.ai/pricing) | Two-B200 node rate and bundled host RAM/storage |
| [DeepSeek API pricing](https://api-docs.deepseek.com/quick_start/pricing/) | 1017 API alias and input/output peak/off-peak token rates |
| [Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) and [rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules) | Notebook runtime and live rule gate; public page extraction did not establish host RAM/driver/storage limits |
| [ARC Prize 2026 AGI 2 track](https://arcprize.org/competitions/2026/arc-agi-2) | Offline notebook requirement and exact two-output scoring |

## Evidence still required from implementation

Additional 19 October follow-up: [CED_CALIBRATION.md](CED_CALIBRATION.md) records the exact prompt/continuation scoring policy derived from the official report and native reference. [COMMUNITY_PORTS.md](COMMUNITY_PORTS.md) pins the catonooka/Maya artifacts and distinguishes public GLM code from author-reported DeepSeek prototype results.

- Native observer-on/off and reference/Marlin parity on actual 1017 weights after the audited collector fixes.
- Long-context query-tiling parity and actual 32K/64K calibration peaks; distributed counting and measured RTX replay cost.
- Actual L4 operator correctness, peak memory and end-to-end throughput.
- The current Kaggle driver, host RAM, P2P topology, attached-input capacity and cold random-read behavior.
- Exact parity of the chosen wheel pair and pinned source build after patches.
- ARC quality for the 1017 checkpoint at E64/E72 under a frozen compact Engram mask, with long trajectories and clean root holdouts; smaller expert budgets are optional extensions.
- The retained capability of the tool-free specialist as expert and Engram budgets shrink; the user's reported tool-free pass@2 baseline is 80% and is accepted as provided context.
- Recoverability under router updates or LoRA with a differentiable native computation path.
- Whether optimized batching, prefix caching, CED or DSpark improves the final competition score within its runtime.
