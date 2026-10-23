#!/usr/bin/env python3
"""Estimate DeepSeek-V4.1-Flash weights, native persistent state and runtime costs.

This is an arithmetic planning tool, not a measured engine memory profiler.
Architecture: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json
Audited native runtime: yhfgyyf/vllm-deepseek-v4-sm89 at
3f3bbaea916707c851ca027b87b97fb6605998b8. Sources are listed in JSON output.

All GiB arguments are aggregate across the GPUs unless their name says per-GPU.
The default packing models W4A16 Marlin: 4-bit values + BF16 scale per 32 values.
Native defaults reserve an OPTIONAL 128-token CED tail for every configured
request slot; use --ced-window 0 when CED is disabled. This planning default
does not imply the runtime enables CED by default. The residual --workspace-gib
reserve is IN ADDITION to the separately counted known shared buffers.

Engram lookup-table GPU residency is EXCLUDED unless --engram-gib is set.
Engram readout weights belong in --fixed-gib. CPU table/remap storage, optional
DSpark and vision, and loading/conversion peaks need separate gates.
Custom cache layouts do not inherit native state or workspace assumptions.
"""

import argparse
import json

GIB = 1024**3
MIB = 1024**2
LAYERS = 40
ORIGINAL_EXPERTS = 384
EXPERT_PARAMETERS = 3 * 5120 * 2304
FIXED_BACKBONE_PARAMETERS_APPROX = 8_298_000_000
ENGRAM_PARAMETERS = (384_006_168 + 384_016_682) * 256
ENGRAM_NATIVE_BYTES = (384_006_168 + 384_016_682) * 264
PACKING_BYTES = {"marlin_fp4": 0.5625, "native_fp4": 0.53125, "fp8_block32": 1.03125}
SOURCE_ROOT = "https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/"
SOURCES = {
    "native_cache": SOURCE_ROOT + "vllm/models/deepseek_v4_1/attention.py",
    "compressor_rings": SOURCE_ROOT + "vllm/models/deepseek_v4_1/compressor.py",
    "indexer_workspace": SOURCE_ROOT + "vllm/model_executor/layers/sparse_attn_indexer.py",
    "workspace_default": SOURCE_ROOT + "vllm/envs.py",
    "candidate_buffers": SOURCE_ROOT + "vllm/models/deepseek_v4_1/nvidia/model.py",
    "ced_tail": SOURCE_ROOT + "vllm/models/deepseek_v4_1/nvidia/ced.py",
    "swa_paging": SOURCE_ROOT + "vllm/v1/kv_cache_interface.py",
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_float(value):
    number = float(value)
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experts", type=positive_int, nargs="+", default=[64, 72])
    p.add_argument("--packing", choices=PACKING_BYTES, default="marlin_fp4")
    p.add_argument("--fixed-gib", type=nonnegative_float, default=12.6, help="aggregate non-routed weight reserve; excludes Engram tables")
    p.add_argument("--engram-gib", type=nonnegative_float, default=0, help="additional aggregate GPU Engram table/cache allocation")
    p.add_argument("--gpus", type=positive_int, default=4)
    p.add_argument("--gpu-gib", type=nonnegative_float, default=22.3, help="physical GiB per GPU; replace with measured value")
    p.add_argument("--utilization", type=nonnegative_float, default=0.90)
    p.add_argument("--workspace-gib", type=nonnegative_float, default=4, help="additional aggregate reserve for activations, graphs, Engram staging, extra SWA pages and other uncounted peaks; excludes separately counted shared buffers")
    p.add_argument("--context", type=positive_int, default=65536)
    p.add_argument("--batch", type=positive_int, default=32)
    p.add_argument("--cache-layout", choices=["native", "custom"], default="native", help="native is the audited mixed FP8/FP4 layout; custom requires explicit component assumptions")
    p.add_argument("--kv-bytes-per-token", type=positive_int, nargs="+", default=None, help="linear global+index history bytes/token/cache copy; native is fixed at 890, custom values are arithmetic hypotheses")
    p.add_argument("--cache-replication", type=positive_int, default=4, help="copies of each request's cache across the deployment; verified TP4 path uses 4")
    p.add_argument("--swa-bytes-per-request", type=nonnegative_float, default=None, help="override logical live local-cache floor; excludes additional replay/padding")
    p.add_argument("--compressor-bytes-per-request", type=nonnegative_float, default=None, help="override ring bytes per request/cache copy; native default derives three FP32 rings from speculative-tokens")
    p.add_argument("--speculative-tokens", type=nonnegative_int, default=0, help="native ring sizing only; does not include draft-model weights or other speculative buffers")
    p.add_argument("--max-num-batched-tokens", type=positive_int, default=8192, help="native shared candidate/top-k output capacity per GPU")
    p.add_argument("--indexer-logits-mib", type=positive_int, default=512, help="native logits workspace MiB per GPU, matching VLLM_SPARSE_INDEXER_MAX_LOGITS_MB")
    p.add_argument("--shared-buffers-gib", type=nonnegative_float, default=None, help="override AGGREGATE indexer scratch plus candidate/top-k buffers; required to complete custom-layout estimates")
    p.add_argument("--max-num-seqs", type=positive_int, default=None, help="reserved request slots for native CED; defaults to batch")
    p.add_argument("--ced-window", type=nonnegative_int, default=None, help="native CED tail capacity: planning default 128; 0 disables; 1152 includes 1024 vision tokens")
    p.add_argument("--ced-gib", type=nonnegative_float, default=None, help="override AGGREGATE CED state; required to complete custom layouts (explicit 0 means absent)")
    p.add_argument("--tokens", type=positive_int, default=1_000_000, help="output tokens for optional cost calculation")
    p.add_argument("--throughput", type=nonnegative_float, help="measured or assumed aggregate output tokens/second across the entire node")
    p.add_argument("--hourly-usd", type=nonnegative_float, help="USD/hour for the entire rented node, not one GPU")
    p.add_argument("--format", choices=["json", "markdown"], default="markdown")
    return p


def runtime_components(args, kv_bytes):
    """Known persistent bytes, with unknown custom components kept explicit."""
    native = args.cache_layout == "native"
    copies = args.cache_replication
    # Three ratio-2 sources and one ratio-1 source own BOTH global and index K.
    # 890 is the asymptotic per-original-token size; retain exact odd-L behavior.
    history_per_request = (
        (288 + 68) * (3 * (args.context // 2) + args.context)
        if native else args.context * kv_bytes
    )
    history = history_per_request * args.batch * copies
    swa_per_request = args.swa_bytes_per_request
    if swa_per_request is None and native:
        swa_per_request = LAYERS * min(args.context, 128) * 528
    ring_rows = max(8, 1 << (args.speculative_tokens + 1).bit_length())
    ring_per_request = args.compressor_bytes_per_request
    if ring_per_request is None and native:
        ring_per_request = 3 * ring_rows * (2 * 512) * 4

    native_buffers = None
    shared = None
    if args.shared_buffers_gib is not None:
        shared = args.shared_buffers_gib * GIB
    elif native:
        logits_elements = args.indexer_logits_mib * MIB // 4
        native_buffers = {
            "logits": logits_elements * 4,
            "block_scores": (logits_elements // 8) * 4,
            "lengths": (logits_elements // (2048 * 8)) * 4,
            "radix_topk": MIB,
            "candidate_and_topk_outputs": args.max_num_batched_tokens * (2048 + 512) * 4,
        }
        shared = sum(native_buffers.values()) * args.gpus

    max_seqs = args.max_num_seqs if args.max_num_seqs is not None else args.batch
    ced_window = args.ced_window if args.ced_window is not None else (128 if native else None)
    ced = None
    if args.ced_gib is not None:
        ced = args.ced_gib * GIB
    elif native:
        ced = max_seqs * ced_window * (4 * 5120 * 2 + 4 * 4 + 8) * args.gpus

    components = {
        "global_and_index_history": history,
        "swa_live_floor": None if swa_per_request is None else swa_per_request * args.batch * copies,
        "compressor_rings": None if ring_per_request is None else ring_per_request * args.batch * copies,
        "shared_buffers": shared,
        "ced_tail": ced,
    }
    unknown = [name for name, value in components.items() if value is None]
    return {
        "bytes": components,
        "unknown_components": unknown,
        "known_nonweight_persistent_bytes": sum(value for value in components.values() if value is not None),
        "native_shared_buffers_bytes_per_gpu": native_buffers,
        "native_compressor_ring_rows": ring_rows if native else None,
        "ced_reserved_request_slots": max_seqs if native else None,
        "ced_window": ced_window,
    }


def estimate(args):
    capacity = args.gpus * args.gpu_gib * args.utilization
    rows = []
    for experts in args.experts:
        routed_parameters = LAYERS * experts * EXPERT_PARAMETERS
        expert_gib = routed_parameters * PACKING_BYTES[args.packing] / GIB
        backbone = routed_parameters + FIXED_BACKBONE_PARAMETERS_APPROX
        for kv_bytes in args.kv_bytes_per_token or [890]:
            runtime = runtime_components(args, kv_bytes)
            components = runtime["bytes"]
            global_bytes = components["global_and_index_history"]
            swa_bytes = components["swa_live_floor"]
            ring_bytes = components["compressor_rings"]
            cache_gib = (
                None if swa_bytes is None or ring_bytes is None
                else (global_bytes + swa_bytes + ring_bytes) / GIB
            )
            weights_gib = expert_gib + args.fixed_gib + args.engram_gib
            known_lower_bound = weights_gib + runtime["known_nonweight_persistent_bytes"] / GIB + args.workspace_gib
            complete = not runtime["unknown_components"]
            required = known_lower_bound if complete else None
            headroom = None if required is None else capacity - required
            rows.append({
                "experts_per_layer": experts,
                "removed_expert_percent": (1 - experts / ORIGINAL_EXPERTS) * 100,
                "backbone_parameters_b_approx": backbone / 1e9,
                "backbone_plus_full_engram_parameters_b_approx": (backbone + ENGRAM_PARAMETERS) / 1e9,
                "global_kv_bytes_per_token_per_copy": kv_bytes,
                "cache_layout": args.cache_layout,
                "routed_expert_gib": expert_gib,
                "resident_weights_gib": weights_gib,
                "resident_weights_decimal_gb": weights_gib * GIB / 1e9,
                "global_kv_gib": global_bytes / GIB,
                "local_kv_floor_gib": None if swa_bytes is None else swa_bytes / GIB,
                "compressor_rings_gib": None if ring_bytes is None else ring_bytes / GIB,
                "cache_gib": cache_gib,
                "shared_buffers_gib": None if components["shared_buffers"] is None else components["shared_buffers"] / GIB,
                "ced_tail_gib": None if components["ced_tail"] is None else components["ced_tail"] / GIB,
                "runtime_components": runtime,
                "known_nonweight_persistent_gib": runtime["known_nonweight_persistent_bytes"] / GIB,
                "residual_workspace_reserve_gib": args.workspace_gib,
                "known_lower_bound_with_residual_reserve_gib": known_lower_bound,
                "estimate_components_complete": complete,
                "required_aggregate_gib": required,
                "headroom_within_utilization_cap_gib": headroom,
                "headroom_per_gpu_if_balanced_gib": None if headroom is None else headroom / args.gpus,
                "fits_aggregate_budget": None if headroom is None else headroom >= 0,
            })
    cost = None
    if args.throughput is not None:
        hours = args.tokens / args.throughput / 3600
        cost = {"output_tokens": args.tokens, "aggregate_tokens_per_second": args.throughput,
                "generation_hours": hours, "node_hourly_usd": args.hourly_usd,
                "generation_usd": None if args.hourly_usd is None else hours * args.hourly_usd,
                "excludes": "prefill, loading, calibration, retries, recovery training, idle time and storage"}
    return {
        "assumptions": {**vars(args), "packing_bytes_per_parameter": PACKING_BYTES[args.packing],
                        "usable_aggregate_gib": capacity, "expert_parameters": EXPERT_PARAMETERS,
                        "fixed_backbone_parameters_approx": FIXED_BACKBONE_PARAMETERS_APPROX,
                        "full_engram_parameters": ENGRAM_PARAMETERS,
                        "full_engram_native_table_gib": ENGRAM_NATIVE_BYTES / GIB,
                        "full_hash_space_uint32_remap_gib": (384_006_168 + 384_016_682) * 4 / GIB},
        "source_pins": SOURCES,
        "limitations": [
            "Aggregate fit assumes balanced shards; the largest rank can still run out of memory.",
            "Fixed weights and the residual workspace reserve are assumptions to replace with engine measurements, not proof of fit.",
            "Native shared buffers are counted separately; workspace-gib covers OTHER activations, graph/allocator peaks, Engram staging and extra SWA pages.",
            "Engram lookup-table GPU memory is only --engram-gib; GPU readout weights belong in fixed-gib. CPU tables/remaps and export/loading peaks are not included.",
            "SWA is live content only. Page alignment, in-flight prefill chunks, extra retained tokens and cached prefixes can add memory.",
            "Native history is 890 bytes/token PER TP rank, including index K; cache replication is independent of query-head sharding.",
            "Native CED128 is a conservative planning reservation, not a runtime default. Use ced-window 0 when disabled; retained vision capacity can require 1152.",
            "Custom layouts never inherit native SWA/rings/shared buffers/CED. Missing overrides make totals and fit unknown; hypothetical layouts are not supported-engine claims.",
            "Optional vision/draft loading and memory peaks during conversion are not included.",
            "Expert-count arithmetic does not establish kernel support; smaller counts may need a router whitelist patch.",
        ],
        "scenarios": rows, "cost": cost,
    }


def markdown(result):
    a = result["assumptions"]
    def number(value):
        return "unknown" if value is None else f"{value:.2f}"

    lines = ["# DeepSeek-V4.1 memory estimate", "",
             f"Budget: {a['gpus']} × {a['gpu_gib']:.2f} GiB at {a['utilization']:.0%} utilization = **{a['usable_aggregate_gib']:.2f} GiB**.",
             f"Packing: {a['packing']}; non-routed weights {a['fixed_gib']:.2f} GiB; **additional residual reserve {a['workspace_gib']:.2f} GiB**; GPU Engram tables {a['engram_gib']:.2f} GiB.",
             f"Layout {a['cache_layout']}; context {a['context']:,}; batch {a['batch']}; cache copies {a['cache_replication']}. Full host Engram payload is {a['full_engram_native_table_gib']:.2f} GiB; an optional full-space uint32 remap is {a['full_hash_space_uint32_remap_gib']:.2f} GiB.", "",
             "| Experts/layer | History B/token/copy | Weights GB (decimal) | KV + SWA + rings GiB | Shared buffers GiB | CED GiB | Required GiB | Headroom GiB | Aggregate fit |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in result["scenarios"]:
        fit = "unknown" if row["fits_aggregate_budget"] is None else ("yes" if row["fits_aggregate_budget"] else "no")
        lines.append(f"| {row['experts_per_layer']} | {row['global_kv_bytes_per_token_per_copy']} | {row['resident_weights_decimal_gb']:.2f} | {number(row['cache_gib'])} | {number(row['shared_buffers_gib'])} | {number(row['ced_tail_gib'])} | {number(row['required_aggregate_gib'])} | {number(row['headroom_within_utilization_cap_gib'])} | {fit} |")
    runtime = result["scenarios"][0]["runtime_components"]
    if a["cache_layout"] == "native":
        lines.extend(["", f"Native CED reservation: **{runtime['ced_window']} tokens × {runtime['ced_reserved_request_slots']} request slots per GPU** (unless ced-gib overrides it). Disable this reservation with `--ced-window 0` when the runtime disables CED.",
                      "Known shared buffers are reused across indexer layers. The additional residual reserve remains necessary for uncounted peaks."])
    elif runtime["unknown_components"]:
        lines.extend(["", "Custom-layout total and fit are unknown until explicit overrides supply: " + ", ".join(runtime["unknown_components"]) + "."])
    if result["cost"] is not None:
        c = result["cost"]
        lines.extend(["", f"Generation time: {c['output_tokens']:,} / {c['aggregate_tokens_per_second']:g} / 3600 = **{c['generation_hours']:.3f} hours**."])
        if c["generation_usd"] is not None:
            lines.append(f"Generation cost at ${c['node_hourly_usd']:g}/node-hour: **${c['generation_usd']:.2f}**.")
        lines.append(f"Cost excludes {c['excludes']}.")
    lines.extend(["", "## Limits", ""] + [f"- {item}" for item in result["limitations"]])
    return "\n".join(lines)


def main():
    p = parser()
    args = p.parse_args()
    if not 0 < args.utilization <= 1 or args.gpu_gib <= 0:
        p.error("utilization must be in (0,1] and gpu-gib must be positive")
    if any(e < 6 or e > ORIGINAL_EXPERTS for e in args.experts):
        p.error("retained expert counts must be between original top-k 6 and 384")
    if args.throughput is not None and args.throughput <= 0:
        p.error("throughput must be positive when specified")
    if args.hourly_usd is not None and args.throughput is None:
        p.error("--hourly-usd requires --throughput")
    if args.max_num_seqs is not None and args.max_num_seqs < args.batch:
        p.error("max-num-seqs cannot be smaller than the estimated active batch")
    if args.cache_layout == "native":
        if args.kv_bytes_per_token is not None and args.kv_bytes_per_token != [890]:
            p.error("native history is fixed at 890 bytes/token; use --cache-layout custom for hypothetical layouts")
        if args.cache_replication != args.gpus:
            p.error("the audited native TP path replicates KV on every GPU; cache-replication must match gpus")
        if args.max_num_batched_tokens < args.batch:
            p.error("max-num-batched-tokens must accommodate the active decode batch")
        if args.shared_buffers_gib is None and args.indexer_logits_mib * MIB // 4 < max(args.context, 2048 * 8):
            p.error("native indexer logits workspace must fit at least one full-context and candidate row")
    elif args.kv_bytes_per_token is None:
        p.error("--cache-layout custom requires --kv-bytes-per-token")
    elif args.ced_window is not None:
        p.error("custom CED allocation is implementation-specific; use --ced-gib, including explicit 0 if absent")
    result = estimate(args)
    print(json.dumps(result, indent=2, allow_nan=False) if args.format == "json" else markdown(result))


if __name__ == "__main__":
    main()
