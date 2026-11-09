"""DeepSeek-V4-Flash GGUF backend for the Qwen3.8 stage: our llama.cpp fork on 4x L4.

Two releases, picked by --dsv4-profile (auto: the GGUF general.architecture):

- 0731 (deepseek4): the DeepSeek-V4-Flash-0731 GGUF.
- v41-1017-l4 (deepseek41): the pruned DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS (288 of 384 routed experts, 75% of
  the engram rows). Launching it first runs dsv41_prep (model links, expert keep file, engram hot sets in /dev/shm
  and /kaggle/tmp), then serves the measured 128-slot decode profile.

Prompts are rendered by the release's official encoder (vendored in third_party, checksum pinned) and sent to
/completion as token ids. llama.cpp's bundled DeepSeek-V4 chat template renders the preview "max" effort text,
which is the 0731 "high" level and not the V4.1 numeric effort, so the server template is not used.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import struct
import subprocess
import sys
import time

import dsv41_prep


SERVED_NAME = "dsv4-flash"
LLAMA_SERVER = "/kaggle/working/llama.cpp/build/bin/llama-server"
THIRD_PARTY = Path(__file__).with_name("third_party")
PREP = Path(__file__).with_name("dsv41_prep.py")

# 4x L4 tensor-split decode profile of the llama.cpp fork (e0feb9d7b), 128k context per slot.
# GGML_CUDA_FA_ROWS_SPARSE is the only path that is not bit-exact (mean |dNLL| 0.038 on real
# prompts); --dsv4-env GGML_CUDA_FA_ROWS_SPARSE=0 turns it off.
ENV_0731 = {
    "LLAMA_DSV4_CSA_HOST": "1", "LLAMA_DSV4_CSA_CACHE": "2560", "LLAMA_DSV4_LID_DCP": "1",
    "GGML_META_THREADS": "1", "GGML_CUDA_AR_CPU": "1",
    "GGML_CUDA_AR_SMALL_MAX_BYTES": "1048576", "GGML_CUDA_AR_RSAG_MIN_BYTES": "32768",
    "GGML_CUDA_MMVQ_MOE_MAX_BATCH": "16", "GGML_CUDA_MMVF_F32_MAX_BATCH": "8", "GGML_CUDA_MMVQ_MMA": "1",
    "GGML_CUDA_FA_KV_ROWS": "1", "GGML_CUDA_FA_UNIFORM": "1", "GGML_CUDA_FA_ROWS_SPARSE": "1",
}

# 4x L4 decode profile of the pruned V4.1-Flash-1017 UD-IQ2_XXS: 128 sequences at 56k depth decode 463.6 t/s
# in gen_v41 (experiments/deepseek_v41_flash_q2/BATCHED_DECODE.md, ER2: elastic expert cache, CPU hybrid experts,
# engram hot sets in RAM). launch() adds the CPU pinning of this machine (dsv41_prep.cpu_layout), the expert keep
# file, the engram hot sets and LLAMA_TP_CAND = top_k.
ENV_V41 = {
    "LLAMA_DSV41_TENSOR_PARALLEL": "1", "LLAMA_DSV41_MULTI_SEQ": "1", "LLAMA_SPLIT_ACCEPT_UNSTAMPED": "1",
    "LLAMA_LOAD_THREADS": "32", "LLAMA_LOAD_INDEPENDENT_READERS": "1", "LLAMA_LOAD_STAGE_AHEAD": "32",
    "GGML_CUDA_EXPERT_CACHE_TENSOR": "1", "GGML_CUDA_EXPERT_CACHE_MIB": "20000",
    "GGML_CUDA_EXPERT_CACHE_RESERVE_MIB": "150", "GGML_CUDA_EXPERT_CACHE_ELASTIC": "1",
    "GGML_CUDA_EXPERT_CACHE_REUSE_PLAN": "1", "GGML_CUDA_EXPERT_CACHE_MIXED_LAYOUTS": "1",
    "GGML_CUDA_EXPERT_CACHE_LAYOUT_POOLS": "1", "GGML_CUDA_EXPERT_CACHE_HOST_REGISTER": "1",
    "GGML_CUDA_EXPERT_CACHE_OVERLAP": "1", "GGML_CUDA_EXPERT_CACHE_POLICY": "lfu",
    "GGML_CUDA_EXPERT_CACHE_LFU_MIN_TOKENS": "129", "GGML_CUDA_EXPERT_CACHE_DECODE_LFU": "1",
    "GGML_CUDA_EXPERT_CACHE_DECODE_LFU_DECAY": "3200", "GGML_CUDA_EXPERT_CACHE_POOL_MIN_SLOTS": "384",
    "GGML_CUDA_EXPERT_CACHE_REGROW": "64", "GGML_CUDA_EXPERT_CACHE_LEAN_FALLBACK": "1",
    "GGML_CUDA_EXPERT_CACHE_POOL_BIAS": "90", "GGML_CUDA_EXPERT_CACHE_RANGE_PERCENT": "80",
    "GGML_CUDA_EXPERT_CACHE_REORDER": "1", "GGML_CUDA_EXPERT_CACHE_CYCLE": "1024",
    "GGML_CUDA_EXPERT_CACHE_PREFETCH": "288", "GGML_CUDA_EXPERT_CACHE_PREFETCH_BLOCKS": "16",
    "GGML_CUDA_EXPERT_CACHE_PREFETCH_DEFER": "1", "GGML_CUDA_EXPERT_CACHE_PREFETCH_MIN_ROUTES": "2",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID": "20", "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_MIN_TOKENS": "16",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_MAX_ROUTES": "2",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_EXPERTS_BY_TYPE": "iq1_s=14,iq2_xxs=18,iq2_s=18",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_SHARE_BY_TYPE": "iq1_s=100,iq2_xxs=100,iq2_s=100",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_COLD": "1", "GGML_CUDA_EXPERT_HYBRID_STREAM": "1",
    "GGML_CUDA_MMVQ_MOE_MAX_BATCH": "128", "GGML_CUDA_MMVQ_MOE_RUN_MIXED_IQ": "1", "GGML_CUDA_MMVQ_MOE_RUN_DOWN": "4",
    "GGML_CUDA_MMVQ_MOE_RUN_OCC": "2", "GGML_CUDA_MMVQ_MOE_RUN_DOWN_ITER": "8",
    "GGML_CUDA_MMQ_EXPERT_COMPACT": "1", "GGML_CUDA_MMQ_EXPERT_TILE_AWARE": "1",
    "GGML_CUDA_PDL": "0", "GGML_META_THREADS": "0", "GGML_CUDA_MMVF_F16_MAX_BATCH": "1",
    "GGML_CUDA_NCCL_ALLREDUCE_FP32": "1", "GGML_CUDA_ALLREDUCE": "internal",
    "GGML_CUDA_AR_CPU": "1", "GGML_CUDA_AR_CPU_THREADS": "4", "GGML_CUDA_AR_CPU_Q8": "1",
    "GGML_CUDA_AR_SMALL_MAX_BYTES": "1073741824", "GGML_CUDA_AR_RSAG_MIN_BYTES": "32768",
    "GGML_CUDA_AR_BF16_THRESHOLD": "1", "GGML_CUDA_AR_BUF_BYTES": "67108864", "GGML_CUDA_AR_SP_SKIP_OWN": "1",
    "GGML_CUDA_AR_PF_KB": "20480", "GGML_CUDA_AR_PF_TOUCH": "1", "GGML_CUDA_AR_PF_BLOCKS": "32",
    "GGML_CUDA_SIDE_STREAM": "1", "GGML_CUDA_FA_KV_ROWS": "1", "GGML_CUDA_FA_ROWS_GQA16": "1",
    "GGML_CUDA_FA_UNIFORM": "1", "GGML_CUDA_FA_PF_KB": "16384", "GGML_CUDA_FA_PF_BLOCKS": "2",
    "GGML_CUDA_MAPPED_SHARED": "1", "GGML_META_PACK_INPUTS": "65536", "GGML_SCHED_INPUT_FENCE": "1",
    "GGML_CUDA_PAGED_RESERVE_MB": "24", "LLAMA_DSV4_KV_PAGED": "1", "LLAMA_DSV4_LID_PAGED": "1",
    "LLAMA_DSV4_CSA_HOST": "1", "LLAMA_DSV4_CSA_CACHE": "1024", "LLAMA_DSV4_LID_DCP": "1",
    "LLAMA_DSV41_GATHER": "1", "LLAMA_DSV41_GATHER_ONCE": "1", "LLAMA_DSV41_IDX_GEMM": "1", "LLAMA_DSV41_EP_TAIL": "1",
    "LLAMA_DSV41_UNION": "1", "LLAMA_DSV41_QKV_A": "1", "LLAMA_DSV41_LID_FUSED": "1", "LLAMA_DSV41_HC_SP": "1024",
    "LLAMA_DSV41_SHEXP_FIRST": "65", "LLAMA_DSV41_PREFETCH": "1", "LLAMA_DSV41_PREFETCH_LEAD": "512",
    "LLAMA_DSV41_PREFETCH_MAX_TOKENS": "8192", "LLAMA_DSV41_PREFETCH_TAIL": "1",
    "LLAMA_DSV41_PREFETCH_MIN_TOKENS": "129",
    "LLAMA_DSV41_ENGRAM_PRUNE": "1", "LLAMA_DSV41_ENGRAM_THREADS": "128", "LLAMA_DSV41_ENGRAM_CACHE_MB": "256",
    "LLAMA_DSV41_ENGRAM_STATS": "2", "LLAMA_DSV41_ENGRAM_ASYNC": "1", "LLAMA_DSV41_ENGRAM_INDEPENDENT_READERS": "2",
    "LLAMA_DSV41_ENGRAM_DIRECT": "1", "LLAMA_DSV41_ENGRAM_PARALLEL": "0", "LLAMA_DSV41_ENGRAM_DECODE_THREADS": "0",
    "MTMD_COMPUTE_RELEASE": "1",
}

PROFILES = {
    "0731": dict(release="DeepSeek-V4-Flash-0731", architecture="deepseek4", env=ENV_0731,
                 encoder=THIRD_PARTY / "deepseek_v4_0731" / "encoding_dsv4.py",
                 sha256="abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6",
                 non_exact=("GGML_CUDA_FA_ROWS_SPARSE",), top_p=0.95, top_k=0, slots=None),
    # top_k 64: the tensor-split output samples from the top LLAMA_TP_CAND logits of each device, exact for
    # top_k <= LLAMA_TP_CAND; full-vocabulary sampling (top_k 0) gathers the whole vocabulary every step.
    "v41-1017-l4": dict(release="DeepSeek-V4.1-Flash-1017", architecture="deepseek41", env=ENV_V41,
                        encoder=THIRD_PARTY / "deepseek_v41_1017" / "encoding_dsv41.py",
                        sha256="13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2",
                        non_exact=("GGML_CUDA_AR_CPU_Q8",), top_p=1.0, top_k=64, slots=128),
}
ARCHITECTURES = {p["architecture"]: name for name, p in PROFILES.items()}
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
# One sequence of a 128-slot V4.1 batch decodes ~3.6 t/s (463.6 t/s / 128).
V41_SEQUENCE_TPS = 3.6
_encoders = {}
_active = dict(profile="0731")


def effort_value(text):
    if text in ("low", "high", "max"):
        return text
    if text.isdigit() and 1 <= int(text) <= 100:
        return int(text)
    raise argparse.ArgumentTypeError("expected low, high, max or (V4.1) 1-100")


def add_arguments(parser):
    g = parser.add_argument_group("DeepSeek-V4 llama.cpp backend (--qwen38-variant dsv4)")
    g.add_argument("--dsv4-model", type=Path, help="GGUF file (first shard of a split model) or the folder holding it; "
                   "for v41-1017-l4 the folder of parts 1-5 (part 6 is the --dsv4-engram-table).")
    g.add_argument("--dsv4-profile", choices=("auto", *PROFILES), default="auto",
                   help="Release profile: encoder, sampling defaults and server settings. auto reads the GGUF "
                        "general.architecture of --dsv4-model (deepseek4: 0731, deepseek41: v41-1017-l4).")
    g.add_argument("--llama-server", type=Path, default=Path(LLAMA_SERVER))
    g.add_argument("--dsv4-reasoning-effort", type=effort_value, default="max",
                   help="Official thinking-mode effort: low, high or max (0731); V4.1 also takes 1-100, "
                        "max is 'Reasoning Effort: 100'.")
    g.add_argument("--dsv4-temperature", type=float, default=1.0)
    g.add_argument("--dsv4-top-p", type=float, help="Profile default: 0.95 (0731), 1.0 (v41-1017-l4).")
    g.add_argument("--dsv4-top-k", type=int, help="0 disables top-k. Profile default: 0 (0731), 64 (v41-1017-l4).")
    g.add_argument("--dsv4-min-p", type=float, default=0.0)
    g.add_argument("--dsv4-seed", type=int, default=1000, help="Request seed = this + planned job number.")
    g.add_argument("--dsv4-ubatch", type=int,
                   help="Default 128; v41-1017-l4: the slot count if larger, so one ubatch decodes every slot.")
    g.add_argument("--dsv4-batch", type=int, default=4096)
    g.add_argument("--dsv4-threads", type=int, default=24)
    g.add_argument("--dsv4-kv-type", default="q8_0")
    g.add_argument("--dsv4-env", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a server environment profile entry; KEY= removes it. Repeatable.")
    g.add_argument("--dsv4-server-arg", action="append", default=[], metavar="ARG",
                   help="Extra llama-server argument, written as --dsv4-server-arg=--flag. Repeatable.")
    g.add_argument("--dsv4-engram-table", type=Path, default=Path(dsv41_prep.ENGRAM_TABLE),
                   help="v41-1017-l4: the IQ4_NL engram table GGUF, served as the model's last part. A missing "
                        "/kaggle/input path is looked up by its file name under /kaggle/input.")
    g.add_argument("--dsv4-imp-dir", type=Path, default=Path(dsv41_prep.IMP_DIR),
                   help="v41-1017-l4: dspk-41-imp (expert scores, engram counts and masks), read as data. A missing "
                        "/kaggle/input path is looked up by its folder name under /kaggle/input.")
    g.add_argument("--dsv4-prep-dir", type=Path, default=Path(dsv41_prep.PREP_DIR),
                   help="v41-1017-l4: model links, expert keep file and prep.json; reused when the inputs match.")
    g.add_argument("--dsv4-engram-dir", default=",".join(dsv41_prep.ENGRAM_DIRS),
                   help="v41-1017-l4: folders of the engram hot sets, comma-separated: tier k goes to the k-th "
                        "folder and the rest to the last. The default puts the 16 and 7 GiB tiers in /dev/shm and "
                        "the rest (54.25 GiB at --dsv4-engram-keep 0.75) on local disk in /kaggle/tmp.")
    g.add_argument("--dsv4-engram-tiers", default=",".join(map(str, dsv41_prep.ENGRAM_TIERS)),
                   help="v41-1017-l4: GiB of the hottest engram tiers; the last tier takes the other kept rows.")
    g.add_argument("--dsv4-engram-keep", type=dsv41_prep.keep_arg, default=dsv41_prep.ENGRAM_KEEP,
                   help="v41-1017-l4: share of each layer's engram rows kept (the others read as zeros), or the "
                        "name of a notool mask of --dsv4-imp-dir such as engram_top48gib.")
    g.add_argument("--dsv4-mmproj", type=Path,
                   help="v41-1017-l4: the vision projector GGUF (e.g. mmproj-BF16.gguf), served with --mmproj; "
                        "required by --input-mode text-image.")
    g.add_argument("--live-interval-seconds", type=float, default=30,
                   help="LIVE TPS/progress interval for all backends; e.g. 30 or 60 seconds.")


def profile_of(args):
    return PROFILES[args.dsv4_profile]


def resolve_profile(args):
    """--dsv4-profile, with auto read from the GGUF architecture of --dsv4-model."""
    if args.dsv4_profile != "auto":
        return args.dsv4_profile
    if args.dsv4_model is None:
        raise ValueError("--dsv4-profile auto reads --dsv4-model; name the profile when using --base-url")
    try:
        arch = gguf_general(first_shard(args.dsv4_model)).get("general.architecture")
    except (OSError, KeyError, struct.error) as exc:
        raise ValueError(f"Cannot read the GGUF header of {args.dsv4_model}: {exc}") from exc
    if arch not in ARCHITECTURES:
        raise ValueError(f"No dsv4 profile for GGUF architecture {arch!r}; pass --dsv4-profile")
    return ARCHITECTURES[arch]


def env_profile(args, layout=None, prep=None):
    """Server environment: the profile, this machine's CPU layout and prepared files (v41), then --dsv4-env."""
    profile = dict(profile_of(args)["env"], **(layout or {}))
    if prep is not None:
        profile.update(LLAMA_DSV41_EXPERT_KEEP=prep["expert_keep"]["file"], LLAMA_DSV41_ENGRAM_LOCAL=prep["engram_local"])
    # Every GPU returns the top k logits of its vocab slice instead of the whole vocabulary: the global top k are among those
    # 4 x k, so with the request's top-k <= k (and no penalty ahead of it in the chain) the sampled token is the same.
    if args.dsv4_profile == "v41-1017-l4" and args.dsv4_top_k:
        profile["LLAMA_TP_CAND"] = str(args.dsv4_top_k)
    for item in args.dsv4_env:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"--dsv4-env expects KEY=VALUE, got {item!r}")
        if value:
            profile[key] = value
        else:
            profile.pop(key, None)
    return profile


def engram_tiers(args):
    try:
        tiers = [float(x) for x in args.dsv4_engram_tiers.split(",") if x.strip()]
    except ValueError:
        tiers = None
    if not tiers or not all(0 < t < 1024 for t in tiers):
        raise ValueError(f"--dsv4-engram-tiers expects GiB per tier, e.g. 16,7; got {args.dsv4_engram_tiers!r}")
    return tiers


def pinned_cpus(profile):
    """The CPUs the v41 profile pins threads to."""
    cpus = set()
    for key in ("GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_CORES", "LLAMA_DSV41_ENGRAM_CPUS"):
        for group in profile.get(key, "").split("/"):
            cpus |= dsv41_prep.cpu_list(group)
    core, threads = profile.get("GGML_CUDA_AR_CPU_CORE", ""), profile.get("GGML_CUDA_AR_CPU_THREADS", "4")
    if core.isdigit() and threads.isdigit():
        cpus |= set(range(int(core), int(core) + int(threads)))
    return cpus


def validate_options(args):
    if args.qwen38_variant != "dsv4":
        return
    repl = args.solve_mode == "tir" and getattr(args, "tir_interface", None) == "repl"
    if args.solve_mode != "direct" and not repl:
        raise ValueError("dsv4 solves directly or with --solve-mode tir --tir-interface repl")
    if args.input_mode not in ("text", "text-image") or args.input_mode == "text-image" and not repl:
        raise ValueError("dsv4 takes text input, or text-image with REPL solving")
    if not args.base_url and args.dsv4_model is None:
        raise ValueError("--dsv4-model is required to launch the dsv4 server")
    args.dsv4_profile = resolve_profile(args)
    _active["profile"] = args.dsv4_profile
    profile, v41 = profile_of(args), args.dsv4_profile == "v41-1017-l4"
    if repl and not v41:
        raise ValueError("dsv4 REPL solving needs the V4.1 encoder (v41-1017-l4): 0731 has no tool round trip here")
    mmproj = getattr(args, "dsv4_mmproj", None)
    if args.input_mode == "text-image" and mmproj is None:
        raise ValueError("--input-mode text-image needs --dsv4-mmproj: its image-token geometry, and the "
                         "server's --mmproj")
    if mmproj is not None and not (Path(mmproj).is_file() and Path(mmproj).stat().st_size):
        raise ValueError(f"Missing or empty --dsv4-mmproj file: {mmproj}")
    if args.dsv4_top_p is None:
        args.dsv4_top_p = profile["top_p"]
    if args.dsv4_top_k is None:
        args.dsv4_top_k = profile["top_k"]
    if args.dsv4_ubatch is None:
        args.dsv4_ubatch = max(128, args.server_max_num_seqs) if v41 else 128
    if not (args.dsv4_temperature >= 0 and 0 < args.dsv4_top_p <= 1 and 0 <= args.dsv4_min_p < 1
            and args.dsv4_top_k >= 0):
        raise ValueError("Invalid dsv4 sampling settings")
    if isinstance(args.dsv4_reasoning_effort, int) and not v41:
        raise ValueError("The 0731 encoder takes --dsv4-reasoning-effort low, high or max")
    if min(args.dsv4_ubatch, args.dsv4_batch, args.dsv4_threads) < 1 or not args.live_interval_seconds > 0:
        raise ValueError("dsv4 batch, ubatch, threads and live interval must be positive")
    load_encoder()
    env = env_profile(args, dsv41_prep.cpu_layout() if v41 and not args.base_url else None)
    if args.base_url:
        return
    cand = env.get("LLAMA_TP_CAND", "")
    if cand and (not cand.isdigit() or not 0 < args.dsv4_top_k <= int(cand)):
        raise ValueError(f"LLAMA_TP_CAND={cand} samples from the top {cand} logits of each GPU: it needs "
                         f"0 < --dsv4-top-k <= {cand}, got {args.dsv4_top_k}")
    if not v41:
        return
    engram_tiers(args)
    # The profile pins threads to these CPUs; a different machine fails here, not hours into the run.
    missing = sorted(pinned_cpus(env) - os.sched_getaffinity(0))
    if missing:
        raise ValueError(f"The v41 profile pins threads to CPUs {missing}, which this process cannot use")
    if args.server_max_num_seqs < profile["slots"]:
        print(f"WARNING: v41-1017-l4 is measured at {profile['slots']} slots; {args.server_max_num_seqs} slots "
              "decode fewer tokens per second in total", flush=True)
    if args.request_timeout_seconds * V41_SEQUENCE_TPS < args.context:
        print(f"WARNING: at ~{V41_SEQUENCE_TPS} t/s per sequence, --request-timeout-seconds "
              f"{args.request_timeout_seconds:.0f} stops a request near "
              f"{args.request_timeout_seconds * V41_SEQUENCE_TPS:.0f} tokens, below --context {args.context}; "
              "e.g. 86400 lets it run to the context", flush=True)


def load_encoder(name=None):
    """The official encoder of the profile (default: the active one), checksum verified."""
    name = name or _active["profile"]
    if name not in _encoders:
        profile = PROFILES[name]
        digest = hashlib.sha256(profile["encoder"].read_bytes()).hexdigest()
        if digest != profile["sha256"]:
            raise ValueError(f"{profile['release']} encoder checksum mismatch: {digest}")
        spec = importlib.util.spec_from_file_location("dsv4_encoding_" + name.replace("-", "_"), profile["encoder"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _encoders[name] = module
    return _encoders[name]


def encoder_info(args):
    profile = profile_of(args)
    return dict(release=profile["release"], path=str(profile["encoder"]), sha256=profile["sha256"],
                thinking_mode="thinking", reasoning_effort=args.dsv4_reasoning_effort)


def source_paths(args):
    """The files that decide what the dsv4 backend serves and sends, for the manifest checksums."""
    return [PREP, *([profile_of(args)["encoder"]] if args.qwen38_variant == "dsv4" else [])]


def messages_for(prompt):
    return [dict(role="user", content=prompt)]


def encode(prompt, effort):
    return load_encoder().encode_messages(messages_for(prompt), thinking_mode="thinking", reasoning_effort=effort)


def sampling(args):
    return dict(temperature=args.dsv4_temperature, top_p=args.dsv4_top_p, top_k=args.dsv4_top_k,
                min_p=args.dsv4_min_p, repeat_penalty=1.0, seed_base=args.dsv4_seed)


def make_payload(args, ids, seed):
    return dict(prompt=ids, max_tokens=0, stream=True, cache_prompt=False, return_tokens=True,
                temperature=args.dsv4_temperature, top_p=args.dsv4_top_p, top_k=args.dsv4_top_k,
                min_p=args.dsv4_min_p, repeat_penalty=1.0, seed=seed)


def first_shard(path):
    path = Path(path).absolute()  # not resolved: the V4.1 parts are links in one folder
    if path.is_dir():
        found = [p for p in path.glob("*.gguf") if "mmproj" not in p.name
                 and (not SPLIT.search(p.name) or SPLIT.search(p.name).group(1) == "00001")]
        if len(found) != 1:
            raise ValueError(f"Expected one GGUF model (or first shard) in {path}, found {sorted(p.name for p in found)}")
        path = found[0]
    match = SPLIT.search(path.name)
    if match and match.group(1) != "00001":
        raise ValueError(f"Pass the first shard of the split GGUF, not {path.name}")
    return path


def resolve_model(path):
    path = first_shard(path)
    match = SPLIT.search(path.name)
    shards = ([path.with_name(f"{path.name[:match.start()]}-{i:05d}-of-{match.group(2)}.gguf")
               for i in range(1, int(match.group(2)) + 1)] if match else [path])
    sizes = {}
    for shard in shards:
        if not shard.is_file() or not shard.stat().st_size:
            raise ValueError(f"Missing or empty GGUF file: {shard}")
        sizes[shard.name] = shard.stat().st_size
    return path, sizes


def gguf_general(path, prefixes=("general.", "split.")):
    """Header scalars whose keys start with prefixes (default general.* and split.*, for the launch receipt)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        _, _, n_kv = struct.unpack("<IQQ", f.read(20))

        def string():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        for _ in range(n_kv):
            key = string()
            kind, = struct.unpack("<I", f.read(4))
            if kind == 9:
                elem, count = struct.unpack("<IQ", f.read(12))
                for _ in range(count if elem == 8 else 0):
                    string()
                if elem != 8:
                    f.seek(count * sizes[elem], 1)
            elif kind == 8:
                value = string()
            else:
                value, = struct.unpack("<" + formats[kind], f.read(sizes[kind]))
            if kind != 9 and key.startswith(prefixes):
                out[key] = value
    return out


def environment(args, layout=None, prep=None):
    profile = env_profile(args, layout, prep)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
    env.update(profile, CUDA_VISIBLE_DEVICES="0,1,2,3")
    removed = sorted(k for k in os.environ if k.startswith(("LLAMA_", "GGML_")) and k not in profile)
    return env, profile, removed


def command_for(args, model):
    slots = args.server_max_num_seqs
    return [str(args.llama_server), "-m", str(model), "--alias", SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(args.qwen38_port),
            "--split-mode", "tensor", "-ngl", "99", "--fit", "off", "-fa", "on",
            "-ctk", args.dsv4_kv_type, "-ctv", args.dsv4_kv_type,
            # -c is the total over all slots: each of the --no-kv-unified slots gets --context.
            "-c", str(args.context * slots), "-np", str(slots), "--no-kv-unified",
            "-ub", str(args.dsv4_ubatch), "-b", str(args.dsv4_batch), "-t", str(args.dsv4_threads),
            # The token-sharded indexer cache (LLAMA_DSV4_LID_DCP) cannot save slot state.
            "--no-context-shift", "--cache-ram", "0", "--ctx-checkpoints", "0",
            "--timeout", "86400", "--metrics", "--no-webui",
            # The measured V4.1 load: direct I/O, no warmup decode (it routes every token through every expert).
            *(["--load-mode", "dio", "--no-warmup"] if args.dsv4_profile == "v41-1017-l4" else []),
            # The warm encode sizes the clip buffers before the elastic expert cache takes the free memory.
            # The projector weights stay in pinned host memory and stream in per encode: on CUDA0 their 0.9 GiB
            # cost it expert cache slots and made it the slowest device of every decode step (296 -> 284 ms at
            # 128 x 12k, bit-exact, an encode takes 0.1-0.25 s longer). The clip compute buffers go after each
            # encode too (MTMD_COMPUTE_RELEASE in ENV_V41), the next encode takes them back from the elastic expert
            # cache: 289 -> 282 ms at 128 x 12k after image requests, bit-exact embeddings.
            *(["--mmproj", str(args.dsv4_mmproj), "--mmproj-warmup-size", "2048", "--mmproj-weights-host"]
              if getattr(args, "dsv4_mmproj", None) else []),
            *args.dsv4_server_arg]


def source_revision(binary):
    """Commit and dirty-file count of the llama.cpp checkout the binary was built from, if any."""
    for root in binary.parents:
        if (root / ".git").exists():
            try:
                head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                      text=True, timeout=20).stdout.strip()
                # File-mode-only differences (e.g. a copied checkout) are not source changes.
                dirty = subprocess.run(["git", "-C", str(root), "-c", "core.fileMode=false", "status", "--porcelain",
                                        "--untracked-files=no"],
                                       capture_output=True, text=True, timeout=60).stdout.splitlines()
                return dict(root=str(root), commit=head, modified_files=len(dirty))
            except (OSError, subprocess.SubprocessError):
                return dict(root=str(root))
    return None


def prepare_v41(args, output):
    """dsv41_prep in an isolated interpreter (it reads the imp dataset, as data); returns its prep.json."""
    command = [sys.executable, "-I", str(PREP), "--model", str(args.dsv4_model),
               "--engram-table", str(args.dsv4_engram_table), "--imp-dir", str(args.dsv4_imp_dir),
               "--prep-dir", str(args.dsv4_prep_dir), "--engram-dirs", args.dsv4_engram_dir,
               "--engram-tiers", ",".join(map(str, engram_tiers(args))), "--engram-keep", str(args.dsv4_engram_keep)]
    started = time.monotonic()
    with (output / "prep.log").open("x") as log:
        process = subprocess.Popen(command, cwd=output, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print("PREP", line.rstrip(), flush=True)
        process.wait()
    if process.returncode:
        raise RuntimeError(f"dsv41_prep exited with {process.returncode}; see {output / 'prep.log'}")
    prep = json.loads((args.dsv4_prep_dir / "prep.json").read_text())
    return dict(prep, seconds=round(time.monotonic() - started, 1))


def launch(args):
    output = (args.output_root / "server").resolve()
    v41 = args.dsv4_profile == "v41-1017-l4"
    try:
        binary = args.llama_server.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"llama-server binary not found or not executable: {binary}")
        if not v41:
            model, shards = resolve_model(args.dsv4_model)
            general = gguf_general(model)
        if output.exists():
            raise ValueError(f"Use a fresh server output directory: {output}")
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=10)
        if processes.strip():
            raise ValueError(f"GPU processes are active; will not stop them: {processes.strip()}")
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", args.qwen38_port))
    except (ValueError, OSError, KeyError, struct.error, subprocess.SubprocessError) as exc:
        raise RuntimeError(str(exc)) from exc
    output.mkdir(parents=True)
    prep = layout = None
    if v41:
        # Model links, expert keep file and engram hot sets (~5 min for new hot sets), then this machine's pinning.
        prep = prepare_v41(args, output)
        layout = dsv41_prep.cpu_layout()
        try:
            model, shards = resolve_model(prep["model"])
            general = gguf_general(model)
        except (ValueError, OSError, KeyError, struct.error) as exc:
            raise RuntimeError(str(exc)) from exc
    command = command_for(args, model)
    env, profile, removed = environment(args, layout, prep)
    release = profile_of(args)["release"]
    name = " ".join(str(general.get(k, "")) for k in ("general.name", "general.basename", "general.version"))
    if release.rsplit("-", 1)[-1] not in name + str(args.dsv4_model) + model.name:
        print(f"WARNING: GGUF metadata does not name the {release} release ({name.strip()!r}); the prompts "
              f"follow the {release} encoder", flush=True)
    receipt = dict(status="prepared", created_utc=datetime.now(timezone.utc).isoformat(),
                   command=command, environment_overrides=profile, environment_removed=removed,
                   non_exact=[k for k in profile_of(args)["non_exact"] if profile.get(k, "0") != "0"],
                   variant="dsv4", profile=args.dsv4_profile, served_model_name=SERVED_NAME, model=str(model),
                   shards_bytes=shards, gguf_general=general, llama_server=str(binary),
                   llama_cpp=source_revision(binary), encoder=encoder_info(args), sampling=sampling(args),
                   slots=args.server_max_num_seqs, context_per_slot=args.context, cpu_layout=layout, prep=prep)
    receipt_path = output / "launch.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "server.log").open("xb") as log:
        process = subprocess.Popen(command, env=env, cwd=output, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(status="launched_not_yet_ready", pid=process.pid)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Command:", " ".join(f"{k}={v}" for k, v in profile.items()), shlex.join(command), flush=True)
    if receipt["non_exact"]:
        print("NOTE: not bit-exact kernels enabled:", ", ".join(receipt["non_exact"]), flush=True)
    print("PID:", process.pid, "Log:", output / "server.log", flush=True)
    return process, receipt


async def tokenize(client, url, text):
    response = await client.post(url + "/tokenize", json=dict(content=text, add_special=False, parse_special=True),
                                 timeout=120)
    response.raise_for_status()
    return response.json()["tokens"]


async def check_server(client, url, args):
    """Slot context/count and the special-token vocabulary the encoder relies on."""
    response = await client.get(url + "/props", timeout=20)
    response.raise_for_status()
    props = response.json()
    n_ctx, slots = props.get("default_generation_settings", {}).get("n_ctx"), props.get("total_slots")
    if not isinstance(n_ctx, int) or n_ctx < args.context:
        raise ValueError(f"Server slot context {n_ctx} is below --context {args.context}")
    if not isinstance(slots, int) or slots < args.concurrency:
        raise ValueError(f"Server has {slots} slots, below --concurrency {args.concurrency}")
    enc = load_encoder()
    if props.get("bos_token") not in (None, enc.bos_token) or props.get("eos_token") not in (None, enc.eos_token):
        raise ValueError("Served BOS/EOS tokens do not match the DeepSeek-V4 encoder")
    specials = {}
    # V4.1 puts the reasoning effort in a system turn.
    for name, text in (("bos", enc.bos_token), ("eos", enc.eos_token), ("user", enc.USER_SP_TOKEN),
                       ("assistant", enc.ASSISTANT_SP_TOKEN), ("think", enc.thinking_start_token),
                       ("think_end", enc.thinking_end_token),
                       *([("system", enc.SYSTEM_SP_TOKEN)] if hasattr(enc, "SYSTEM_SP_TOKEN") else [])):
        ids = await tokenize(client, url, text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token in the served vocabulary: {ids}")
        specials[name] = ids[0]
    return dict(id=SERVED_NAME, n_ctx=n_ctx, total_slots=slots, special_token_ids=specials,
                **{k: props.get(k) for k in ("model_path", "model_alias", "model_ftype", "build_info")})


async def tokenize_prompt(client, url, text, specials):
    ids = await tokenize(client, url, text)
    # One BOS (V4.1: then the effort system turn), the user turn, ending with the thinking-mode generation prompt.
    if (len(ids) < 4 or ids[0] != specials["bos"] or ids.count(specials["bos"]) != 1
            or ("system" in specials and ids[1] != specials["system"])
            or ids[-2:] != [specials["assistant"], specials["think"]] or specials["user"] not in ids):
        raise ValueError("Encoded prompt does not tokenize to BOS ... <User> ... <Assistant><think>")
    return ids


async def smoke(client, url, specials, output):
    """Short chat-mode request: proves token prompts, EOS and the stream format before long runs."""
    text = load_encoder().encode_messages(messages_for("Return only the integer: 17 + 25."), thinking_mode="chat")
    ids = await tokenize(client, url, text)
    if ids[0] != specials["bos"] or ids[-1] != specials["think_end"]:
        raise ValueError("Chat-mode smoke prompt does not end with </think>")
    payload = dict(prompt=ids, max_tokens=32, stream=True, cache_prompt=False, return_tokens=True,
                   temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, repeat_penalty=1.0, seed=0)
    result = await stream_completion(client, url, payload, output, 300, parse_grid=None, specials=specials,
                                     thinking=False)
    if result["error"] or result["stop_type"] != "eos" or "42" not in result["content"]:
        raise ValueError(f"Completion smoke failed ({result['error'] or result['stop_type']}): "
                         f"{result['content'][:200]!r}; see {output}")
    return dict(content=result["content"], usage=result["usage"], elapsed_s=result["elapsed_s"])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_partial(folder):
    """The generated events of an unfinished output, to continue it: its resume.json prefix, then stream.jsonl.

    With return_tokens every event carries all token ids since the previous one (a split UTF-8 piece goes
    with the next event), so the concatenated ids are exactly the tokens generated so far.
    """
    folder = Path(folder)
    prior = json.loads((folder / "resume.json").read_text()) if (folder / "resume.json").exists() else {}
    events, last, count = list(prior.get("events", [])), 0.0, 0
    # A continuation stopped during its prompt has resume.json but no stream yet.
    lines = (folder / "stream.jsonl").read_text().splitlines() if (folder / "stream.jsonl").exists() else []
    for n, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if n == len(lines) - 1:  # A killed process can leave half a line.
                break
            raise
        if isinstance(event, list):
            last, predicted, tokens, text = event
            count += len(tokens)
            if predicted is not None and predicted != count:
                raise ValueError(f"{folder}: {count} streamed token ids but tokens_predicted {predicted}")
            events.append([tokens, text])
    result = folder / "result.json"
    elapsed = (json.loads(result.read_text())["elapsed_s"] if result.exists()
               else prior.get("elapsed_s", 0.0) + last)
    return dict(source=str(folder.resolve()), tokens=sum(len(t) for t, _ in events), elapsed_s=elapsed,
                events=events)


async def stream_completion(client, url, payload, output, timeout, *, parse_grid, specials,
                            progress=None, files=None, thinking=True, marker=None, loop_guard=None,
                            loop_action="off", prefix=None, prompt_tokens=None, request_record=None):
    """Stream one /completion request, keeping every event (elapsed, decoded, token ids, text).

    loop_guard (qwen38_loop.StreamTextLoopGuard) watches the generated text; with loop_action "stop" a
    confirmed hit closes the stream, which makes llama-server cancel the request and free its slot.
    prefix (read_partial of an interrupted output) continues it: payload["prompt"] already ends with its
    token ids. Its text, think state and guard windows are restored first (resume.json keeps it), and the
    result counts the whole output: usage, streamed tokens and elapsed include the prefix.
    A multimodal prompt (prompt_string + multimodal_data) passes its expanded prompt_tokens and a
    request_record for request.json instead of the payload with its images.
    """
    if loop_guard is not None and loop_action not in ("observe", "stop"):
        raise ValueError("A gzip guard needs observe or stop mode")
    output.mkdir()
    write_json(output / "request.json", payload if request_record is None else request_record)
    if prefix is not None:
        (output / "resume.json").write_text(json.dumps(prefix) + "\n")
    for name, value in (files or {}).items():
        (output / name).write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
    progress = {} if progress is None else progress
    progress.update(prompt_tokens=len(payload["prompt"]) if prompt_tokens is None else prompt_tokens,
                    decoded=0, started=time.monotonic())
    start = time.monotonic()
    marker, think_end = marker or load_encoder().thinking_end_token, specials["think_end"]
    # Official parser: reasoning runs to the first </think>, the answer follows it. Without --special
    # llama-server may not render control tokens, so the </think> token id also closes the reasoning.
    reasoning, answer, think_closed = [], [], not thinking
    final, error, cancelled = None, None, False
    first, last, think_end_tokens = None, None, 0
    loop_stopped = False

    def split(tokens, text):
        nonlocal think_closed, think_end_tokens
        think_end_tokens += tokens.count(think_end)
        if think_closed:
            answer.append(text)
        elif marker in text:
            head, tail = text.split(marker, 1)
            reasoning.append(head)
            answer.append(tail)
            think_closed = True
        else:
            reasoning.append(text)
            think_closed = think_end in tokens

    n_prefix = prefix["tokens"] if prefix is not None else 0
    for tokens, text in prefix["events"] if prefix is not None else []:
        split(tokens, text)
        # The same settings stopped no earlier window; a hit here means they changed.
        loop_stopped = bool(loop_guard is not None and loop_guard.feed_event(tokens, text)
                            and loop_action == "stop") or loop_stopped
    try:
        if loop_stopped:
            raise RuntimeError("The resumed prefix already trips the stop guard")
        async with asyncio.timeout(timeout):
            async with client.stream("POST", url + "/completion", json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread())[:2000].decode("utf-8", "replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {body}")
                with (output / "stream.jsonl").open("x", buffering=1) as raw:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        elapsed = time.monotonic() - start
                        if event.get("error"):
                            raise RuntimeError(str(event["error"]))
                        text, tokens = event.get("content") or "", event.get("tokens") or []
                        split(tokens, text)
                        if event.get("stop"):
                            final = {k: v for k, v in event.items() if k != "prompt"}
                            if tokens or text:
                                # The stop event can carry the last ids (a held-back UTF-8 tail): an event
                                # line keeps them, so read_partial sees every generated token.
                                raw.write(json.dumps([round(elapsed, 3), event.get("tokens_predicted"), tokens,
                                                      text]) + "\n")
                                first = elapsed if first is None else first
                                last = elapsed
                            raw.write(json.dumps(dict(elapsed_s=elapsed, final=final)) + "\n")
                            break
                        raw.write(json.dumps([round(elapsed, 3), event.get("tokens_predicted"), tokens, text]) + "\n")
                        if tokens or text:
                            first = elapsed if first is None else first
                            last = elapsed
                        progress["decoded"] = event.get("tokens_predicted") or progress["decoded"]
                        hit = loop_guard.feed_event(tokens, text) if loop_guard is not None else None
                        if hit:
                            write_json(output / "loop_detection.json", dict(action=loop_action, **hit))
                            print(f"GZIP {output.name}: ratio={hit['ratio']:.2f}, action={loop_action}, "
                                  f"token={hit['observed_completion_tokens']}", flush=True)
                            if loop_action == "stop":
                                loop_stopped = True
                                break  # Closing the stream cancels this request, not the server.
                if final is None and not loop_stopped:
                    raise RuntimeError("Stream ended without a stop event")
    except asyncio.CancelledError:
        error, cancelled = "Cancelled: pipeline stopped or server exited", True
    except TimeoutError:
        error = f"TimeoutError: request budget {timeout:.0f} s (deadline or --request-timeout-seconds)"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    reasoning, content = "".join(reasoning), "".join(answer)
    stop_type = final.get("stop_type") if final else None
    finish = "loop_detected" if loop_stopped else {"eos": "stop", "word": "stop", "limit": "length"}.get(stop_type, stop_type)
    evaluated, predicted = (final.get("tokens_evaluated"), final.get("tokens_predicted")) if final else (None, None)
    if final and n_prefix:  # The prefix was sent as prompt but is part of the output.
        evaluated, predicted = (evaluated or 0) - n_prefix, n_prefix + (predicted or 0)
    usage = (dict(prompt_tokens=evaluated, completion_tokens=predicted,
                  total_tokens=(evaluated or 0) + (predicted or 0)) if final else {})
    grid = (parse_grid(content) if parse_grid is not None and error is None and finish == "stop"
            and think_closed else None)
    result = dict(content=content, reasoning=reasoning, usage=usage, finish_reason=finish, stop_type=stop_type,
                  truncated=final.get("truncated") if final else None, think_closed=think_closed,
                  think_end_tokens=think_end_tokens, tool_calls=[], stream_done=final is not None, error=error,
                  cancelled=cancelled, final_grid=grid, streamed_tokens=n_prefix + progress["decoded"],
                  elapsed_s=(prefix["elapsed_s"] if prefix is not None else 0) + time.monotonic() - start,
                  first_content_s=first, last_content_s=last,
                  timings=final.get("timings") if final else None,
                  generation_settings=final.get("generation_settings") if final else None)
    # An interrupted stream has no final usage; its streamed count still measures the decode.
    generated = (predicted - n_prefix if predicted else 0) or progress["decoded"]
    result["decode_tokens_per_s_estimate"] = (
        generated / (last - first) if first is not None and last is not None and last > first and generated else None)
    if loop_guard is not None:
        result["loop_detection"] = dict(action=loop_action, stopped=loop_stopped, **loop_guard.snapshot())
    if prefix is not None:
        result["resumed"] = dict(source=prefix["source"], prior_tokens=n_prefix, prior_elapsed_s=prefix["elapsed_s"],
                                 segment_elapsed_s=result["elapsed_s"] - prefix["elapsed_s"])
    write_json(output / "result.json", result)
    (output / "reasoning.txt").write_text(reasoning)
    (output / "answer.txt").write_text(content)
    return result


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def generated(row):
    """Server-reported completion tokens, else the streamed count of an interrupted request."""
    return (row.get("usage") or {}).get("completion_tokens") or row.get("streamed_tokens")


class LiveView:
    """LIVE log lines plus live.md: throughput, context depth, finishes and running accuracy."""

    def __init__(self, output_root, planned, test_counts, routed_tasks, labelled, title="DeepSeek-V4", prior=None):
        self.output_root, self.planned, self.test_counts, self.title = output_root, planned, test_counts, title
        # Tokens per output decoded by the run this one resumes: not part of this run's throughput.
        self.prior = prior or {}
        self.routed_tasks, self.labelled = routed_tasks, labelled
        self.started = self.prev_time = time.monotonic()
        self.prev_total = 0
        self.peak_rate = 0.0

    def decoded(self, results, progress):
        # Rows of one multi-turn task share its usage (usage_shared_task); a running task's finished
        # requests are in decoded_before.
        rows = {r.get("usage_shared_task") or f"{r['task']}_{r['test_index']}": r for r in results}
        return (sum(max(0, (generated(r) or 0) - self.prior.get(f"{r['task']}_{r['test_index']}", 0))
                    for r in rows.values())
                + sum(p.get("decoded_before", 0) + p.get("decoded", 0) for p in progress.values()))

    def accuracy(self, results):
        correct = [r for r in results if r.get("correct")]
        score = sum(1 / self.test_counts[r["task"]] for r in correct)
        return len(correct), score

    def result_line(self, row, results):
        correct, score = self.accuracy(results)
        verdict = ("correct" if row.get("correct") else "wrong" if row.get("valid") else "no-grid") \
            if self.labelled else ("grid" if row.get("valid") else "no-grid")
        return (f"RESULT {row['task']}_{row['test_index']}: {verdict} finish={row.get('finish_reason')} "
                f"think={'closed' if row.get('think_closed') else 'open'} gen={generated(row)} "
                f"t/s={row.get('decode_tokens_per_s_estimate') or 0:.1f} {clock(row.get('elapsed_s') or 0)} | "
                f"done {len(results)}/{self.planned} correct {correct} score {score:.2f}/{self.routed_tasks}"
                + ("" if row.get("error") is None else f" error={row['error'][:120]}"))

    async def tick(self, results, progress, queued):
        now = time.monotonic()
        gpu = await asyncio.to_thread(gpu_memory)
        total = self.decoded(results, progress)
        rate = (total - self.prev_total) / max(1e-6, now - self.prev_time)
        self.prev_total, self.prev_time = total, now
        self.peak_rate = max(self.peak_rate, rate)
        average = total / max(1e-6, now - self.started)
        depths = [p["prompt_tokens"] + p.get("decoded", 0) for p in progress.values()]
        correct, score = self.accuracy(results)
        finishes = [r.get("finish_reason") for r in results]
        errors = sum(r.get("error") is not None for r in results)
        snap = dict(elapsed_s=now - self.started, running=len(progress), queued=queued, done=len(results),
                    planned=self.planned, valid=sum(bool(r.get("valid")) for r in results),
                    correct=correct if self.labelled else None, routed_score=score if self.labelled else None,
                    natural_stop=finishes.count("stop"), length=finishes.count("length"),
                    loop=finishes.count("loop_detected"), errors=errors,
                    decoded_tokens=total, tokens_per_s=rate, tokens_per_s_avg=average,
                    max_depth=max(depths, default=0), mean_depth=sum(depths) / len(depths) if depths else 0,
                    gpu_mib=gpu)
        accuracy = (f"correct {correct}/{len(results)} score {score:.2f}/{self.routed_tasks}" if self.labelled
                    else f"valid {snap['valid']}/{len(results)}")
        print(f"LIVE {clock(snap['elapsed_s'])} | run {snap['running']} queue {queued} done {len(results)}/{self.planned}"
              f" | {accuracy} | stop {snap['natural_stop']} len {snap['length']} loop {snap['loop']} err {errors}"
              f" | {rate:.1f} t/s now {average:.1f} avg | ctx max {snap['max_depth']/1000:.1f}k"
              f" mean {snap['mean_depth']/1000:.1f}k | gpu MiB {' '.join(map(str, gpu or []))}", flush=True)
        self.write_markdown(snap, results, progress)
        return snap

    def write_markdown(self, snap, results, progress):
        lines = [f"# {self.title} live ({datetime.now(timezone.utc).isoformat(timespec='seconds')})", "",
                 f"Elapsed {clock(snap['elapsed_s'])}; running {snap['running']}, queued {snap['queued']}, "
                 f"done {snap['done']}/{snap['planned']}; valid {snap['valid']}; "
                 + (f"correct {snap['correct']}, routed score {snap['routed_score']:.2f}/{self.routed_tasks}; "
                    if self.labelled else "")
                 + f"natural stop {snap['natural_stop']}, length {snap['length']}, errors {snap['errors']}.", "",
                 f"Decode {snap['tokens_per_s']:.1f} t/s (last interval), {snap['tokens_per_s_avg']:.1f} t/s average, "
                 f"{snap['decoded_tokens']:,} tokens; GPU MiB {snap['gpu_mib']}.", "",
                 "| Output | State | Prompt | Generated | t/s | Elapsed | Finish | Think | Grid | Correct |",
                 "|---|---|---:|---:|---:|---:|---|---|---|---|"]
        now = time.monotonic()
        for label, p in sorted(progress.items(), key=lambda item: -item[1].get("decoded", 0)):
            spent, done = now - p.get("task_started", p["started"]), p.get("decoded_before", 0) + p.get("decoded", 0)
            lines.append(f"| {label} | running | {p['prompt_tokens']} | {done} | "
                         f"{done / max(1e-6, spent):.1f} | {clock(spent)} | | | | |")
        for r in reversed(results):
            lines.append(f"| {r['task']}_{r['test_index']} | {'error' if r.get('error') else 'done'} | "
                         f"{r.get('prompt_tokens', '')} | {generated(r) or ''} | "
                         f"{r.get('decode_tokens_per_s_estimate') or 0:.1f} | {clock(r.get('elapsed_s') or 0)} | "
                         f"{r.get('finish_reason') or r.get('status', '')} | "
                         f"{'closed' if r.get('think_closed') else 'open'} | {'yes' if r.get('valid') else 'no'} | "
                         f"{r.get('correct', '')} |")
        (self.output_root / "live.md").write_text("\n".join(lines) + "\n")
