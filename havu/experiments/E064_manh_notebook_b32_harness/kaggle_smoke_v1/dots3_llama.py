"""dots3-note-prev Q3_K_XL GGUF backend for the Qwen3.8 stage: our llama.cpp fork on 4x L4.

Prompts are rendered by the GGUF's own chat template (/apply-template: one user message, enable_thinking),
checked, and sent to /completion as token ids, so the stream, live view and timings are the dsv4 ones.
The server profile is the fastest measured setting of the fork (experiments/dots3_note_gguf/Q3_SPEED.md, HC6):
24 slots x 32768, 24 real ARC prompts of 2-24k tokens, MTP 3 drafts: about 325 t/s of steady decode at a median
depth of about 7k, about 296 t/s at 26-31k (nsL30b).
"""

import os
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import socket
import struct
import subprocess

import dsv4_llama


SERVED_NAME = "dots3"
MODEL = "/kaggle/input/dots-studio_dots3-note-prev-Q3_K_XL"
# KV (q8_0) and the expert cache share each GPU's memory: 24 x 32768 is the measured fit with the cache at
# GGML_CUDA_EXPERT_CACHE_MIB=15600. Other splits of the same total (e.g. 12 x 65536) were not run.
TOKEN_BUDGET = 24 * 32768
MAX_SLOTS = 24

# HC6 of Q3_SPEED.md. Model: context-parallel MLA KV (DCP), f16 router, fused q/kv-a projection, experts split in
# heavy/light ff blocks. Expert cache: per GPU 15.6 GB of the 30 GB of expert slices stay in VRAM (LFU over decode
# counts); up to 16 missing experts of a lookup (4 routes each) run on 6 physical cores per GPU (CPU_HYBRID, AVX-512
# VNNI) and are added to the FFN output just before the all-reduce (CPU_HYBRID_LATE) instead of being copied over
# PCIe. Decode: the expert products in the MMQ decode kernels, sparse DSA attention over a quarter of the cells,
# each GPU returns the top 40 logits of its vocab slice (TP_CAND: the sampled token is the same for a request
# top-k <= 40).
ENV_PROFILE = {
    "LLAMA_DOTS3_DCP": "1", "LLAMA_DOTS3_ROUTER_F16": "1", "LLAMA_DOTS3_QKV_A_FUSE": "1",
    "LLAMA_DOTS3_EXPS_SPLIT": "1", "LLAMA_DOTS3_SHEXP_HEAVY": "1", "LLAMA_LOAD_THREADS": "8", "LLAMA_TP_CAND": "40",
    "NCCL_PROTO": "Simple",
    "GGML_CUDA_MMVQ_MMA": "1", "GGML_CUDA_MMVF_F32_MAX_BATCH": "8", "GGML_CUDA_MMVQ_MOE_MAX_BATCH": "16",
    "GGML_CUDA_AR_SMALL_MAX_BYTES": "1048576", "GGML_CUDA_AR_RSAG_MIN_BYTES": "32768",
    "GGML_CUDA_FATTN_SPARSE_MIN_KV": "1024", "GGML_CUDA_FATTN_SPARSE_RATIO": "0.25", "GGML_CUDA_FA_Q8_NATIVE_MLA": "1",
    "GGML_CUDA_MMQ_EXPERT_COMPACT": "1", "GGML_CUDA_MMQ_EXPERT_TILE_AWARE": "1",
    "GGML_CUDA_MMQ_DEC": "1", "GGML_CUDA_MMQ_DEC_CFG": "4222", "GGML_CUDA_MMQ_DEC_NSUB_GU": "2",
    "GGML_CUDA_MMQ_DEC_DOWN_CFG": "4212", "GGML_CUDA_MMQ_DEC_NSUB": "16",
    "GGML_CUDA_EXPERT_CACHE_TENSOR": "1", "GGML_CUDA_EXPERT_CACHE_TENSOR_ARCH": "dots3note",
    "GGML_CUDA_EXPERT_CACHE_MIB": "15600", "GGML_CUDA_EXPERT_CACHE_RESERVE_MIB": "700",
    "GGML_CUDA_EXPERT_CACHE_MIXED_LAYOUTS": "1", "GGML_CUDA_EXPERT_CACHE_LAYOUT_POOLS": "1",
    "GGML_CUDA_EXPERT_CACHE_POOL_BIAS": "112", "GGML_CUDA_EXPERT_CACHE_REUSE_PLAN": "1",
    "GGML_CUDA_EXPERT_CACHE_OVERLAP": "1", "GGML_CUDA_EXPERT_CACHE_EARLY_SPLIT": "1",
    "GGML_CUDA_EXPERT_CACHE_POLICY": "lfu", "GGML_CUDA_EXPERT_CACHE_LFU_MIN_TOKENS": "128",
    "GGML_CUDA_EXPERT_CACHE_DECODE_LFU": "1", "GGML_CUDA_EXPERT_CACHE_HOST_REGISTER": "1",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID": "16", "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_MAX_ROUTES": "4",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_LATE": "1",
    "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_CORES": "0-5/6-11/12-17/18-23",
}
# The MTP draft passes pick their tokens on the device from the TP_CAND candidates and hand them with the hidden
# row to the next pass there, without a host round trip.
MTP_ENV = {"LLAMA_SPEC_MTP_GPU_CHAIN": "1"}
# Not bit-identical to the GPU-only path (Q3_SPEED.md): the CPU computes its experts on Q8_K activations, and the late
# merge changes the sum order. The other profile keys were not compared token for token.
NON_EXACT = ("GGML_CUDA_EXPERT_CACHE_CPU_HYBRID", "GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_LATE")
HYBRID_CPU_FLAGS = ("avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vnni")
# The template renders no thinking prefix: the model opens <think> itself.
GENERATION_PROMPT = "<|endofuser|><|assistant|>"
SPECIALS = (("system", "<|system|>"), ("end_user", "<|endofuser|>"), ("assistant", "<|assistant|>"),
            ("eos", "<|endofassistant|>"), ("think", "<think>"), ("think_end", "</think>"))


def add_arguments(parser):
    # --llama-server and --live-interval-seconds come from dsv4_llama.add_arguments.
    g = parser.add_argument_group("dots3-note llama.cpp backend (--qwen38-variant dots3)")
    g.add_argument("--d3-model", type=Path, default=Path(MODEL),
                   help="GGUF file (first shard of a split model) or the folder holding it; the MTP layer is in it.")
    g.add_argument("--d3-mtp-draft-max", type=int, default=3, help="MTP draft tokens; 0 disables.")
    g.add_argument("--d3-temperature", type=float, default=1.0)
    g.add_argument("--d3-top-p", type=float, default=0.95)
    g.add_argument("--d3-top-k", type=int, default=40, help="At most LLAMA_TP_CAND (40).")
    g.add_argument("--d3-min-p", type=float, default=0.0)
    g.add_argument("--d3-seed", type=int, default=1000, help="Request seed = this + planned job number.")
    g.add_argument("--d3-ubatch", type=int, default=512)
    g.add_argument("--d3-batch", type=int, default=2048)
    g.add_argument("--d3-threads", type=int, default=16)
    g.add_argument("--d3-threads-batch", type=int, default=32)
    g.add_argument("--d3-kv-type", default="q8_0")
    g.add_argument("--d3-token-budget", type=int, default=TOKEN_BUDGET,
                   help="Refuse --context x --server-max-num-seqs above this; 0 disables.")
    g.add_argument("--d3-env", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a server environment profile entry; KEY= removes it. Repeatable.")
    g.add_argument("--d3-server-arg", action="append", default=[], metavar="ARG",
                   help="Extra llama-server argument, written as --d3-server-arg=--flag. Repeatable.")


def env_profile(args):
    profile = dict(ENV_PROFILE, **(MTP_ENV if args.d3_mtp_draft_max else {}))
    for item in args.d3_env:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"--d3-env expects KEY=VALUE, got {item!r}")
        if value:
            profile[key] = value
        else:
            profile.pop(key, None)
    return profile


def cpu_list(text):
    cpus = set()
    for part in filter(None, text.split(",")):
        first, _, last = part.partition("-")
        if not first.isdigit() or not (last or first).isdigit() or int(first) > int(last or first):
            raise ValueError(f"Invalid cpu list entry {part!r}")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def available_cpus():
    return os.sched_getaffinity(0)


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as info:
            return next((set(line.split(":", 1)[1].split()) for line in info if line.startswith("flags")), None)
    except OSError:
        return None


def validate_options(args):
    if args.qwen38_variant != "dots3":
        return
    if args.input_mode != "text" or args.solve_mode != "direct":
        raise ValueError("dots3 supports text input and direct solving only")
    if not (args.d3_temperature >= 0 and 0 < args.d3_top_p <= 1 and 0 <= args.d3_min_p < 1 and args.d3_top_k >= 0):
        raise ValueError("Invalid dots3 sampling settings")
    if (min(args.d3_ubatch, args.d3_batch, args.d3_threads, args.d3_threads_batch) < 1 or args.d3_mtp_draft_max < 0
            or not args.live_interval_seconds > 0):
        raise ValueError("dots3 batch, ubatch and threads must be positive, MTP tokens non-negative")
    profile = env_profile(args)
    if args.base_url:
        return
    cand = profile.get("LLAMA_TP_CAND", "")
    if cand and (not cand.isdigit() or not 0 < args.d3_top_k <= int(cand)):
        raise ValueError(f"LLAMA_TP_CAND={cand} samples from the top {cand} logits of each GPU: it needs "
                         f"0 < --d3-top-k <= {cand}, got {args.d3_top_k}")
    if profile.get("LLAMA_SPEC_MTP_GPU_CHAIN", "0") not in ("", "0") and not cand:
        raise ValueError("LLAMA_SPEC_MTP_GPU_CHAIN needs LLAMA_TP_CAND")
    if args.server_max_num_seqs > MAX_SLOTS:
        raise ValueError(f"{args.server_max_num_seqs} slots is above the {MAX_SLOTS} measured on 4x L4")
    if args.d3_token_budget and args.context * args.server_max_num_seqs > args.d3_token_budget:
        raise ValueError(f"{args.server_max_num_seqs} slots x {args.context} context exceeds the 4x L4 KV budget of "
                         f"{args.d3_token_budget} tokens next to the expert cache (measured: 24 x 32768)")
    if profile.get("GGML_CUDA_EXPERT_CACHE_CPU_HYBRID", "0") not in ("", "0"):
        flags = cpu_flags()
        if flags is not None and not set(HYBRID_CPU_FLAGS) <= flags:
            raise ValueError("GGML_CUDA_EXPERT_CACHE_CPU_HYBRID needs AVX-512 VNNI, which this CPU lacks; "
                             "--d3-env GGML_CUDA_EXPERT_CACHE_CPU_HYBRID= turns it off (much slower)")
        # The workers are pinned to these CPUs; a different machine fails here, not hours into a cascade.
        cores = profile.get("GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_CORES", "")
        groups = [cpu_list(group) for group in cores.split("/")] if cores else []
        if cores and (len(groups) != 4 or not all(groups)):
            raise ValueError(f"GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_CORES needs one cpu list per GPU, got {cores!r}")
        missing = sorted(set().union(*groups) - available_cpus())
        if missing:
            raise ValueError(f"CPUs {missing} are not available here; set "
                             "--d3-env GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_CORES=<4 lists separated by />")


def sampling(args):
    return dict(temperature=args.d3_temperature, top_p=args.d3_top_p, top_k=args.d3_top_k,
                min_p=args.d3_min_p, repeat_penalty=1.0, seed_base=args.d3_seed)


def make_payload(args, ids, seed):
    return dict(prompt=ids, max_tokens=0, stream=True, cache_prompt=False, return_tokens=True,
                temperature=args.d3_temperature, top_p=args.d3_top_p, top_k=args.d3_top_k,
                min_p=args.d3_min_p, repeat_penalty=1.0, seed=seed)


def messages_for(prompt):
    return [dict(role="user", content=prompt)]


def template_request(prompt, thinking=True):
    return dict(messages=messages_for(prompt), chat_template_kwargs=dict(enable_thinking=thinking))


def environment(args):
    profile = env_profile(args)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
    env.update(profile, CUDA_VISIBLE_DEVICES="0,1,2,3")
    removed = sorted(k for k in os.environ if k.startswith(("LLAMA_", "GGML_")) and k not in profile)
    return env, profile, removed


def command_for(args, model):
    slots = args.server_max_num_seqs
    draft = ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.d3_mtp_draft_max)]
    draft = draft if args.d3_mtp_draft_max else []
    return [str(args.llama_server), "-m", str(model), "--alias", SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(args.qwen38_port),
            "--split-mode", "tensor", "--device", "CUDA0,CUDA1,CUDA2,CUDA3", "-ngl", "99", "--fit", "off",
            "-fa", "on", "-ctk", args.d3_kv_type, "-ctv", args.d3_kv_type,
            # -c is the total over all slots: each --no-kv-unified slot gets --context.
            "-c", str(args.context * slots), "-np", str(slots), "--no-kv-unified",
            "-b", str(args.d3_batch), "-ub", str(args.d3_ubatch),
            "-t", str(args.d3_threads), "-tb", str(args.d3_threads_batch), "--jinja", "--reasoning", "on",
            "--no-context-shift", "--cache-ram", "0", "--ctx-checkpoints", "0", "--load-mode", "dio",
            "--timeout", "86400", "--metrics", "--no-webui", *draft, *args.d3_server_arg]


def launch(args):
    output = (args.output_root / "server").resolve()
    try:
        model, shards = dsv4_llama.resolve_model(args.d3_model)
        binary = args.llama_server.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"llama-server binary not found or not executable: {binary}")
        general = dsv4_llama.gguf_general(model)
        command = command_for(args, model)
        env, profile, removed = environment(args)
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
    receipt = dict(status="prepared", created_utc=datetime.now(timezone.utc).isoformat(),
                   command=command, environment_overrides=profile, environment_removed=removed,
                   non_exact=[k for k in NON_EXACT if profile.get(k, "0") not in ("", "0")],
                   variant="dots3", served_model_name=SERVED_NAME, model=str(model), shards_bytes=shards,
                   mtp_draft_max=args.d3_mtp_draft_max, gguf_general=general, llama_server=str(binary),
                   llama_cpp=dsv4_llama.source_revision(binary), sampling=sampling(args),
                   slots=args.server_max_num_seqs, context_per_slot=args.context)
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


async def render(client, url, prompt, thinking=True):
    response = await client.post(url + "/apply-template", json=template_request(prompt, thinking), timeout=120)
    response.raise_for_status()
    return response.json()["prompt"]


async def check_server(client, url, args):
    """Slot context/count, the special tokens the parser relies on, and the thinking generation prompt."""
    response = await client.get(url + "/props", timeout=20)
    response.raise_for_status()
    props = response.json()
    n_ctx, slots = props.get("default_generation_settings", {}).get("n_ctx"), props.get("total_slots")
    if not isinstance(n_ctx, int) or n_ctx < args.context:
        raise ValueError(f"Server slot context {n_ctx} is below --context {args.context}")
    if not isinstance(slots, int) or slots < args.concurrency:
        raise ValueError(f"Server has {slots} slots, below --concurrency {args.concurrency}")
    specials = {}
    for name, text in SPECIALS:
        ids = await dsv4_llama.tokenize(client, url, text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token in the served vocabulary: {ids}")
        specials[name] = ids[0]
    probe = await render(client, url, "Probe.")
    if not probe.endswith(GENERATION_PROMPT) or "<no_think>" in probe:
        raise ValueError(f"Chat template does not end with the thinking generation prompt: {probe[-80:]!r}")
    return dict(id=SERVED_NAME, n_ctx=n_ctx, total_slots=slots, special_token_ids=specials, template_probe=probe,
                **{k: props.get(k) for k in ("model_path", "model_alias", "model_ftype", "build_info")})


async def tokenize_prompt(client, url, text, specials):
    if not text.endswith(GENERATION_PROMPT) or "<no_think>" in text:
        raise ValueError(f"Rendered prompt does not end with the thinking generation prompt: {text[-80:]!r}")
    ids = await dsv4_llama.tokenize(client, url, text)
    # <|system|> ... the user turn, ending with <|endofuser|><|assistant|> and no thinking block.
    if (len(ids) < 4 or ids[0] != specials["system"] or ids[-2:] != [specials["end_user"], specials["assistant"]]
            or specials["think"] in ids or specials["think_end"] in ids):
        raise ValueError("Rendered prompt does not tokenize to <|system|> ... <|endofuser|><|assistant|>")
    return ids


async def smoke(client, url, specials, output):
    """Short non-thinking request: proves token prompts, EOS and the stream format before long runs."""
    text = await render(client, url, "Return only the integer: 17 + 25.", thinking=False)
    ids = await dsv4_llama.tokenize(client, url, text)
    if ids[0] != specials["system"] or specials["assistant"] not in ids:
        raise ValueError(f"Non-thinking smoke prompt has no system and assistant turn: {text[-80:]!r}")
    payload = dict(prompt=ids, max_tokens=64, stream=True, cache_prompt=False, return_tokens=True,
                   temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, repeat_penalty=1.0, seed=0)
    result = await dsv4_llama.stream_completion(client, url, payload, output, 300, parse_grid=None,
                                                specials=specials, thinking=False, marker="</think>")
    if result["error"] or result["stop_type"] != "eos" or "42" not in result["content"]:
        raise ValueError(f"Completion smoke failed ({result['error'] or result['stop_type']}): "
                         f"{result['content'][:200]!r}; see {output}")
    return dict(content=result["content"], usage=result["usage"], elapsed_s=result["elapsed_s"],
                timings=result["timings"])
