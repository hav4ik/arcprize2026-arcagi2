"""Qwen3.8-Flash-Next IQ3_S GGUF backend for the Qwen3.8 stage: our llama.cpp fork on 4x L4.

Prompts are rendered by the GGUF's own chat template (/apply-template: one user message,
enable_thinking, reasoning_effort, as the vLLM Flash-Next requests), checked, and sent to
/completion as token ids, so the stream, live view and timings are the dsv4 ones.
The server profile is the fastest measured setting of fork 57d25a26c (experiments/qwen38_iq3_l4/NOTES.md there):
llama-server at 26 x 128000, 26 real ARC prompts x 3072 tokens: decode 476 t/s with all 26 decoding (the profile of
fork 9be7f391d: 428 t/s on the same build), 259 s for the whole batch (265 s).
"""

import os
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import socket
import struct
import subprocess

import dsv4_llama


SERVED_NAME = "qwen38-flash-llama"
MODEL = "/mnt/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
MTP_MODEL = "/mnt/models/mtp-Qwen3.8-Flash-Next-Q4_0.gguf"
CPUS = "0-13,24-37"  # The all-reduce thread runs alone on core 14 (GGML_CUDA_AR_CPU_CORE).
# Without paged KV every slot reserves its full context: KV + indexer + pool ~6.4 KB/token/GPU at q8_0
# next to ~15.7 GB of weights (32 x 29k loaded, 32 x 32k ran out of memory). Paged KV (LLAMA_KV_PAGED,
# CUDA VMM) maps pages as sequences grow and spills to host memory when VRAM runs out, so there the
# limit is the live tokens of all slots, not slots x context.
TOKEN_BUDGET = 930_000
# B=26 is the largest decode batch measured to fit (B=27 failed in cublasCreate at 64k).
MAX_SLOTS = 26
# The server reserves its prompt graph for -ub tokens attending to LLAMA_IDX_PP_CAP rows (all --context rows
# without it), so that compute buffer grows with ubatch x rows: 256 x 131072 fits at B=26 (+694 MB/GPU),
# 2048 x 131072 failed to allocate 9.5 GB on GPU 0. The cap keeps ub 2048 within the reserve that fits; a
# prompt chunk deeper than the cap runs in smaller ubatches (ubatch x depth <= ubatch x cap), decode keeps
# the whole context. Needs a llama.cpp build with LLAMA_IDX_PP_CAP.
PP_RESERVE = 256 * 131072

# PROD_ENV + E_sl2 + the B=32 kernel work of experiments/qwen38_iq3_l4 (NOTES.md; the GEN_* keys are gen_prof4c only):
# decode: CPU all-reduce on core 14 with the q8 wire, split in push / pull around the next independent nodes
# (META_AR_SPLIT) and prefetching 8 MB of weights into L2 meanwhile (AR_PF_KB), context-parallel QSA KV and indexer
# keeping each device's top 256 blocks (DCP_TOPLOC), q8_0 pooled keys scored by their own kernel (IDX_SCORE),
# uniform FA mode 2, bf16 recurrent states, the shared expert split over the GPUs like the routed ones;
# prefill (ub 2048): sequence-parallel hc, expert MMQ grid on real tiles, 16 MB SP gather buffer;
# memory: paged KV with the raw indexer keys in host memory (decode reads only the pooled ones), cuBLAS created at
# init so it cannot fail once paged KV has the memory; LLAMA_LAZY_WARM reads the 27 GB PLE shard into the page cache
# (cold pages cost ~18%). LLAMA_KV_IDX_TYPE keeps the indexer cache at q8_0 under --qf-kv-type q4_0.
ENV_PROFILE = {
    "CUDA_MODULE_LOADING": "EAGER", "GGML_META_THREADS": "1",
    "GGML_CUDA_AR_CPU": "1", "GGML_CUDA_AR_CPU_Q8": "1", "GGML_CUDA_AR_CPU_CORE": "14",
    "GGML_CUDA_AR_RSAG_MIN_BYTES": "32768", "GGML_CUDA_AR_SMALL_MAX_BYTES": "67108864",
    "GGML_CUDA_AR_COPY_THRESHOLD": "1000000000000", "GGML_CUDA_AR_BUF_BYTES": "16777216",
    "GGML_CUDA_AR_CPU_BLOCKS": "24", "GGML_CUDA_AR_SP_SKIP_OWN": "1", "GGML_CUDA_AR_PF_KB": "8192",
    "GGML_META_AR_SPLIT": "1",
    "GGML_CUDA_MMVQ_MOE_MAX_BATCH": "32", "GGML_CUDA_MMVF_F32_MAX_BATCH": "8", "GGML_CUDA_MMVQ_MMA": "1",
    "GGML_CUDA_MMQ_EXPERT_TILE_AWARE": "1", "GGML_CUDA_MMQ_EXPERT_COMPACT": "1",
    "GGML_CUDA_MMQ_EXPERT_COMPACT_ALL": "1", "GGML_CUDA_FA_UNIFORM": "2", "GGML_CUDA_IDX_SCORE": "1",
    "GGML_CUDA_CUBLAS_INIT": "1",
    "LLAMA_QSA_DCP": "1", "LLAMA_QSA_IDX_SHARD": "1", "LLAMA_QSA_DCP_TOPLOC": "256", "LLAMA_QSA_POOL_TYPE": "q8_0",
    "LLAMA_QWEN4EXP_Q8": "hc", "LLAMA_QWEN4EXP_HC_SP": "256", "LLAMA_QWEN4EXP_HC_SP_LAST": "1",
    "LLAMA_QWEN4EXP_SHEXP_SPLIT": "1", "LLAMA_RS_TYPE_S": "bf16",
    "LLAMA_KV_PAGED": "1", "LLAMA_KV_IDX_HOST": "1", "LLAMA_KV_IDX_TYPE": "q8_0", "LLAMA_LAZY_WARM": "16",
}
# A draft-verify batch is B x (n+1) tokens; keep it on the MoE mmvq path.
MTP_ENV = {"GGML_CUDA_MMVQ_MOE_MAX_BATCH": "64"}
# Not bit-identical to the plain model (outputs stay coherent; measured in the notes): Q8 all-reduce wire,
# Q8 hidden-connection weights, sequence-parallel hc through the q8 gather, bf16 recurrent states, the DCP top-256
# cap, the indexer score kernel with the q8_0 pool, the split shared expert (with the q8 wire: 84/3168 teacher-forced
# argmax flips at B=32). The rest are opt-in (--qf-env): token-parallel QSA prefill (+3%), bf16 MoE down, replicated
# QSA KV, lossy indexer, the IQ4 tensor-core GEMV (GGML_CUDA_MMVQ_MMA_MASK=253, -0.3 ms at B=32) and the fused
# shared-expert gate (GGML_CUDA_FUSE_SHEXP_GATE=1, no measured gain).
NON_EXACT = ("GGML_CUDA_AR_CPU_Q8", "LLAMA_QWEN4EXP_Q8", "LLAMA_QWEN4EXP_HC_SP", "LLAMA_RS_TYPE_S",
             "LLAMA_QSA_DCP_TOPLOC", "GGML_CUDA_IDX_SCORE", "LLAMA_QSA_POOL_TYPE", "LLAMA_QWEN4EXP_SHEXP_SPLIT",
             "LLAMA_QSA_TSP", "GGML_CUDA_MOE_DOWN_BF16", "LLAMA_QSA_KV_REP", "LLAMA_QSA_IDX_LOSSY",
             "GGML_CUDA_MMVQ_MMA_MASK", "GGML_CUDA_FUSE_SHEXP_GATE")
EFFORT_TEXT = {"xhigh": "Reasoning effort is set to xhigh.", "low": "Reasoning effort is set to low.", "medium": None}
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"
SPECIALS = (("im_start", "<|im_start|>"), ("eos", "<|im_end|>"), ("think", "<think>"), ("think_end", "</think>"))


def add_arguments(parser):
    # --llama-server and --live-interval-seconds come from dsv4_llama.add_arguments.
    g = parser.add_argument_group("Qwen3.8-Flash llama.cpp backend (--qwen38-variant flash-llama)")
    g.add_argument("--qf-model", type=Path, default=Path(MODEL),
                   help="GGUF file (first shard of a split model) or the folder holding it.")
    g.add_argument("--qf-mtp-model", type=Path, default=Path(MTP_MODEL))
    g.add_argument("--qf-mtp-draft-max", type=int, default=0,
                   help="MTP draft tokens (draft on CUDA0); 0 disables. 2 pays below ~16 running requests "
                        "but its draft model and context need GPU0 memory: use fewer slots.")
    g.add_argument("--qf-reasoning-effort", choices=("xhigh", "medium", "low"), default="xhigh")
    g.add_argument("--qf-temperature", type=float, default=1.0)
    g.add_argument("--qf-top-p", type=float, default=0.95)
    g.add_argument("--qf-top-k", type=int, default=20, help="0 disables top-k.")
    g.add_argument("--qf-min-p", type=float, default=0.0)
    g.add_argument("--qf-seed", type=int, default=1000, help="Request seed = this + planned job number.")
    g.add_argument("--qf-ubatch", type=int, default=None,
                   help="Default: 2048 (the fastest prefill; 2560: -3%%, 4096: -18%%). LLAMA_IDX_PP_CAP "
                        "keeps ubatch x rows within the 256 x 131072 prompt-graph reserve measured to fit.")
    g.add_argument("--qf-batch", type=int, default=2048)
    g.add_argument("--qf-kv-type", default="q8_0")
    g.add_argument("--qf-token-budget", type=int, default=TOKEN_BUDGET,
                   help="Without paged KV, refuse --context x --server-max-num-seqs above this (q8_0 KV, "
                        "no MTP); 0 disables.")
    g.add_argument("--qf-cpus", default=CPUS, help="taskset CPU list for llama-server; empty disables pinning.")
    g.add_argument("--qf-env", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a server environment profile entry; KEY= removes it. Repeatable.")
    g.add_argument("--qf-server-arg", action="append", default=[], metavar="ARG",
                   help="Extra llama-server argument, written as --qf-server-arg=--flag. Repeatable.")


def ubatch(args):
    return args.qf_ubatch or 2048


def pp_cap(args):
    """LLAMA_IDX_PP_CAP: the rows of the prompt-graph reserve, a multiple of 256; 0 when all of --context fits."""
    if ubatch(args) * args.context <= PP_RESERVE:
        return 0
    return PP_RESERVE // ubatch(args) // 256 * 256


def env_profile(args):
    profile = dict(ENV_PROFILE, **(MTP_ENV if args.qf_mtp_draft_max else {}))
    if pp_cap(args):
        profile["LLAMA_IDX_PP_CAP"] = str(pp_cap(args))
    # Every GPU returns the top k logits of its vocab slice instead of all 248k: sampling a full row per sequence on
    # the CPU cost ~7.5 ms per step at B=24. The global top k are among those 4 x k, so with the request's top-k <= k
    # (and no repeat/DRY penalty ahead of it in the chain) the sampled token is the same. Not with the MTP draft.
    if args.qf_top_k and not args.qf_mtp_draft_max:
        profile["LLAMA_TP_CAND"] = str(args.qf_top_k)
    for item in args.qf_env:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"--qf-env expects KEY=VALUE, got {item!r}")
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
            raise ValueError(f"Invalid --qf-cpus entry {part!r}")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def available_cpus():
    return os.sched_getaffinity(0)


def free_cpus(args):
    """CPUs llama-server and its all-reduce thread are not pinned to, for tool code; [] runs it unpinned."""
    if args.base_url or not args.qf_cpus:
        return []
    core = env_profile(args).get("GGML_CUDA_AR_CPU_CORE", "")
    return sorted(available_cpus() - cpu_list(args.qf_cpus) - ({int(core)} if core.isdigit() else set()))


def validate_options(args):
    if args.qwen38_variant != "flash-llama":
        return
    if args.input_mode != "text" or not (args.solve_mode == "direct" or args.tir_interface == "repl"):
        raise ValueError("flash-llama supports text input with direct solving or --tir-interface repl only")
    if not (args.qf_temperature >= 0 and 0 < args.qf_top_p <= 1 and 0 <= args.qf_min_p < 1 and args.qf_top_k >= 0):
        raise ValueError("Invalid flash-llama sampling settings")
    if min(ubatch(args), args.qf_batch) < 1 or args.qf_mtp_draft_max < 0 or not args.live_interval_seconds > 0:
        raise ValueError("flash-llama batch and ubatch must be positive, MTP tokens non-negative")
    profile = env_profile(args)
    if args.base_url:
        return
    cand = profile.get("LLAMA_TP_CAND", "")
    if cand and (not cand.isdigit() or not 0 < args.qf_top_k <= int(cand)):
        raise ValueError(f"LLAMA_TP_CAND={cand} samples from the top {cand} logits of each GPU: it needs "
                         f"0 < --qf-top-k <= {cand}, got {args.qf_top_k}")
    paged = profile.get("LLAMA_KV_PAGED", "0") not in ("", "0")
    if paged and args.server_max_num_seqs > MAX_SLOTS:
        raise ValueError(f"{args.server_max_num_seqs} slots is above the {MAX_SLOTS} measured to fit on 4x L4")
    cap = profile.get("LLAMA_IDX_PP_CAP", "")
    if cap and (not cap.isdigit() or int(cap) % 256):
        raise ValueError(f"LLAMA_IDX_PP_CAP must be a multiple of 256, got {cap!r}")
    rows = min(args.context, int(cap or 0) or args.context)
    if ubatch(args) * rows > PP_RESERVE:
        raise ValueError(f"ubatch {ubatch(args)} x {rows} prompt-graph rows is above the {PP_RESERVE} reserve "
                         "measured to fit on 4x L4 (2048 x 131072 ran out of memory at startup)")
    if not paged and args.qf_token_budget and args.context * args.server_max_num_seqs > args.qf_token_budget:
        raise ValueError(f"{args.server_max_num_seqs} slots x {args.context} context exceeds the 4x L4 KV budget "
                         f"of {args.qf_token_budget} tokens without paged KV; use fewer slots (e.g. 7 x 131072)")
    # The profile pins threads to these CPUs; a different machine fails here, not hours into a cascade.
    pinned = cpu_list(args.qf_cpus) | ({int(profile["GGML_CUDA_AR_CPU_CORE"])}
                                       if profile.get("GGML_CUDA_AR_CPU_CORE", "").isdigit() else set())
    missing = sorted(pinned - available_cpus())
    if missing:
        raise ValueError(f"CPUs {missing} are not available here; set --qf-cpus and "
                         "--qf-env GGML_CUDA_AR_CPU_CORE=<free core> (or empty values to disable pinning)")


def sampling(args):
    return dict(temperature=args.qf_temperature, top_p=args.qf_top_p, top_k=args.qf_top_k,
                min_p=args.qf_min_p, repeat_penalty=1.0, seed_base=args.qf_seed)


def make_payload(args, ids, seed):
    return dict(prompt=ids, max_tokens=0, stream=True, cache_prompt=False, return_tokens=True,
                temperature=args.qf_temperature, top_p=args.qf_top_p, top_k=args.qf_top_k,
                min_p=args.qf_min_p, repeat_penalty=1.0, seed=seed)


def messages_for(prompt):
    return [dict(role="user", content=prompt)]


def template_request(prompt, effort, thinking=True):
    kwargs = dict(enable_thinking=thinking, **({"reasoning_effort": effort} if thinking else {}))
    return dict(messages=messages_for(prompt), chat_template_kwargs=kwargs,
                **({"reasoning_effort": effort} if thinking else {}))


def environment(args):
    profile = env_profile(args)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
    env.update(profile, CUDA_VISIBLE_DEVICES="0,1,2,3")
    removed = sorted(k for k in os.environ if k.startswith(("LLAMA_", "GGML_")) and k not in profile)
    return env, profile, removed


def command_for(args, model, mtp=None):
    slots = args.server_max_num_seqs
    pin = ["taskset", "-c", args.qf_cpus] if args.qf_cpus else []
    draft = (["--spec-type", "draft-mtp", "-md", str(mtp), "--spec-draft-n-max", str(args.qf_mtp_draft_max),
              "-devd", "CUDA0", "-ngld", "99"] if args.qf_mtp_draft_max else [])
    return [*pin, str(args.llama_server), "-m", str(model), "--alias", SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(args.qwen38_port),
            "--split-mode", "tensor", "-ngl", "999", "--fit", "off", "-fa", "on",
            "-ctk", args.qf_kv_type, "-ctv", args.qf_kv_type,
            # -c is the total over all slots: each --no-kv-unified slot gets --context. A full unified
            # cache would end every running request with "Context size has been exceeded".
            "-c", str(args.context * slots), "-np", str(slots), "--no-kv-unified",
            "-b", str(args.qf_batch), "-ub", str(ubatch(args)), "--jinja",
            "--no-context-shift", "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-warmup",
            "--timeout", "86400", "--metrics", "--no-webui", *draft, *args.qf_server_arg]


def launch(args):
    output = (args.output_root / "server").resolve()
    try:
        model, shards = dsv4_llama.resolve_model(args.qf_model)
        mtp, mtp_shards = dsv4_llama.resolve_model(args.qf_mtp_model) if args.qf_mtp_draft_max else (None, None)
        binary = args.llama_server.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"llama-server binary not found or not executable: {binary}")
        if args.qf_cpus and shutil.which("taskset") is None:
            raise ValueError("taskset is not installed; pass --qf-cpus= to run unpinned")
        general = dsv4_llama.gguf_general(model)
        command = command_for(args, model, mtp)
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
                   variant="flash-llama", served_model_name=SERVED_NAME, model=str(model), shards_bytes=shards,
                   mtp_model=str(mtp) if mtp else None, mtp_shards_bytes=mtp_shards,
                   mtp_draft_max=args.qf_mtp_draft_max, gguf_general=general, llama_server=str(binary),
                   llama_cpp=dsv4_llama.source_revision(binary), reasoning_effort=args.qf_reasoning_effort,
                   sampling=sampling(args), slots=args.server_max_num_seqs, context_per_slot=args.context)
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


async def render(client, url, prompt, effort, thinking=True):
    response = await client.post(url + "/apply-template", json=template_request(prompt, effort, thinking),
                                 timeout=120)
    response.raise_for_status()
    return response.json()["prompt"]


async def check_server(client, url, args):
    """Slot context/count, the special tokens the parser relies on, and the effort the template renders."""
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
    probe = await render(client, url, "Probe.", args.qf_reasoning_effort)
    expected = EFFORT_TEXT[args.qf_reasoning_effort]
    if not probe.endswith(GENERATION_PROMPT):
        raise ValueError(f"Chat template does not end with the thinking generation prompt: {probe[-80:]!r}")
    if expected is not None and expected not in probe:
        raise ValueError(f"Chat template ignores reasoning_effort={args.qf_reasoning_effort}: {probe[:300]!r}")
    return dict(id=SERVED_NAME, n_ctx=n_ctx, total_slots=slots, special_token_ids=specials, template_probe=probe,
                **{k: props.get(k) for k in ("model_path", "model_alias", "model_ftype", "build_info")})


async def tokenize_prompt(client, url, text, specials):
    if not text.endswith(GENERATION_PROMPT):
        raise ValueError(f"Rendered prompt does not end with the thinking generation prompt: {text[-80:]!r}")
    ids = await dsv4_llama.tokenize(client, url, text)
    # The user turn, ending with <|im_start|>assistant\n<think>\n and no closed thinking block.
    if (len(ids) < 4 or ids[0] != specials["im_start"] or specials["think"] not in ids[-2:]
            or specials["think_end"] in ids):
        raise ValueError("Rendered prompt does not tokenize to <|im_start|> ... assistant <think>")
    return ids


async def smoke(client, url, specials, output):
    """Short non-thinking request: proves token prompts, EOS and the stream format before long runs."""
    text = await render(client, url, "Return only the integer: 17 + 25.", None, thinking=False)
    ids = await dsv4_llama.tokenize(client, url, text)
    if ids[0] != specials["im_start"] or specials["think_end"] not in ids[-4:]:
        raise ValueError(f"Non-thinking smoke prompt does not end with an empty think block: {text[-80:]!r}")
    payload = dict(prompt=ids, max_tokens=48, stream=True, cache_prompt=False, return_tokens=True,
                   temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, repeat_penalty=1.0, seed=0)
    result = await dsv4_llama.stream_completion(client, url, payload, output, 300, parse_grid=None,
                                                specials=specials, thinking=False, marker="</think>")
    if result["error"] or result["stop_type"] != "eos" or "42" not in result["content"]:
        raise ValueError(f"Completion smoke failed ({result['error'] or result['stop_type']}): "
                         f"{result['content'][:200]!r}; see {output}")
    return dict(content=result["content"], usage=result["usage"], elapsed_s=result["elapsed_s"],
                timings=result["timings"])
