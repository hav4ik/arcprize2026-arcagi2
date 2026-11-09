"""DeepSeek-V4.1 Flash REPL-TIR and vision on our vLLM fork (--qwen38-variant dsv4-vllm).

The official V4.1 encoder renders every REPL turn as on llama-server (dsv41_repl); vLLM's /tokenize encodes
its text, so the prompt ids are the served tokenizer's and the REPL ledger extends them turn by turn.
Text-only prompts stream from /v1/completions as ids. With images, /inference/v1/generate receives the ids
with each image's placeholder run collapsed to one <｜deepseek_image｜> plus the PNGs; the server expands
each run again and reports the prompt it ran, which must equal ours. The CoT second attempt uses vLLM's
chat path (the fork's V4.1 tokenizer renders it) with the same effort and sampling.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile

import dsv41_repl
import dsv4_llama
import qwen38_repl as repl
import qwen38_server
from qwen38_repl_vision import ChatPrompt


MODEL = Path("/mnt/deepseek-E128-2bit")
SERVED_NAME = "dsv41"


def add_arguments(parser):
    g = parser.add_argument_group("DeepSeek-V4.1 on our vLLM fork (--qwen38-variant dsv4-vllm); "
                                  "sampling and effort come from the --dsv4-* options")
    g.add_argument("--dsv41-vllm-model", type=Path, default=MODEL, help="HF checkpoint folder (E128 2-bit GPTQ).")
    g.add_argument("--dsv41-vllm-block-size", type=int, default=128)
    g.add_argument("--dsv41-vllm-linear-backend", default="humming")
    g.add_argument("--dsv41-vllm-engram-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--dsv41-vllm-cudagraph-max", type=int, default=None,
                   help="Largest FULL CUDA graph in tokens; defaults to the server batch (times the decode "
                        "query length under a --vllm-arg=--speculative-config).")
    g.add_argument("--dsv41-vllm-replay-graph-max", type=int, default=0,
                   help="Run the decoder replay layers (20-39) of eager prefill/mixed steps in CUDA graphs "
                        "of 32, 64, ... up to this many rows (a multiple of 32; wheel g70867e279f+); 0 = off.")
    g.add_argument("--dsv41-vllm-triton-cache", type=Path, default=None,
                   help="A warmed TRITON_CACHE_DIR; defaults to the environment's, else a fresh temp folder.")
    g.add_argument("--dsv41-vllm-vision-weights", type=Path, nargs="+", default=[],
                   help="Safetensors holding the vision tower (vision.*, aligner.*, image_start/end/newline) for a "
                        "checkpoint without it; the server loads a symlinked view of --dsv41-vllm-model with them.")
    g.add_argument("--dsv41-vllm-engram-model", type=Path, default=None,
                   help="A build of the same checkpoint holding the Engram tables (model-engram-*.safetensors) that "
                        "--dsv41-vllm-model lacks; the server loads a symlinked view with them linked in.")


def validate_options(args):
    if args.qwen38_variant != "dsv4-vllm":
        return
    # --dsv41-harness havu runs Vu's solver on this server instead (dsv41_havu).
    if getattr(args, "dsv41_harness", "repl") != "havu" and (
            args.solve_mode != "tir" or getattr(args, "tir_interface", None) != "repl"):
        raise ValueError("dsv4-vllm runs --solve-mode tir --tir-interface repl (its CoT second attempt included)")
    args.dsv4_profile = dsv41_repl.PROFILE
    dsv4_llama._active["profile"] = args.dsv4_profile
    profile = dsv4_llama.profile_of(args)
    if args.dsv4_top_p is None:
        args.dsv4_top_p = profile["top_p"]
    if args.dsv4_top_k is None:
        args.dsv4_top_k = profile["top_k"]
    if not (args.dsv4_temperature >= 0 and 0 < args.dsv4_top_p <= 1 and 0 <= args.dsv4_min_p < 1
            and args.dsv4_top_k >= 0):
        raise ValueError("Invalid dsv4 sampling settings")
    if args.dsv41_vllm_block_size < 1 or (args.dsv41_vllm_cudagraph_max or 1) < 1:
        raise ValueError("Block size and CUDA graph batch must be positive")
    if args.dsv41_vllm_replay_graph_max < 0 or args.dsv41_vllm_replay_graph_max % 32:
        raise ValueError("--dsv41-vllm-replay-graph-max must be 0 or a positive multiple of 32")
    if not args.base_url and not (args.dsv41_vllm_model / "config.json").is_file():
        raise ValueError(f"No config.json in --dsv41-vllm-model {args.dsv41_vllm_model}")
    if args.dsv41_vllm_vision_weights and (args.input_mode != "text-image" or args.base_url):
        raise ValueError("--dsv41-vllm-vision-weights serves our vision tower: use --input-mode text-image "
                         "without --base-url")
    if args.dsv41_vllm_engram_model and args.base_url:
        raise ValueError("--dsv41-vllm-engram-model serves our checkpoint: drop --base-url")
    if not args.base_url and (args.input_mode == "text-image" or args.dsv41_vllm_engram_model):
        served_index(args)
    # The gzip loop guard reads this checkpoint's tokenizer.json.
    args.model_qwen38 = str(args.dsv41_vllm_model)
    dsv4_llama.load_encoder()


def sampling(args):
    """The request fields of the V4.1 profile; unseeded, as the other vLLM backends."""
    return dict(temperature=args.dsv4_temperature, top_p=args.dsv4_top_p, top_k=args.dsv4_top_k,
                min_p=args.dsv4_min_p)


def make_payload(args, ids):
    return dict(prompt=ids, max_tokens=0, **sampling(args))


def chat_overrides(args):
    """vLLM chat fields for a non-REPL request: effort to the fork's V4.1 template, profile sampling."""
    # The top-level field would override the template's effort and takes no integer effort.
    return dict(reasoning_effort=None, chat_template_kwargs=dict(thinking=True, reasoning_effort=args.dsv4_reasoning_effort),
                **sampling(args))


def source_paths(args):
    """The files that decide what dsv4-vllm sends, for the manifest checksums."""
    if args.qwen38_variant != "dsv4-vllm":
        return []
    return [Path(__file__), Path(dsv41_repl.__file__), dsv4_llama.profile_of(args)["encoder"]]


def read_metadata(model):
    """The image-token geometry of the checkpoint's vision config, as the fork's processor reads it."""
    path = Path(model) / "config.json"
    config = json.loads(path.read_text())
    vision = config.get("vision_config") or {}
    if not vision or config.get("image_token_id") is None:
        raise ValueError(f"{model} has no DeepSeek-V4.1 vision config")
    if vision.get("max_wh_ratio") is not None:
        raise ValueError("The checkpoint resizes wide images (max_wh_ratio); the renderer only pads")
    factor = vision.get("patch_size", 14) * vision.get("downsample_ratio", 3)
    max_tokens = vision.get("max_image_tokens", 1024)
    return dict(token_layout=dsv41_repl.TOKEN_LAYOUT, spatial_factor=factor,
                min_pixels=min(vision.get("min_pixels", 295936), max_tokens * factor * factor),
                max_tokens=max_tokens, image_token_id=config["image_token_id"],
                placeholder=dsv41_repl.encoder().IMAGE_PLACEHOLDER, model=str(model),
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


INDEX = "model.safetensors.index.json"
ENGRAM_TABLE = re.compile(r"\.engram\.(embed\.(weight|scale)(\.rows_\d+)?|embed_tokens\.row_map)$")
ROW_CHUNK = re.compile(r"\.rows_\d+$")  # a table split by rows: vLLM loads either form


def safetensors_shapes(path):
    """Tensor name -> shape from a safetensors header, without reading the tensors."""
    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    return {name: entry["shape"] for name, entry in header.items() if name != "__metadata__"}


def vision_shapes(config):
    """The vision tower's tensors and shapes for this config.json, as vLLM's DeepseekV41ForCausalLM loads them."""
    vision, dim = config["vision_config"], (config.get("text_config") or config)["hidden_size"]
    width, inter, r = vision["hidden_size"], vision["intermediate_size"], vision["downsample_ratio"]
    shapes = {"image_start": [dim], "image_end": [dim], "image_newline": [dim],
              "aligner.w1.weight": [dim, width * r * r], "aligner.w1.bias": [dim],
              "aligner.w2.weight": [dim, dim], "aligner.w2.bias": [dim],
              "vision.patch_embed.proj.weight": [width, 3 * vision["patch_size"] ** 2],
              "vision.patch_embed.proj.bias": [width], "vision.norm.weight": [width]}
    for i in range(vision["num_hidden_layers"]):
        shapes.update({f"vision.blocks.{i}.{name}": shape for name, shape in (
            ("attn.wqkv.weight", [3 * width, width]), ("attn.wqkv.bias", [3 * width]),
            ("attn.wo.weight", [width, width]), ("attn.wo.bias", [width]),
            ("mlp.w1.weight", [2 * inter, width]), ("mlp.w2.weight", [width, inter]),
            ("norm1.weight", [width]), ("norm2.weight", [width]))})
    return shapes


def engram_settings(model):
    config = json.loads((Path(model) / "config.json").read_text())
    return {key: value for part in (config, config.get("text_config") or {}) for key, value in part.items()
            if "engram" in key}


def engram_tables(model, donor, weight_map):
    """weight_map with the Engram tables of `donor` (the same checkpoint with them) in place of the model's absent
    ones, and the donor files to link; vLLM loads whole files, so those must hold nothing but the tables."""
    model, donor = Path(model), Path(donor)
    if engram_settings(model) != engram_settings(donor):
        raise ValueError(f"--dsv41-vllm-engram-model {donor} has another Engram config than {model}")
    if (donor / "gptq").is_dir() and not (model / "gptq").is_dir():
        raise ValueError(f"{model} has no gptq/ expert files (vLLM would load MXFP4 experts and run out of memory)")
    own = {name: file for name, file in weight_map.items() if ENGRAM_TABLE.search(name)}
    for file in sorted(set(own.values())):
        if (model / file).exists():
            raise ValueError(f"{model} has its own Engram tables ({file}): drop --dsv41-vllm-engram-model")
        if any(f == file and name not in own for name, f in weight_map.items()):
            raise ValueError(f"{model} lacks {file}, which holds more than Engram tables")
    tables = {name: file for name, file in json.loads((donor / INDEX).read_text())["weight_map"].items()
              if ENGRAM_TABLE.search(name)}
    layers = {name.split(".engram.")[0] for name in weight_map if ".engram." in name and name not in own}
    whole = lambda names: {ROW_CHUNK.sub("", name) for name in names}
    if not tables or {name.split(".engram.")[0] for name in tables} != layers or (own and whole(own) != whole(tables)):
        raise ValueError(f"--dsv41-vllm-engram-model {donor} does not hold the Engram tables of {model}'s layers")
    files = {}
    for file in sorted(set(tables.values())):
        if set(safetensors_shapes(donor / file)) != {name for name, f in tables.items() if f == file}:
            raise ValueError(f"{donor / file} holds more than its indexed Engram tables")
        if (model / file).exists():
            raise ValueError(f"{donor / file} collides with {model / file}")
        files[file] = donor / file
    return {**{name: file for name, file in weight_map.items() if name not in own}, **tables}, files


def served_index(args):
    """The safetensors index vLLM loads and the files to link beside --dsv41-vllm-model's: the
    --dsv41-vllm-vision-weights files and --dsv41-vllm-engram-model's Engram tables. Text-image needs the whole
    vision tower in it, shaped as the config says."""
    model = Path(args.dsv41_vllm_model)
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / INDEX).read_text())
    weight_map, files = dict(index["weight_map"]), {}
    if args.dsv41_vllm_engram_model:
        weight_map, files = engram_tables(model, args.dsv41_vllm_engram_model, weight_map)
    for path in map(Path, args.dsv41_vllm_vision_weights):
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError(f"--dsv41-vllm-vision-weights {path} is not a safetensors file")
        if path.name in files or (model / path.name).exists():
            raise ValueError(f"--dsv41-vllm-vision-weights {path.name} collides with another checkpoint file")
        files[path.name] = path
        for name in safetensors_shapes(path):
            if name in weight_map:
                raise ValueError(f"{name} in {path} is already in {weight_map[name]}")
            weight_map[name] = path.name
    absent = sorted({file for file in weight_map.values() if file not in files and not (model / file).is_file()})
    if absent:
        raise ValueError(f"{model} lacks {absent[:3]}, which its index lists")
    if args.input_mode == "text-image":
        if not config.get("vision_config"):
            raise ValueError(f"{model} has no DeepSeek-V4.1 vision config")
        expected, shapes = vision_shapes(config), {}
        missing = sorted(set(expected) - set(weight_map))
        if missing:
            raise ValueError(f"No weights for {len(missing)} vision tensors, e.g. {missing[:4]}: the vision tower "
                             "needs them all (the aligner too)")
        for file in {weight_map[name] for name in expected}:
            shapes.update(safetensors_shapes(files.get(file, model / file)))
        for name, shape in expected.items():
            if shapes.get(name) != shape:
                raise ValueError(f"{name} in {weight_map[name]} has shape {shapes.get(name)}, the config needs {shape}")
    return dict(index, weight_map=weight_map), files


def checkpoint_view(args):
    """--dsv41-vllm-model with the files served_index adds: symlinks plus the merged index. The path depends only on
    the inputs, so restarts reuse it (vLLM's compile cache keys on the model path). Folders (gptq/) are real ones of
    file links: vLLM's post-load page-cache drop walks the view with rglob, which skips linked folders."""
    model = args.dsv41_vllm_model.resolve()
    index, files = served_index(args)
    index = json.dumps(index, indent=2) + "\n"
    inputs = [model, *args.dsv41_vllm_vision_weights, *filter(None, [args.dsv41_vllm_engram_model])]
    digest = hashlib.sha256(json.dumps([str(Path(p).resolve()) for p in inputs]).encode()).hexdigest()[:12]
    view = args.temp_root.resolve() / f"dsv41-model-{digest}"
    if not view.exists():
        view.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=view.name + ".", dir=view.parent))
        for item in model.iterdir():
            if item.is_dir():
                (staging / item.name).mkdir()
                for sub in item.iterdir():
                    (staging / item.name / sub.name).symlink_to(sub)
            elif item.name != INDEX:
                (staging / item.name).symlink_to(item)
        for name, path in files.items():
            (staging / name).symlink_to(path.resolve())
        (staging / INDEX).write_text(index)
        staging.rename(view)
    if (view / INDEX).read_text() != index:
        raise ValueError(f"{view} is a view of another checkpoint")
    return view


def speculative_tokens(vllm_args):
    """num_speculative_tokens of a --speculative-config among the pass-through vLLM args, else 0."""
    for i, arg in enumerate(vllm_args):
        name, eq, value = arg.partition("=")
        if name.replace("_", "-") == "--speculative-config":
            config = json.loads(value if eq else vllm_args[i + 1])
            return int(config.get("num_speculative_tokens") or 0)
    return 0


def capture_sizes(args):
    """FULL CUDA graph sizes in tokens under speculative decoding: whole decode batches of 1, 2, 4, 8, 16, ...
    requests. vLLM keeps an explicit max on its token grid, which needs ~3x the graphs once rounded to the
    decode query length and stops short of the full batch. None without speculative decoding."""
    query_len = 1 + speculative_tokens(args.vllm_arg)
    if query_len == 1:
        return None
    batch = args.server_max_num_seqs
    limit = args.dsv41_vllm_cudagraph_max or batch * query_len
    counts = sorted(n for n in {1, 2, 4, *range(8, batch, 8), batch} if n <= batch)
    sizes = [n * query_len for n in counts if n * query_len <= limit]
    if not sizes:
        raise ValueError(f"--dsv41-vllm-cudagraph-max {limit} is below the decode query length {query_len}")
    return sizes


def command_for(args, model=None):
    batch = args.server_max_num_seqs
    compilation = dict(cudagraph_mode="FULL", max_cudagraph_capture_size=args.dsv41_vllm_cudagraph_max or batch)
    if sizes := capture_sizes(args):
        compilation.update(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=sizes[-1])
    if args.dsv41_vllm_replay_graph_max:
        compilation["decoder_replay_cudagraph_capture_sizes"] = list(
            range(32, args.dsv41_vllm_replay_graph_max + 1, 32))
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model or args.dsv41_vllm_model),
               "--served-model-name", SERVED_NAME, "--host", "127.0.0.1", "--port", str(args.qwen38_port),
               "--trust-remote-code", "--tensor-parallel-size", str(args.tensor_parallel_size),
               "--distributed-executor-backend", "mp", "--kv-cache-dtype", "fp8",
               "--block-size", str(args.dsv41_vllm_block_size), "--max-model-len", str(args.context),
               "--max-num-seqs", str(batch), "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--enable-prefix-caching", "--enable-batch-sharded-sampling",
               "--engram-config", json.dumps({"cpu_offload": args.dsv41_vllm_engram_cpu_offload}),
               "--linear-backend", args.dsv41_vllm_linear_backend,
               "--compilation-config", json.dumps(compilation),
               "--reasoning-parser", "deepseek_v41",
               # Exact-id multimodal prompts and the per-request prefix cache hit.
               "--enable-scale-out", "--enable-prompt-tokens-details"]
    if args.kv_offload_gib_per_rank:
        command += ["--kv-transfer-config", json.dumps({
            "kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": int(args.kv_offload_gib_per_rank * 1024**3),
                                          "lazy_offload": args.kv_offload_mode == "lazy"}})]
    if getattr(args, "dsv41_harness", "repl") == "havu":  # Vu's chat turns (images, final turns): server tool calls
        command += ["--enable-auto-tool-choice", "--tool-call-parser", "deepseek_v41"]
    if args.input_mode == "text-image":
        command += ["--limit-mm-per-prompt", json.dumps({"image": args.vision_max_images})]
    else:
        command.append("--language-model-only")
    if args.eager:
        command.append("--enforce-eager")
    return command + args.vllm_arg


def environment(args, temp):
    env = os.environ.copy()
    overrides = {"CUDA_VISIBLE_DEVICES": qwen38_server.selected_gpu_ids(args), "VLLM_USE_V2_MODEL_RUNNER": "1",
                 "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_PLUGINS": "", "TOKENIZERS_PARALLELISM": "false",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                 "VLLM_NO_USAGE_STATS": "1", "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp)}
    for key, value in (("TORCH_CUDA_ARCH_LIST", "8.9"), ("FLASHINFER_CUDA_ARCH_LIST", "8.9"), ("MAX_JOBS", "8")):
        overrides[key] = env.get(key, value)
    overrides["TRITON_CACHE_DIR"] = str(args.dsv41_vllm_triton_cache or env.get("TRITON_CACHE_DIR") or temp / "triton")
    # Batch-sharded sampling sends each rank's top k logits only when every request samples within them.
    if int(env.get("VLLM_BATCH_SHARD_TOPK") or 0) < args.dsv4_top_k:
        overrides["VLLM_BATCH_SHARD_TOPK"] = str(args.dsv4_top_k)
    env.update(overrides)
    return env, overrides


def launch(args, name="server"):
    output = (args.output_root / name).resolve()
    temp_root = args.temp_root.resolve()
    try:
        metadata = read_metadata(args.dsv41_vllm_model) if args.input_mode == "text-image" else None
        if not (args.dsv41_vllm_model / "tokenizer.json").is_file():
            raise ValueError("Local tokenizer.json is required; no downloads")
        if len(os.fsencode(temp_root / ("d41-" + "x" * 8) / ("x" * 36))) > 107:
            raise ValueError("Temp root too long for vLLM Unix sockets")
        if output.exists():
            raise ValueError(f"Use a fresh server output directory: {output}")
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=10)
        if processes.strip():
            raise ValueError(f"GPU processes are active; will not stop them: {processes.strip()}")
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", args.qwen38_port))
        served = (checkpoint_view(args) if args.dsv41_vllm_vision_weights or args.dsv41_vllm_engram_model
                  else args.dsv41_vllm_model)
        command = command_for(args, served)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        raise RuntimeError(str(exc)) from exc
    output.mkdir(parents=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="d41-", dir=temp_root))
    env, overrides = environment(args, temp)
    receipt = dict(status="prepared", created_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   environment_overrides=overrides, variant="dsv4-vllm", served_model_name=SERVED_NAME,
                   model=str(args.dsv41_vllm_model), served_model=str(served),
                   vision_weights=[str(p) for p in args.dsv41_vllm_vision_weights],
                   engram_model=args.dsv41_vllm_engram_model and str(args.dsv41_vllm_engram_model), vision=metadata,
                   input_mode=args.input_mode,
                   tensor_parallel_size=args.tensor_parallel_size, batch=args.server_max_num_seqs,
                   context=args.context, kv_cache_dtype="fp8",
                   cpu_kv_offload_gib_total=args.tensor_parallel_size * args.kv_offload_gib_per_rank,
                   capacity_note="Use measured vLLM capacity; CPU KV offload does not add resident GPU KV.",
                   versions={})
    for package in ("vllm", "torch", "transformers", "triton", "flashinfer-python"):
        try:
            receipt["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            receipt["versions"][package] = None
    receipt_path = output / "launch.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    process = None
    try:
        with (output / "server.log").open("xb") as log:
            process = subprocess.Popen(command, env=env, cwd=output, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(status="launched_not_yet_ready", pid=process.pid)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print("Command:", shlex.join(command), flush=True)
        print("PID:", process.pid, "Log:", output / "server.log", flush=True)
    except BaseException:
        qwen38_server.stop_owned_process(process)
        raise
    return process, receipt


class Dsv41VllmServer(dsv41_repl.Dsv41ReplServer, repl.VllmServer):
    """The V4.1 encoder's REPL turns (dsv41_repl) on vLLM: /tokenize ids, /v1/completions or generate."""

    def __init__(self, client, url, model, *, root, metadata=None, max_images=0):
        repl.VllmServer.__init__(self, client, url, model, native_template=True)
        dsv41_repl.Dsv41ReplServer.__init__(self, client, url, root=root, metadata=metadata, max_images=max_images)
        self.special_ids = None

    tokenize = repl.VllmServer.tokenize

    async def specials(self):
        if self.special_ids is None:
            enc, ids = self.enc, {}
            for name, text in (("bos", enc.bos_token), ("eos", enc.eos_token), ("user", enc.USER_SP_TOKEN),
                               ("assistant", enc.ASSISTANT_SP_TOKEN), ("think", enc.thinking_start_token),
                               ("think_end", enc.thinking_end_token), ("image", enc.IMAGE_PLACEHOLDER),
                               *([("system", enc.SYSTEM_SP_TOKEN)] if hasattr(enc, "SYSTEM_SP_TOKEN") else [])):
                tokens = await self.tokenize(text)
                if len(tokens) != 1:
                    raise ValueError(f"{text!r} is not a single token in the served vocabulary: {tokens}")
                ids[name] = tokens[0]
            if self.metadata is not None and ids["image"] != self.metadata["image_token_id"]:
                raise ValueError("The encoder's image placeholder is not the checkpoint's image_token_id")
            self.special_ids = ids
        return self.special_ids

    async def generate_request(self, payload, images):
        """/inference/v1/generate body: each image's placeholder run collapsed to one placeholder."""
        prompt, placeholder = payload["prompt"], self.placeholder_id
        ids, i, sizes = [], 0, iter(images)
        while i < len(prompt):
            if prompt[i] != placeholder:
                ids.append(prompt[i])
                i += 1
                continue
            image = next(sizes, None)
            if image is None or prompt[i:i + image["tokens"]] != [placeholder] * image["tokens"]:
                raise ValueError("The prompt's image placeholder runs do not match its images")
            ids.append(placeholder)
            i += image["tokens"]
        if next(sizes, None) is not None:
            raise ValueError("The prompt holds fewer image placeholder runs than images")
        # The sampling parameters /v1/completions derives from this payload (vllm_xargs included).
        body = {k: v for k, v in payload.items() if k not in ("prompt", "cache_salt")}
        (rendered,) = await self.post("/v1/completions/render", dict(body, prompt=ids[:1]))
        sampling = dict(rendered["sampling_params"])
        sampling.setdefault("max_tokens", 16)  # /render omits SamplingParams defaults; see VllmVisionServer.
        request = dict(token_ids=ids, sampling_params=sampling, content_parts=[
            dict(type="image_url", url="data:image/png;base64," + im["data"]) for im in images])
        if payload.get("cache_salt") is not None:
            request["cache_salt"] = payload["cache_salt"]
        return request

    async def check_transport(self, images):
        """One-token request with rendered images: the server must run exactly our expanded ids."""
        content = [dict(type="text", text="Transport check.")] + [p for im in images for p in im.content_parts()]
        prompt = await self.render_tokens(dict(messages=[dict(role="user", content=content)], tools=[]))
        if [im["tokens"] for im in prompt.request["images"]] != [im.vision_tokens for im in images]:
            raise ValueError("The renderer and the checkpoint count the images' tokens differently")
        result = await self.complete(dict(prompt=prompt, max_tokens=1, temperature=0.0),
                                     self.root / "vision_transport_check", 300, specials=await self.specials(),
                                     progress={}, thinking=True)
        if result["error"]:
            raise ValueError(f"Vision transport check failed: {result['error']}")
        return dict(prompt_tokens=len(prompt), image_tokens=[im.vision_tokens for im in images])

    async def complete(self, payload, output, timeout, *, specials, progress, thinking, **guard):
        prompt = payload["prompt"]
        images = prompt.request["images"] if isinstance(prompt, ChatPrompt) else []
        generate = await self.generate_request(payload, images) if images else None
        result = await repl.VllmServer.complete(self, dict(payload, prompt=list(prompt)), output, timeout,
                                                specials=specials, progress=progress, thinking=thinking,
                                                generate=generate, **guard)
        usage = result.get("usage") or {}
        # A stream the loop guard closed carries no usage.
        if not result["error"] and usage.get("prompt_tokens", len(prompt)) != len(prompt):
            result["error"] = (f"PromptCountError: vLLM ran {usage.get('prompt_tokens')} prompt tokens for "
                               f"{len(prompt)} counted ids")
            dsv4_llama.write_json(output / "result.json", result)
        if images:
            dsv4_llama.write_json(output / "multimodal_audit.json", dict(
                endpoint="/inference/v1/generate", images=len(images), image_sha256=[im["sha256"] for im in images],
                image_tokens=[im["tokens"] for im in images], counted_prompt_tokens=len(prompt),
                served_prompt_tokens=usage.get("prompt_tokens"),
                cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens")))
        return result


def main():
    """Start only the server, with the pipeline's flags, and leave it running (qwen38_server.py --launch). With
    --supervise N a detached supervisor keeps it up through N restarts, in output_root/server_restart{n}; its status
    is output_root/supervisor.json unless --supervise-status says otherwise."""
    import qwen38_pipeline
    parser = qwen38_pipeline.build_parser()
    qwen38_server.add_supervise_arguments(parser)
    args = parser.parse_args()
    qwen38_pipeline.validate_options(args)
    if args.supervise is None:
        launch(args)
    else:
        def start(n):
            name = f"server_restart{n}" if n else "server"
            return launch(args, name)[0], (args.output_root / name / "server.log").resolve()

        qwen38_server.supervise(args, start, port=args.qwen38_port, model=SERVED_NAME,
                                status_path=args.supervise_status or args.output_root / "supervisor.json")
    print("Base URL:", f"http://127.0.0.1:{args.qwen38_port}/v1", "Model:", SERVED_NAME, flush=True)


if __name__ == "__main__":
    main()
