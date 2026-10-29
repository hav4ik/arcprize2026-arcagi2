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
                   help="Largest FULL CUDA graph batch; defaults to the server batch.")
    g.add_argument("--dsv41-vllm-triton-cache", type=Path, default=None,
                   help="A warmed TRITON_CACHE_DIR; defaults to the environment's, else a fresh temp folder.")


def validate_options(args):
    if args.qwen38_variant != "dsv4-vllm":
        return
    if args.solve_mode != "tir" or getattr(args, "tir_interface", None) != "repl":
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
    if not args.base_url and not (args.dsv41_vllm_model / "config.json").is_file():
        raise ValueError(f"No config.json in --dsv41-vllm-model {args.dsv41_vllm_model}")
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


def command_for(args):
    batch = args.server_max_num_seqs
    compilation = dict(cudagraph_mode="FULL", max_cudagraph_capture_size=args.dsv41_vllm_cudagraph_max or batch)
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(args.dsv41_vllm_model),
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


def launch(args):
    output = (args.output_root / "server").resolve()
    temp_root = args.temp_root.resolve()
    try:
        command = command_for(args)
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
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        raise RuntimeError(str(exc)) from exc
    output.mkdir(parents=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="d41-", dir=temp_root))
    env, overrides = environment(args, temp)
    receipt = dict(status="prepared", created_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   environment_overrides=overrides, variant="dsv4-vllm", served_model_name=SERVED_NAME,
                   model=str(args.dsv41_vllm_model), vision=metadata, input_mode=args.input_mode,
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
    """Start only the server, with the pipeline's flags, and leave it running (qwen38_server.py --launch)."""
    import qwen38_pipeline
    args = qwen38_pipeline.build_parser().parse_args()
    qwen38_pipeline.validate_options(args)
    launch(args)
    print("Base URL:", f"http://127.0.0.1:{args.qwen38_port}/v1", "Model:", SERVED_NAME, flush=True)


if __name__ == "__main__":
    main()
