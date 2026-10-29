"""ARC Qwen3 axial rotary bridge; helpers are testable without importing vLLM."""

from functools import wraps

import torch

from arc_rope_3d import ArcRotaryEmbedding3D
from arc_rope_3d_positions import ArcRequestPositionCache, scheduled_positions
from pre_sft_rope_3d import CONFIG_KEY, read_rope_3d_config, rope_3d_config


def configure_vllm_rope_3d(model_path, options, engine_kwargs):
    """Select the offline worker extension only for saved 3D checkpoints."""
    config = read_rope_3d_config(model_path)
    if config is None:
        return False
    if options.get("bidirectional_prompt_attention", False):
        raise ValueError("vLLM 3D RoPE + bidirectional prompt attention is not supported yet")
    if options.get("vllm_async_scheduling", False):
        raise ValueError("vLLM 3D RoPE requires synchronous scheduling")
    if int(options.get("vllm_speculative_ngram_tokens", 0) or 0):
        raise ValueError("vLLM 3D RoPE does not yet support speculative draft positions")
    if int(engine_kwargs.get("tensor_parallel_size", 1)) != 1:
        raise ValueError("vLLM ARC 3D RoPE currently requires TP=1")
    overrides = dict(engine_kwargs.get("hf_overrides") or {})
    overrides[CONFIG_KEY] = config
    engine_kwargs.update(
        hf_overrides=overrides,
        worker_cls="arc_vllm_rope_3d_worker.ArcRope3DWorker",
        async_scheduling=False,
    )
    return True


def runner_positions(model_runner, num_reqs=None):
    """Slice each authoritative CPU token prefix at the runner's KV offset."""
    batch = model_runner.input_batch
    count = int(batch.num_reqs if num_reqs is None else num_reqs)
    if count == 0:
        return None  # explicitly marked dummy profiling/warmup, not real text
    boundaries = model_runner.query_start_loc.np[:count + 1]
    if len(boundaries) != count + 1 or int(boundaries[0]) != 0:
        raise RuntimeError("Unsupported vLLM 3D RoPE query layout")
    use_cache = getattr(model_runner, "arc_rope_position_cache_enabled", True)
    cache = getattr(model_runner, "arc_rope_position_cache", None)
    if use_cache and cache is None:
        cache = model_runner.arc_rope_position_cache = ArcRequestPositionCache()
    if cache is not None:
        # Only scheduled requests are retained: memory cannot grow with the
        # number of completed DFS candidates or queued requests.
        cache.retain(batch.req_ids[:count] if use_cache else ())
    prefixes, starts, lengths, xyz = [], [], [], []
    for row, request_id in enumerate(batch.req_ids[:count]):
        request = model_runner.requests[request_id]
        if request.prompt_token_ids is None:
            raise RuntimeError("ARC 3D RoPE cannot derive coordinates from prompt embeddings")
        start = int(batch.num_computed_tokens_cpu[row])
        length = int(boundaries[row + 1]) - int(boundaries[row])
        if use_cache:
            xyz.extend(cache.positions(request_id, request.prompt_token_ids,
                                       request.output_token_ids, start, length))
        else:
            prefixes.append(list(request.prompt_token_ids) + list(request.output_token_ids))
            starts.append(start)
            lengths.append(length)
    if not use_cache:
        xyz = scheduled_positions(prefixes, starts, lengths)
    return torch.tensor(xyz, dtype=torch.long, device=model_runner.device)


class VllmArcRotary3D(torch.nn.Module):
    """Pure-tensor rotary; xyz uses the existing model positions argument.

    Qwen3's vLLM compile contract already supports positions [3, tokens]. No
    Python request context is accessed inside either compiled or captured code.
    """

    def __init__(self, head_dim, base):
        super().__init__()
        self.head_dim = head_dim
        self.base = float(base)
        pairs = head_dim // 2
        self.axis_pairs = (pairs // 3, pairs // 3, pairs - 2 * (pairs // 3))

    def forward(self, positions, query, key, offsets=None):
        if offsets is not None or positions.ndim != 2 or positions.shape[0] != 3:
            raise RuntimeError("vLLM 3D RoPE requires the ARC coordinate-aware runner")
        if query.shape[0] != positions.shape[1]:
            raise ValueError("vLLM 3D RoPE token/table shapes disagree")
        # Frequency constants stay FP32 even if model.to(bfloat16) is called.
        # No mutable table buffer, .item(), or data-dependent Python control.
        with torch.autocast(device_type=query.device.type, enabled=False):
            angles = torch.cat([
                positions[axis, :, None].float() * self.base ** (
                    -torch.arange(count, device=query.device, dtype=torch.float32) / count
                ) for axis, count in enumerate(self.axis_pairs)
            ], -1)
            angles = torch.cat((angles, angles), -1)
            cos, sin = angles.cos(), angles.sin()

        def rotate(tensor):
            shaped = tensor.reshape(-1, tensor.shape[-1] // self.head_dim, self.head_dim)
            c = cos.to(dtype=tensor.dtype).unsqueeze(1)
            s = sin.to(dtype=tensor.dtype).unsqueeze(1)
            half = self.head_dim // 2
            rotated = torch.cat((-shaped[..., half:], shaped[..., :half]), dim=-1)
            return (shaped * c + rotated * s).reshape_as(tensor)

        return rotate(query), rotate(key)


def install_vllm_rope_3d(model, config):
    """Replace only rotary operations; preserve QKNorm, LoRA and paged attention.

    Works with ordinary Qwen3 and the existing Delta-MH Qwen3 plugin. Install
    on the unwrapped model BEFORE the first graph capture/compile execution.
    """
    if config != rope_3d_config(config["base"]):
        raise ValueError("Unsupported vLLM ARC 3D RoPE checkpoint contract")
    if model.config.model_type != "qwen3" or model.config.vocab_size != 16:
        raise ValueError("vLLM ARC 3D RoPE requires dense Qwen3 with ARC16 vocabulary")
    if getattr(model, "_arc_vllm_rope_3d", None) is not None:
        raise RuntimeError("vLLM 3D RoPE is already installed")
    head_dim = model.config.head_dim
    ArcRotaryEmbedding3D(head_dim, base=config["base"])  # Validate partition.
    layers = model.model.layers
    if len(layers) != model.config.num_hidden_layers:
        raise ValueError("vLLM ARC 3D RoPE requires the complete Qwen3 layer stack (PP=1)")
    for layer in layers:
        layer.self_attn.rotary_emb = VllmArcRotary3D(head_dim, config["base"])
    original = model.forward

    @wraps(original)
    def forward(input_ids, positions, *args, arc_rope_positions=None, **kwargs):
        if arc_rope_positions is None:
            raise RuntimeError("Real vLLM 3D RoPE request is missing coordinates")
        # The runner always passes ONE persistent full-capacity buffer. Only
        # this view's token dimension varies, as in vLLM's native positions.
        xyz = arc_rope_positions
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < positions.numel():
            raise ValueError("vLLM 3D RoPE buffer is shorter than the physical batch")
        return original(input_ids, xyz[:positions.numel()].T, *args, **kwargs)

    model.forward = forward
    model._arc_vllm_rope_3d = dict(config)


def refresh_position_buffer(buffer, xyz):
    """Update stable CUDA-graph input storage outside capture/replay.

    None denotes an explicitly scheduled dummy/profile call. All unused tail
    entries are neutral, including padding added by the CUDA-graph dispatcher.
    """
    if xyz is not None and (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] > buffer.shape[0]):
        raise ValueError("3D RoPE scheduled coordinates exceed the persistent buffer")
    buffer.zero_()
    if xyz is not None:
        buffer[:xyz.shape[0]].copy_(xyz)
    return buffer
