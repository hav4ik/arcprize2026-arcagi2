"""Opt-in bounded-shard fast loading; no checkpoint or installed-file edits.

Stock fastsafetensors stages a whole shard on a GPU. Keep its distributed
loader for small files, but use vLLM's CPU safetensors iterator for oversized
files. The ordinary model weight loaders still perform TP/EP slicing and
copying. No tensor filtering, casting, quantization or name rewriting occurs.
"""

from functools import wraps
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def partition_shards(files, max_gpu_shard_bytes):
    if max_gpu_shard_bytes <= 0:
        raise ValueError("GPU shard byte limit must be positive")
    small, oversized = [], []
    # Every rank receives the same filenames/sizes. Do not choose paths using
    # rank-local free VRAM: divergent fast-loader collectives would deadlock.
    for filename in files:
        size = Path(filename).stat().st_size
        if size <= 0:
            raise ValueError(f"Empty weight shard: {filename}")
        (small if size <= max_gpu_shard_bytes else oversized).append(filename)
    return small, oversized


def mixed_weights_iterator(files, use_tqdm, max_gpu_shard_bytes, fast_iterator, cpu_iterator):
    small, oversized = partition_shards(files, max_gpu_shard_bytes)
    logger.info("ARC guarded fastsafetensors: %d GPU shards, %d CPU shards; GPU shard cap %.3f GiB",
                len(small), len(oversized), max_gpu_shard_bytes / 1024**3)
    if small:
        yield from fast_iterator(small, use_tqdm)
    if oversized:
        # The native CPU iterator maps one file at a time and yields CPU
        # tensors. It does not stage a whole shard in GPU memory. Do not use
        # an expert filter here: the model's own loaders handle ownership.
        yield from cpu_iterator(oversized, use_tqdm)


def install_guard(max_gpu_shard_bytes):
    """Called inside each owned CUDA worker before its model is constructed."""
    from vllm.model_executor.model_loader import default_loader, weight_utils

    original = weight_utils.fastsafetensors_weights_iterator
    existing = getattr(original, "_arc_max_gpu_shard_bytes", None)
    if existing is not None:
        if existing != max_gpu_shard_bytes or default_loader.fastsafetensors_weights_iterator is not original:
            raise RuntimeError("A different fastsafetensors guard is already installed")
        return
    if default_loader.fastsafetensors_weights_iterator is not original:
        raise RuntimeError("Unknown fastsafetensors iterator override; refusing to replace it")
    if max_gpu_shard_bytes <= 0:
        raise ValueError("GPU shard byte limit must be positive")
    cpu_iterator = weight_utils.safetensors_weights_iterator

    @wraps(original)
    def guarded(files, use_tqdm_on_load):
        yield from mixed_weights_iterator(files, use_tqdm_on_load, max_gpu_shard_bytes,
                                          original, cpu_iterator)

    guarded._arc_max_gpu_shard_bytes = max_gpu_shard_bytes
    weight_utils.fastsafetensors_weights_iterator = guarded
    default_loader.fastsafetensors_weights_iterator = guarded
