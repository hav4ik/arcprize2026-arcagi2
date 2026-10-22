"""Fast checkpoint reader and fail-fast Pathways C++ loader for DeepSeek-V4.1-Flash-1017.

Design invariants:
  * Zero silent fallbacks: any missing shard, unexpected dtype/shape, or broken
    Pathways persistence configuration raises immediately.
  * Transient CNS / TFHub read timeouts use bounded exponential-backoff retries
    over persistent gfile handles with f.readinto into preallocated memoryviews.
"""

from __future__ import annotations

# pylint: disable=g-bare-generic,missing-function-docstring,g-complex-comprehension,g-doc-args,g-doc-return-or-yield,g-doc-exception

from collections.abc import Iterable, Mapping, Sequence
from concurrent import futures
import dataclasses
import datetime
import importlib
import json
import os
import struct
import time
from typing import Any

from absl import logging
import jax
import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_layout as layout_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_ref as ref
from google3.pyglib import gfile

DEFAULT_CHECKPOINT = "/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1"

_DTYPES = {
    "BF16": ref.BF16,
    "F32": ref.F32,
    "F8_E4M3": ref.FP8,
    "F8_E8M0": np.dtype(np.uint8),
    "I8": np.dtype(np.uint8),
    "U8": np.dtype(np.uint8),
    "U32": np.dtype(np.uint32),
}
_READ_DEADLINE_S = 60.0
_READ_SLICE = 32 << 20
_MAX_ATTEMPTS = 8


@dataclasses.dataclass(frozen=True)
class TensorEntry:
  shard: str
  dtype: np.dtype
  shape: tuple[int, ...]
  start: int
  end: int


class SafetensorsReader:
  """Parallel byte-range reader for a HuggingFace safetensors directory."""

  def __init__(self, path: str, num_threads: int = 16):
    self.path = path
    self.num_threads = num_threads
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not gfile.Exists(index_path):
      raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with gfile.Open(index_path, "r") as f:
      weight_map: dict[str, str] = json.load(f)["weight_map"]
    self._tensors: dict[str, TensorEntry] = {}
    for shard in sorted(set(weight_map.values())):
      header, base = self._read_header(shard)
      for raw_name, entry in header.items():
        if raw_name == "__metadata__":
          continue
        name = raw_name.removeprefix("model.")
        dtype_str = entry["dtype"]
        if dtype_str not in _DTYPES:
          raise ValueError(
              f"Unsupported safetensors dtype {dtype_str!r} for {name}"
          )
        a, b = entry["data_offsets"]
        self._tensors[name] = TensorEntry(
            shard=shard,
            dtype=_DTYPES[dtype_str],
            shape=tuple(int(d) for d in entry["shape"]),
            start=base + int(a),
            end=base + int(b),
        )
    expected = {k.removeprefix("model.") for k in weight_map}
    missing = expected - set(self._tensors)
    if missing:
      raise ValueError(
          f"{len(missing)} tensors listed in index.json missing from shards: "
          f"{sorted(missing)[:5]}"
      )

  def _read_header(self, shard: str) -> tuple[dict[str, Any], int]:
    shard_path = os.path.join(self.path, shard)
    with gfile.Open(shard_path, "rb", deadline=120.0) as f:
      raw_len = f.read(8, deadline=120.0)
      if len(raw_len) != 8:
        raise IOError(f"Short header length read from {shard_path}")
      (n,) = struct.unpack("<Q", raw_len)
      hdr_bytes = f.read(n, deadline=120.0)
      if len(hdr_bytes) != n:
        raise IOError(f"Short header read from {shard_path}")
      return json.loads(hdr_bytes), 8 + n

  def __contains__(self, name: str) -> bool:
    return name.removeprefix("model.") in self._tensors

  def names(self) -> list[str]:
    return sorted(self._tensors)

  def entry(self, name: str) -> TensorEntry:
    key = name.removeprefix("model.")
    if key not in self._tensors:
      raise KeyError(f"Tensor {name!r} not found in {self.path}")
    return self._tensors[key]

  def read_many(self, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Reads requested tensor names in parallel using persistent gfile handles."""
    norm_names = sorted(
        {n.removeprefix("model.") for n in names},
        key=lambda n: (self.entry(n).shard, self.entry(n).start),
    )
    out = {
        n: np.empty(self._tensors[n].shape, dtype=self._tensors[n].dtype)
        for n in norm_names
    }
    jobs: list[tuple[str, int, memoryview]] = []
    for n in norm_names:
      t = self._tensors[n]
      view = memoryview(out[n].reshape(-1).view(np.uint8))
      if len(view) != t.end - t.start:
        raise IOError(
            f"{n}: header byte span {t.end - t.start} != array bytes"
            f" {len(view)}"
        )
      for off in range(0, t.end - t.start, _READ_SLICE):
        stop = min(off + _READ_SLICE, t.end - t.start)
        jobs.append((t.shard, t.start + off, view[off:stop]))

    if not jobs:
      return out
    workers = max(1, min(self.num_threads, len(jobs)))

    def _run(worker_id: int) -> None:
      handles: dict[str, Any] = {}
      try:
        for shard, pos, dst in jobs[worker_id::workers]:
          self._read_into(handles, shard, pos, dst)
      finally:
        for f in handles.values():
          try:
            f.close()
          except Exception:  # pylint: disable=broad-exception-caught
            pass

    with futures.ThreadPoolExecutor(workers) as pool:
      list(pool.map(_run, range(workers)))
    return out

  def _read_into(
      self, handles: dict[str, Any], shard: str, pos: int, dst: memoryview
  ) -> None:
    got = 0
    for attempt in range(_MAX_ATTEMPTS):
      try:
        f = handles.get(shard)
        if f is None:
          f = gfile.Open(
              os.path.join(self.path, shard), "rb", deadline=_READ_DEADLINE_S
          )
          handles[shard] = f
        while got < len(dst):
          f.seek(pos + got)
          with gfile._TemporaryDeadline(deadline_length=_READ_DEADLINE_S):  # pylint: disable=protected-access
            n = f.readinto(dst[got:])
          if not n:
            raise IOError(f"{shard}: unexpected EOF at offset {pos + got}")
          got += n
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        f = handles.pop(shard, None)
        if f is not None:
          try:
            f.close()
          except Exception:  # pylint: disable=broad-exception-caught
            pass
        if attempt + 1 == _MAX_ATTEMPTS:
          raise
        logging.warning(
            "Transient read error on %s @%d (attempt %d/%d): %s",
            shard,
            pos + got,
            attempt + 1,
            _MAX_ATTEMPTS,
            e,
        )
        time.sleep(min(20.0, 1.5**attempt))


def write_safetensors_dir(
    out_dir: str,
    tensors: Mapping[str, np.ndarray],
    config_dict: Mapping[str, Any] | None = None,
) -> None:
  """Writes a single-shard safetensors directory with index.json."""
  os.makedirs(out_dir, exist_ok=True)
  dtype_rev = {
      np.dtype(ref.BF16): "BF16",
      np.dtype(ref.F32): "F32",
      np.dtype(ref.FP8): "F8_E4M3",
      np.dtype(np.uint8): "F8_E8M0",
      np.dtype(np.int8): "I8",
  }
  shard_name = "model-00001-of-00001.safetensors"
  header: dict[str, Any] = {}
  weight_map: dict[str, str] = {}
  offset = 0
  for name in sorted(tensors):
    arr = np.ascontiguousarray(tensors[name])
    if (
        name.endswith(".weight")
        and ".experts." in name
        and arr.dtype == np.uint8
    ):
      dt_str = "I8"
    else:
      dt_str = dtype_rev[arr.dtype]
    header[name] = {
        "dtype": dt_str,
        "shape": list(arr.shape),
        "data_offsets": [offset, offset + arr.nbytes],
    }
    weight_map[name] = shard_name
    offset += arr.nbytes

  blob = json.dumps(header).encode("utf-8")
  blob += b" " * ((-len(blob)) % 8)
  shard_path = os.path.join(out_dir, shard_name)
  with open(shard_path, "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    for name in sorted(tensors):
      f.write(np.ascontiguousarray(tensors[name]).tobytes())

  index_data = {
      "metadata": {"total_size": offset},
      "weight_map": weight_map,
  }
  with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index_data, f, indent=2)
  if config_dict is not None:
    with open(os.path.join(out_dir, "config.json"), "w") as f:
      json.dump(dict(config_dict), f, indent=2)


def load_device_arrays_from_reader(
    reader: SafetensorsReader,
    lay: layout_lib.DeviceLayout,
    device_ids: Sequence[int],
) -> dict[int, dict[str, np.ndarray]]:
  """Reads only the tensors required by device_ids and slices per-device arrays."""
  wanted_bb_experts = {
      e for d in device_ids for e in lay.backbone_experts_of(d)
  }
  wanted_mtp_experts = {e for d in device_ids for e in lay.dspark_experts_of(d)}
  keep_names = []
  for name in reader.names():
    if (
        name.startswith("vision.")
        or name.startswith("aligner.")
        or name.startswith("image_")
    ):
      continue
    if ".ffn.experts." in name:
      parts = name.split(".")
      exp_idx = int(parts[parts.index("experts") + 1])
      if name.startswith("layers.") and exp_idx not in wanted_bb_experts:
        continue
      if name.startswith("mtp.") and exp_idx not in wanted_mtp_experts:
        continue
    keep_names.append(name)

  raw = reader.read_many(keep_names)
  return {d: layout_lib.build_device_arrays(raw, lay, d) for d in device_ids}


def load_exported_params(
    export_params_dir: str,
    abstract_params: Any,
    shardings: Any,
) -> Any:
  """Fail-fast Pathways C++ restore of pre-sliced OCDBT leaves.

  Raises RuntimeError immediately on any missing path, leaf mismatch, or
  unavailable Pathways persistence backend (no Python deserialization fallback).
  """
  if not export_params_dir:
    raise ValueError("export_params_dir must be non-empty")
  if not gfile.Exists(export_params_dir):
    raise FileNotFoundError(
        f"Exported params directory does not exist: {export_params_dir}"
    )
  leaves, treedef = jax.tree.flatten(abstract_params)
  shard_leaves = treedef.flatten_up_to(shardings)
  if len(leaves) != len(shard_leaves):
    raise ValueError(
        f"Leaf count {len(leaves)} != sharding count {len(shard_leaves)}"
    )

  t0 = time.perf_counter()
  pw_persistence = importlib.import_module(
      "google3.third_party.pathways.jax.experimental.persistence"
  )
  persistence_pb2 = importlib.import_module(
      "google3.third_party.pathways.persistence.persistence_pb2"
  )
  xla_client = importlib.import_module(
      "google3.third_party.tensorflow.compiler.xla.python.xla_client"
  )
  location = pw_persistence.PersistedLocation(
      backend_type=(
          persistence_pb2.StorageBackendType.BACKEND_TENSORSTORE_CUSTOM_SPEC
      ),
      base_path=export_params_dir,
  )
  persisted = []
  for i, x in enumerate(leaves):
    buf = persistence_pb2.PersistedShardedBufferProto()
    shape = xla_client.Shape.array_shape(
        xla_client.PrimitiveType(xla_client.dtype_to_etype(x.dtype)),
        x.shape,
    ).with_major_to_minor_layout_if_absent()
    buf.shape.ParseFromString(shape.to_serialized_proto())
    buf.name = json.dumps({
        "driver": "zarr",
        "kvstore": {
            "driver": "ocdbt",
            "base": {"driver": "gfile", "path": ""},
            "path": str(i),
        },
    })
    persisted.append(
        location.read(buf, timeout=datetime.timedelta(seconds=600))
    )
  arrays = pw_persistence.read_persisted_arrays(
      persisted, shardings=shard_leaves
  )
  logging.info(
      "Restored %d exported leaves via Pathways C++ workers from %s in %.1fs",
      len(arrays),
      export_params_dir,
      time.perf_counter() - t0,
  )
  return jax.tree.unflatten(treedef, arrays)
