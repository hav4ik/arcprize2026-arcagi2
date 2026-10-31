r"""Header-only parameter census of a DeepSeek-V4.1 E128 2-bit checkpoint on CNS.

Reads only the safetensors headers of the backbone shards, every `gptq/` layer
file, and the Engram shards, then aggregates logical parameter counts per block.
Packed 2-bit tensors (`int32` `q`/`z`) are counted as 16 parameters per word.

Example:
  blaze run //experimental/users/arazdai/inspect_tools:safetensors_param_census -- \
    --model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit
"""

import collections
from collections.abc import Sequence
from concurrent import futures
import json
import re
import struct

from absl import app
from absl import flags

from google3.pyglib import gfile

_MODEL_DIR = flags.DEFINE_string("model_dir", None, "Checkpoint dir.", required=True)
_SHOW_TEMPLATES = flags.DEFINE_bool(
    "show_templates", True, "List distinct tensor-name templates (layer index -> N)."
)

_PACKED_U2 = (".q", ".z")


def _read_header(path: str) -> dict[str, dict]:
  with gfile.Open(path, "rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))
  header.pop("__metadata__", None)
  return header


def _numel(shape: list[int]) -> int:
  n = 1
  for d in shape:
    n *= d
  return n


def _logical(name: str, dtype: str, shape: list[int]) -> int:
  n = _numel(shape)
  if dtype == "I32" and name.endswith(_PACKED_U2):
    return n * 16
  return n


def _template(name: str) -> str:
  t = re.sub(r"^layers\.\d+\.", "layers.N.", name)
  t = re.sub(r"^e\d+\.", "e*.", t)
  t = re.sub(r"\.experts\.\d+\.", ".experts.*.", t)
  return t


def _block(name: str) -> str:
  """Coarse block label for a tensor name."""
  if re.match(r"^e\d+\.w[123]\.q$", name):
    return "routed_experts: 2-bit codes q"
  if re.match(r"^e\d+\.w[123]\.s$", name):
    return "routed_experts: group scales s (BF16)"
  if re.match(r"^e\d+\.w[123]\.z$", name):
    return "routed_experts: zero points z (2-bit)"
  if "shared_experts" in name:
    return "shared_experts (FP8 + UE8M0 scale)" if name.endswith((".weight", ".scale")) else "shared_experts: other"
  if ".ffn.gate." in name:
    return "router gate"
  if "engram" in name:
    return "engram (layers 1 & 14)"
  if ".attn." in name:
    return "attention (MLA/SWA/compressor/indexer)"
  if name.startswith("embed"):
    return "token embeddings"
  if name.startswith("head"):
    return "lm head"
  if "norm" in name:
    return "norms"
  return "other (mHC / misc)"


def main(argv: Sequence[str]) -> None:
  del argv
  root = _MODEL_DIR.value.rstrip("/")
  entries = sorted(gfile.ListDir(root))
  backbone = [e for e in entries if e.startswith("model-") and e.endswith(".safetensors")]
  gptq = sorted(gfile.ListDir(f"{root}/gptq")) if gfile.Exists(f"{root}/gptq") else []
  paths = [f"{root}/{e}" for e in backbone] + [f"{root}/gptq/{e}" for e in gptq if e.endswith(".safetensors")]
  print(f"Reading {len(paths)} headers ({len(backbone)} backbone/engram shards, {len(gptq)} gptq files)...")

  with futures.ThreadPoolExecutor(max_workers=16) as pool:
    headers = list(pool.map(_read_header, paths))

  by_block: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
  by_tmpl: dict[tuple[str, str, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0])
  dtypes_by_block: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
  for path, header in zip(paths, headers):
    is_gptq = "/gptq/" in path
    for name, meta in header.items():
      n = _logical(name, meta["dtype"], meta["shape"])
      blk = _block(name)
      by_block[blk][0] += n
      by_block[blk][1] += 1
      dtypes_by_block[blk][meta["dtype"]] += 1
      key = (("gptq/" if is_gptq else "") + _template(name), meta["dtype"], str(meta["shape"]))
      by_tmpl[key][0] += n
      by_tmpl[key][1] += 1

  print(f"\n{'block':44s} {'tensors':>8s} {'params':>18s} {'share':>8s}  dtypes")
  total = sum(v[0] for v in by_block.values())
  for blk, (n, t) in sorted(by_block.items(), key=lambda kv: -kv[1][0]):
    dts = ", ".join(f"{d}x{c}" for d, c in dtypes_by_block[blk].most_common())
    print(f"{blk:44s} {t:>8,d} {n:>18,d} {100.0 * n / total:>7.3f}%  {dts}")
  print(f"{'TOTAL':44s} {sum(v[1] for v in by_block.values()):>8,d} {total:>18,d}")

  if _SHOW_TEMPLATES.value:
    print(f"\n{'tensor template':64s} {'dtype':8s} {'shape':22s} {'count':>6s} {'params':>16s}")
    for (tmpl, dt, shp), (n, c, _) in sorted(by_tmpl.items(), key=lambda kv: kv[0][0]):
      print(f"{tmpl:64s} {dt:8s} {shp:22s} {c:>6,d} {n:>16,d}")


if __name__ == "__main__":
  app.run(main)
