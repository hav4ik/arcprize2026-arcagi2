"""Image-span bookkeeping and preprocessing for DeepSeek-V4.1 vision (exact port, no torch).

Mirrors `inference/image_processor.py` of the released model so that training, data
preparation, and evaluation agree bit-for-bit on:

  * how an image of (width, height) pixels is resized/padded and tiled into 14x14 ViT patches,
  * how many LLM positions it occupies after the 3x3 aligner downsample, and
  * the layout of those positions:

        [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every position of a span carries `IMAGE_TOKEN_ID` (`<｜deepseek_image｜>`) in `input_ids`;
`token_types` (int8) distinguishes them (`TEXT = -1` elsewhere).
"""

from __future__ import annotations

import base64
import dataclasses
import io
import math
import struct
from typing import Sequence

import numpy as np
from PIL import Image
from PIL import ImageOps

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_TOKEN_ID = 129264

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = 0, 1, 2, 3

# DeepSeek-V4.1-Flash vision defaults (`config.json: vision_config` + `inference/model.py`).
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 3
MAX_N_TOKEN = 1024
MIN_PIXELS = 544 * 544
MAX_WH_RATIO: int | None = None
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclasses.dataclass(frozen=True)
class ImagePlan:
  """Resize plan for one image."""

  width: int  # original pixels
  height: int
  best_width: int  # resized/padded pixels (multiples of PATCH_SIZE)
  best_height: int
  n_vit_h: int  # ViT patch grid
  n_vit_w: int
  n_llm_h: int  # LLM token grid after the aligner downsample
  n_llm_w: int

  @property
  def num_tokens(self) -> int:
    return num_image_tokens(self.n_llm_h, self.n_llm_w)

  @property
  def num_patches(self) -> int:
    return self.n_vit_h * self.n_vit_w


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
  return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int) -> tuple[int, int]:
  return (
      math.ceil((best_height // PATCH_SIZE) / DOWNSAMPLE_RATIO),
      math.ceil((best_width // PATCH_SIZE) / DOWNSAMPLE_RATIO),
  )


def _solve_resize_ratio(height: int, width: int) -> tuple[int, int]:
  r = height / width
  max_w_float = math.sqrt((MAX_N_TOKEN - 2) / r + 0.25) - 0.5
  max_h_float = max_w_float * r
  cell = PATCH_SIZE * DOWNSAMPLE_RATIO
  if max_w_float < 1.0:
    return (MAX_N_TOKEN - 2) // 2 * cell, cell
  if max_h_float < 1.0:
    return cell, (MAX_N_TOKEN - 3) * cell
  beta = min(
      math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height
  )
  return (
      math.floor(height * beta / PATCH_SIZE) * PATCH_SIZE,
      math.floor(width * beta / PATCH_SIZE) * PATCH_SIZE,
  )


def plan_image_grid(width: int, height: int) -> ImagePlan:
  """Pure function of the original pixel size (== `image_processor.plan_image_grid`)."""
  w, h = width, height
  if MAX_WH_RATIO is not None and w > h * MAX_WH_RATIO:
    w = h * MAX_WH_RATIO
  if 0 < w * h < MIN_PIXELS:
    ratio = (MIN_PIXELS / (w * h)) ** 0.5
    w = int(w * ratio)
    h = int(h * ratio)
  best_w = math.ceil(w / PATCH_SIZE) * PATCH_SIZE
  best_h = math.ceil(h / PATCH_SIZE) * PATCH_SIZE
  n_h, n_w = llm_grid(best_h, best_w)
  if num_image_tokens(n_h, n_w) > MAX_N_TOKEN:
    best_h, best_w = _solve_resize_ratio(h, w)
    n_h, n_w = llm_grid(best_h, best_w)
    assert num_image_tokens(n_h, n_w) <= MAX_N_TOKEN
  return ImagePlan(
      width=width,
      height=height,
      best_width=best_w,
      best_height=best_h,
      n_vit_h=best_h // PATCH_SIZE,
      n_vit_w=best_w // PATCH_SIZE,
      n_llm_h=n_h,
      n_llm_w=n_w,
  )


def image_token_types(n_llm_h: int, n_llm_w: int) -> np.ndarray:
  types = [IMAGE_START]
  types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
  types.append(IMAGE_END)
  return np.asarray(types, dtype=np.int8)


def decode_data_url(url: str) -> bytes:
  """Decodes an OpenAI-style `data:image/png;base64,...` URL into raw bytes."""
  header, _, payload = url.partition(",")
  if not header.startswith("data:") or ";base64" not in header:
    raise ValueError(f"Unsupported image URL header: {header[:40]!r}")
  return base64.b64decode(payload)


def png_size(data: bytes) -> tuple[int, int]:
  """(width, height) from a PNG header without decoding; falls back to PIL otherwise."""
  if data[:8] == PNG_MAGIC and data[12:16] == b"IHDR":
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)
  with Image.open(io.BytesIO(data)) as im:
    return int(im.width), int(im.height)


def plan_from_bytes(data: bytes) -> ImagePlan:
  w, h = png_size(data)
  return plan_image_grid(w, h)


def preprocess_image(data: bytes) -> tuple[np.ndarray, ImagePlan]:
  """Decodes + pads/resizes an image and tiles it into ViT patches.

  Returns `(patches float32 [n_vit_h * n_vit_w, 3, PATCH, PATCH] normalised to [-1, 1], plan)`,
  matching `image_processor.load_image` (which then casts to bfloat16).
  """
  with Image.open(io.BytesIO(data)) as source:
    image = source.convert("RGB")
  plan = plan_image_grid(image.width, image.height)
  if MAX_WH_RATIO is not None and image.width >= MAX_WH_RATIO * image.height:
    image = image.resize((plan.best_width, plan.best_height))
  else:
    image = ImageOps.pad(image, (plan.best_width, plan.best_height), color=(127, 127, 127))
  x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
  x = (x - 0.5) / 0.5
  p = PATCH_SIZE
  patches = (
      x.reshape(3, plan.n_vit_h, p, plan.n_vit_w, p)
      .transpose(1, 3, 0, 2, 4)
      .reshape(plan.n_vit_h * plan.n_vit_w, 3, p, p)
  )
  return np.ascontiguousarray(patches), plan


def expand_image_placeholders(
    ids: Sequence[int],
    image_blobs: Sequence[bytes],
    image_token_id: int = IMAGE_TOKEN_ID,
    per_token_arrays: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[ImagePlan]]:
  """Expands each `image_token_id` placeholder in `ids` into its full image span.

  Args:
    ids: token ids with exactly one `image_token_id` per image.
    image_blobs: encoded images in prompt order.
    image_token_id: placeholder id (also used for every span position).
    per_token_arrays: optional 1-D arrays aligned with `ids` (e.g. `loss_mask`, `loss_weights`)
      that are expanded alongside; span positions receive 0.

  Returns:
    `(new_ids int32, token_types int8, expanded_arrays, plans)`.
  """
  ids = np.asarray(ids, dtype=np.int32)
  pos = np.flatnonzero(ids == image_token_id)
  if len(pos) != len(image_blobs):
    raise ValueError(
        f"Found {len(pos)} image placeholders but {len(image_blobs)} images."
    )
  extra = [np.asarray(a) for a in (per_token_arrays or [])]
  for a in extra:
    if a.shape != ids.shape:
      raise ValueError(f"per_token_array shape {a.shape} != ids shape {ids.shape}")
  out_ids: list[np.ndarray] = []
  out_types: list[np.ndarray] = []
  out_extra: list[list[np.ndarray]] = [[] for _ in extra]
  plans: list[ImagePlan] = []
  prev = 0
  for p_idx, blob in zip(pos.tolist(), image_blobs):
    out_ids.append(ids[prev:p_idx])
    out_types.append(np.full(p_idx - prev, TEXT, dtype=np.int8))
    for k, a in enumerate(extra):
      out_extra[k].append(a[prev:p_idx])
    plan = plan_from_bytes(blob)
    plans.append(plan)
    types = image_token_types(plan.n_llm_h, plan.n_llm_w)
    out_ids.append(np.full(types.shape[0], image_token_id, dtype=np.int32))
    out_types.append(types)
    for k, a in enumerate(extra):
      out_extra[k].append(np.zeros(types.shape[0], dtype=a.dtype))
    prev = p_idx + 1
  out_ids.append(ids[prev:])
  out_types.append(np.full(len(ids) - prev, TEXT, dtype=np.int8))
  for k, a in enumerate(extra):
    out_extra[k].append(a[prev:])
  return (
      np.concatenate(out_ids).astype(np.int32),
      np.concatenate(out_types).astype(np.int8),
      [np.concatenate(parts) for parts in out_extra],
      plans,
  )


def image_spans(token_types: np.ndarray) -> list[tuple[int, int]]:
  """Returns `[(start, end))` of each image span in `token_types` (START..END inclusive)."""
  starts = np.flatnonzero(token_types == IMAGE_START).tolist()
  ends = np.flatnonzero(token_types == IMAGE_END).tolist()
  if len(starts) != len(ends):
    raise ValueError(f"Unbalanced image spans: {len(starts)} starts, {len(ends)} ends")
  return [(s, e + 1) for s, e in zip(starts, ends)]
