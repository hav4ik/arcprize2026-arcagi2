r"""Inspects E065 (text+image) rollout parquet files: schema, SFT filters, image layout and token budget.

Example:
  blaze run //experimental/users/arazdai/kaggle/sft:inspect_e065_rollouts -- \
    --rollouts_dir=/cns/lu-d/home/arazdai/kaggle/arc_sft_data/livobereshka_e065_rollouts/rollouts \
    --files=e065_images_arc2_train.parquet
"""

from __future__ import annotations

import base64
import collections
import io
import json
import math
import os
import struct

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from google3.pyglib import gfile

FLAGS = flags.FLAGS
flags.DEFINE_string("rollouts_dir", None, "CNS dir with parquet files.", required=True)
flags.DEFINE_list("files", None, "Parquet basenames.", required=True)
flags.DEFINE_integer("max_rows", 0, "Row cap per file (0 = all).")
flags.DEFINE_integer("show_examples", 1, "How many message layouts to print.")

# DeepSeek-V4.1 vision defaults (inference/model.py ModelArgs + config.json vision_config).
PATCH = 14
DOWNSAMPLE = 3
MIN_PIXELS = 544 * 544
MAX_N_TOKEN = 1024


def _png_size(data: bytes) -> tuple[int, int]:
  if data[:8] != b"\x89PNG\r\n\x1a\n":
    return (-1, -1)
  w, h = struct.unpack(">II", data[16:24])
  return int(w), int(h)


def _llm_grid(bh: int, bw: int) -> tuple[int, int]:
  return math.ceil((bh // PATCH) / DOWNSAMPLE), math.ceil((bw // PATCH) / DOWNSAMPLE)


def _num_tokens(n_h: int, n_w: int) -> int:
  return n_h * (n_w + 1) + 2


def _plan(width: int, height: int) -> tuple[int, int, int, int]:
  """Mirror of image_processor.plan_image_grid (+ safe_resize) for max_wh_ratio=None."""
  if 0 < width * height < MIN_PIXELS:
    ratio = (MIN_PIXELS / (width * height)) ** 0.5
    width = int(width * ratio)
    height = int(height * ratio)
  bw = math.ceil(width / PATCH) * PATCH
  bh = math.ceil(height / PATCH) * PATCH
  n_h, n_w = _llm_grid(bh, bw)
  if _num_tokens(n_h, n_w) > MAX_N_TOKEN:
    r = height / width
    max_w = math.sqrt((MAX_N_TOKEN - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = PATCH * DOWNSAMPLE
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    bh = math.floor(height * beta / PATCH) * PATCH
    bw = math.floor(width * beta / PATCH) * PATCH
    n_h, n_w = _llm_grid(bh, bw)
  return n_h, n_w, bh, bw


def _iter_images(messages: list[dict]):
  for mi, m in enumerate(messages):
    c = m.get("content")
    if isinstance(c, list):
      for pi, part in enumerate(c):
        if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
          url = (part.get("image_url") or {}).get("url", "") if part.get("type") == "image_url" else ""
          yield mi, m.get("role"), pi, url


def main(argv):
  del argv
  for fn in FLAGS.files:
    path = os.path.join(FLAGS.rollouts_dir, fn)
    with gfile.Open(path, "rb") as f:
      df = pd.read_parquet(io.BytesIO(f.read()))
    if FLAGS.max_rows:
      df = df.head(FLAGS.max_rows)
    print(f"\n===== {fn}: {len(df)} rows; columns: {list(df.columns)}")
    for col in ("batch", "correct", "train_pass", "answer_source", "stop_reason", "harness", "model"):
      if col in df:
        print(f"  {col}: {dict(collections.Counter(df[col].astype(str)).most_common(8))}")
    err = df["error"].notna() & (df["error"].astype(str).str.len() > 0) if "error" in df else pd.Series(False, index=df.index)
    print(f"  rows with error: {int(err.sum())}")
    p0 = df[df["pair"] == 0] if "pair" in df else df
    ok = p0[(p0["correct"] == True) & (p0["train_pass"] == True) & (p0.get("answer_source", "submit") == "submit") & ~err.loc[p0.index]]  # noqa: E712
    print(f"  pair==0 rows: {len(p0)}; 3-gate verified (correct & train_pass & submit & !error): {len(ok)} "
          f"({100 * len(ok) / max(1, len(p0)):.1f}%)")
    if "completion_tokens" in ok:
      ct = ok["completion_tokens"].astype(float)
      print(f"  verified completion_tokens: mean {ct.mean():.0f}, p50 {ct.median():.0f}, p90 {ct.quantile(.9):.0f}, max {ct.max():.0f}")

    n_img, img_tok, img_px, per_row, roles, parts_layout = [], [], [], [], collections.Counter(), collections.Counter()
    shown = 0
    for _, row in ok.iterrows():
      msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
      cnt = 0
      toks = 0
      for mi, role, pi, url in _iter_images(msgs):
        cnt += 1
        roles[(mi, role)] += 1
        if url.startswith("data:image/png;base64,"):
          data = base64.b64decode(url.split(",", 1)[1])
          w, h = _png_size(data)
          n_h, n_w, bh, bw = _plan(w, h)
          t = _num_tokens(n_h, n_w)
          toks += t
          img_px.append((w, h))
          img_tok.append(t)
      n_img.append(cnt)
      per_row.append(toks)
      # Layout of the first user message's parts (text/image sequence).
      for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
          parts_layout["".join("I" if p.get("type") in ("image_url", "image") else "T" for p in m["content"])] += 1
          break
      if shown < FLAGS.show_examples and cnt:
        shown += 1
        print("  --- example first user message parts:")
        for m in msgs:
          if m.get("role") == "user" and isinstance(m.get("content"), list):
            for p in m["content"][:12]:
              if p.get("type") == "text":
                t = p["text"]
                print(f"      TEXT({len(t)}): {t[:160]!r}{' ...' if len(t) > 160 else ''}")
              else:
                print(f"      IMAGE({p.get('type')}) url[:40]={((p.get('image_url') or {}).get('url', ''))[:40]!r}")
            if len(m["content"]) > 12:
              print(f"      ... {len(m['content']) - 12} more parts")
            break
    if n_img:
      n_img = np.array(n_img)
      per_row = np.array(per_row)
      print(f"  images/rollout: mean {n_img.mean():.1f}, max {n_img.max()}, rows w/o images {(n_img == 0).sum()}")
      print(f"  image tokens/rollout: mean {per_row.mean():.0f}, p90 {np.percentile(per_row, 90):.0f}, max {per_row.max()}")
      if img_tok:
        it = np.array(img_tok)
        print(f"  tokens/image: mean {it.mean():.0f}, p50 {np.median(it):.0f}, max {it.max()} ; "
              f"sizes px (w,h) sample: {img_px[:6]} ; min-pixel upscales: {sum(1 for w, h in img_px if w * h < MIN_PIXELS)}/{len(img_px)}")
      print(f"  image message positions (msg_idx, role): {dict(roles.most_common(6))}")
      print(f"  first-user-message part layouts: {dict(parts_layout.most_common(6))}")

    # Sequence-length cap report: estimated total tokens per verified rollout.
    # total ~= system+first-user text (chars/3.6) + image span tokens + completion_tokens
    # (+ tool outputs, approximated by 25% of completion). Exact numbers come from data prep.
    est = []
    for _, row in ok.iterrows():
      msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
      text_chars = 0
      img_t = 0
      for m in msgs[:2]:
        c = m.get("content")
        if isinstance(c, str):
          text_chars += len(c)
        elif isinstance(c, list):
          for p in c:
            if p.get("type") == "text":
              text_chars += len(p.get("text", ""))
            elif p.get("type") in ("image_url", "image"):
              url = (p.get("image_url") or {}).get("url", "")
              if url.startswith("data:image/png;base64,"):
                w, h = _png_size(base64.b64decode(url.split(",", 1)[1]))
                n_h, n_w, _, _ = _plan(w, h)
                img_t += _num_tokens(n_h, n_w)
      comp = float(row.get("completion_tokens") or 0)
      est.append(text_chars / 3.6 + img_t + 1.25 * comp)
    if est:
      est = np.array(est)
      print("  est. total tokens/rollout: "
            f"p50 {np.median(est):.0f}, p90 {np.percentile(est, 90):.0f}, max {est.max():.0f}")
      for cap in (24576, 28672, 32768, 36864, 49152):
        print(f"    kept at max_seq_len={cap}: {int((est <= cap).sum())}/{len(est)} "
              f"({100 * (est <= cap).mean():.1f}%)")


if __name__ == "__main__":
  app.run(main)
