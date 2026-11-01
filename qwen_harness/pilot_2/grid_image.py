"""Grid -> PNG rendering for the E058 "text+image" ARC harness (exact port of Havu's `grid_image.py`).

Each grid cell is a solid `cell` x `cell` square in the standard ARC palette, no grid lines, no
border. For DeepSeek-V4.1 vision (14-px patches merged 3x3 -> one LLM token per 42x42-px tile) the
cell side is `42 * k` px with the smallest `k` such that the image has >= 169 tiles, so every cell
is exactly `k x k` tokens and the image processor never rescales it (grids up to 30x30).
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

Grid = list[list[int]]

# The standard ARC palette (arcprize.org / ARC-AGI-2 viewer).
PALETTE = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}
COLOUR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "light blue",
    9: "maroon",
}
LEGEND = "Colour key (cell value: colour): " + ", ".join(
    f"{k} {v}" for k, v in COLOUR_NAMES.items()
) + "."

# DeepSeek V4.1 vision (E058): 14-px patches merged 3x3, so one LLM token covers a 42x42-pixel tile.
# The API rescales images to >= ~169 tiles (min_pixels 295,936) and <= ~1,024 tiles, and charges
# tiles_w * tiles_h + tiles_h + 2 tokens (one per tile, one per tile row, two markers).
DS_TILE, DS_MIN_TILES, DS_MAX_TILES = 42, 169, 1024
ALIGN = {"deepseek": (DS_TILE, DS_MIN_TILES, DS_MAX_TILES), "qwen": (32, 64, 2500)}

# `run_deepseek_tools.IMAGE_NOTE` (E058) and the E064 "strict" variant.
IMAGE_NOTE = (
    "\n\nThe images below show the SAME training examples and test input(s) as the JSON above, in the same "
    "order; they are not additional examples. Each image is one grid, drawn as flat coloured squares "
    "without lines (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 light blue, "
    "9 maroon). Use them to see shapes and layout; the JSON grids specify the exact cells."
)
IMAGE_NOTES = {
    "": IMAGE_NOTE,
    "strict": IMAGE_NOTE + (
        " The images give only an overall impression and are easy to misread: before "
        "you build on anything you see in them (a symmetry, a count, which cells "
        "connect, sizes or positions), confirm it in code on the JSON grids."
    ),
}


def render(grid: Grid, cell: int = 32) -> bytes:
  """PNG bytes of `grid` with each cell a solid `cell` x `cell` square."""
  h, w = len(grid), len(grid[0])
  small = Image.new("RGB", (w, h))
  small.putdata([PALETTE[v] for row in grid for v in row])
  img = small.resize((w * cell, h * cell), Image.NEAREST)  # exact: every cell becomes a solid square
  buf = io.BytesIO()
  img.save(buf, format="PNG", optimize=True)
  return buf.getvalue()


def aligned_cell(grid: Grid, profile: str = "deepseek") -> int:
  """Pixels per cell (tile * k, smallest k reaching the provider's minimum) for an unresized image."""
  tile, lo, hi = ALIGN[profile]
  cells = len(grid) * len(grid[0])
  k = 1
  while cells * k * k < lo:
    k += 1
  if cells * k * k > hi:
    raise ValueError(f"grid of {cells} cells exceeds {hi} tiles even at one tile per cell")
  return tile * k


def deepseek_cell(grid: Grid) -> int:
  return aligned_cell(grid, "deepseek")


def deepseek_tokens(grid: Grid, cell: int | None = None) -> int:
  """Prompt tokens DeepSeek charges for the image of `grid` drawn at `cell` px (default aligned)."""
  cell = cell or deepseek_cell(grid)
  tw, th = len(grid[0]) * cell // DS_TILE, len(grid) * cell // DS_TILE
  return tw * th + th + 2


def data_uri(grid: Grid, cell: int = 32) -> str:
  return "data:image/png;base64," + base64.b64encode(render(grid, cell)).decode()


def text_image_content(
    prompt: str,
    train: list[dict[str, Any]],
    test_inputs: list[Grid],
    cell: int = 0,
    align: str = "deepseek",
    note: str = IMAGE_NOTE,
) -> list[dict[str, Any]]:
  """E058 `--grid-format text+image`: the full text prompt + note, then one labelled image per grid.

  `cell == 0` -> `aligned_cell(align)` (each cell exactly k x k model tiles, never rescaled);
  otherwise a fixed cell size in pixels. Exact port of `run_deepseek_tools.text_image_content`.
  """

  def image(label: str, g: Grid) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": f"\n{label} ({len(g)}x{len(g[0])}):"},
        {"type": "image_url", "image_url": {"url": data_uri(g, cell or aligned_cell(g, align))}},
    ]

  parts: list[dict[str, Any]] = [{"type": "text", "text": prompt + note}]
  for i, pair in enumerate(train):  # numbered from 0, like the text prompt's --Example 0--
    parts += image(f"Example {i} input", pair["input"]) + image(f"Example {i} output", pair["output"])
  for j, g in enumerate(test_inputs, 1):
    parts += image("Test input" if len(test_inputs) == 1 else f"Test input {j - 1}", g)
  return parts


def content_images(content: Any) -> list[bytes]:
  """PNG bytes of every `image_url` part of a message content (prompt order); `[]` for plain text."""
  if not isinstance(content, list):
    return []
  out: list[bytes] = []
  for part in content:
    if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
      url = part.get("image_url", {}).get("url", "") if part.get("type") == "image_url" else ""
      header, _, payload = url.partition(",")
      if not header.startswith("data:") or ";base64" not in header:
        raise ValueError(f"Unsupported image URL: {header[:40]!r}")
      out.append(base64.b64decode(payload))
  return out
