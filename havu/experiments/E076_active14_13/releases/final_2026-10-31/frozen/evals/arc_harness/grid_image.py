"""E033: ARC grids as images for vision-language models, one model patch token per grid cell.

Qwen3.5/3.8's vision encoder cuts an image into 16x16-pixel ViT patches and merges 2x2 neighbours into one token for
the language model, so one LLM visual token covers 32x32 pixels. With `cell = patch_size * merge_size` (32) every grid
cell is exactly one token (cell = 16 gives one ViT patch per cell, 4 cells per token, but grids with an odd side then
make image sides that are not multiples of 32, which the processor resizes; use 32). Images are drawn without grid
lines or borders, so cell boundaries fall exactly on patch boundaries. The server must not resize the images: Qwen's
default processor enlarges images below 65,536 pixels, so it needs a lower `min_pixels` (see E033 README).
"""
import base64
import io
import json

from PIL import Image

# The standard ARC palette (arcprize.org / ARC-AGI-2 viewer).
PALETTE = {0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64), 4: (255, 220, 0),
           5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37)}
COLOUR_NAMES = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "grey", 6: "magenta", 7: "orange",
                8: "light blue", 9: "maroon"}
LEGEND = "Colour key (cell value: colour): " + ", ".join(f"{k} {v}" for k, v in COLOUR_NAMES.items()) + "."


def render(grid, cell=32):
    """PNG bytes of `grid` with each cell a solid `cell` x `cell` square."""
    h, w = len(grid), len(grid[0])
    small = Image.new("RGB", (w, h))
    small.putdata([PALETTE[v] for row in grid for v in row])
    img = small.resize((w * cell, h * cell), Image.NEAREST)  # exact: every cell becomes a solid square
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# DeepSeek V4.1 vision (E058): 14-px patches merged 3x3, so one LLM token covers a 42x42-pixel tile. The API rescales
# images to >= ~169 tiles (min_pixels 295,936) and <= ~1,024 tiles, and charges tiles_w * tiles_h + tiles_h + 2
# tokens (one per tile, one per tile row, two markers; measured 2026-10-26). A cell side of 42*k px with
# k = ceil(sqrt(169 / cells)) keeps every cell exactly k x k tokens and avoids any rescale for grids up to 30x30.
DS_TILE, DS_MIN_TILES, DS_MAX_TILES = 42, 169, 1024
# Qwen3.8 Flash via OpenRouter -> Alibaba (E058, measured 2026-10-27): 16-px patches merged 2x2 = 32-px tiles; images are
# rescaled to >= 64 tiles (256^2 px) and <= ~2,500 tiles; charged tiles + 2 tokens.
ALIGN = {"deepseek": (DS_TILE, DS_MIN_TILES, DS_MAX_TILES), "qwen": (32, 64, 2500)}


def aligned_cell(grid, profile="deepseek"):
    """Pixels per cell (tile * k, smallest k reaching the provider's minimum) for an unresized, token-aligned image."""
    tile, lo, hi = ALIGN[profile]
    cells = len(grid) * len(grid[0])
    k = 1
    while cells * k * k < lo:
        k += 1
    if cells * k * k > hi:
        raise ValueError(f"grid of {cells} cells exceeds {hi} tiles even at one tile per cell")
    return tile * k


def deepseek_cell(grid):
    """Pixels per cell for an unresized, token-aligned DeepSeek image of `grid`."""
    return aligned_cell(grid, "deepseek")


def deepseek_tokens(grid, cell=None):
    """Prompt tokens DeepSeek charges for the image of `grid` drawn at `cell` px (default deepseek_cell)."""
    cell = cell or deepseek_cell(grid)
    tw, th = len(grid[0]) * cell // DS_TILE, len(grid) * cell // DS_TILE
    return tw * th + th + 2


def qwen_tokens(grid, cell=None):
    """Prompt tokens Qwen3.8 Flash (Alibaba) charges for the image of `grid` at `cell` px (default aligned)."""
    cell = cell or aligned_cell(grid, "qwen")
    return (len(grid[0]) * cell // 32) * (len(grid) * cell // 32) + 2


def data_uri(grid, cell=32):
    return "data:image/png;base64," + base64.b64encode(render(grid, cell)).decode()


# E070 (Manh's --vision-layout): several grids in one image. Rows of grids, left to right with GAP white cells between
# grids and between rows, on white (not an ARC colour); the image is padded with white to whole DeepSeek tiles so the
# API never rescales it. Cell sides stay aligned to DeepSeek's geometry: 42*k px (k x k tiles per cell, smallest k
# reaching 169 tiles) when one tile per cell fits in 1,024 tiles, else 28 or 14 px (one ViT patch).
ROW_GAP = 2
SUB_TILE_CELLS = (28, 14)


def rows_cells(rows):
    """(width, height) in cells of an image of `rows` (lists of grids)."""
    w = max(sum(len(g[0]) for g in row) + ROW_GAP * (len(row) - 1) for row in rows)
    h = sum(max(len(g) for g in row) for row in rows) + ROW_GAP * (len(rows) - 1)
    return w, h


def _tiles(px):
    return -(-px // DS_TILE)


def rows_cell(rows):
    """Pixels per cell for the multi-grid image of `rows`, or None if it cannot fit in DS_MAX_TILES tiles."""
    w, h = rows_cells(rows)
    k = 1
    while w * h * k * k < DS_MIN_TILES:
        k += 1
    if w * h * k * k <= DS_MAX_TILES:
        return DS_TILE * k
    for cell in SUB_TILE_CELLS:
        tw, th = _tiles(w * cell), _tiles(h * cell)
        if DS_MIN_TILES <= tw * th <= DS_MAX_TILES:
            return cell
    return None


def render_rows(rows, cell):
    """PNG bytes of `rows` drawn at `cell` px per cell, padded to whole 42-px tiles."""
    w, h = rows_cells(rows)
    img = Image.new("RGB", (_tiles(w * cell) * DS_TILE, _tiles(h * cell) * DS_TILE), (255, 255, 255))
    y = 0
    for row in rows:
        x = 0
        for g in row:
            gh, gw = len(g), len(g[0])
            small = Image.new("RGB", (gw, gh))
            small.putdata([PALETTE[v] for r in g for v in r])
            img.paste(small.resize((gw * cell, gh * cell), Image.NEAREST), (x * cell, y * cell))
            x += gw + ROW_GAP
        y += max(len(g) for g in row) + ROW_GAP
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def rows_data_uri(rows, cell):
    return "data:image/png;base64," + base64.b64encode(render_rows(rows, cell)).decode()


def rows_tokens(rows, cell):
    """Prompt tokens DeepSeek charges for the image of `rows` at `cell` px (see deepseek_tokens)."""
    w, h = rows_cells(rows)
    tw, th = _tiles(w * cell), _tiles(h * cell)
    return tw * th + th + 2


def sheet_pages(pairs):
    """Training pairs (input, output) packed greedily, in order, into as few sheet images as fit DeepSeek's tile cap;
    returns a list of (first_index, [pairs]). A pair that fits nowhere alone raises ValueError."""
    pages, cur, start = [], [], 0
    for i, p in enumerate(pairs):
        if cur and rows_cell([[a, b] for a, b in cur + [p]]) is None:
            pages.append((start, cur))
            cur, start = [], i
        if rows_cell([[p[0], p[1]]]) is None:
            raise ValueError(f"training pair {i} does not fit in {DS_MAX_TILES} tiles even at {SUB_TILE_CELLS[-1]} px")
        cur.append(p)
    if cur:
        pages.append((start, cur))
    return pages


def visual_tokens(grid, cell=32, patch=16, merge=2):
    """LLM visual tokens for an unresized image of `grid` (excluding the vision start/end markers)."""
    h, w = len(grid) * cell, len(grid[0]) * cell
    return (h // (patch * merge)) * (w // (patch * merge))


def image_parts(grid, cell=32, label=""):
    """Chat content parts for one grid: a short text line with its size, then the image."""
    h, w = len(grid), len(grid[0])
    return [{"type": "text", "text": f"{label}({h} rows x {w} columns; one {cell}x{cell}-pixel square per cell):"},
            {"type": "image_url", "image_url": {"url": data_uri(grid, cell)}}]


def build_image_prompt(template, train, test_input, cell=32, suffix=""):
    """The ARC Prize prompt with every JSON grid replaced by its image: a list of chat content parts."""
    head, rest = template.split("{training_examples}")
    middle, tail = rest.split("{test_input}")
    parts = [{"type": "text", "text": head.rstrip("\n") + "\n" + LEGEND + "\n"}]
    for i, pair in enumerate(train):
        parts.append({"type": "text", "text": f"--Example {i}-- \n\n INPUT: "})
        parts += image_parts(pair["input"], cell)
        parts.append({"type": "text", "text": "\n\nOUTPUT: "})
        parts += image_parts(pair["output"], cell)
        parts.append({"type": "text", "text": "\n\n"})
    parts.append({"type": "text", "text": middle})
    parts += image_parts(test_input, cell)
    parts.append({"type": "text", "text": tail + suffix})
    return merge_text(parts)


def merge_text(parts):
    """Join adjacent text parts (fewer, larger text segments)."""
    out = []
    for p in parts:
        if p["type"] == "text" and out and out[-1]["type"] == "text":
            out[-1] = {"type": "text", "text": out[-1]["text"] + p["text"]}
        else:
            out.append(p)
    return out


def prompt_text_for_log(parts):
    """A JSON-safe copy of the prompt for transcripts, with image data replaced by a short marker."""
    return json.loads(json.dumps([p if p["type"] == "text" else {"type": "image_url", "image_url": {"url": "<png>"}}
                                  for p in parts]))
