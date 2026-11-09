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


def data_uri(grid, cell=32):
    return "data:image/png;base64," + base64.b64encode(render(grid, cell)).decode()


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
