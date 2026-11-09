"""Deterministic, label-blind ARC images for the optional Qwen3.8 vision path.

No model, CUDA, Transformers or Pillow import is needed by text-only callers.
Images supplement the full numerical prompt; they never replace examples.
"""

import argparse
from dataclasses import dataclass
import base64
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import struct
import uuid


PALETTE = ("#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
           "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25")
MIN_PIXELS = 65536
DEFAULT_MAX_PIXELS = 2097152
LAYOUTS = ("sheets", "pairs-plain")
CELL_PIXELS = 24
PLAIN_CELL_PIXELS = 16  # arc3-2026 vision_context upscale

# Text part between the numerical prompt and the images; it depends on how they are drawn.
TEXT_NOTE = ("The following images illustrate the SAME training examples and test input above, "
             "not additional examples. Example numbers match the text (starting at 0). "
             "Each cell's color represents its numerical ID in the legend. Labels, arrows, "
             "legends and margins are not grid cells. Use both representations; the numerical "
             "grids specify the exact cells.")
GRIDS_NOTE = ("The following images show the SAME training examples and test input above, not additional "
              "examples, in the same order; the text before each image says what it shows. Images contain "
              "only grids: within an example the input is left of the output, and examples sharing an image "
              "are stacked top to bottom. Use both representations; the numerical grids specify the exact cells.")
PLAIN_NOTE = ("The following images illustrate the SAME training examples and test input above, not additional "
              "examples. Example numbers match the text (starting at 0). Each grid cell is drawn as a flat colored "
              "square (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 gray, 6 magenta, 7 orange, 8 light blue, "
              "9 maroon); an example's input is on the left and its output on the right, separated by white, "
              "which is not a grid color. Use both representations; the numerical grids specify the exact cells.")


def image_note(layout="sheets", text=True):
    if layout == "pairs-plain":
        return PLAIN_NOTE
    return TEXT_NOTE if text else GRIDS_NOTE


def add_arguments(parser):
    parser.add_argument("--input-mode", choices=("text", "text-image"), default="text",
                        help="Qwen3.8 only: keep full text grids and optionally add PNG illustrations.")
    parser.add_argument("--vision-cell-pixels", type=int, default=None,
                        help=f"Cell side in pixels (default: {CELL_PIXELS} for sheets, {PLAIN_CELL_PIXELS} for "
                             "pairs-plain). A Qwen token covers 32x32 pixels, so 32 draws one cell per token.")
    parser.add_argument("--vision-max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help="Per-image area cap. Split sheets without shrinking/cropping grids.")
    parser.add_argument("--vision-max-images", type=int, default=8,
                        help="Training sheets plus test image per request; exceeding this skips the request.")
    parser.add_argument("--vision-layout", choices=LAYOUTS, default="sheets",
                        help="sheets: several example pairs per image up to --vision-max-pixels; pairs-plain: "
                             "one image per pair, flat cells without lines or text (arc3 style).")
    parser.add_argument("--vision-text", action=argparse.BooleanOptionalAction, default=True,
                        help="sheets: draw the title, color legend and size headers; --no-vision-text draws grids only.")


def validate_options(args):
    if args.input_mode not in ("text", "text-image"):
        raise ValueError("Unknown Qwen3.8 input mode")
    if args.vision_cell_pixels is not None and not 16 <= args.vision_cell_pixels <= 64:
        raise ValueError("vision-cell-pixels must be between 16 and 64")
    if not MIN_PIXELS <= args.vision_max_pixels <= 16777216:
        raise ValueError("vision-max-pixels must be between 65536 and 16777216")
    if not 2 <= args.vision_max_images <= 32:
        raise ValueError("vision-max-images must be between 2 and 32")
    if getattr(args, "vision_layout", "sheets") not in LAYOUTS:
        raise ValueError(f"vision-layout must be one of {LAYOUTS}")


def processor_kwargs(max_pixels):
    # Qwen's 'shortest/longest_edge' size fields are pixel AREAS, not lengths.
    # Both endpoints receive the same overrides as the server's processor.
    return {"min_pixels": MIN_PIXELS, "max_pixels": max_pixels}


def read_metadata(model):
    """Read only local processor/config assets, including for --base-url runs."""
    model = Path(model)
    config_path, processor_path = model / "config.json", model / "preprocessor_config.json"
    if not processor_path.is_file():
        raise ValueError("Vision requires local preprocessor_config.json; copy the checkpoint's "
                         "processor assets during setup (no runtime downloads)")
    config = json.loads(config_path.read_text())
    processor = json.loads(processor_path.read_text())
    vision = config.get("vision_config") or {}
    if config.get("model_type") not in ("qwen3_5", "qwen4_exp") or config.get("language_model_only"):
        raise ValueError("Vision requires a full qwen3_5 or qwen4_exp multimodal target checkpoint")
    for name in ("patch_size", "spatial_merge_size", "depth"):
        if type(vision.get(name)) is not int or vision[name] <= 0:
            raise ValueError(f"Missing/invalid vision_config.{name}")
    ids = {name: config.get(name) for name in
           ("image_token_id", "vision_start_token_id", "vision_end_token_id")}
    if any(type(v) is not int or v < 0 for v in ids.values()) or len(set(ids.values())) != 3:
        raise ValueError("Missing/invalid target vision token IDs")
    if not processor.get("image_processor_type"):
        raise ValueError("preprocessor_config.json must declare an image processor")
    for name, expected in (("patch_size", vision["patch_size"]),
                           ("merge_size", vision["spatial_merge_size"])):
        if name in processor and processor[name] != expected:
            raise ValueError(f"Image processor {name} differs from target vision_config")
    return dict(**ids, spatial_factor=vision["patch_size"] * vision["spatial_merge_size"],
                vision_depth=vision["depth"],
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (config_path, processor_path)})


def validate_weights(model, metadata, *, resolved_shards=None):
    """Check actual safetensors headers, without loading tensors or using CUDA.

    The launcher can supply its validated shard map for flattened Kaggle mounts.
    Still inspect every indexed shard; never trust the index alone for vision.
    """
    names = set()
    model = Path(model)
    index = model / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
              if index.exists() else sorted(p.name for p in model.glob("*.safetensors")))
    if resolved_shards is not None and set(resolved_shards) != set(shards):
        raise ValueError("Resolved vision shard map must match every checkpoint shard")
    for name in shards:
        path = Path(resolved_shards[name]) if resolved_shards is not None else model / name
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header size: {path}")
            names.update(json.loads(stream.read(size)))
    vision_names = {n[n.index("visual.") + len("visual."):] for n in names
                    if n.startswith(("model.visual.", "visual."))}
    blocks = {int(n.split(".")[1]) for n in vision_names
              if n.startswith("blocks.") and n.split(".")[1].isdigit()}
    if (not any(n.startswith("patch_embed.") for n in vision_names)
            or not any(n.startswith("merger.") for n in vision_names)
            or blocks != set(range(metadata["vision_depth"]))):
        raise ValueError("Vision weights missing/incomplete: expected patch embedding, all "
                         "vision blocks and merger. Use the full multimodal checkpoint.")
    return dict(vision_tensor_count=len(vision_names), vision_block_count=len(blocks))


class RenderLimitError(ValueError):
    """Cannot represent every grid under explicit image limits; never truncate."""


@dataclass(frozen=True)
class RenderedImage:
    png: bytes
    label: str
    kind: str
    width: int
    height: int
    grids: list
    vision_tokens: int

    @property
    def sha256(self):
        return hashlib.sha256(self.png).hexdigest()

    def content_parts(self):
        return [{"type": "text", "text": self.label},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")}}]

    def save(self, root):
        relative = Path("images") / f"{self.sha256}.png"
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != self.png:
                raise ValueError(f"Existing image artifact differs: {path}")
        else:
            temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
            try:
                with temp.open("xb") as stream:
                    stream.write(self.png)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        return dict(path=str(relative), sha256=self.sha256, label=self.label, kind=self.kind,
                    width=self.width, height=self.height, grids=self.grids,
                    vision_tokens=self.vision_tokens)


class Renderer:
    PAD, GAP, HEADER, ROW_HEADER, ROW_GAP = 24, 64, 152, 32, 24

    def __init__(self, args, metadata):
        validate_options(args)
        from PIL import Image, ImageDraw, ImageFont, __version__
        self.Image, self.ImageDraw = Image, ImageDraw
        # Pillow bundles this font: no host fonts, internet or generated artwork.
        try:
            self.font = ImageFont.load_default(size=18)
            self.title_font = ImageFont.load_default(size=22)
        except TypeError as exc:
            raise ValueError("Vision rendering requires Pillow >= 10.1, installed offline") from exc
        self.max_pixels = args.vision_max_pixels
        self.max_images, self.factor = args.vision_max_images, metadata["spatial_factor"]
        # deepseek41v (DeepSeek-V4.1 mmproj): the server resizes an image below its minimum area or above
        # its token cap, so images are padded to the 42-pixel token grid and kept inside both instead.
        self.token_layout = metadata.get("token_layout", "qwen")
        self.min_pixels = metadata["min_pixels"] if self.token_layout == "deepseek41v" else MIN_PIXELS
        self.max_tokens = metadata.get("max_tokens")
        # Callers built before these options existed keep the labelled sheets.
        self.layout = getattr(args, "vision_layout", "sheets")
        self.text = getattr(args, "vision_text", True)
        self.cell = args.vision_cell_pixels or (PLAIN_CELL_PIXELS if self.layout == "pairs-plain" else CELL_PIXELS)
        self.note = image_note(self.layout, self.text)
        self.train_cache = {}
        self.settings = dict(renderer_version=2, pillow_version=__version__,
                             cell_pixels=self.cell,
                             layout=self.layout, text=self.text, max_pixels=self.max_pixels,
                             max_images=self.max_images, spatial_factor=self.factor, palette=PALETTE)
        if self.token_layout != "qwen":
            self.settings.update(token_layout=self.token_layout, min_pixels=self.min_pixels,
                                 max_tokens=self.max_tokens)

    def rounded(self, value):
        return ((value + self.factor - 1) // self.factor) * self.factor

    def tokens(self, width, height):
        if self.token_layout == "deepseek41v":
            # [start] ([image] * columns [newline]) * rows [end], one token per 42x42 pixels.
            return (height // self.factor) * (width // self.factor + 1) + 2
        return width * height // self.factor**2

    def oversized(self, width, height):
        return (width * height > self.max_pixels
                or self.max_tokens is not None and self.tokens(width, height) > self.max_tokens)

    def fit(self, width, height):
        """Factor-aligned size, white-padded up to the minimum area; never shrunk."""
        if self.token_layout == "deepseek41v":
            while width * height < self.min_pixels:
                if width <= height:
                    width += self.factor
                else:
                    height += self.factor
        if width * height < self.min_pixels or self.oversized(width, height):
            raise RenderLimitError("A grid/sheet exceeds --vision-max-pixels or the image token limit; raise the "
                                   "limit. No grid was shrunk, cropped or omitted.")
        return width, height

    def shape(self, grid):
        if (not isinstance(grid, list) or not 1 <= len(grid) <= 30
                or any(not isinstance(row, list) or not 1 <= len(row) <= 30 for row in grid)
                or len({len(row) for row in grid}) != 1
                or any(type(v) is not int or not 0 <= v <= 9 for row in grid for v in row)):
            raise ValueError("Vision expects rectangular ARC grids with integers 0..9 and sides 1..30")
        return len(grid), len(grid[0])

    def canvas(self, width, height, title):
        width, height = self.fit(self.rounded(max(704, width)), self.rounded(max(256, height)))
        image = self.Image.new("RGB", (width, height), "white")
        draw = self.ImageDraw.Draw(image)
        draw.text((self.PAD, 12), title, fill="black", font=self.title_font)
        draw.text((self.PAD, 44), "Color IDs (labels and legend are outside the grids)", fill="black", font=self.font)
        for value, color in enumerate(PALETTE):
            x = self.PAD + 64 * value
            draw.text((x + 10, 68), str(value), fill="black", font=self.font)
            draw.rectangle((x, 94, x + 32, 126), fill=color, outline="#606060")
        return image, draw

    def draw_grid(self, draw, grid, x, y, *, kind, index, side):
        rows, cols = self.shape(grid)
        draw.text((x, y - self.ROW_HEADER), f"{side.upper()}  {rows} rows x {cols} cols",
                  fill="black", font=self.font)
        return self.draw_cells(draw, grid, x, y, kind=kind, index=index, side=side)

    def draw_cells(self, draw, grid, x, y, *, kind, index, side, cell=None, lines=True):
        """Cells only; a None cell stays white (the Luna harness's changed-cells grid uses it)."""
        cell = cell or self.cell
        rows, cols = self.shape([[0 if v is None else v for v in row] for row in grid])
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if value is not None:
                    left, top = x + c*cell, y + r*cell
                    draw.rectangle((left, top, left+cell-1, top+cell-1), fill=PALETTE[value])
        if lines:  # One-pixel boundaries; cell interiors retain the exact ARC color.
            for c in range(cols + 1):
                draw.line((x+c*cell, y, x+c*cell, y+rows*cell), fill="#606060")
            for r in range(rows + 1):
                draw.line((x, y+r*cell, x+cols*cell, y+r*cell), fill="#606060")
        return dict(kind=kind, index=index, side=side, rows=rows, cols=cols,
                    x=x, y=y, cell_pixels=cell)

    def blank(self, width, height):
        width, height = self.fit(self.rounded(max(256, width)), self.rounded(max(256, height)))
        image = self.Image.new("RGB", (width, height), "white")
        return image, self.ImageDraw.Draw(image)

    def grids_size(self, rows):
        """Unrounded size of grids_image(rows): a one-cell margin and two-cell gaps."""
        return (self.cell * max(sum(len(g[0]) for g, _, _ in row) + 2*len(row) for row in rows),
                self.cell * sum(max(len(g) for g, _, _ in row) + 2 for row in rows))

    def grids_image(self, rows, label, kind):
        """Rows of (grid, index, side), each row left to right and rows top to bottom, on white
        with no title, legend, size header or arrow; the text label says what the image shows."""
        image, draw = self.blank(*self.grids_size(rows))
        placements, y = [], self.cell
        for row in rows:
            x = self.cell
            for grid, index, side in row:
                placements.append(self.draw_cells(draw, grid, x, y, kind=kind, index=index, side=side))
                x += (len(grid[0]) + 2) * self.cell
            y += (max(len(g) for g, _, _ in row) + 2) * self.cell
        return self.encoded(image, label, kind, placements)

    def plain_image(self, items, label, kind):
        """arc3-2026 style: (grid, index, side) items side by side as flat squares without lines,
        two blank cells apart, padded with white on the right and bottom to the spatial factor."""
        cell = self.cell
        width = cell * (sum(len(g[0]) for g, _, _ in items) + 2*(len(items) - 1))
        image, draw = self.blank(width, cell * max(len(g) for g, _, _ in items))
        placements, x = [], 0
        for grid, index, side in items:
            placements.append(self.draw_cells(draw, grid, x, 0, kind=kind, index=index, side=side,
                                              cell=cell, lines=False))
            x += (len(grid[0]) + 2) * cell
        return self.encoded(image, label, kind, placements)

    def encoded(self, image, label, kind, grids):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
        return RenderedImage(buffer.getvalue(), label, kind, image.width, image.height, grids,
                             self.tokens(image.width, image.height))

    def pair_size(self, pair):
        hi, wi = self.shape(pair["input"])
        ho, wo = self.shape(pair["output"])
        # Reserve at least 256px under each side's dimension label.
        return (2*self.PAD + max(256, wi*self.cell) + self.GAP + max(256, wo*self.cell),
                self.ROW_HEADER*2 + max(hi, ho)*self.cell + self.ROW_GAP)

    def sheet_size(self, pairs):
        sizes = [self.pair_size(pair) for _, pair in pairs]
        return (self.rounded(max(704, max(w for w, h in sizes))),
                self.rounded(max(256, self.HEADER + self.PAD + sum(h for w, h in sizes))))

    def grids_sheet_size(self, pairs):
        w, h = self.grids_size([[(p["input"], i, "input"), (p["output"], i, "output")] for i, p in pairs])
        return self.rounded(max(256, w)), self.rounded(max(256, h))

    def training_images(self, train):
        # Project explicitly: hidden test labels and other metadata cannot enter
        # pixels, identities, ordering or caching.
        pairs = [{"input": p["input"], "output": p["output"]} for p in train]
        key = json.dumps(pairs, separators=(",", ":"))
        if key in self.train_cache:
            return self.train_cache[key]
        if self.layout == "pairs-plain":
            pages = [[item] for item in enumerate(pairs)]
        else:
            size = self.sheet_size if self.text else self.grids_sheet_size
            pages, page = [], []
            for item in enumerate(pairs):
                w, h = size(page + [item])
                if self.oversized(w, h) and page:
                    pages.append(page)
                    page = []
                w, h = size(page + [item])
                if self.oversized(w, h):
                    raise RenderLimitError("A training pair exceeds --vision-max-pixels; "
                                           "raise the limit. No grids were omitted.")
                page.append(item)
            if page:
                pages.append(page)
        if len(pages) + 1 > self.max_images:
            raise RenderLimitError(f"Need {len(pages)+1} images including test input, but "
                                   f"--vision-max-images={self.max_images}; no examples were omitted.")
        if self.layout == "pairs-plain" or not self.text:
            images = [self.page_image(page) for page in pages]
            self.train_cache[key] = images
            return images
        images = []
        for number, page in enumerate(pages):
            width, height = self.sheet_size(page)
            image, draw = self.canvas(width, height, f"Training examples - sheet {number+1} of {len(pages)}")
            y, grids = self.HEADER, []
            for index, pair in page:
                draw.text((self.PAD, y), f"Example {index}", fill="black", font=self.font)
                y += self.ROW_HEADER*2
                output_x = self.PAD + max(256, self.shape(pair["input"])[1]*self.cell) + self.GAP
                grids.append(self.draw_grid(draw, pair["input"], self.PAD, y,
                                            kind="training", index=index, side="input"))
                grids.append(self.draw_grid(draw, pair["output"], output_x, y,
                                            kind="training", index=index, side="output"))
                draw.text((output_x - 46, y), "->", fill="black", font=self.font)
                y += max(len(pair["input"]), len(pair["output"]))*self.cell + self.ROW_GAP
            images.append(self.encoded(image, "Training sheet: " + ", ".join(f"Example {i}" for i, _ in page),
                                       "training", grids))
        self.train_cache[key] = images
        return images

    def page_image(self, page):
        """(index, pair) items without text: one pair per row, input left of output."""
        label = (f"Example {page[0][0]}: input (left) -> output (right)." if len(page) == 1 else
                 f"Examples {', '.join(str(i) for i, _ in page)}, one per row from top to bottom: "
                 "input (left) -> output (right).")
        rows = [[(p["input"], i, "input"), (p["output"], i, "output")] for i, p in page]
        if self.layout == "pairs-plain":
            return self.plain_image(rows[0], label, "training")
        return self.grids_image(rows, label, "training")

    def test_image(self, grid, test_index, label):
        """The test input alone, drawn like the training images."""
        if self.layout == "pairs-plain":
            return self.plain_image([(grid, test_index, "input")], label, "test")
        if not self.text:
            return self.grids_image([[(grid, test_index, "input")]], label, "test")
        rows, cols = self.shape(grid)
        width = 2*self.PAD + cols*self.cell
        height = self.HEADER + self.ROW_HEADER + rows*self.cell + self.PAD
        image, draw = self.canvas(width, height, "Test input - predict its output")
        placement = self.draw_grid(draw, grid, self.PAD, self.HEADER + self.ROW_HEADER,
                                   kind="test", index=test_index, side="input")
        return self.encoded(image, label, "test", [placement])

    def render(self, task, test_index):
        training = self.training_images(task["train"])
        return training + [self.test_image(task["test"][test_index]["input"], test_index,
                                           "Test input only (no test output is shown).")]


def audit_tokenization(response, images, metadata):
    """Reject ignored images/resizing instead of undercounting the 81K budget."""
    tokens, count = response.get("tokens"), response.get("count")
    if (type(count) is not int or count <= 0 or not isinstance(tokens, list)
            or len(tokens) != count or any(type(t) is not int for t in tokens)):
        raise ValueError("Vision requires /tokenize to return expanded token IDs and their exact count")
    image_id = metadata["image_token_id"]
    runs = [len(list(group)) for token, group in itertools.groupby(tokens) if token == image_id]
    expected = [image.vision_tokens for image in images]
    if (runs != expected
            or tokens.count(metadata["vision_start_token_id"]) != len(images)
            or tokens.count(metadata["vision_end_token_id"]) != len(images)):
        raise ValueError(f"Multimodal tokenization mismatch: expected image-token runs {expected}, got {runs}. "
                         "Check the target/processor and vision-enabled server; no text-only fallback.")
    return dict(image_count=len(images), image_tokens=sum(runs), non_image_prompt_tokens=count-sum(runs))
