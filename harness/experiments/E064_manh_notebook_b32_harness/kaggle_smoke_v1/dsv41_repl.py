"""DeepSeek-V4.1 REPL-TIR on llama-server: the official encoder renders every turn, tools call in DSML.

Prompts are exact ids counted here: the encoder's text tokenized with special parsing, each image
placeholder expanded to the mmproj's token count. llama-server receives the same ids with every
placeholder run collapsed to its media marker (the mtmd mixed prompt), and reports the expanded
count back (tokens_evaluated), which must match. Text-only prompts are sent as plain ids.

With tools the encoder keeps every turn's reasoning and only appends, so the REPL's prompt ledger
extends the served ids and llama-server's slot cache resumes each turn where the last one ended.
Tool images are therefore kept first-N, not latest-N: an image once shown stays in place.
"""

import base64
from copy import deepcopy
import functools
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct

import dsv4_llama
import qwen38_repl as repl
from qwen38_repl_vision import ChatPrompt


PROFILE = "v41-1017-l4"  # The encoder with the V4.1 tool round trip and image placeholders.
TOKEN_LAYOUT = "deepseek41v"
MAX_TOKENS = 1024  # clip.cpp's fixed V4.1 cap on one image's token block (vision_max_n_token).

VISUAL_GUIDANCE = """

# Targeted visual inspection
The original numerical grids and colored images describe the SAME puzzle.
Python also provides view(*grids, reason="specific uncertainty to inspect").
Pass 1-3 ARC grids/crops (integer colors 0..9), at most two view calls per cell.
The next turn receives flat-color PNGs, ordered left to right, with the reason as a label.
Use this for an ambiguous relation, object crop, or a side-by-side predicted/known TRAINING output.
Use check(solve) and numerical measurements for exact verification; visual similarity is not proof.
Do not reprint/re-render all original training grids. Original images remain in context. Tool images
stay visible until the image budget is spent; later views arrive as labels only, so view sparingly
and write observations in your progress note.
Never call a test prediction "expected": test answers are unknown.
"""

FULLWIDTH = re.compile(r"<｜([^｜<>\n]{1,80})｜>")
ASCII = re.compile(r"<\|([^|<>\n]{1,80})\|>")
# USER_DEFINED tokens: tokenized as specials even without special parsing.
LITERALS = (("｜DSML｜", "｜DSML\\｜"), ("</think>", "<\\/think>"), ("<think>", "<\\think>"),
            ("</dsml:", "<\\/dsml:"), ("<dsml:", "<\\dsml:"))


def encoder():
    return dsv4_llama.load_encoder(PROFILE)


def escape(text):
    """Model, tool or task text made safe for special parsing; idempotent. Lone surrogates are replaced."""
    text = text.encode("utf-8", "replace").decode("utf-8")
    text = ASCII.sub(r"<\\|\1|>", FULLWIDTH.sub(r"<\\｜\1｜>", text))
    for literal, escaped in LITERALS:
        text = text.replace(literal, escaped)
    return text


def read_metadata(mmproj):
    """The mmproj's image-token geometry, read from its GGUF header."""
    keys = dsv4_llama.gguf_general(mmproj, prefixes=("clip.",))
    if keys.get("clip.projector_type") != TOKEN_LAYOUT:
        raise ValueError(f"{mmproj} is not a DeepSeek-V4.1 mmproj (projector {keys.get('clip.projector_type')!r})")
    patch, merge = keys.get("clip.vision.patch_size"), keys.get("clip.vision.projector.scale_factor")
    min_pixels = keys.get("clip.vision.image_min_pixels")
    if not all(isinstance(v, int) and v > 0 for v in (patch, merge, min_pixels)):
        raise ValueError(f"{mmproj} lacks the V4.1 patch size, merge factor or minimum image area")
    factor = patch * merge
    return dict(token_layout=TOKEN_LAYOUT, spatial_factor=factor,
                min_pixels=min(min_pixels, MAX_TOKENS * factor * factor), max_tokens=MAX_TOKENS,
                placeholder=encoder().IMAGE_PLACEHOLDER, mmproj=str(mmproj),
                mmproj_bytes=Path(mmproj).stat().st_size)


def image_tokens(width, height, factor):
    """[start] ([image] * columns [newline]) * rows [end]; the server pads to the factor grid."""
    return -(-height // factor) * (-(-width // factor) + 1) + 2


def audit_tokenization(tokenization, images, metadata):
    """The counted prompt holds one placeholder run per image, of that image's token count."""
    tokens, count, placeholder = tokenization["tokens"], tokenization["count"], tokenization["placeholder_id"]
    runs = [len(list(group)) for token, group in itertools.groupby(tokens) if token == placeholder]
    expected = [image.vision_tokens for image in images]
    if count != len(tokens) or runs != expected:
        raise ValueError(f"Multimodal tokenization mismatch: expected image-token runs {expected}, got {runs}")
    return dict(image_count=len(images), image_tokens=sum(runs), non_image_prompt_tokens=count - sum(runs))


# ----- DSML tool calls ---------------------------------------------------------------------


def parse_calls(content):
    """repl.parse_calls for DSML: (text before the first call block, [(name, {param: text})], malformed).

    Non-string parameter values (string="false") are kept as their JSON text, like the XML dialect's."""
    enc = encoder()
    opening = f"<{enc.dsml_token}{enc.tool_calls_block_name}"
    closing = f"</{enc.dsml_token}{enc.tool_calls_block_name}>"
    head, _, _ = content.partition(opening)
    calls, at = [], 0
    while (start := content.find(opening, at)) >= 0:
        try:
            at, stop, parsed = enc.parse_tool_calls(start + len(opening), content)
            if stop != closing:
                raise ValueError("Unterminated tool-call block")
            for call in parsed:
                arguments = json.loads(call["arguments"])
                name = call["name"] if call.get("namespace") is None else f"{call['namespace']}::{call['name']}"
                calls.append((name, {k: v if isinstance(v, str) else json.dumps(v) for k, v in arguments.items()}))
        except (ValueError, AssertionError):
            return head.strip(), calls, True
    return head.strip(), calls, False


@functools.cache
def dsml_dialect():
    """Built on first use: importing this module must not need the encoder file."""
    enc = encoder()
    dsml, block = enc.dsml_token, enc.tool_calls_block_name
    example = (f"<{dsml}{block}>\n<{dsml}{enc.tool_call_tag_name} name=\"python\">\n"
               f"<{dsml}{enc.tool_parameter_tag_name} name=\"code\" string=\"true\">print(1)"
               f"</{dsml}{enc.tool_parameter_tag_name}>\n</{dsml}{enc.tool_call_tag_name}>\n</{dsml}{block}>")
    return repl.Dialect(
        call_open=f"<{dsml}{block}", call_close=f"</{dsml}{block}>",
        # repl.THINK_CLOSE's early-stop text, closed the way this model closes its thinking.
        think_close="\n\nConsidering the limited time by the user, I have to give the solution based on the "
                    "thinking directly now." + enc.thinking_end_token,
        empty_think=enc.thinking_end_token,
        nudge_format=("Your tool call could not be parsed. Call a tool in exactly this format, with a "
                      "progress_note parameter (Known: ...\nHypothesis: ...\nNext test: ...) before code:\n"
                      + example + "\nReplace the ellipses with actual working state, not placeholders."),
        nudge_optional=f"Call python like this:\n{example}\nprogress_note is optional.",
        parse=parse_calls)


# ----- server ------------------------------------------------------------------------------


def png_size(png):
    if png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("Invalid PNG")
    return struct.unpack(">II", png[16:24])


class Dsv41ReplServer:
    """llama-server /completion with the V4.1 encoder: the REPL's native_template server interface."""
    native_template = True
    prompt_ledger = True  # Extend the served ids; the encoder only appends when tools are present.
    dialect = property(lambda self: dsml_dialect())

    def __init__(self, client, url, *, root, metadata=None, max_images=0):
        self.client, self.url = client, url
        self.root, self.metadata, self.max_images = Path(root).resolve(), metadata, max_images
        self.enc = encoder()
        self.marker, self.placeholder_id = None, None

    async def tokenize(self, text):
        return await dsv4_llama.tokenize(self.client, self.url, text)

    async def props(self):
        if self.marker is None:
            response = await self.client.get(self.url + "/props", timeout=20)
            response.raise_for_status()
            props = response.json()
            if not (props.get("modalities") or {}).get("vision") or not props.get("media_marker"):
                raise ValueError("REPL vision needs llama-server with the V4.1 --mmproj loaded")
            self.marker = props["media_marker"]
        return self.marker

    def view(self, artifact):
        path = (self.root / artifact["path"]).resolve()
        if not path.is_relative_to(self.root / "images") or path.suffix != ".png":
            raise ValueError("Visual artifact must be a PNG under the run's images directory")
        png = path.read_bytes()
        if hashlib.sha256(png).hexdigest() != artifact["sha256"]:
            raise ValueError("Visual artifact checksum mismatch")
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}

    def messages(self, request):
        """The encoder's messages: tools on the system turn, escaped text, first-N tool images."""
        def text(value):
            return escape(value or "")

        out, shown = [], 0
        for i, message in enumerate(deepcopy(request["messages"])):
            role, content = message["role"], message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(dict(type="text", text=text(part.get("text"))))
                    elif part.get("type") == "image_url":
                        parts.append(part)
                        shown += 1  # Task and revealed test images are never omitted.
                    else:
                        raise ValueError(f"Unsupported REPL content part {part.get('type')!r}")
                content = parts
            else:
                content = text(content)
            if message.get("_vision"):
                parts = [dict(type="text", text=content)]
                for artifact in message["_vision"]:
                    label = text(artifact["label"])
                    if shown < self.max_images:
                        parts += [dict(type="text", text=label), self.view(artifact)]
                        shown += 1
                    else:
                        parts.append(dict(type="text", text=f"[Image not shown, image budget spent: {label}]"))
                content = parts
            converted = dict(role=role, content=content)
            if role == "system" and i == 0 and request.get("tools"):
                converted["tools"] = request["tools"]
            elif role == "assistant":
                converted["reasoning_content"] = text(message.get("reasoning_content"))
                calls = []
                for call in message.get("tool_calls") or []:
                    arguments = call["function"]["arguments"]
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    calls.append(dict(id=call.get("id", ""), type="function", function=dict(
                        name=call["function"]["name"],
                        arguments={k: text(v) if isinstance(v, str) else v for k, v in arguments.items()})))
                if calls:
                    converted["tool_calls"] = calls
            elif role == "tool":
                converted["tool_call_id"] = message.get("tool_call_id", "")
            elif role not in ("system", "user"):
                raise ValueError(f"Unexpected message role {role!r}")
            out.append(converted)
        return out

    def encode(self, request):
        """(prompt text, image records in prompt order); the generation prompt follows a final user turn."""
        messages = self.messages(request)
        text, media = self.enc.encode_messages(messages, thinking_mode="thinking", drop_thinking=False,
                                               reasoning_effort=request.get("reasoning_effort"),
                                               return_multi_modal_data=True)
        opened = self.enc.ASSISTANT_SP_TOKEN + self.enc.thinking_start_token
        if request.get("add_generation_prompt", True) is False and messages[-1]["role"] != "assistant":
            if not text.endswith(opened):
                raise ValueError("The V4.1 encoder did not end the user turn with its generation prompt")
            text = text[:-len(opened)]
        return text, media["images"]

    async def render_tokens(self, request):
        text, records = self.encode(request)
        ids = await self.tokenize(text)
        images = []
        for record in records:
            url = record.get("url") or ""
            if not url.startswith("data:image/png;base64,"):
                raise ValueError("REPL vision accepts local PNG data only")
            data = url.split(",", 1)[1]
            png = base64.b64decode(data, validate=True)
            width, height = png_size(png)
            factor = self.metadata["spatial_factor"]
            tokens = image_tokens(width, height, factor)
            if (width % factor or height % factor or width * height < self.metadata["min_pixels"]
                    or tokens > self.metadata["max_tokens"]):
                raise ValueError(f"A {width}x{height} image would be resized by the server; the renderer "
                                 f"pads to {factor}-pixel cells within the minimum area and token cap")
            images.append(dict(data=data, width=width, height=height, tokens=tokens,
                               sha256=hashlib.sha256(png).hexdigest()))
        if images and self.placeholder_id is None:
            (self.placeholder_id,) = await self.tokenize(self.enc.IMAGE_PLACEHOLDER)
        if ids.count(self.placeholder_id) != len(images):
            raise ValueError("The encoded prompt does not hold one image placeholder per image")
        sizes = iter(images)
        expanded = [t for token in ids
                    for t in ([token] * next(sizes)["tokens"] if token == self.placeholder_id else [token])]
        return ChatPrompt(expanded, dict(images=images),
                          dict(tokens=expanded, count=len(expanded), placeholder_id=self.placeholder_id))

    def with_ids(self, prompt, ids):
        return ChatPrompt(ids, prompt.request, dict(prompt.tokenization, tokens=list(ids), count=len(ids)))

    async def append_prompt(self, prompt, tokens):
        """Exact ids extend the same images: llama-server takes ids in the mixed prompt as they are."""
        if not isinstance(prompt, ChatPrompt):
            raise TypeError("A V4.1 continuation lost its counted prompt")
        return self.with_ids(prompt, list(prompt) + list(tokens))

    async def render_prefix(self, request, specials):
        ids = await self.render_tokens(request)
        text, _ = self.encode(request)
        if not ids or ids[0] != specials["bos"] or list(ids[-2:]) != [specials["assistant"], specials["think"]]:
            raise ValueError("REPL prompt does not tokenize to BOS ... <Assistant><think>")
        return repl.CheckpointPrefix(request, text), ids

    async def check_template(self, system, effort, *, tools):
        """The encoder's DSML round trip of the REPL tools; it is local code, so this checks the wiring."""
        head = repl.template_request(system, repl.render_task(repl.SAMPLE_TASK), effort, tools=tools)
        served, _ = self.encode(head)
        multi, _ = self.encode(repl.CheckpointPrefix(head, served).with_history(repl.SAMPLE_TURNS))
        suffix = self.enc.ASSISTANT_SP_TOKEN + self.enc.thinking_start_token
        if not served.endswith(suffix) or not multi.endswith(suffix) or not multi.startswith(served[:-len(suffix)]):
            raise ValueError("The V4.1 encoder does not render the REPL prompt append-only")
        if any(self.enc.to_json(t) not in served for t in self.enc.tools_from_openai_format(tools)):
            raise ValueError("The V4.1 encoder does not preserve the REPL tool schemas")
        history = multi[len(served):]
        expected = [(c["function"]["name"], c["function"]["arguments"])
                    for m in repl.SAMPLE_TURNS for c in m.get("tool_calls", [])]
        _, calls, malformed = parse_calls(history)
        if malformed or calls != expected:
            raise ValueError("The V4.1 encoder does not round-trip the REPL DSML tool calls")
        if any(m["content"] not in multi for m in repl.SAMPLE_TURNS if m["role"] == "tool"):
            raise ValueError("The V4.1 encoder omitted REPL tool feedback")
        eos = await self.tokenize(self.enc.eos_token)
        turns = sum(m["role"] == "assistant" for m in repl.SAMPLE_TURNS)
        if len(eos) != 1 or (await self.tokenize(multi)).count(eos[0]) != turns:
            raise ValueError("Each REPL assistant turn must end with exactly one served EOS token")
        return dict(renderer="checkpoint-every-turn", encoder="deepseek-v4.1", prefix_exact=None, turns_exact=None,
                    tools_equal=True, turns_checked=True, tool_calls_roundtrip=True,
                    reasoning_preserved="Mirror?" in history, generation_prompt=suffix,
                    prefix_sha256=hashlib.sha256(served.encode()).hexdigest(),
                    turns_sha256=hashlib.sha256(multi.encode()).hexdigest())

    async def wire(self, prompt):
        """The /completion prompt: plain ids, or ids with each image's run collapsed to the media marker."""
        images = prompt.request["images"] if isinstance(prompt, ChatPrompt) else []
        if not images:
            return list(prompt)
        placeholder, marker = self.placeholder_id, await self.props()
        parts, i, sizes = [], 0, iter(images)
        while i < len(prompt):
            if prompt[i] != placeholder:
                parts.append(prompt[i])
                i += 1
                continue
            image = next(sizes, None)
            if image is None or prompt[i:i + image["tokens"]] != [placeholder] * image["tokens"]:
                raise ValueError("The prompt's image placeholder runs do not match its images")
            parts.append(marker)
            i += image["tokens"]
        if next(sizes, None) is not None:
            raise ValueError("The prompt holds fewer image placeholder runs than images")
        return dict(prompt_string=parts, multimodal_data=[im["data"] for im in images])

    async def check_transport(self, images):
        """One-token request with rendered images: the server must count exactly our expanded ids."""
        content = [dict(type="text", text="Transport check.")] + [p for im in images for p in im.content_parts()]
        prompt = await self.render_tokens(dict(messages=[dict(role="user", content=content)], tools=[]))
        if [im["tokens"] for im in prompt.request["images"]] != [im.vision_tokens for im in images]:
            raise ValueError("The renderer and the mmproj count the images' tokens differently")
        response = await self.client.post(self.url + "/completion", timeout=300, json=dict(
            prompt=await self.wire(prompt), n_predict=1, cache_prompt=False, temperature=0.0))
        response.raise_for_status()
        evaluated = response.json().get("tokens_evaluated")
        if evaluated != len(prompt):
            raise ValueError(f"llama-server counted {evaluated} prompt tokens for {len(prompt)} expanded ids; "
                             "the mmproj resizes or tokenizes images differently")
        return dict(prompt_tokens=len(prompt), image_tokens=[im.vision_tokens for im in images])

    async def complete(self, payload, output, timeout, *, specials, progress, thinking, **guard):
        prompt = payload["prompt"]
        images = prompt.request["images"] if isinstance(prompt, ChatPrompt) else []
        wire = dict(payload, prompt=await self.wire(prompt))
        record = {**payload, "prompt": None, "prompt_tokens": len(prompt),
                  "images": [{k: v for k, v in im.items() if k != "data"} for im in images]}
        result = await dsv4_llama.stream_completion(
            self.client, self.url, wire, output, timeout, parse_grid=None, specials=specials, progress=progress,
            thinking=thinking, marker=self.enc.thinking_end_token, prompt_tokens=len(prompt),
            request_record=record, **guard)
        usage = result.get("usage") or {}
        if not result["error"] and usage.get("prompt_tokens") != len(prompt):
            result["error"] = (f"PromptCountError: llama-server evaluated {usage.get('prompt_tokens')} prompt "
                               f"tokens for {len(prompt)} counted ids")
        cached = (result.get("timings") or {}).get("cache_n")
        if usage and type(cached) is int:
            usage["prompt_tokens_details"] = dict(cached_tokens=cached)
        if images:
            dsv4_llama.write_json(output / "multimodal_audit.json", dict(
                endpoint="/completion", images=len(images), image_sha256=[im["sha256"] for im in images],
                image_tokens=[im["tokens"] for im in images], counted_prompt_tokens=len(prompt),
                served_prompt_tokens=usage.get("prompt_tokens"), cached_prompt_tokens=cached))
        return result
