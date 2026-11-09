"""Opt-in multimodal REPL transport and bounded, label-blind Python visual feedback.

The numerical task remains authoritative. Text-only REPL uses its existing transport.
Images are never replaced by token IDs on /v1/completions: every multimodal request
goes through /v1/chat/completions with the original images and exact token counting,
or, with token_transport, through /inference/v1/generate with the exact ids and images.
"""

import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import qwen38_repl as repl
import qwen38_vision as vision


VISUAL_GUIDANCE = """

# Targeted visual inspection
The original numerical grids and colored images describe the SAME puzzle.
Python also provides view(*grids, reason="specific uncertainty to inspect").
Pass 1-3 ARC grids/crops (integer colors 0..9), at most two view calls per cell.
The next turn receives flat-color PNGs, ordered left to right, with the reason as a label.
Use this for an ambiguous relation, object crop, or a side-by-side predicted/known TRAINING output.
Use check(solve) and numerical measurements for exact verification; visual similarity is not proof.
Do not reprint/re-render all original training grids. Original images remain in context. Only the
latest two tool images are retained visually; write observations in your progress note.
Never call a test prediction "expected": test answers are unknown.
"""


def task_images(renderer, task, *, test_reveal="all", visual_tools=False):
    """Render only disclosed inputs, reserving capacity for eventual complete disclosure."""
    images = list(renderer.training_images(task["train"]))
    eventual = len(images) + len(task["test"])
    if eventual > renderer.max_images:
        raise vision.RenderLimitError("All TIR training/test images exceed --vision-max-images; no inputs omitted")
    if visual_tools and eventual >= renderer.max_images:
        raise vision.RenderLimitError("No room for Python views; raise --vision-max-images")
    visible, indices = repl.disclosure.initial_view(task, test_reveal)
    for position, original in enumerate(indices):
        label = f"Test input {position} only (no test output)."
        if len(indices) < len(task["test"]):
            label += f" Python test_inputs[{position}], original test {original}."
        images.append(replace(renderer.render(visible, position)[-1], label=label))
    return images


class ChatPrompt(list):
    """List-compatible counted IDs plus the multimodal request that actually produced them."""
    def __init__(self, ids, request, tokenization, *, assistant_prefix=None):
        super().__init__(ids)
        self.request = request
        self.tokenization = tokenization
        self.assistant_prefix = assistant_prefix


class Visualizer:
    def __init__(self, renderer, root):
        self.renderer, self.root = renderer, Path(root)

    def save(self, visuals, *, turn, call_id):
        artifacts = []
        for item in visuals[:2]:
            grids = item["grids"]
            if not 1 <= len(grids) <= 3:
                raise ValueError("Visual feedback must contain 1-3 grids")
            for grid in grids:
                self.renderer.shape(grid)
            label = f"Python view, turn {turn}, {call_id}: {item['reason'][:400]}. Grids left to right."
            image = self.renderer.plain_image(
                [(grid, i, "requested") for i, grid in enumerate(grids)], label, "python-view")
            artifacts.append(image.save(self.root))
        return artifacts


class VllmVisionServer(repl.VllmServer):
    def __init__(self, client, url, model, *, root, metadata, max_images, token_transport=False):
        super().__init__(client, url, model, native_template=True)
        self.root, self.metadata, self.max_images = Path(root).resolve(), metadata, max_images
        # Exact served ids (the REPL's prompt ledger) instead of server-side re-rendering of messages.
        self.token_transport = token_transport

    def wire_request(self, request):
        request = deepcopy(request)
        # Recent OpenAI-compatible chat endpoints reject tools=[], although
        # /tokenize accepts it. Omission has the same no-tools template meaning.
        if request.get("tools") == []:
            request.pop("tools")
        # The owned server pins these limits globally; every PNG is already
        # rendered within them. Recent vLLM disables per-request overrides.
        # For external servers, the image-token audit below fails closed if
        # their resizing policy changes our expected dimensions.
        request.pop("mm_processor_kwargs", None)
        messages = request["messages"]
        initial_count = sum(p.get("type") == "image_url" for m in messages
                            for p in m.get("content", []) if isinstance(p, dict))
        extra = [(i, im) for i, m in enumerate(messages) for im in m.get("_vision", [])]
        slots = max(0, min(2, self.max_images - initial_count))
        keep = extra[-slots:] if slots else []
        for i, message in enumerate(messages):
            if "_vision" in message:
                artifacts = message.pop("_vision")
                parts = [{"type": "text", "text": message["content"]}]
                for artifact in artifacts:
                    if (i, artifact) not in keep:
                        parts.append({"type": "text", "text": f"[Earlier image omitted: {artifact['label']}]"})
                        continue
                    path = (self.root / artifact["path"]).resolve()
                    if not path.is_relative_to(self.root / "images") or path.suffix != ".png":
                        raise ValueError("Visual artifact must be a PNG under the run's images directory")
                    png = path.read_bytes()
                    if hashlib.sha256(png).hexdigest() != artifact["sha256"]:
                        raise ValueError("Visual artifact checksum mismatch")
                    parts += [{"type": "text", "text": artifact["label"]},
                              {"type": "image_url", "image_url": {
                                  "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}]
                message["content"] = parts
            if "reasoning_content" in message:
                message["reasoning"] = message.pop("reasoning_content")
        request.setdefault("add_generation_prompt", True)
        return request

    async def render_tokens(self, request):
        wire = self.wire_request(request)
        fields = ("messages", "tools", "add_generation_prompt", "continue_final_message",
                  "chat_template_kwargs", "mm_processor_kwargs")
        tokenization = await self.post("/tokenize", {k: wire[k] for k in fields if k in wire})
        images = []
        for message in wire["messages"]:
            for part in message.get("content", []):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                url = part["image_url"]["url"]
                if not url.startswith("data:image/png;base64,"):
                    raise ValueError("REPL vision accepts local PNG data only")
                png = base64.b64decode(url.split(",", 1)[1], validate=True)
                if png[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Invalid PNG")
                w, h = struct.unpack(">II", png[16:24])
                images.append(SimpleNamespace(vision_tokens=w*h // self.metadata["spatial_factor"]**2))
        if len(images) > self.max_images:
            raise ValueError("REPL image limit exceeded; original task images are never omitted")
        if images:
            vision.audit_tokenization(tokenization, images, self.metadata)
        return ChatPrompt(tokenization["tokens"], wire, tokenization)

    def with_ids(self, prompt, ids):
        return ChatPrompt(ids, prompt.request, prompt.tokenization)

    async def append_prompt(self, prompt, tokens):
        if not isinstance(prompt, ChatPrompt):
            raise TypeError("Multimodal continuation lost its image-bearing request")
        if self.token_transport:
            return self.with_ids(prompt, list(prompt) + list(tokens))
        suffix = (await self.post("/detokenize", dict(tokens=tokens)))["prompt"]
        request = deepcopy(prompt.request)
        if prompt.assistant_prefix is None:
            rendered = (await self.post("/detokenize", dict(tokens=list(prompt))))["prompt"]
            opened = repl.native_generation_prompt(rendered).split("<|im_start|>assistant\n", 1)[1]
            raw = opened + suffix
        else:
            request["messages"].pop()  # Replace the existing partial assistant, not duplicate it.
            raw = prompt.assistant_prefix + suffix
        reasoning, marker, content = raw.partition("</think>")
        if not marker:
            raise ValueError("Multimodal continuation requires a closed thinking block")
        reasoning = reasoning.removeprefix("<think>").strip()
        request["messages"].append(dict(role="assistant", reasoning=reasoning, content=content))
        request.update(add_generation_prompt=False, continue_final_message=True)
        counted = await self.render_tokens(request)
        counted.assistant_prefix = raw
        return counted

    async def check_transport(self, token_id):
        """Fail at startup, not inside every trajectory, when the exact-id endpoints are disabled."""
        response = await self.client.post(self.url + "/v1/completions/render", timeout=120,
                                          json=dict(model=self.model, prompt=[token_id], max_tokens=1))
        if response.status_code != 200:
            raise RuntimeError(f"Exact-id vision transport needs vLLM's scale-out endpoints (--enable-scale-out); "
                               f"/v1/completions/render returned HTTP {response.status_code}")

    async def generate_request(self, payload):
        """/inference/v1/generate body: one placeholder per image, which the server expands again."""
        prompt, image = payload["prompt"], self.metadata["image_token_id"]
        ids = [t for i, t in enumerate(prompt) if t != image or not i or prompt[i - 1] != image]
        urls = [p["image_url"]["url"] for m in prompt.request["messages"] for p in m.get("content", [])
                if isinstance(p, dict) and p.get("type") == "image_url"]
        if ids.count(image) != len(urls) or ids.count(self.metadata["vision_start_token_id"]) != len(urls):
            raise ValueError("Exact-id prompt does not hold one image placeholder per request image")
        # The sampling parameters /v1/completions derives from this payload (generation config defaults,
        # vllm_xargs); generate applies none itself. They do not depend on the prompt.
        body = {k: v for k, v in payload.items() if k not in ("prompt", "cache_salt")}
        (rendered,) = await self.post("/v1/completions/render", dict(body, prompt=ids[:1]))
        sampling = dict(rendered["sampling_params"])
        # /render omits SamplingParams defaults, including max_tokens=16.
        # /generate deliberately treats an omitted cap as the server default
        # instead. Restore the serialized default, not the original request
        # cap: the renderer may already have clipped that cap to 16.
        sampling.setdefault("max_tokens", 16)
        request = dict(token_ids=ids, content_parts=[dict(type="image_url", url=url) for url in urls],
                       sampling_params=sampling)
        if payload.get("cache_salt") is not None:
            request["cache_salt"] = payload["cache_salt"]
        return request

    async def complete(self, payload, output, timeout, **kwargs):
        prompt = payload["prompt"]
        if not isinstance(prompt, ChatPrompt):
            raise TypeError("Multimodal REPL requires a counted ChatPrompt; token-only requests drop images")
        if self.token_transport:
            result = await super().complete(payload, output, timeout,
                                            generate=await self.generate_request(payload), **kwargs)
        else:
            result = await super().complete(payload, output, timeout, chat_request=prompt.request, **kwargs)
        # IDs can be identical for different image pixels. Include the actual request identity in the audit.
        (output / "multimodal_audit.json").write_text(json.dumps({
            "endpoint": "/inference/v1/generate" if self.token_transport else "/v1/chat/completions",
            "request_sha256": hashlib.sha256(json.dumps(prompt.request, sort_keys=True).encode()).hexdigest(),
            "images": sum(p.get("type") == "image_url" for m in prompt.request["messages"]
                          for p in m.get("content", []) if isinstance(p, dict)),
            "counted_prompt_tokens": len(prompt), "continued_assistant": prompt.assistant_prefix is not None,
        }, indent=2) + "\n")
        return result
