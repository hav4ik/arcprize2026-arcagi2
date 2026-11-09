#!/usr/bin/env python3
"""E066 test (CPU): the chat requests of replay_mm.py / nll_eval_mm.py make the vLLM fork consume exactly the offline
token stream of tokenize_image_rollouts.py.

Emulates the fork's chat path on each request body: _normalize_messages (copied below from the fork's
vllm/tokenizers/deepseek_v41.py), tools on the system message, the fork's own encode_messages
(vllm/tokenizers/deepseek_v41_encoding.py, loaded from the wheel; defaults as the server calls it), tokenizer.json,
and the image-span expansion (the fork's mm_preprocess.py uses the same sizing as the model's image_processor.py).

  test_mm_requests.py --tok DIR [...] --parquet e065_image.parquet --model-dir DIR --fork-encoding FILE [--n 40]
"""
import argparse
import copy
import importlib.util
import json
import random

import numpy as np
from tokenizers import Tokenizer

from mm_common import KEY, chat_body, load_index, load_rows
from tokenize_image_rollouts import IMAGE_TOKEN_ID, num_image_tokens, plan_image_grid, png_size, vision_args


def normalize_messages(messages, placeholder):  # the fork's _normalize_messages
    result = [dict(message) for message in copy.deepcopy(messages)]
    for message in result:
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type == "text":
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(placeholder)
                else:
                    raise ValueError(f"unsupported part {part_type!r}")
            message["content"] = "\n\n".join(parts)
    return result


def server_stream(body, enc, tok, vargs):
    conv = normalize_messages(body["messages"], enc.IMAGE_PLACEHOLDER)
    system = next((m for m in conv if m["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": ""}
        conv.insert(0, system)
    system["tools"] = body["tools"]
    kw = body["chat_template_kwargs"]
    prompt = enc.encode_messages(conv, thinking_mode="thinking" if kw.get("thinking", True) else "chat",
                                 drop_thinking=kw.get("drop_thinking", True), reasoning_effort=kw["reasoning_effort"])
    ids = np.asarray(tok.encode(prompt, add_special_tokens=False).ids, np.int64)
    urls = [p["image_url"]["url"] for m in body["messages"] if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]
    sizes = []
    for u in urls:
        w, h = png_size(u)
        n_h, n_w, _, _ = plan_image_grid(w, h, vargs)
        sizes.append(num_image_tokens(n_h, n_w))
    pos = np.flatnonzero(ids == IMAGE_TOKEN_ID)
    assert len(pos) == len(sizes), (len(pos), len(sizes))
    reps = np.ones(len(ids), np.int64)
    reps[pos] = sizes
    return np.repeat(ids, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fork-encoding", required=True, help="vllm/tokenizers/deepseek_v41_encoding.py of the fork")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("fork_v41_encoding", a.fork_encoding)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    tok = Tokenizer.from_file(f"{a.model_dir}/tokenizer.json")
    vargs = vision_args(a.model_dir)
    recs = load_index(a.tok)
    random.Random(0).shuffle(recs)
    recs = recs[: a.n]
    rows = load_rows(a.parquet)
    ok = n_tok = n_img = 0
    for r in recs:
        body = chat_body(rows[tuple(r[k] for k in KEY)], "dsv41")
        got = server_stream(body, enc, tok, vargs)
        want = np.fromfile(f"{r['dir']}/tokens.u32", np.uint32)[r["offset"]: r["offset"] + r["length"]].astype(np.int64)
        same = len(got) == len(want) and np.array_equal(got, want)
        ok += same
        n_tok += len(want)
        n_img += r["n_img"]
        if not same:
            k = int(np.flatnonzero(got[: min(len(got), len(want))] != want[: min(len(got), len(want))])[0]) \
                if not np.array_equal(got[: min(len(got), len(want))], want[: min(len(got), len(want))]) \
                else min(len(got), len(want))
            print(f"MISMATCH {r['task_id']}: server {len(got)} vs offline {len(want)} tokens, first diff at {k}")
    print(json.dumps({"requests": len(recs), "identical": ok, "tokens": n_tok, "image_tokens": n_img}))
    assert ok == len(recs), "server-side rendering differs from the offline stream"
    print("all requests render to the offline stream")


if __name__ == "__main__":
    main()
