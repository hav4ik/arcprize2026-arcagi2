#!/usr/bin/env python3
"""E066: exact DeepSeek-V4.1 token sequences for tool rollouts with images (E065, E058 harness).

Same output format as E043's tokenize_rollouts.py (tokens.u32, roles.u8, index.jsonl), with:
- tools: rebuilt by the harness's own tools_for() from each rollout's recorded config (multi-test changes the text);
- images: encoding.py renders each image block as one `<｜deepseek_image｜>` (id 129264); each is expanded here to
  the model's image span, n_llm_h * (n_llm_w + 1) + 2 positions that all carry id 129264 (inference/image_processor.py:
  IMAGE_START, rows of IMAGE + IMAGE_NEW_LINE, IMAGE_END). Image positions have role 0; find them as ids == 129264.
  The span size comes from the PNG size through a verbatim port of image_processor.py's resize plan.
- index.jsonl adds n_img (image positions), n_images, n_pairs, batch, parent, stop_reason.

The vLLM fork serves the same token ids: its chat path calls the same encode_messages (tools present => earlier
reasoning kept), and its processor expands placeholders the same way; replay_mm.py checks that on the box.

  tokenize_image_rollouts.py --model-dir DIR --parquet e065_image.parquet --out data/tok_img/shard0 --shard 0/8
"""
import argparse
import base64
import json
import math
import struct
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/E043_dsv41_real_pruning/scripts"))
sys.path.insert(0, str(ROOT / "evals"))
from tokenize_rollouts import load_encoding  # noqa: E402

IMAGE_TOKEN_ID = 129264


# --- verbatim port of inference/image_processor.py (pinned 1017 revision): token grid of an image -------------------
def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio)


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width, height, args):
    p = args.vision_patch_size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)
# ---------------------------------------------------------------------------------------------------------------------


def vision_args(model_dir):
    vc = json.load(open(Path(model_dir) / "config.json"))["vision_config"]
    return Namespace(vision_patch_size=vc["patch_size"], vision_downsample_ratio=vc["downsample_ratio"],
                     vision_min_pixels=vc["min_pixels"], vision_max_n_token=vc["max_image_tokens"],
                     vision_max_wh_ratio=vc.get("max_wh_ratio"))


def png_size(url):
    b = base64.b64decode(url.split(",", 1)[1][:64])
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "expected a PNG data URI"
    return struct.unpack(">II", b[16:24])  # width, height


def harness_tools(cfg):
    import run_deepseek_tools as h  # the harness's own schema builder (read-only import)
    ns = Namespace(mode=cfg["mode"], tool_timeout=cfg["tool_timeout"], answer_tool=cfg.get("answer_tool", False),
                   multi_test=cfg.get("multi_test", False), helpers=cfg.get("helpers", False),
                   show_tool=cfg.get("show_tool", False))
    return h.tools_for(ns)


def render_with_roles(enc, messages, effort, tools):
    """(text, spans, images): spans = char ranges of assistant renderings, images = records in prompt order."""
    messages = [dict(m) for m in messages]
    last_asst = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    messages = messages[: last_asst + 1]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    messages[0]["tools"] = tools
    merged = enc.merge_tool_messages(messages)
    merged = enc.sort_tool_results_by_call_order(merged)
    merged, images = enc.process_image_messages(merged)
    text, spans = enc.bos_token, []
    for idx, msg in enumerate(merged):
        piece = enc.render_message(idx, merged, thinking_mode="thinking", drop_thinking=False,
                                   reasoning_effort=effort)
        if msg["role"] == "assistant":
            spans.append((len(text), len(text) + len(piece)))
        text += piece
    ref = enc.encode_messages(messages, thinking_mode="thinking", reasoning_effort=effort)
    assert text == ref, "incremental render differs from encode_messages"
    return text, spans, images


def expand_images(ids, roles, images, vargs):
    pos = np.flatnonzero(ids == IMAGE_TOKEN_ID)
    assert len(pos) == len(images), (len(pos), len(images))
    sizes = []
    for rec in images:
        w, h = png_size(rec["url"])
        n_h, n_w, _, _ = plan_image_grid(w, h, vargs)
        sizes.append(num_image_tokens(n_h, n_w))
    reps = np.ones(len(ids), dtype=np.int64)
    reps[pos] = sizes
    return np.repeat(ids, reps), np.repeat(roles, reps), int(sum(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="needs encoding/encoding.py, tokenizer.json, config.json")
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", type=int, default=50, help="numeric reasoning effort prefix (API 'low' = 50)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: keep rows with global row index % n == i")
    args = ap.parse_args()

    enc = load_encoding(args.model_dir)
    tok = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    assert tok.token_to_id(enc.IMAGE_PLACEHOLDER) == IMAGE_TOKEN_ID
    vargs = vision_args(args.model_dir)
    tools_cache = {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ftok, frole = open(out / "tokens.u32", "wb"), open(out / "roles.u8", "wb")
    findex = open(out / "index.jsonl", "w")
    offset, n_rows, global_row = 0, 0, -1
    shard_i, shard_n = map(int, args.shard.split("/"))
    for path in args.parquet:
        for row in pq.read_table(path).to_pylist():
            if args.limit and n_rows >= args.limit:
                break
            global_row += 1
            if global_row % shard_n != shard_i:
                continue
            messages = json.loads(row["messages"])
            if not any(m["role"] == "assistant" for m in messages):
                continue
            if row.get("tools"):  # stored by make_e065_parquet.py
                tools = json.loads(row["tools"])
            else:
                cfg = json.loads(row["config"])
                key = (cfg["mode"], cfg["tool_timeout"], cfg.get("answer_tool"), cfg.get("multi_test"),
                       cfg.get("helpers"), cfg.get("show_tool"))
                if key not in tools_cache:
                    tools_cache[key] = harness_tools(cfg)
                tools = tools_cache[key]
            text, spans, images = render_with_roles(enc, messages, args.effort, tools)
            encoded = tok.encode(text, add_special_tokens=False)
            ids = np.asarray(encoded.ids, dtype=np.uint32)
            starts = np.asarray([s for s, _ in encoded.offsets], dtype=np.int64)
            roles = np.zeros(len(ids), dtype=np.uint8)
            for s, e in spans:
                roles[(starts >= s) & (starts < e)] = 1
            ids, roles, n_img = expand_images(ids, roles, images, vargs)
            events = json.loads(row["events"]) if row.get("events") else []
            usage = [(e["usage"].get("prompt_tokens"), e["usage"].get("completion_tokens"))
                     for e in events if isinstance(e, dict) and isinstance(e.get("usage"), dict)]
            rec = {"split": Path(path).stem, "task_id": row["task_id"], "pair": row["pair"],
                   "attempt": row["attempt"], "batch": row.get("batch"), "parent": row.get("parent"),
                   "n_pairs": row.get("n_pairs"), "correct": bool(row["correct"]), "train_pass": row.get("train_pass"),
                   "stop_reason": row.get("stop_reason"), "offset": offset, "length": int(len(ids)),
                   "n_gen": int(roles.sum()), "n_asst": len(spans), "n_img": n_img, "n_images": len(images),
                   "api_usage": usage}
            ftok.write(ids.astype(np.uint32).tobytes())
            frole.write(roles.astype(np.uint8).tobytes())
            findex.write(json.dumps(rec) + "\n")
            offset += len(ids)
            n_rows += 1
            if n_rows % 500 == 0:
                print(f"{n_rows} rollouts, {offset / 1e6:.1f}M tokens", file=sys.stderr, flush=True)
    for f in (ftok, frole, findex):
        f.close()
    print(json.dumps({"rollouts": n_rows, "tokens": offset}))


if __name__ == "__main__":
    main()
