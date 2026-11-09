"""E033: check that a vision server keeps one LLM visual token per ARC cell (no resizing).

Sends prompts with grid images of several sizes to an OpenAI-compatible server and compares the reported prompt
tokens with a text-only baseline: the difference must equal rows*cols (plus the constant per-image marker tokens).
Usage: .venv/bin/python data_tools/check_image_tokens.py http://127.0.0.1:18301/v1 [cell=32]
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from arc_harness.grid_image import data_uri, visual_tokens  # noqa: E402

URL, CELL = sys.argv[1].rstrip("/"), int(sys.argv[2]) if len(sys.argv) > 2 else 32


def prompt_tokens(content):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}], "max_tokens": 1,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["usage"]["prompt_tokens"]


base = prompt_tokens([{"type": "text", "text": "Grid:"}])
rows, ok = [], True
for h, w in ((1, 1), (3, 3), (2, 7), (5, 11), (10, 10), (13, 29), (30, 30)):
    grid = [[(r * 3 + c) % 10 for c in range(w)] for r in range(h)]
    n = prompt_tokens([{"type": "text", "text": "Grid:"}, {"type": "image_url", "image_url": {"url": data_uri(grid, CELL)}}])
    rows.append((h, w, n - base, visual_tokens(grid, CELL)))
markers = {d - e for _, _, d, e in rows}
print(f"text-only prompt {base} tokens; per-image extra tokens beyond rows*cols: {sorted(markers)}")
for h, w, d, e in rows:
    print(f"  {h:2d}x{w:<2d}: +{d} prompt tokens, expected {e} cell tokens")
ok = len(markers) == 1 and min(markers) >= 0
print("ALIGNED: one visual token per cell" if ok else "NOT ALIGNED: the processor resized some images")
sys.exit(0 if ok else 1)
