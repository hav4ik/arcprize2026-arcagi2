"""E058 perception probe: can DeepSeek V4.1-Flash-1017 (or Qwen3.8 Flash via OpenRouter, PROBE_MODEL=qwen) read an
ARC grid from a token-aligned image?

20 eval grids (training inputs/outputs, varied sizes; fixed seed), each drawn by grid_image.deepseek_cell (one cell =
k x k 42-px tiles, no API rescale), transcribed as digit rows with thinking disabled and with thinking at low effort.
Reports shape accuracy, cell accuracy, exact grids, and whether the API's image token count matches the prediction
(i.e. the image was not rescaled). Usage: perception_probe.py [n=20]
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
from openai import OpenAI  # noqa: E402

from arc_harness.digit_grid_parse import parse_digit_rows  # noqa: E402
from arc_harness.grid_image import LEGEND, aligned_cell, data_uri, deepseek_tokens, qwen_tokens  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
tasks = json.loads((ROOT / "data/arc-agi_evaluation_challenges.json").read_text())
random.seed(58)
grids = []
for tid in random.sample(sorted(tasks), N):
    pair = random.choice(tasks[tid]["train"])
    grids.append((tid, random.choice(["input", "output"]), pair[random.choice(["input", "output"])]))
QWEN = os.environ.get("PROBE_MODEL") == "qwen"
if QWEN:
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=1800)
    MODEL, PROFILE, TOKENS = "qwen/qwen3.8-flash", "qwen", qwen_tokens
else:
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com", timeout=1800)
    MODEL, PROFILE, TOKENS = "deepseek-flash", "deepseek", deepseek_tokens


def extra_body(think):
    if QWEN:
        return {"provider": {"order": ["Alibaba"], "allow_fallbacks": False}, "reasoning": {"enabled": think}}
    return {"thinking": {"type": "enabled"}, "reasoning_effort": "low"} if think else {"thinking": {"type": "disabled"}}
PROMPT = ("The image shows one ARC grid: each cell is a flat square of one colour, with no lines between cells. "
          f"{LEGEND} Transcribe the grid exactly, one line per row, digits separated by spaces. "
          "Reply with only the digit rows.")


def base_tokens():
    r = client.chat.completions.create(model=MODEL, max_tokens=16, extra_body=extra_body(False),
                                       messages=[{"role": "user", "content": PROMPT}])
    return r.usage.prompt_tokens


def run(item, mode, base):
    tid, side, g = item
    for i in range(4):  # Alibaba rate-limits; retry API errors
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=32000, extra_body=extra_body(mode != "nothink"),
                messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                       {"type": "image_url", "image_url": {"url": data_uri(g, aligned_cell(g, PROFILE))}}]}])
            break
        except Exception as exc:
            if i == 3:
                return dict(task=tid, h=len(g), w=len(g[0]), mode=mode, shape_ok=False, exact=False, cell_acc=0,
                            image_tokens=-1, predicted_tokens=TOKENS(g), out_tokens=0, error=type(exc).__name__)
            __import__("time").sleep(10)
    pred = parse_digit_rows(r.choices[0].message.content) or []
    shape_ok = len(pred) == len(g) and all(len(row) == len(g[0]) for row in pred)
    cells = sum(a == b for pr, gr in zip(pred, g) for a, b in zip(pr, gr)) if pred else 0
    return dict(task=tid, h=len(g), w=len(g[0]), mode=mode, shape_ok=shape_ok, exact=pred == g,
                cell_acc=cells / (len(g) * len(g[0])), image_tokens=r.usage.prompt_tokens - base,
                predicted_tokens=TOKENS(g), out_tokens=r.usage.completion_tokens)


base = base_tokens()
with ThreadPoolExecutor(8 if QWEN else 16) as pool:
    rows = list(pool.map(lambda a: run(*a, base), [(g, m) for m in ("nothink", "low") for g in grids]))
out = ROOT / f"experiments/E058_deepseek_vision/perception_probe{'_qwenflash' if QWEN else ''}.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
for mode in ("nothink", "low"):
    rs = [r for r in rows if r["mode"] == mode]
    print(f"{mode:8s} shape ok {sum(r['shape_ok'] for r in rs)}/{len(rs)} | exact {sum(r['exact'] for r in rs)}/{len(rs)} | "
          f"mean cell acc {sum(r['cell_acc'] for r in rs) / len(rs):.1%} | out tokens mean {sum(r['out_tokens'] for r in rs) / len(rs):.0f}")
aligned = sum(r["image_tokens"] == r["predicted_tokens"] for r in rows)
print(f"image tokens == prediction (no rescale) for {aligned}/{len(rows)} calls")
for r in sorted(rows, key=lambda r: (r["h"] * r["w"], r["mode"])):
    print(f"  {r['task']} {r['h']:2d}x{r['w']:<2d} {r['mode']:7s} shape={'Y' if r['shape_ok'] else 'n'} "
          f"cells={r['cell_acc']:.0%} tokens {r['image_tokens']}/{r['predicted_tokens']}")
