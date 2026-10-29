"""Build one E058-style text+image chat request (the harness's own prompt + text_image_content, --image-cell 0) for an
ARC eval task, plus a twin with the image parts removed, and the expected image-token count per grid
(grid_image.deepseek_tokens). Writes req_img.json, req_txt.json, expect.json into OUT."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/vu/Workspace/ARC-AGI-2")
sys.path.insert(0, str(ROOT / "evals"))
import run_deepseek_tools as H  # noqa: E402
from arc_harness.grid_image import deepseek_tokens  # noqa: E402

out = Path(sys.argv[1])
task_id = sys.argv[2] if len(sys.argv) > 2 else None
out.mkdir(parents=True, exist_ok=True)
tasks = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))
task_id = task_id or next(iter(tasks))
task = tasks[task_id]
args = H.parser().parse_args(["--mode", "grid", "--prompt", "lean2sub", "--answer-tool", "--grid-format", "text+image",
                              "--image-cell", "0", "--effort-header", "minimal", "--sandbox-url", "x",
                              "--base-url", "http://127.0.0.1:8100/v1"])
test_input = task["test"][0]["input"]
prompt = H.build_prompt(task["train"], test_input) + H.PROMPT_SUFFIXES[args.prompt]
content = H.text_image_content(prompt, task["train"], [test_input], 0, "deepseek", H.IMAGE_NOTES[""])
system = {"role": "system", "content": H.EFFORT_HEADERS["minimal"]}
tools = H.tools_for(args)
extra = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "temperature": 1.0, "top_p": 1.0}
grids = [g for p in task["train"] for g in (p["input"], p["output"])] + [test_input]
req = {"model": "dsv41", "messages": [system, {"role": "user", "content": content}], "tools": tools,
       "max_tokens": 64, **extra}
txt = {**req, "messages": [system, {"role": "user", "content": [p for p in content if p["type"] == "text"]}]}
json.dump(req, open(out / "req_img.json", "w"))
json.dump(txt, open(out / "req_txt.json", "w"))
exp = {"task": task_id, "n_images": len(grids), "image_tokens_expected": sum(deepseek_tokens(g) for g in grids),
       "per_image": [deepseek_tokens(g) for g in grids], "grid_shapes": [[len(g), len(g[0])] for g in grids]}
json.dump(exp, open(out / "expect.json", "w"), indent=1)
print(json.dumps({k: exp[k] for k in ("task", "n_images", "image_tokens_expected")}),
      "request bytes", (out / "req_img.json").stat().st_size)
