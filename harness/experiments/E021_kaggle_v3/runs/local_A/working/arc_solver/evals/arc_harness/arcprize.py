"""ARC Prize public-eval prompt and answer parser (github.com/arcprize/arc-agi-benchmarking), verbatim."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"
TEMPLATE = (ROOT / "notes/prompts/arcprize_official_system_prompt.txt").read_text()


def build_prompt(train, test_input):
    ex = ""
    for i, pair in enumerate(train):
        ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(pair["input"]) + "\n\n"
        ex += "OUTPUT: \n\n" + json.dumps(pair["output"]) + "\n\n"
    return TEMPLATE.format(training_examples=ex, test_input=json.dumps(test_input))


def is_grid(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(r, list) for r in x)


def extract_from_boxed(s):
    m = re.search(r"\\boxed\{(.*?)\}", s, re.DOTALL)
    if m:
        try:
            g = json.loads(m.group(1).strip())
            return g if is_grid(g) else None
        except json.JSONDecodeError:
            return None
    return None


def backscan_json_parser(s):
    end = max(s.rfind("]"), s.rfind("}"))
    if end < 0:
        return None
    close = s[end]
    open_ = "[" if close == "]" else "{"
    depth = 1
    for i in range(end - 1, -1, -1):
        depth += (s[i] == close) - (s[i] == open_)
        if depth == 0:
            try:
                g = json.loads(s[i:end + 1])
                return g if is_grid(g) else None
            except json.JSONDecodeError:
                return None
    return None


def parse(s):
    return extract_from_boxed(s) or backscan_json_parser(s or "")


