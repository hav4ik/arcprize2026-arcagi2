"""Tool specs exactly as E025's DeepSeek harness sent them, in OpenAI form ({"type": "function", "function": ...}).

The wrapped form matters: the HF Qwen3.8 template and our Qwen harness (agent_qwen.render_system) both write
the wrapper into <tools>, so training must too. Snapshot of run_deepseek_tools.tools_for() with the E025
config (helpers + answer tool, grid mode); regenerate with --regen if the harness changes.
"""
import json
from pathlib import Path

SPEC = Path(__file__).parent / "e025_tools.json"


def e025_tools():
    return json.load(open(SPEC))


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="re-snapshot from evals/run_deepseek_tools.py (project .venv)")
    if ap.parse_args().regen:
        root = Path("/home/vu/Workspace/ARC-AGI-2")
        sys.path.insert(0, str(root / "evals"))
        sys.argv = [sys.argv[0]]
        import run_deepseek_tools as h
        jsonl = root / "evals/results/tools_persistent_grid_lean2roles_deepseek-flash_low_e025_arc2_train.jsonl"
        cfg = json.loads(open(jsonl).readline())["config"]
        json.dump(h.tools_for(argparse.Namespace(**cfg)), open(SPEC, "w"), indent=1)
    print(json.dumps([t["function"]["name"] for t in e025_tools()]))
