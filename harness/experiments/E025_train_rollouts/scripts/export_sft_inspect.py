"""Export E025 rollouts for SFT inspection: JSON + rendered/tokenized text under DeepSeek-V4.1 and Qwen3.8.

For each transcript writes, to --out:
  <name>.json                      meta row, tool specs, raw DeepSeek messages, converted Tinker messages
  <name>.<renderer>.txt            the exact rendered training sequence; trained tokens (weight 1) are
                                   wrapped in ⟦ ⟧ so masking is visible
  <name>.<renderer>.tokens.tsv     one token per line: index, id, weight, repr(piece)

Renderers come from tinker-cookbook (the code Tinker SFT uses), with
TrainOnWhat.ALL_ASSISTANT_MESSAGES on a single sequence (thinking kept on every turn).

Needs a venv with tinker-cookbook installed, e.g.
  python -I export_sft_inspect.py --tools e025_tools.json --out DIR TRACE.messages.json.gz ...
"""
import argparse
import gzip
import json
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer

RENDERERS = {  # name -> (tokenizer model, renderer name)
    "deepseekv4_1_low": ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseekv4_1_low_reasoning"),
    "qwen3_8_xhigh": ("Qwen/Qwen3.8-27B", "qwen3_8_xhigh_reasoning"),
}
RESULTS = Path("/home/vu/Workspace/ARC-AGI-2/evals/results")


def to_tinker(raw):
    """DeepSeek API messages -> tinker_cookbook Messages (thinking as ThinkingPart, tool calls as ToolCall)."""
    names = {}
    out = []
    for m in raw:
        if m["role"] == "assistant":
            parts = []
            if (m.get("reasoning_content") or "").strip():
                parts.append({"type": "thinking", "thinking": m["reasoning_content"]})
            if (m.get("content") or "").strip():
                parts.append({"type": "text", "text": m["content"]})
            msg = {"role": "assistant", "content": parts}
            if m.get("tool_calls"):
                msg["tool_calls"] = [ToolCall(id=c["id"], function=ToolCall.FunctionBody(
                    name=c["function"]["name"], arguments=c["function"]["arguments"])) for c in m["tool_calls"]]
                names.update({c["id"]: c["function"]["name"] for c in m["tool_calls"]})
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_call_id": m["tool_call_id"],
                        "name": names.get(m["tool_call_id"], "")})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def jsonable(msgs):
    return [{**m, "tool_calls": [c.model_dump() for c in m["tool_calls"]]} if "tool_calls" in m else m for m in msgs]


def find_row(trace):
    jsonl = RESULTS / (trace.parent.name + ".jsonl")
    for line in open(jsonl):
        row = json.loads(line)
        if Path(row.get("trace_path") or "").name == trace.name:
            return row
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--tools", type=Path, required=True, help="tool specs as sent to the API (tools_for)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tools = [t["function"] for t in json.load(open(args.tools))]
    renderers = {k: get_renderer(r, get_tokenizer(tok)) for k, (tok, r) in RENDERERS.items()}

    for trace in args.traces:
        raw = json.load(gzip.open(trace))["messages"]
        row = find_row(trace) or {}
        name = trace.name.removesuffix(".messages.json.gz")
        msgs = to_tinker(raw)
        summary = {}
        for key, rend in renderers.items():
            convo = rend.create_conversation_prefix_with_tools(tools) + msgs
            model_input, weights = rend.build_supervised_example(convo, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            ids, w = model_input.to_ints(), [int(x) for x in weights.tolist()]
            tok = rend.tokenizer
            pieces = [tok.decode([i]) for i in ids]
            text, cur = [], 0
            for p, wi in zip(pieces, w):
                if wi != cur:
                    text.append("⟦" if wi else "⟧")
                    cur = wi
                text.append(p)
            if cur:
                text.append("⟧")
            (args.out / f"{name}.{key}.txt").write_text("".join(text))
            with open(args.out / f"{name}.{key}.tokens.tsv", "w") as f:
                f.write("idx\tid\tweight\tpiece\n")
                for j, (i, p, wi) in enumerate(zip(ids, pieces, w)):
                    f.write(f"{j}\t{i}\t{wi}\t{p!r}\n")
            summary[key] = {"tokens": len(ids), "trained_tokens": sum(w)}
            print(f"{name} {key}: {len(ids)} tokens, {sum(w)} trained")
        meta = {k: v for k, v in row.items() if k not in ("content",)}
        json.dump({"meta": meta, "token_counts": summary, "tools": tools, "raw_messages": raw,
                   "tinker_messages": jsonable(msgs)},
                  open(args.out / f"{name}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
