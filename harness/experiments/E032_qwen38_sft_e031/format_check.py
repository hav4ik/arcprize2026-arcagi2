"""E032: check that the HF chat template and an OpenAI-compatible server build exactly the training prompt for
E031-format conversations (system effort header + E031 tools + lean2sub task, then assistant/tool turns).

For a few E032 training rollouts, render the first k messages (k = 2: system + task; k = 4: + assistant turn with
thinking and a tool call + tool result) with the training renderer as a generation prompt, and compare:
1. token ids against the HF tokenizer's chat template (tools = E031 specs, reasoning_effort = xhigh), exactly;
2. (with a URL) the token count against usage.prompt_tokens reported by the server for the same messages.

  python -I format_check.py [URL MODEL]
"""
import gzip
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import RENDERER, RESULTS, TOKENIZER, PREFIX, TOOLS, to_tinker  # noqa: E402


def server_prompt_tokens(url, model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 1,
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["usage"]["prompt_tokens"]


def api_messages(raw):
    """Messages as the DeepSeek harness sends them back to the server (it re-sends reasoning_content)."""
    return [{k: v for k, v in m.items() if k in ("role", "content", "reasoning_content", "tool_calls", "tool_call_id")}
            for m in raw]


def hf_messages(raw):
    """The same messages for the HF template: tool-call arguments as dicts, as the server passes them."""
    out = []
    for m in api_messages(raw):
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def main():
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    url, model = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (None, None)
    tok = get_tokenizer(TOKENIZER)
    rend = get_renderer(RENDERER, tok)
    man = [json.loads(l) for l in open("/home/vu/Workspace/ARC-AGI-2/data/sft/e032/train_manifest.jsonl")][:6]
    ok = True
    for m in man:
        raw = json.load(gzip.open(RESULTS / f"{PREFIX}{m['subset']}" / m["trace"]))["messages"]
        assert raw[0]["role"] == "system"
        for k in (2, 4):
            if len(raw) < k:
                continue
            convo = rend.create_conversation_prefix_with_tools(TOOLS, system_prompt=raw[0]["content"]) + to_tinker(raw[1:k])
            want = rend.build_generation_prompt(convo).to_ints()
            hf = tok.apply_chat_template(hf_messages(raw[:k]), tools=TOOLS, add_generation_prompt=True, tokenize=True,
                                         reasoning_effort="xhigh")
            hf = list(hf["input_ids"] if hasattr(hf, "keys") else hf)
            same = hf == want
            line = f"{m['task_id']:<22} k={k}: training {len(want):6d}  HF template {len(hf):6d} {'IDENTICAL' if same else 'DIFFERENT'}"
            if not same:
                i = next((j for j, (a, b) in enumerate(zip(hf, want)) if a != b), min(len(hf), len(want)))
                line += f" (first diff at {i}: HF {tok.decode(hf[i:i + 12])!r} vs training {tok.decode(want[i:i + 12])!r})"
            ok &= same
            if url:
                got = server_prompt_tokens(url, model, api_messages(raw[:k]))
                ok &= got == len(want)
                line += f"  server {got:6d} {'OK' if got == len(want) else 'MISMATCH'}"
            print(line)
    print("ALL MATCH" if ok else "MISMATCHES FOUND")


if __name__ == "__main__":
    main()
