"""E028: check that an OpenAI-compatible server builds exactly the training prompt for harness conversations.

For a few E025 rollouts, render the first k messages (k = 1: system+tools+task; k = 3: + assistant turn with
thinking and a tool call + tool result) with the training renderer as a generation prompt, and compare the token
count with usage.prompt_tokens the server reports for the same messages sent through /v1/chat/completions
(tools = E025 tool specs, chat_template_kwargs.reasoning_effort = xhigh, max_tokens = 1).

  python -I format_check.py http://127.0.0.1:31002/v1 sft
"""
import gzip
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import RENDERER, RESULTS, TOKENIZER, PREFIX, to_tinker  # noqa: E402
from tools_spec import e025_tools  # noqa: E402


def server_prompt_tokens(url, model, messages):
    body = {"model": model, "messages": messages, "tools": e025_tools(), "max_tokens": 1,
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["usage"]["prompt_tokens"]


def api_messages(raw):
    """Messages as the DeepSeek harness sends them back to the server (it re-sends reasoning_content)."""
    out = []
    for m in raw:
        m = {k: v for k, v in m.items() if k in ("role", "content", "reasoning_content", "tool_calls", "tool_call_id")}
        out.append(m)
    return out


def main():
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    url, model = sys.argv[1], sys.argv[2]
    rend = get_renderer(RENDERER, get_tokenizer(TOKENIZER))
    man = [json.loads(l) for l in open("/home/vu/Workspace/ARC-AGI-2/data/sft/e028/train_manifest.jsonl")][:6]
    ok = True
    for m in man:
        raw = json.load(gzip.open(RESULTS / f"{PREFIX}{m['subset']}" / m["trace"]))["messages"]
        for k in (1, 3):
            if len(raw) < k:
                continue
            convo = rend.create_conversation_prefix_with_tools(e025_tools()) + to_tinker(raw[:k])
            want = rend.build_generation_prompt(convo).length
            got = server_prompt_tokens(url, model, api_messages(raw[:k]))
            ok &= want == got
            print(f"{m['task_id']:<22} k={k}: training render {want:6d}  server {got:6d}  {'OK' if want == got else 'MISMATCH'}")
    print("ALL MATCH" if ok else "MISMATCHES FOUND")


if __name__ == "__main__":
    main()
