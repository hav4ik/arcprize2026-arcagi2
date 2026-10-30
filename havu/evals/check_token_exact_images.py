"""Live check of --token-exact with images (E043 + E058 --grid-format text+image) on a DeepSeek-V4.1 vision server.

    .venv/bin/python evals/check_token_exact_images.py --base-url http://127.0.0.1:8000/v1 \
        --challenges data/arc-agi_evaluation_challenges.json [--task ID] [--tokens 64]

The server needs the vision weights, --enable-scale-out and --limit-mm-per-prompt '{"image": N}' (N >= the task's
grid count) instead of --language-model-only; run it idle. One text+image prompt goes three ways at temperature 0, each
under its own cache_salt (no prefix reuse between paths):
  chat   /v1/chat/completions with the messages: the server renders the prompt and attaches the images
  exact  turn_token_exact: /inference/v1/generate with one placeholder per image plus the images
  ids    /v1/completions with the same ids (the old --token-exact path: no pixels reach the model)
exact must run chat's prompt ids and sample chat's greedy tokens; ids is expected to diverge. A second exact turn
continues the history. Exit status 1 when exact does not match chat.
"""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_deepseek_arcprize import build_prompt
from run_deepseek_tools import load_tasks, parser, text_image_content, tools_for, turn_token_exact


def first_diff(a, b):
    """First index where the two id lists differ (None: equal)."""
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return None if a == b else i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="dsv41", help="served model name")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--task", help="task id (default: the first task)")
    ap.add_argument("--tokens", type=int, default=64, help="greedy tokens compared per path")
    ap.add_argument("--chat-template-kwargs", default='{"thinking": true, "reasoning_effort": 50}')
    a = ap.parse_args()
    tasks = load_tasks(a.challenges)
    tid = a.task or next(iter(tasks))
    train, test_input = tasks[tid]["train"], tasks[tid]["test"][0]["input"]
    content = text_image_content(build_prompt(train, test_input), train, [test_input])
    kw, run = json.loads(a.chat_template_kwargs), uuid4().hex[:12]
    hargs = parser().parse_args(["--base-url", a.base_url, "--model", a.model, "--token-exact", "--extra-body",
                                 json.dumps({"chat_template_kwargs": kw, "temperature": 0.0,
                                             "cache_salt": f"check-exact-{run}"})])
    client = OpenAI(api_key="none", base_url=a.base_url, timeout=3600, max_retries=0)
    messages = [{"role": "user", "content": content}]

    chat = client.chat.completions.create(
        model=a.model, messages=messages, tools=tools_for(hargs), max_tokens=a.tokens, temperature=0.0,
        extra_body={"chat_template_kwargs": kw, "return_token_ids": True, "skip_special_tokens": False,
                    "cache_salt": f"check-chat-{run}"})
    chat_prompt, chat_out = chat.model_extra["prompt_token_ids"], chat.choices[0].model_extra["token_ids"]

    tok = {"ids": None, "n": 0, "prev_len": 0, "prev_sampled": [], "resync": False, "turns": 0, "breaks": 0,
           "rerenders": 0, "resyncs": 0, "parse_fallbacks": 0}
    assistant, usage, finish = turn_token_exact(client, hargs, messages, a.tokens, tok)
    exact_prompt, exact_out = tok["ids"][:tok["prev_len"]], tok["prev_sampled"]

    old = client.completions.create(model=a.model, prompt=exact_prompt, max_tokens=a.tokens, temperature=0.0,
                                    extra_body={"return_token_ids": True, "skip_special_tokens": False,
                                                "cache_salt": f"check-ids-{run}"})
    old_out = old.choices[0].model_extra["token_ids"]

    messages += [assistant, {"role": "user", "content": "Continue."}]
    _, usage2, finish2 = turn_token_exact(client, hargs, messages, 8, tok)

    ok = exact_prompt == chat_prompt and exact_out == chat_out and usage["prompt_tokens"] == len(chat_prompt)
    report = {
        "task": tid, "images": sum(p["type"] == "image_url" for p in content), "prompt_ids": len(chat_prompt),
        "exact_prompt_matches_chat": exact_prompt == chat_prompt, "exact_served_prompt_tokens": usage["prompt_tokens"],
        "greedy_tokens": a.tokens, "exact_vs_chat_first_diff": first_diff(exact_out, chat_out),
        "ids_only_vs_chat_first_diff": first_diff(old_out, chat_out), "exact_finish": finish,
        "turn2": {"prompt_tokens": usage2["prompt_tokens"], "finish": finish2,
                  "cached_tokens": (usage2.get("prompt_tokens_details") or {}).get("cached_tokens")},
        "token_exact": {k: v for k, v in tok.items() if k not in ("ids", "n", "prev_len", "prev_sampled", "resync")},
        "ok": ok,
    }
    print(json.dumps(report, indent=1))
    if report["ids_only_vs_chat_first_diff"] is None:
        print("note: the ids-only path matched chat too; the images may not change these first tokens", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
