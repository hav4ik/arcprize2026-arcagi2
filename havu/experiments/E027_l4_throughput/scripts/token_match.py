"""E034 test B: do the server's generated token ids equal the ids of the same text re-tokenized inside the next prompt,
and are generated tokens reused by the prefix cache on the next turn? Real E032 prompts; one request at a time."""
import json, sys, glob, urllib.request
B = "http://127.0.0.1:8001"
def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(B + path, data=json.dumps(body).encode(),
                                                                   headers={"Content-Type": "application/json"}), timeout=3600))
def tok(text): return post("/tokenize", {"model": "qwen3.8-27b", "prompt": text, "add_special_tokens": False})["tokens"]
out = []
traces = sorted(glob.glob("/root/arc/e032_traces/trace_*.txt"))
for f in traces[:6]:
    t = open(f).read()
    i = t.find("<|im_start|>assistant\n"); P0 = t[:i + len("<|im_start|>assistant\n")]
    for n, ignore in ((3000, False), (4000, True)):
        r = post("/v1/completions", {"model": "qwen3.8-27b", "prompt": P0, "max_tokens": n, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                     "logprobs": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False,
                                     "stop": [] if ignore else ["</tool_call>"], "ignore_eos": ignore, "include_stop_str_in_output": True})
        c = r["choices"][0]; text = c["text"]
        gen_ids = [int(x.split(":", 1)[1]) for x in c["logprobs"]["tokens"]]
        p_ids = tok(P0); full_ids = tok(P0 + text)
        match = full_ids == p_ids + gen_ids
        first_diff = next((k for k in range(min(len(full_ids), len(p_ids) + len(gen_ids))) if full_ids[k] != (p_ids + gen_ids)[k]), None)
        nxt = P0 + text + "<|im_end|>\n<|im_start|>user\n<tool_response>\n[completed]\nok\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        r2 = post("/v1/completions", {"model": "qwen3.8-27b", "prompt": nxt, "max_tokens": 1})
        u = r2["usage"]; cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        row = dict(trace=f.rsplit("/", 1)[1], ignore_eos=ignore, prompt_tokens=len(p_ids), generated=len(gen_ids),
                   retokenized_equal=match, first_diff=None if first_diff is None else first_diff - len(p_ids),
                   diff_context=None if first_diff is None else [full_ids[first_diff:first_diff + 4], (p_ids + gen_ids)[first_diff:first_diff + 4]],
                   next_prompt_tokens=u["prompt_tokens"], cached_next=cached,
                   cached_if_prompt_only=(len(p_ids) // 1568) * 1568, cached_if_gen_reused=((len(p_ids) + len(gen_ids)) // 1568) * 1568)
        print(json.dumps(row), flush=True); out.append(row)
json.dump(out, open("/root/arc/runs/e034_cache/token_match.json", "w"), indent=1)
print("TOKEN_MATCH_DONE")
