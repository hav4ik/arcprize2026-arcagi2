"""Tokenization stability of the agent's multi-turn prompts (prefix caching needs token-level prefixes).
For each trace (full transcript text of one attempt), the k-th request prompt is the text up to the k-th
"<|im_start|>assistant\\n<think>\\n". Checks: (1) tokens(prompt_k) is a prefix of tokens(prompt_k+1); (2) where not,
the first differing token position relative to the end of prompt_k. Prints a JSON summary and token ids hash so two
tokenizer versions can be compared. usage: python tok_check.py <model dir> <trace files...>"""
import hashlib, json, sys
from transformers import AutoTokenizer
import transformers

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
GEN = "<|im_start|>assistant\n<think>\n"
res = {"transformers": transformers.__version__, "turns": 0, "prefix_ok": 0, "breaks": [], "hash": hashlib.sha1()}
for f in sys.argv[2:]:
    text = open(f).read()
    cuts, i = [], 0
    while True:
        j = text.find(GEN, i)
        if j < 0:
            break
        cuts.append(j + len(GEN)); i = j + 1
    prev = None
    for c in cuts:
        ids = tok(text[:c], add_special_tokens=False)["input_ids"]
        res["hash"].update(json.dumps(ids).encode())
        if prev is not None:
            res["turns"] += 1
            if ids[:len(prev)] == prev:
                res["prefix_ok"] += 1
            else:
                d = next(k for k, (a, b) in enumerate(zip(prev, ids)) if a != b) if any(a != b for a, b in zip(prev, ids)) else min(len(prev), len(ids))
                res["breaks"].append({"file": f.split("/")[-1], "prev_len": len(prev), "first_diff": d, "tokens_before_end": len(prev) - d,
                                      "context": tok.decode(prev[max(0, d - 3):d + 3])})
        prev = ids
res["hash"] = res["hash"].hexdigest()
res["n_breaks"] = len(res["breaks"]); res["breaks"] = res["breaks"][:12]
print(json.dumps(res, indent=1))
