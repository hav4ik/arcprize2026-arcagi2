#!/bin/bash
# E042: one greedy completion with logprobs; checks tokens come back and logprobs are finite.
curl -s -m 600 localhost:${PORT:-8100}/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv41-e64", "prompt": "The quick brown fox jumps over the lazy dog. Count to ten:",
  "max_tokens": 48, "temperature": 0, "ignore_eos": true, "logprobs": 1, "return_token_ids": true}' |
/root/dsv41/venv/bin/python -I -c '
import json, math, sys
d = json.load(sys.stdin); c = d["choices"][0]; lp = c["logprobs"]["token_logprobs"]
print("usage", d["usage"]); print("text", repr(c["text"][:200]))
print("token_ids", c.get("token_ids", [])[:48])
print("logprob min/max", min(lp), max(lp), "all finite", all(math.isfinite(x) for x in lp))'
