"""Fetch safetensors headers of every .safetensors file in a HF repo via HTTP range requests (no weight download)."""
import json, os, struct, sys
from concurrent.futures import ThreadPoolExecutor
import requests
from huggingface_hub import HfApi, hf_hub_url

repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = os.environ.get("HF_TOKEN")
files = [f for f in HfApi().list_repo_files(repo, revision=rev) if f.endswith(".safetensors")]
H = {"Authorization": f"Bearer {tok}"} if tok else {}

def head(f):
    u = hf_hub_url(repo, f, revision=rev)
    r = requests.get(u, headers={**H, "Range": "bytes=0-7"}, allow_redirects=True, timeout=60)
    n = struct.unpack("<Q", r.content[:8])[0]
    r = requests.get(u, headers={**H, "Range": f"bytes=8-{7 + n}"}, allow_redirects=True, timeout=120)
    return f, json.loads(r.content)

with ThreadPoolExecutor(16) as ex:
    res = dict(ex.map(head, files))
json.dump(res, open(out, "w"))
print(len(res), "headers")
