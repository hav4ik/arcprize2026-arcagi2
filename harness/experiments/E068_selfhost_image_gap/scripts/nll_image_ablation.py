#!/usr/bin/env python3
"""E068: does the self-hosted vision path deliver what the API's model saw? E066's teacher-forced NLL of the API's image
rollouts (nll_eval_mm.py, chat path, prompt_logprobs) with the images replaced, everything else unchanged.

  --images real   E066's request as is
  --images grey   every image -> a uniform (127,127,127) PNG of the same pixel size: same resize plan, same token
                  count and ids (all 129264), so the stream check still holds and the runs pair position by position
Adds per-rollout NLL of the first assistant turn's generated tokens (`n_t1`, `nll_t1`), where the API's reasoning
describes the pictures most. All other options are nll_eval_mm.py's; run both modes with the same --seed/--n-seq.

  nll_image_ablation.py --images grey --tok ... --parquet ... --per-task --n-seq 32 --tag full_grey --out ab.jsonl
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "E066_dsv41_image_pruning/scripts"))
import nll_eval_mm as N  # noqa: E402

MODE = "real"
if "--images" in sys.argv:
    i = sys.argv.index("--images")
    MODE = sys.argv[i + 1]
    del sys.argv[i:i + 2]
assert MODE in ("real", "grey"), MODE


def grey_url(url):
    header, _, payload = url.partition(",")
    with Image.open(io.BytesIO(base64.b64decode(payload))) as im:
        size = im.size
    buf = io.BytesIO()
    Image.new("RGB", size, (127, 127, 127)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def replace_images(messages):
    n = 0
    for m in messages:
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "image_url":
                    p["image_url"]["url"] = grey_url(p["image_url"]["url"])
                    n += 1
    return n


_chat_body = N.chat_body


def chat_body(row, model, effort=50, **extra):
    body = _chat_body(row, model, effort, **extra)
    if MODE == "grey":
        assert replace_images(body["messages"]) > 0
    return body




def score_t1(server, body, ids, roles):
    import requests
    r = requests.post(f"{server}/v1/chat/completions", json=body, timeout=7200)
    r.raise_for_status()
    plp = r.json()["prompt_logprobs"]
    if len(plp) != len(ids):
        raise SystemExit(f"server prompt has {len(plp)} positions, offline stream {len(ids)}")
    lp = np.zeros(len(ids))
    for t in range(1, len(ids)):
        d = plp[t]
        e = d.get(str(int(ids[t]))) or d.get(int(ids[t]))
        if e is None:
            raise SystemExit(f"position {t}: offline token {int(ids[t])} not in the server's prompt logprobs")
        lp[t] = e["logprob"]
    gen = roles == 1
    first = int(np.argmax(gen)) if gen.any() else len(ids)
    end = first
    while end < len(ids) and gen[end]:
        end += 1
    res = {"n_gen": int(gen[1:].sum()), "nll_gen": float(-lp[1:][gen[1:]].sum()),
           "n_t1": end - first, "nll_t1": float(-lp[first:end].sum()),
           "n_all": len(ids) - 1, "nll_all": float(-lp[1:].sum()),
           "top1_gen": 0, "n_blk": 0, "nll_blk": 0.0, "top1_blk": 0}
    return res


N.chat_body = chat_body
N.score = score_t1

if __name__ == "__main__":
    N.main()
