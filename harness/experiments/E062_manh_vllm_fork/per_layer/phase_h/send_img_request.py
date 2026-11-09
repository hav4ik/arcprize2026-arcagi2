"""POST req_img.json and req_txt.json to the local server; report status, prompt_tokens, the image-token difference vs
the expected count, and the first part of the output."""
import json, sys, time, urllib.request, urllib.error
d = sys.argv[1]
exp = json.load(open(f"{d}/expect.json"))
res = {}
for name in ("req_txt", "req_img"):
    body = open(f"{d}/{name}.json", "rb").read()
    r = urllib.request.Request("http://127.0.0.1:8100/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        out = json.load(urllib.request.urlopen(r, timeout=600))
        msg = out["choices"][0]["message"]
        res[name] = {"ok": True, "s": round(time.time() - t0, 1), "usage": out.get("usage"),
                     "finish": out["choices"][0].get("finish_reason"),
                     "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:300],
                     "content": (msg.get("content") or "")[:200]}
    except urllib.error.HTTPError as e:
        res[name] = {"ok": False, "http": e.code, "body": e.read()[:600].decode(errors="replace")}
print(json.dumps(res, indent=1))
if res.get("req_img", {}).get("ok") and res.get("req_txt", {}).get("ok"):
    diff = res["req_img"]["usage"]["prompt_tokens"] - res["req_txt"]["usage"]["prompt_tokens"]
    print(f"image prompt tokens = {diff} (expected {exp['image_tokens_expected']} for {exp['n_images']} images; "
          f"per image {exp['per_image']})")
