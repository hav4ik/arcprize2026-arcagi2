"""Chat-path cache checks from attempt traces (out/dsv41_logs/traces/*.json.gz: messages + events with per-turn usage):
(a) turn >= 2 whose cached prefix covers the first user message's image span (cached >= turn-1 prompt - 512);
(c) turns >= 2 whose cached prefix stops inside the previous assistant turn (re-render break) or before it
    (eviction), vs full reuse (cached >= previous prompt + completion - 512);
plus images per prompt. Usage: turns.py <traces dir>"""
import collections, glob, gzip, json, sys
n_att = 0; imgs = []; cover = [0, 0]; cls = collections.Counter(); first_prompt = []
for f in glob.glob(sys.argv[1] + "/*.json*"):
    try:
        d = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
    except Exception:
        continue
    msgs = d.get("messages") or []
    users = [m for m in msgs if m.get("role") == "user"]
    if users and isinstance(users[0].get("content"), list):
        imgs.append(sum(1 for p in users[0]["content"] if p.get("type") == "image_url"))
    us = [e["usage"] for e in d.get("events", []) if e.get("usage")]
    if not us:
        continue
    n_att += 1
    p1 = us[0].get("prompt_tokens") or 0; first_prompt.append(p1)
    for k in range(1, len(us)):
        cached = ((us[k].get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        prev_p, prev_c = us[k - 1].get("prompt_tokens") or 0, us[k - 1].get("completion_tokens") or 0
        cover[1] += 1; cover[0] += cached >= p1 - 512
        if cached >= prev_p + prev_c - 512:
            cls["full reuse"] += 1
        elif cached >= prev_p - 512:
            cls["break inside last assistant turn"] += 1
        else:
            cls["evicted before last prompt"] += 1
tot = sum(cls.values())
print(f"attempts with usage: {n_att}; turns>=2: {tot}")
print(f"(a) image span covered on turns>=2: {cover[0]}/{cover[1]} = {cover[0] / max(1, cover[1]):.3f}")
print("(c) " + ", ".join(f"{k}: {v} ({v / max(1, tot):.3f})" for k, v in cls.most_common()))
if imgs:
    print(f"images per prompt: max {max(imgs)}, mean {sum(imgs) / len(imgs):.1f}, n {len(imgs)}; turn-1 prompt mean {sum(first_prompt) / len(first_prompt):.0f} tok")
