#!/usr/bin/env python3
"""E051 parity on KV bytes: a prefix restored from host RAM must put back exactly the bytes a GPU prefix hit uses.

Why bytes and not logprobs: on this stack two GPU-hit runs of the same prompt already differ at token 0 (also with
VLLM_BATCH_INVARIANT=1), so sampled logprobs cannot separate offload bugs from kernel nondeterminism. The KV bytes a
request attends to are deterministic given the run that computed them, and offloading must move them unchanged.

Needs a server started with `--worker-extension-cls e051_ext.E051Ext` and VLLM_SERVER_DEV_MODE=1, and no other
traffic. Per case, one cold run computes and stores the prefix; then
  GPU path:  the next request hits the GPU prefix cache  -> snapshot of its block pages (all 4 ranks);
  host path: reset the GPU prefix cache, poison the whole KV pool (0xA5), the same request restores from the host
             -> snapshot.
Checks, per rank: for every logical block the host path restored (MLA/indexer blocks below the hit; each SWA group's
4 blocks ending at the hit), the page hash equals the GPU path's and is not the poison page. Cases:
  prompt L:  the same prompt again (resume at round_down(L - 1, 128));
  resume L:  next turn P + sampled ids + 200 tool ids after a decode of --max-tokens (rolling checkpoint); the
             reference is a snapshot of P's own request once its decode covers the boundary, since a GPU-path
             run of the next turn would store deeper checkpoints of its own.

    python kv_parity.py --lengths 3000 20000 60000 120000 --json kv_parity.json
"""
import argparse
import hashlib
import http.client
import json
import random
import sys
import time
import urllib.request

PAGE = 113_920
POISON = hashlib.sha1(bytes([0xA5]) * PAGE).hexdigest()[:16]


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw else None


def rpc(a, method):
    return post(a.url, "/collective_rpc", {"method": method})["results"]


def reset(a, external):
    post(a.url, "/reset_prefix_cache" + ("?reset_external=true" if external else ""), {})
    time.sleep(1.0)


def metric(a, prefix):
    with urllib.request.urlopen(a.url + "/metrics", timeout=60) as r:
        text = r.read().decode()
    return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines()
               if l.startswith(prefix) and not l.startswith("#"))


def wait_stores(a, quiet=3.0, timeout=180):
    last, t_last, t0 = None, time.time(), time.time()
    while time.time() - t0 < timeout:
        v = metric(a, "vllm:kv_offload_store_bytes_total")
        if v != last:
            last, t_last = v, time.time()
        elif time.time() - t_last >= quiet:
            return
        time.sleep(0.5)


def complete(a, prompt, max_tokens):
    r = post(a.url, "/v1/completions", {"model": a.model, "prompt": prompt, "max_tokens": max_tokens,
                                         "temperature": 0.0, "ignore_eos": True, "return_token_ids": True})
    return r["choices"][0]["token_ids"], (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")


def run_with_snapshot(a, prompt, max_tokens=None, snap_at=0):
    """Stream one request; snapshot block pages at its token event `snap_at` (0 = first token, prefill done), then let
    it finish. The snapshot runs between engine steps while the request is still decoding."""
    host, port = a.url.split("//")[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=3600)
    body = {"model": a.model, "prompt": prompt, "max_tokens": max_tokens or a.snap_tokens, "temperature": 0.0,
            "ignore_eos": True, "return_token_ids": True,
            "stream": True, "stream_options": {"include_usage": True}}
    conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    snap, cached, n_ev, ids = None, None, 0, []
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("choices"):
            ids += ev["choices"][0].get("token_ids") or []
            if snap is None and n_ev >= snap_at:
                snap = rpc(a, "e051_snapshot")
            n_ev += 1
        if ev.get("usage"):
            cached = (ev["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    conn.close()
    return snap, cached, ids


def only_request(snap_rank):
    assert len(snap_rank) == 1, f"expected exactly one request in the runner, got {len(snap_rank)}"
    return next(iter(snap_rank.values()))


def compare(layout, snap_g, snap_h, hit):
    """Per rank: hashes of the blocks the host path restored must equal the GPU path's and not be poison."""
    sizes = layout["block_sizes"]
    res = {"ranks": [], "ok": True}
    for r, (sg, sh) in enumerate(zip(snap_g, snap_h)):
        gg, hh = only_request(sg), only_request(sh)
        rank = {"checked": 0, "mismatch": [], "poison": []}
        for g, bs in enumerate(sizes):
            if bs == 128:            # MLA + indexer: every block below the hit
                idx = range(0, hit // 128)
            elif bs == 32:           # SWA: the 4-block window ending at the hit
                idx = range(max(hit // 32 - 4, 0), hit // 32)
            else:                    # compressor ring: never restored
                continue
            for i in idx:
                if i >= len(gg[g]["sha"]) or i >= len(hh[g]["sha"]):
                    rank["mismatch"].append([g, i, "missing"])
                    continue
                rank["checked"] += 1
                if hh[g]["sha"][i] == POISON:
                    rank["poison"].append([g, i])
                elif hh[g]["sha"][i] != gg[g]["sha"][i]:
                    rank["mismatch"].append([g, i, gg[g]["ids"][i], hh[g]["ids"][i]])
        rank["ok"] = rank["checked"] > 0 and not rank["mismatch"] and not rank["poison"]
        res["ok"] &= rank["ok"]
        res["ranks"].append(rank)
    return res


def case(a, layout, name, prompt_cold, prompt_next, report):
    """One cold run of prompt_cold computes + stores; prompt_next then runs via GPU hit and via host restore."""
    reset(a, external=True)
    complete(a, prompt_cold, a.max_tokens)
    wait_stores(a)
    snap_g, cached_g, _ = run_with_snapshot(a, prompt_next)
    wait_stores(a)
    reset(a, external=False)
    rpc(a, "e051_poison")
    hits0 = metric(a, "vllm:external_prefix_cache_hits_total")
    snap_h, cached_h, _ = run_with_snapshot(a, prompt_next)
    ext = metric(a, "vllm:external_prefix_cache_hits_total") - hits0
    cmp = compare(layout, snap_g, snap_h, cached_h or 0)
    row = {"case": name, "len": len(prompt_next), "cached_gpu": cached_g, "cached_host": cached_h,
           "external_hits": ext, "aligned": (cached_h or 0) % 128 == 0,
           "checked_per_rank": [r["checked"] for r in cmp["ranks"]],
           "mismatch": [r["mismatch"][:5] for r in cmp["ranks"]], "poison": [r["poison"][:5] for r in cmp["ranks"]],
           "ok": cmp["ok"] and ext > 0 and (cached_h or 0) % 128 == 0 and cached_h == cached_g}
    print(json.dumps(row), flush=True)
    report.append(row)
    return row["ok"]


def resume_case(a, layout, n, prompt, rng, report):
    """Rolling checkpoint of a finished request: snapshot the cold request P itself just after its decode covers the
    boundary b = round_down(L + max_tokens - 1, 128), then restore the next turn P + ids + tool ids from the host
    (only P's stores are there) into a poisoned pool, and compare the restored pages with P's own."""
    b = (n + a.max_tokens - 1) // 128 * 128
    snap_at = min(max(b - n + 1, 0), a.max_tokens - 2)
    reset(a, external=True)
    snap_c, _, ids = run_with_snapshot(a, prompt, max_tokens=a.max_tokens, snap_at=snap_at)
    wait_stores(a)
    nxt = prompt + ids + [rng.randrange(1000, 100000) for _ in range(200)]
    reset(a, external=False)
    rpc(a, "e051_poison")
    hits0 = metric(a, "vllm:external_prefix_cache_hits_total")
    snap_h, cached_h, _ = run_with_snapshot(a, nxt)
    ext = metric(a, "vllm:external_prefix_cache_hits_total") - hits0
    cmp = compare(layout, snap_c, snap_h, cached_h or 0)
    row = {"case": f"resume{n}", "len": len(nxt), "expected_hit": b, "cached_host": cached_h, "external_hits": ext,
           "checked_per_rank": [r["checked"] for r in cmp["ranks"]],
           "mismatch": [r["mismatch"][:5] for r in cmp["ranks"]], "poison": [r["poison"][:5] for r in cmp["ranks"]],
           "ok": cmp["ok"] and ext > 0 and cached_h == b}
    print(json.dumps(row), flush=True)
    report.append(row)
    return row["ok"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--lengths", type=int, nargs="+", default=[3000, 20000, 60000])
    p.add_argument("--max-tokens", type=int, default=64, help="decode length of the cold run (resume case)")
    p.add_argument("--snap-tokens", type=int, default=48, help="decode length of the snapshotted runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    a = p.parse_args()
    layout = rpc(a, "e051_layout")[0]
    print("layout:", json.dumps({k: layout[k] for k in ("block_sizes", "num_groups", "storages")}), flush=True)
    rng = random.Random(a.seed)
    report, ok = [], True
    for n in a.lengths:
        prompt = [rng.randrange(1000, 100000) for _ in range(n)]
        ok &= case(a, layout, f"prompt{n}", prompt, prompt, report)
        ok &= resume_case(a, layout, n, prompt, rng, report)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    print("KV PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
