"""Content checks of the uploaded tool E128 2-bit repo against the original checkpoint, by HTTP range reads."""
import json, os, random, struct, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
from huggingface_hub import hf_hub_url

D, KEEP = sys.argv[1], sys.argv[2]
OURS, ORIG, REV = "chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
H = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
oh = json.load(open(f"{D}/ours_headers.json")); gh = json.load(open(f"{D}/orig_headers.json"))
loc = {}
for repo, hh in ((OURS, oh), (ORIG, gh)):
    for f, h in hh.items():
        for k, v in h.items():
            if k != "__metadata__": loc[(repo, k)] = (f, v)
hlen = {}
S = requests.Session()

def url(repo, f): return hf_hub_url(repo, f, revision="main" if repo == OURS else REV)

def get(repo, f, a, b):
    for _ in range(4):
        try:
            r = S.get(url(repo, f), headers={**H, "Range": f"bytes={a}-{b - 1}"}, timeout=300)
            if len(r.content) == b - a: return r.content
        except Exception: pass
    raise RuntimeError((repo, f, a, b))

def base(repo, f):
    if (repo, f) not in hlen: hlen[(repo, f)] = 8 + struct.unpack("<Q", get(repo, f, 0, 8))[0]
    return hlen[(repo, f)]

DT = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "I32": (np.int32, 4), "U8": (np.uint8, 1), "I8": (np.uint8, 1),
      "F8_E4M3": (np.uint8, 1), "F8_E8M0": (np.uint8, 1)}

def tensor(repo, k, row0=None, nrows=None):
    f, v = loc[(repo, k)]
    a, b = v["data_offsets"]; shape = v["shape"]; dt, sz = DT[v["dtype"]]
    if row0 is not None:
        rb = int(np.prod(shape[1:])) * sz
        a, b, shape = a + row0 * rb, a + (row0 + nrows) * rb, [nrows] + shape[1:]
    raw = get(repo, f, base(repo, f) + a, base(repo, f) + b)
    return np.frombuffer(raw, dtype=dt).reshape(shape)

def bf16(x): return (x.astype(np.uint32) << 16).view(np.float32)

man = json.load(open(f"{D}/ours/e043_manifest.json"))
keep = np.load(KEEP)
ids = {int(l): [int(i) for i in v] for l, v in man["keep_original_ids"].items()}
print("1. manifest ids == local keep set:", all(ids[l] == list(np.where(keep[l])[0]) for l in range(40)),
      "| sorted:", all(ids[l] == sorted(ids[l]) for l in range(40)))

def router(l):
    bad = [nm for nm in ("weight", "bias", "bias_vl")
           if not np.array_equal(tensor(OURS, f"layers.{l}.ffn.gate.{nm}"), tensor(ORIG, f"layers.{l}.ffn.gate.{nm}")[ids[l]])]
    return l, bad
with ThreadPoolExecutor(8) as ex: rr = list(ex.map(router, range(40)))
print("2. router gate weight/bias/bias_vl == original rows at kept ids, 40 layers:", [x for x in rr if x[1]] or "all exact")

random.seed(0)
nbytes = lambda v: v["data_offsets"][1] - v["data_offsets"][0]
common = [k for (r, k) in loc if r == OURS and (ORIG, k) in loc and "gate." not in k and "engram.embed" not in k]
small = [k for k in common if nbytes(loc[(OURS, k)][1]) < 16 << 20]
named = [k for k in common if k.startswith(("embed", "head", "norm"))]
sample = random.sample(small, 60) + named
def same(k):
    o, g = loc[(OURS, k)], loc[(ORIG, k)]
    if o[1]["shape"] != g[1]["shape"] or o[1]["dtype"] != g[1]["dtype"]: return k, False
    ra = get(OURS, o[0], base(OURS, o[0]) + o[1]["data_offsets"][0], base(OURS, o[0]) + o[1]["data_offsets"][1])
    rb = get(ORIG, g[0], base(ORIG, g[0]) + g[1]["data_offsets"][0], base(ORIG, g[0]) + g[1]["data_offsets"][1])
    return k, ra == rb
with ThreadPoolExecutor(8) as ex: sm = list(ex.map(same, sample))
print(f"3. unchanged tensors byte-equal: {sum(x[1] for x in sm)}/{len(sm)} (sample of {len(common)} unchanged keys, incl. {named})",
      [k for k, ok in sm if not ok][:5])

for l in (1, 14):
    n = loc[(OURS, f"layers.{l}.engram.embed_tokens.row_map")][1]["shape"][0]
    st = random.randrange(0, n - 400000); rm = tensor(OURS, f"layers.{l}.engram.embed_tokens.row_map", st, 400000)
    kept = np.nonzero(rm >= 0)[0]
    mono = bool((rm[kept][1:] > rm[kept][:-1]).all())
    ok = 0
    for i in random.sample(kept.tolist(), 40):
        c = int(rm[i])
        ok += np.array_equal(tensor(OURS, f"layers.{l}.engram.embed.weight", c, 1), tensor(ORIG, f"layers.{l}.engram.embed.weight", st + i, 1)) and \
              np.array_equal(tensor(OURS, f"layers.{l}.engram.embed.scale", c, 1), tensor(ORIG, f"layers.{l}.engram.embed.scale", st + i, 1))
    print(f"4. Engram L{l}: row_map window kept {len(kept)/4e5:.1%}, compact ids increasing {mono}, sampled rows equal {ok}/40, "
          f"dropped rows map to {sorted(set(rm[rm < 0].tolist()))[:3]}")

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def mxfp4(l, e, w):
    q = tensor(ORIG, f"layers.{l}.ffn.experts.{e}.{w}.weight"); s = tensor(ORIG, f"layers.{l}.ffn.experts.{e}.{w}.scale")
    c = np.stack([q & 15, q >> 4], -1).reshape(q.shape[0], -1)
    v = np.where((c & 8) != 0, -E2M1[c & 7], E2M1[c & 7])
    return v * np.repeat(np.exp2(s.astype(np.float32) - 127), 32, -1)
def gq(l, c, w, p):
    f = f"gptq/layer_{l:02d}.safetensors"; v = oh[f][f"e{c}.{w}.{p}"]
    a, b = v["data_offsets"]; dt, _ = DT[v["dtype"]]
    return np.frombuffer(get(OURS, f, base(OURS, f) + a, base(OURS, f) + b), dtype=dt).reshape(v["shape"])
def unpack(p, bits, order):
    sh = np.arange(0, 32, bits) if order == "lsb" else np.arange(32 - bits, -1, -bits)
    return ((p.view(np.uint32)[..., None] >> sh.astype(np.uint32)) & ((1 << bits) - 1)).reshape(p.shape[0], -1)
def gptq(l, c, w, g, order):
    q = unpack(gq(l, c, w, "q"), 2, order).astype(np.float32)
    z = unpack(np.ascontiguousarray(gq(l, c, w, "z").T), 2, order).T.astype(np.float32)
    s = bf16(gq(l, c, w, "s"))
    return (q - np.repeat(z, g, -1)) * np.repeat(s, g, -1)
cos = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
W0 = mxfp4(0, ids[0][0], "w1")
orders = {o: round(cos(gptq(0, 0, "w1", 128, o), W0), 4) for o in ("lsb", "msb")}
order = max(orders, key=orders.get); print("5. GPTQ bit order probe (layer 0, e0, w1):", orders)
def one(args):
    l, c, w, g = args
    Wq = gptq(l, c, w, g, order); Wo = mxfp4(l, ids[l][c], w); Wx = mxfp4(l, ids[l][(c + 1) % 128], w)
    return (l, c, ids[l][c], w, cos(Wq, Wo), float(np.linalg.norm(Wq - Wo) / np.linalg.norm(Wo)), cos(Wq, Wx))
jobs = [(l, c, w, g) for l in (0, 13, 27, 39) for c in (0, 41, 87, 127) for w, g in (("w1", 128), ("w3", 128), ("w2", 64))]
with ThreadPoolExecutor(6) as ex: rows = list(ex.map(one, jobs))
for r in rows: print("   L%02d c%3d orig%3d %s cos(right)=%.4f relerr=%.3f cos(next expert)=%.4f" % r)
print("   min cos right:", round(min(r[4] for r in rows), 4), " max |cos| wrong:", round(max(abs(r[6]) for r in rows), 4))
