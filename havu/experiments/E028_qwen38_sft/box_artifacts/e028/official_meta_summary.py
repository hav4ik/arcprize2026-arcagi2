import json, re, collections
from huggingface_hub import get_safetensors_metadata
c = json.load(open("/workspace/e028/official_fp8_meta/config.json"))
q = c["quantization_config"]
print({k: v for k, v in q.items() if k != "modules_to_not_convert"})
mn = q.get("modules_to_not_convert", [])
pat = collections.Counter(re.sub(r"\d+", "N", m) for m in mn)
print("modules_to_not_convert", len(mn), dict(pat))
print("top-level config keys", list(c.keys()))
print("arch", c.get("architectures"), "tie", c.get("tie_word_embeddings"))
md = get_safetensors_metadata("Qwen/Qwen3.8-27B-FP8")
info = {}
for fn, fm in md.files_metadata.items():
    for k, t in fm.tensors.items():
        info[k] = (t.dtype, list(t.shape), fn)
json.dump({k: [v[0], v[1], v[2]] for k, v in info.items()}, open("/workspace/e028/official_fp8_meta/tensors.json", "w"))
print("n tensors", len(info))
pc = collections.Counter((re.sub(r"\.\d+\.", ".N.", k), v[0]) for k, v in info.items())
for (p, d), n in sorted(pc.items()):
    print(f"{n:5d} {d:8s} {p}")
