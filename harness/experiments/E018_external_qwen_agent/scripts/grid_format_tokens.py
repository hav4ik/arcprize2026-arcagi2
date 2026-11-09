import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ch = json.load(open(sys.argv[2]))
def rows(g): return "\n".join("".join(map(str, r)) for r in g)
J = D = 0
for t in ch.values():
    for p in t["test"]:
        js = "".join(json.dumps(x["input"]) + json.dumps(x["output"]) for x in t["train"]) + json.dumps(p["input"])
        ds = "".join(f"Input ({len(x['input'])}x{len(x['input'][0])}):\n" + rows(x["input"]) + f"\nOutput ({len(x['output'])}x{len(x['output'][0])}):\n" + rows(x["output"]) + "\n" for x in t["train"]) + rows(p["input"])
        J += len(tok.encode(js).ids); D += len(tok.encode(ds).ids)
n = sum(len(t["test"]) for t in ch.values())
print(f"pairs {n}: JSON grids mean {J/n:.0f} tok, digit rows+dims mean {D/n:.0f} tok, ratio {D/J:.2f}")
