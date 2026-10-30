# Rebuild E066's preflight selection (shard0, 200k tokens, seed 7) as an exclude manifest, then check that the main
# selection (shard*, 12M tokens, seed 1, --exclude) reproduces E066's 374 rollouts / 12,005,133 tokens.
import glob, json, random, sys
sys.path.insert(0, "repo/experiments/E066_dsv41_image_pruning/scripts")
import replay_mm as m
def pick(tok, tokens, seed, skip=()):
    recs = [r for r in m.load_index(tok, 120_000) if (r["dir"], r["offset"]) not in skip]
    random.Random(seed).shuffle(recs)
    chosen, n = [], 0
    for r in recs:
        if n >= tokens: break
        chosen.append(r); n += r["length"]
    return chosen, n
pre, npre = pick(["data/e066/tok_img_calib_s/shard0"], 200_000, 7)
with open("data/e066/e066_pre_manifest.jsonl", "w") as f:
    for r in pre: f.write(json.dumps(r) + "\n")
main, nmain = pick(sorted(glob.glob("data/e066/tok_img_calib_s/shard*")), 12_000_000, 1, {(r["dir"], r["offset"]) for r in pre})
print(json.dumps({"preflight": len(pre), "preflight_tokens": npre, "main": len(main), "main_tokens": nmain}))
