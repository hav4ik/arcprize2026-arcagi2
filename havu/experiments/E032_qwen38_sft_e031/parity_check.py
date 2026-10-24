"""E032 parity: served models (SGLang, tunnelled) vs Tinker compute_logprobs on the same token sequences.

NLL over trained (assistant) positions for 6 sequences: 3 from the E032 training set and 3 held-out E031 rollouts
(incorrect but train_pass), 5k-20k tokens. Arms: served e032 (block FP8) vs Tinker E032 final; served base (official
FP8) vs Tinker base.

  python -I parity_check.py E032_URL BASE_URL     (e.g. http://127.0.0.1:32002 http://127.0.0.1:32001)
"""
import json, random, sys
from pathlib import Path
import numpy as np
import requests
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import build_dataset as B  # noqa: E402
import train as T  # noqa: E402

FINAL = "tinker://99c4c413-7a89-5c0b-9d9a-30bcfa9a9e1d:train:0/sampler_weights/final"


def served(url, tokens):
    r = requests.post(f"{url}/generate", json={"input_ids": tokens, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                               "return_logprob": True, "logprob_start_len": 0}, timeout=1800)
    r.raise_for_status()
    return [x[0] for x in r.json()["meta_info"]["input_token_logprobs"]]


def main():
    e032_url, base_url = sys.argv[1], sys.argv[2]
    z = np.load(T.DATA / "train.npz")
    t, w, o = z["tokens"], z["weights"], z["offsets"]
    lens = np.diff(o)
    rng = random.Random(1)
    seqs = [("train", t[o[i]:o[i + 1]].tolist(), w[o[i]:o[i + 1]]) for i in
            rng.sample([i for i in range(len(lens)) if 5000 <= lens[i] <= 20000], 3)]
    rows = []
    for sub in B.SUBSETS:
        rows += [r | {"subset": sub} for r in map(json.loads, open(B.RESULTS / f"{B.PREFIX}{sub}.jsonl"))
                 if r.get("correct") is not True and r.get("train_pass") is True and r.get("answer_source") == "submit"]
    rng.shuffle(rows)
    B._init()
    for r in rows:
        _, res, why = B.process((r, r["trace_path"], False, 20000, "drop"))
        if res is not None and len(res[0]) >= 5000:
            seqs.append(("heldout", res[0].tolist(), res[1]))
        if len(seqs) == 6:
            break
    T.tinker_key()
    import tinker
    from tinker import types
    sc = tinker.ServiceClient()
    tk = {"e032": sc.create_sampling_client(model_path=FINAL), "base": sc.create_sampling_client(base_model="Qwen/Qwen3.8-27B")}
    urls = {"e032": e032_url, "base": base_url}
    for arm in ("e032", "base"):
        futs = [tk[arm].compute_logprobs(types.ModelInput.from_ints(s[1])) for s in seqs]
        print(f"== {arm}: Tinker vs served")
        for (kind, toks, ww), f in zip(seqs, futs):
            ref = f.result()
            got = served(urls[arm], toks)
            idx = np.nonzero(ww)[0]
            a = -np.mean([ref[j] for j in idx]); b = -np.mean([got[j] for j in idx])
            print(f"  {kind:8s} {len(toks):6d} tok  NLL Tinker {a:.4f}  served {b:.4f}  diff {b - a:+.4f}")


if __name__ == "__main__":
    main()
