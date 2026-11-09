"""vLLM offline decode throughput at fixed concurrency (TP from --tp).

For each concurrency c: c random 64-token prompts, greedy, ignore_eos, run
once with max_tokens=SHORT and once with LONG; decode step time =
(t_long - t_short) / (LONG - SHORT), which cancels prefill and scheduling
start-up. Reports gen tok/s = c / step time. NCCL env and any comm patch come
from the environment (see sitecustomize in har_vllm/).

Usage: python e2e_decode.py --model DIR --tp 2 --label x --out res.json
"""
import argparse
import json
import os
import time

SHORT, LONG = 16, 272


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", default="1,8,32")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import random

    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=0.80,
              max_model_len=2048, max_num_seqs=64, enable_prefix_caching=False, seed=0,
              enforce_eager=args.eager)
    load_s = time.time() - t0
    rng = random.Random(0)
    res = {"label": args.label, "tp": args.tp, "eager": args.eager, "load_s": round(load_s, 1),
           "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "HAR_"))}, "rows": []}
    # warm-up
    llm.generate([TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(64)])] * 2,
                 SamplingParams(max_tokens=32, ignore_eos=True, temperature=0), use_tqdm=False)
    for c in [int(v) for v in args.conc.split(",")]:
        prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(64)])
                   for _ in range(c)]
        times = {}
        for mt in (SHORT, LONG):
            sp = SamplingParams(max_tokens=mt, ignore_eos=True, temperature=0)
            ts = []
            for _ in range(args.reps):
                a = time.perf_counter()
                outs = llm.generate(prompts, sp, use_tqdm=False)
                ts.append(time.perf_counter() - a)
                assert all(len(o.outputs[0].token_ids) == mt for o in outs)
            times[mt] = min(ts)
        step = (times[LONG] - times[SHORT]) / (LONG - SHORT)
        sample = outs[0].outputs[0].token_ids[:24]
        row = {"conc": c, "step_ms": round(step * 1e3, 3), "gen_tok_s": round(c / step, 1),
               "t_short_s": round(times[SHORT], 3), "t_long_s": round(times[LONG], 3),
               "first_ids": list(sample)}
        res["rows"].append(row)
        print(args.label, json.dumps(row), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
