"""Build the agentic replay workload (E014) from E012 Qwen tool-use transcripts (CPU, local).

    .venv/bin/python kaggle/datasets/agentic_replay/build.py --tokenizer /path/to/tokenizer.json

Each E012 trace is the final raw prompt of one attempt: initial prompt (ends with GEN_PROMPT), then per turn the
generated text, "<|im_end|>\n" + tool responses + GEN_PROMPT. We keep, per attempt:
  prompt0: the initial prompt text
  turns:   [{"gen": generated tokens (turn_log), "suffix": text appended after the generation (tool part +
             GEN_PROMPT; "" on the last turn), "ctx": prompt tokens at the start of the turn (recorded text)}]
  hist:    the recorded generated text of each turn (only used to warm-start an agent mid-transcript)
The replay (kaggle/src/replay_bench.py) sends prompt_t, asks for exactly `gen` tokens (ignore_eos), and builds
prompt_{t+1} = prompt_t + <server's generated text> + suffix_t, which is what the real agent sends.
Output: agentic_replay.json.gz (+ stats.json). Only medium-tier 48k runs of qwen-tools-3 / -2 harnesses.
"""
import argparse
import gzip
import json
import os
import random
import statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
HERE = os.path.dirname(os.path.abspath(__file__))
GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
TOOL_START = "<|im_end|>\n<|im_start|>user\n"
RUNS = ["qwen_tools_medium_48k_dev_s24", "qwen_tools_medium_48k_dev_s24strong", "qwen_tools_medium_48k_full_v3",
        "qwen_tools_medium_48k_pilot_s24"]


def segment(text, n_turns):
    parts = text.split(GEN_PROMPT)
    if len(parts) != n_turns + 1:
        return None
    prompt0 = parts[0] + GEN_PROMPT
    gens, suffixes = [], []
    for i, p in enumerate(parts[1:]):
        last = i == n_turns - 1
        j = p.rfind(TOOL_START)
        if last:
            gens.append(p)
            suffixes.append("")
        elif j < 0:
            return None
        else:
            gens.append(p[:j])
            suffixes.append(p[j:] + GEN_PROMPT)
    return prompt0, gens, suffixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "upload/agentic_replay.json.gz"))
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    out, skipped = [], 0
    for run in RUNS:
        for line in open(os.path.join(ROOT, "evals/results", run + ".jsonl")):
            r = json.loads(line)
            tl = r.get("turn_log") or []
            if not r.get("trace") or not tl or not os.path.exists(os.path.join(ROOT, r["trace"])):
                skipped += 1
                continue
            text = gzip.open(os.path.join(ROOT, r["trace"]), "rt").read()
            seg = segment(text, len(tl))
            if seg is None:
                skipped += 1
                continue
            prompt0, gens, suffixes = seg
            turns, ctx = [], ntok(prompt0)
            for t, g, s in zip(tl, gens, suffixes):
                turns.append({"gen": int(t["tokens"]), "suffix": s, "ctx": ctx})
                ctx += ntok(g) + ntok(s)
            if sum(t["gen"] for t in turns) < 64:
                skipped += 1
                continue
            out.append({"id": f"{run}/{r['task_id']}_p{r['pair']}_a{r['attempt']}", "prompt0": prompt0,
                        "turns": turns, "hist": gens, "final_ctx": ctx})
    random.Random(0).shuffle(out)
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    gen = [sum(t["gen"] for t in x["turns"]) for x in out]
    # Steady-state (time ~ generated tokens) context of a live agent: weight each turn's mean context by its gen.
    num = sum(t["gen"] * (t["ctx"] + t["gen"] / 2) for x in out for t in x["turns"])
    ss_ctx = num / sum(gen)
    stats = {"attempts": len(out), "skipped": skipped, "gen_mean": st.mean(gen), "gen_median": st.median(gen),
             "turns_mean": st.mean(len(x["turns"]) for x in out),
             "prompt0_mean": st.mean(x["turns"][0]["ctx"] for x in out),
             "final_ctx_mean": st.mean(x["final_ctx"] for x in out),
             "final_ctx_p90": sorted(x["final_ctx"] for x in out)[int(0.9 * len(out))],
             "steady_state_ctx_mean": ss_ctx,
             "prefill_per_gen": (sum(x["final_ctx"] for x in out) - sum(gen)) / sum(gen),
             "bytes_gz": os.path.getsize(a.out)}
    json.dump(stats, open(os.path.join(HERE, "upload/stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
