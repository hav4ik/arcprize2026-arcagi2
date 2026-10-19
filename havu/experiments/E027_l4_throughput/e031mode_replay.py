"""E032 parity of agent_qwen's e031 mode (E031_CONFIG) with the E032 training data.

1. Training replay: every E032 training rollout (data/sft/e032/train_manifest.jsonl) is run through the real agent
   loop (aq.solve; recorded DeepSeek turns as completions with their DeepSeek token counts, so the harness writes its
   own budget/turns notes; recorded tool outputs with the notes stripped as executor results). The transcript is
   tokenized with the official tokenizer (Qwen/Qwen3.8-27B-FP8) and compared with train.npz token for token.
2. Other E031 rollouts (not in the training set: wrong answers, max_turns, budget ends, unknown tools): the harness
   transcript is compared as text with the recorded conversation rendered the same way, and the answer source with
   the recorded one (session fallback included).
3. --fixtures N: write N small training rollouts to evals/tests/fixtures/e031_replay.json.gz for test_qwen_e031.py.

Run with a venv that has transformers (tokenizer files only; HF_HUB_OFFLINE=1):
  HF_HUB_OFFLINE=1 <tokvenv>/bin/python -I e031mode_replay.py [--limit N] [--others N] [--fixtures]
"""
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "evals/tests"), str(ROOT / "experiments/E032_qwen38_sft_e031")]
from arc_harness import agent_qwen as aq  # noqa: E402
import build_dataset as bd  # noqa: E402  (E032 builder: clean(), PREFIX; read only)
from test_qwen_e031 import e031_cfg, replay  # noqa: E402

TASKS = {"arc2_train": ROOT / "data/arc-agi_training_challenges.json",
         "nvarc_b0": ROOT / "experiments/E025_train_rollouts/data/nvarc_b0_challenges.json"}


def load_trace(subset, trace):
    return json.load(gzip.open(ROOT / f"evals/results/{bd.PREFIX}{subset}/{trace}"))


def tags_of(d, m=None):
    tool = [x["content"] for x in d["messages"] if x["role"] == "tool"]
    t = set()
    for s in tool:
        t |= {"error"} if s.startswith("[error]") else set()
        t |= {"timeout"} if s.startswith("[timeout]") else set()
        t |= {"tool_error"} if s.startswith("[tool error") else set()
        t |= {"past"} if "Past the target" in s else set()
        t |= {"warn"} if "Budget nearly spent" in s else set()
        t |= {"turns_note"} if " left: call submit_answer" in s else set()
        t |= {"not_accepted"} if "[answer not accepted" in s else set()
        t |= {"out_trunc"} if "[output truncated]" in s or "<output cut>" in s else set()
        t |= {"skipped"} if s.startswith("[skipped") else set()
    if any(len(x.get("tool_calls") or []) > 1 for x in d["messages"]):
        t.add("multi_call")
    if any(x["role"] == "assistant" and not (x.get("reasoning_content") or "").strip() for x in d["messages"]):
        t.add("empty_think")
    if m is not None and m["tokens"] > 40000:
        t.add("long")
    return t


def rendered(messages, train, test_input, cfg):
    """The recorded conversation in the e031 render (initial prompt + render_assistant / render_tool_responses)."""
    out, pending = [aq.initial_prompt(train, test_input, cfg)[:-len(aq.GEN_PROMPT)]], []
    for x in [m for m in messages if m["role"] not in ("system", "user")]:
        if x["role"] == "tool":
            pending.append(x["content"])
            continue
        if pending:
            out.append(aq.render_tool_responses(pending))
            pending = []
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in x.get("tool_calls") or []]
        out.append(aq.render_assistant(x.get("reasoning_content") or "", x.get("content") or "", calls))
    if pending:
        out.append(aq.render_tool_responses(pending))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="training rollouts to replay (0 = all)")
    ap.add_argument("--others", type=int, default=300, help="non-training E031 rollouts to compare as text")
    ap.add_argument("--fixtures", action="store_true", help="write evals/tests/fixtures/e031_replay.json.gz")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B-FP8")
    from test_qwen_e031 import official_tokenizer
    raw_tok = official_tokenizer()  # what the unit test uses
    tasks = {k: json.load(open(v)) for k, v in TASKS.items()}
    z = np.load(ROOT / "data/sft/e032/train.npz")
    toks, offs = z["tokens"], z["offsets"]
    man = [json.loads(line) for line in open(ROOT / "data/sft/e032/train_manifest.jsonl")]
    nl = tok.encode("\n", add_special_tokens=False)
    sys_ids = tok.encode(aq.initial_prompt(tasks["arc2_train"][man[0]["task_id"]]["train"],
                                           tasks["arc2_train"][man[0]["task_id"]]["test"][0]["input"], e031_cfg())
                         .split("<|im_start|>user")[0], add_special_tokens=False)
    print("system turn tokens == training prefix:", toks[offs[0]:offs[0] + len(sys_ids)].tolist() == sys_ids,
          len(sys_ids), "tokens")

    # ---- 1. training replay
    n = len(man) if not args.limit else args.limit
    same = text_same = raw_same = 0
    tag_n, tag_ok, mism, picks = Counter(), Counter(), [], {}
    for i, m in enumerate(man[:n]):
        d = load_trace(m["subset"], m["trace"])
        msgs = d["messages"]
        system = [x for x in msgs if x["role"] == "system"]
        cleaned, why, cut = bd.clean({**m}, [x for x in msgs if x["role"] != "system"], d.get("events", []), False)
        assert why is None and cut is None, (m["trace"], why, cut)
        assert system and system[0]["content"] == aq.EFFORT_HEADERS["minimal"]
        usage = [e["usage"]["completion_tokens"] for e in d["events"] if "usage" in e]
        task = tasks[m["subset"]][m["task_id"]]
        test_input = task["test"][m["pair"]]["input"]
        text, res = replay(cleaned, usage, task["train"], test_input, [[0]])
        ref = toks[offs[i]:offs[i + 1]].tolist()
        ids = tok.encode(text, add_special_tokens=False)
        ok = ids == ref + nl and res["stop_reason"] == "submitted"
        ref_text = tok.decode(ref)
        text_same += text == ref_text + "\n"
        same += ok
        if raw_tok is not None:
            raw_same += raw_tok.encode(text, add_special_tokens=False).ids == ids
        tg = tags_of(d, m) or {"plain"}
        for t in tg:
            tag_n[t] += 1
            tag_ok[t] += ok
            if m["tokens"] < picks.get(t, (10 ** 9,))[0]:
                picks[t] = (m["tokens"], i, m, d, task, test_input, usage, ref_text, ref)
        if not ok:
            k = next((j for j, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
            mism.append({"i": i, "trace": m["trace"], "tags": sorted(tg), "len_ours": len(ids), "len_train": len(ref),
                         "stop": res["stop_reason"], "first_diff": k, "ours": tok.decode(ids[max(0, k - 30):k + 30]),
                         "train": tok.decode(ref[max(0, k - 30):k + 30])})
    print(f"\n[training replay] token-identical {same}/{n}; text-identical {text_same}/{n}; "
          f"tokenizers.Tokenizer(tokenizer.json) == AutoTokenizer on {raw_same}/{n}")
    print("by tag (identical/total):", {t: f"{tag_ok[t]}/{tag_n[t]}" for t in sorted(tag_n)})
    for x in mism[:10]:
        print("MISMATCH", json.dumps(x)[:1500])

    # ---- 2. other E031 rollouts (not trained on): harness-written tool messages and answer source
    in_train = {(m["subset"], m["trace"]) for m in man}
    cmp_n = cmp_ok = src_ok = 0
    skipped, bad, kinds = Counter(), [], Counter()
    rows = [(sub, json.loads(line)) for sub in TASKS for line in open(ROOT / f"evals/results/{bd.PREFIX}{sub}.jsonl")]
    rows = [(sub, r) for sub, r in rows if r.get("trace_path") and (sub, Path(r["trace_path"]).name) not in in_train]
    rows.sort(key=lambda x: x[1].get("stop_reason") == "submitted")  # every budget / max_turns end first
    for sub, row in rows:
        if cmp_n >= args.others:
            break
        d = json.load(gzip.open(row["trace_path"]))
        ev = d.get("events", [])
        if row.get("error") or any(e.get("finish_reason") == "length" for e in ev):
            skipped["error_or_length"] += 1
            continue
        try:
            for x in d["messages"]:
                for c in x.get("tool_calls") or []:
                    a = json.loads(c["function"]["arguments"])
                    assert isinstance(a, dict) and isinstance(a.get("code"), str)
        except Exception:
            skipped["bad_tool_json"] += 1
            continue
        msgs = d["messages"]
        usage = [e["usage"]["completion_tokens"] for e in ev if "usage" in e]
        fallback = [e["output"] for e in ev if e.get("fallback") == "session_answer"]
        task = tasks[sub][row["task_id"]]
        test_input = task["test"][row["pair"]]["input"]
        # the recorded executor outputs, then the session-answer fallback's output (when the attempt had one)
        cfg = e031_cfg(salvage=False, force_submit_tokens=0)
        text, res = _replay_fb(msgs, usage, task, test_input, row, fallback[0] if fallback else None, cfg)
        want = rendered(msgs, task["train"], test_input, cfg)
        if text.endswith(aq.GEN_PROMPT):
            text = text[:-len(aq.GEN_PROMPT)]
        cmp_n += 1
        ok = text == want
        cmp_ok += ok
        src_ok += res["answer_source"] == row.get("answer_source")
        kinds[(row.get("stop_reason"), res["stop_reason"], ok)] += 1
        for t in tags_of(d):
            kinds[("tag", t, ok)] += 1
        if not ok and len(bad) < 5:
            k = next((j for j, (a, b) in enumerate(zip(text, want)) if a != b), min(len(text), len(want)))
            bad.append({"trace": row["trace_path"][-50:], "first_diff_char": k, "ours": text[k - 200:k + 200],
                        "recorded": want[k - 200:k + 200]})
    print(f"\n[other E031 rollouts] text-identical {cmp_ok}/{cmp_n}; answer_source equal {src_ok}/{cmp_n}; "
          f"skipped {dict(skipped)}")
    print("(DeepSeek stop, harness stop, identical) / (tag, name, identical):",
          {str(k): v for k, v in sorted(kinds.items(), key=str)})
    for x in bad:
        print("DIFF", json.dumps(x)[:2000])

    # ---- 3. fixtures for the unit test
    if args.fixtures:
        want_tags = ["not_accepted", "multi_call", "timeout", "past"]
        fx = []
        for t in want_tags:
            _, i, m, d, task, test_input, usage, ref_text, ref = picks[t]
            if any(f["trace"] == m["trace"] for f in fx):
                continue
            cleaned, _, _ = bd.clean({**m}, [x for x in d["messages"] if x["role"] != "system"], d["events"], False)
            fx.append({"trace": m["trace"], "tag": t, "train": task["train"], "test_input": test_input, "pred": [[0]],
                       "messages": d["messages"][:1] + cleaned, "completion_tokens": usage,
                       "expected_text": ref_text, "ids_sha256": hashlib.sha256(json.dumps(ref).encode()).hexdigest()})
        out = ROOT / "evals/tests/fixtures/e031_replay.json.gz"
        out.parent.mkdir(exist_ok=True)
        with gzip.open(out, "wt") as f:
            json.dump(fx, f)
        print(f"\nwrote {len(fx)} fixtures ({[(f['trace'], f['tag']) for f in fx]}) to {out} "
              f"({out.stat().st_size / 1024:.0f} KiB)")


def _replay_fb(msgs, usage, task, test_input, row, fb_output, cfg):
    """replay() with the session-answer fallback's recorded output queued after the conversation's outputs."""
    import test_qwen_e031 as T
    real = T._ReplayExecutor.__init__

    def init(self, outputs, pred):
        real(self, list(outputs) + ([fb_output] if fb_output is not None else []), pred)
    T._ReplayExecutor.__init__ = init
    try:
        return replay(msgs, usage, task["train"], test_input, row.get("pred") or [[0]], cfg)
    finally:
        T._ReplayExecutor.__init__ = real


if __name__ == "__main__":
    main()
