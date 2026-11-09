"""E032: build the Qwen3.8-27B SFT set from E031 DeepSeek-V4.1-Flash-1017 tool-use rollouts.

Copy of E028's builder with three changes: E031 results (lean2sub, no helpers), E031's own tool specs
(e031_tools.json, snapshot of run_deepseek_tools.tools_for() with the E031 config), and the E031 system
message ("Reasoning Effort: Minimal ...", --effort-header minimal) placed after the tools block in the
system turn, as the HF template does with a system message. Default --out data/sft/e032.

All rollouts by default (correct or not), cleaned per the E025 SFT audits, rendering with tinker-cookbook's qwen3_8_xhigh_reasoning
renderer (one sequence per rollout, TrainOnWhat.ALL_ASSISTANT_MESSAGES), an independent
mask check on the token ids, a train/val split by parent task, and a token report.

Output (--out, default data/sft/e032): {train,val}.npz with
  tokens  int32  all sequences concatenated
  weights uint8  1 = trained (aligned with tokens, before the next-token shift)
  offsets int64  sequence boundaries (len n+1)
and manifest.jsonl (one row per kept rollout) + report.json.

Run with a venv that has tinker-cookbook:
  python -I build_dataset.py [--max-tokens 65536] [--limit N]
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/vu/Workspace/ARC-AGI-2")
RESULTS = ROOT / "evals/results"
PREFIX = "tools_persistent_grid_lean2sub_deepseek-flash_low_e031_"
SUBSETS = ["arc2_train", "nvarc_b0"]
RENDERER = "qwen3_8_xhigh_reasoning"
TOKENIZER = "Qwen/Qwen3.8-27B"
# Training outputs that match eval task 89565ca0 up to rotation/flip/colour (E025 data audit).
EXCLUDE_TASKS = {"2753e76c", "81c0276b"}
BAD_STRINGS = ("<｜", "DSML", "end▁of▁thinking", "<|im_", "<|endoftext|>", "<think>", "</think>",
               "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")

sys.path.insert(0, str(Path(__file__).parent))
TOOLS = json.load(open(Path(__file__).parent / "e031_tools.json"))


# ---------------------------------------------------------------- conversion

def to_tinker(raw):
    """DeepSeek API messages -> tinker_cookbook Messages (thinking as ThinkingPart, tool calls as ToolCall)."""
    from tinker_cookbook.renderers.base import ToolCall
    names, out = {}, []
    for m in raw:
        if m["role"] == "assistant":
            parts = []
            if (m.get("reasoning_content") or "").strip():
                parts.append({"type": "thinking", "thinking": m["reasoning_content"]})
            if (m.get("content") or "").strip():
                parts.append({"type": "text", "text": m["content"]})
            msg = {"role": "assistant", "content": parts}
            if m.get("tool_calls"):
                msg["tool_calls"] = [ToolCall(id=c["id"], function=ToolCall.FunctionBody(
                    name=c["function"]["name"], arguments=c["function"]["arguments"])) for c in m["tool_calls"]]
                names.update({c["id"]: c["function"]["name"] for c in m["tool_calls"]})
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_call_id": m["tool_call_id"],
                        "name": names.get(m["tool_call_id"], "")})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# ---------------------------------------------------------------- filters

def bad_turn(m, event):
    """Why an assistant turn is unusable as a training target (None if fine)."""
    if event.get("finish_reason") == "length":
        return "length_stop"  # reasoning/tool call cut off at DeepSeek's per-turn cap
    for c in m.get("tool_calls") or []:
        try:
            args = json.loads(c["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            return "bad_tool_json"
        if not isinstance(args, dict) or not str(args.get("code", "")).strip():
            return "empty_tool_args"
        if c["function"]["name"] not in ("run_python", "submit_answer"):
            return "unknown_tool"
    return None


def clean(row, raw, events, correct_only):
    """Return (messages to train on, reject reason or None, cut reason or None).

    All rollouts are kept by default (correct or not: the target is the reasoning process, not only
    lucky solves). A rollout is cut right before its first unusable assistant turn (length stop, broken
    or empty tool call), keeping the clean prefix; it is rejected if nothing trainable remains, if the
    usage events don't line up with the turns, or if any kept text leaks chat markup.
    """
    if correct_only and (row.get("correct") is not True or row.get("train_pass") is not True
                         or row.get("answer_source") != "submit"):
        return None, "not_correct", None
    if row.get("task_id", "").split("_")[0] in EXCLUDE_TASKS:
        return None, "eval_near_match", None
    asst_idx = [i for i, m in enumerate(raw) if m["role"] == "assistant"]
    turn_events = [e for e in events if "usage" in e]
    if len(turn_events) != len(asst_idx):
        return None, "events_mismatch", None
    cut = None
    for i, e in zip(asst_idx, turn_events):
        why = bad_turn(raw[i], e)
        if why:
            raw, cut = raw[:i], why
            break
    if not any(m["role"] == "assistant" for m in raw):
        return None, f"no_clean_turns_{cut}", cut
    # Trailing tool messages (e.g. the final "[answer accepted]" reply) are kept although they are untrained
    # (~0.09% of tokens): if the conversation ended on an assistant turn, the renderer would format that last
    # turn as a generation (open <think>) instead of as history, unlike every other turn.
    asst = [m for m in raw if m["role"] == "assistant"]
    texts = [m.get("content") or "" for m in raw] + [m.get("reasoning_content") or "" for m in asst] + \
        [c["function"]["arguments"] for m in asst for c in m.get("tool_calls") or []]
    if any(b in t for t in texts for b in BAD_STRINGS):
        return None, "markup_leak", cut
    return raw, None, cut


# ---------------------------------------------------------------- independent mask check

def check_mask(ids, w, tok, n_asst):
    """Verify weights from the token ids alone, without trusting the renderer.

    Walks the <|im_start|>role ... <|im_end|> segments and asserts:
    - weight 1 only inside assistant segments, never in the '<|im_start|>assistant\\n' header,
      never in an empty-think header, never on the '\\n' between segments;
    - every token of an assistant segment after its header is trained, through <|im_end|>;
    - no weight-1 token in system / user / tool-response segments;
    - exactly n_asst assistant segments, each with one contiguous trained run.
    Returns a Counter of token categories.
    """
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    role_ids = {r: tok.encode(r, add_special_tokens=False) for r in ("system", "user", "assistant")}
    nl = tok.encode("\n", add_special_tokens=False)
    empty_think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    cats = Counter()
    starts = [i for i, t in enumerate(ids) if t == im_start]
    assert starts and starts[0] == 0, "sequence must start with <|im_start|>"
    n_seen, turn_ends = 0, []
    for si, s in enumerate(starts):
        e = starts[si + 1] if si + 1 < len(starts) else len(ids)
        role = next((r for r, rid in role_ids.items() if ids[s + 1:s + 1 + len(rid)] == rid), None)
        assert role is not None, f"unknown role at {s}"
        seg_w = w[s:e]
        if role != "assistant":
            assert not any(seg_w), f"trained token inside {role} segment at {s}"
            kind = "system" if role == "system" else (
                "tool_response" if "<tool_response>" in tok.decode(ids[s:s + 12]) else "user")
            cats[kind] += e - s
            continue
        n_seen += 1
        hdr = 1 + len(role_ids["assistant"]) + len(nl)
        assert ids[s + 1 + len(role_ids["assistant"]):s + hdr] == nl
        if ids[s + hdr:s + hdr + len(empty_think)] == empty_think:
            hdr += len(empty_think)
            cats["empty_think_turns"] += 1
        assert im_end in ids[s + hdr:e], f"assistant turn at {s} has no <|im_end|>"
        end = ids.index(im_end, s + hdr)  # first <|im_end|> after header ends the turn
        assert not any(w[s:s + hdr]), f"trained header at {s}"
        assert all(w[s + hdr:end + 1]), f"untrained assistant token in turn at {s}"
        assert not any(w[end + 1:e]), f"trained token after <|im_end|> at {end}"
        cats["assistant_header"] += hdr
        cats["assistant_trained"] += end + 1 - (s + hdr)
        turn_ends.append(end)
        cats["separator"] += e - (end + 1)
    assert n_seen == n_asst, f"{n_seen} assistant segments, expected {n_asst}"
    assert sum(w) == cats["assistant_trained"]
    return cats, turn_ends


# ---------------------------------------------------------------- worker

_st = {}


def _init():
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(TOKENIZER)
    _st["tok"] = tok
    _st["rend"] = get_renderer(RENDERER, tok)
    _st["special"] = set(tok.all_special_ids) | {i for i in tok.get_added_vocab().values()}


def process(job):
    from tinker_cookbook.renderers import TrainOnWhat
    row, path, correct_only, max_tokens, overlong = job
    try:
        d = json.load(gzip.open(path))
    except Exception as exc:
        return row, None, f"unreadable:{type(exc).__name__}"
    raw, events = d["messages"], d.get("events", [])
    system = raw[0]["content"] if raw and raw[0]["role"] == "system" else ""
    if system:  # the E031 effort header; rendered into the system turn after the tools block
        raw = raw[1:]
    raw, why, cut = clean(row, raw, events, correct_only)
    row = {**row, "_cut": cut}
    if why:
        return row, None, why
    tok, rend = _st["tok"], _st["rend"]
    # No raw text may tokenize to a special token (e.g. a literal <|im_end|> inside reasoning).
    for m in raw:
        args = [c["function"]["arguments"] for c in m.get("tool_calls") or []]
        for t in [m.get("content") or "", m.get("reasoning_content") or ""] + args:
            if t and _st["special"] & set(tok.encode(t, add_special_tokens=False)):
                return row, None, "special_token_in_text"
    convo = rend.create_conversation_prefix_with_tools(TOOLS, system_prompt=system) + to_tinker(raw)
    try:
        mi, weights = rend.build_supervised_example(convo, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    except Exception as exc:
        return row, None, f"render_error:{type(exc).__name__}"
    ids = mi.to_ints()
    w = [int(x) for x in weights.tolist()]
    n_asst = sum(m["role"] == "assistant" for m in raw)
    try:
        cats, turn_ends = check_mask(ids, w, tok, n_asst)
        if len(ids) > max_tokens:
            if overlong == "drop":
                return row, None, "over_max_tokens"
            # Cut after the last complete assistant turn that fits: the prefix is exactly the rendering of
            # the conversation up to that turn (thinking is kept in history), so nothing ends mid-turn.
            fit = [k for k, e in enumerate(turn_ends) if e + 1 <= max_tokens]
            if not fit:
                return row, None, "first_turn_over_max"
            n_keep = fit[-1] + 1
            ids, w = ids[:turn_ends[fit[-1]] + 1], w[:turn_ends[fit[-1]] + 1]
            cats, _ = check_mask(ids, w, tok, n_keep)
            row["_truncated_turns"] = (n_asst, n_keep)
    except AssertionError as exc:
        return row, None, f"mask_check:{exc}"
    return row, (np.asarray(ids, np.int32), np.asarray(w, np.uint8), dict(cats)), None


# ---------------------------------------------------------------- main

def parent_of(row):
    return row["task_id"].split("_")[0]  # nvarc ids are <parent>_<hash>


def is_val(parent, frac):
    return int(hashlib.sha1(parent.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def save(path, seqs):
    lens = [len(s[0]) for s in seqs]
    np.savez_compressed(path, tokens=np.concatenate([s[0] for s in seqs]),
                        weights=np.concatenate([s[1] for s in seqs]),
                        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/sft/e032")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--overlong", choices=["cut", "drop"], default="cut",
                    help="cut = keep complete turns up to --max-tokens; drop = discard the rollout")
    ap.add_argument("--sample", type=int, default=0,
                    help="keep N rollouts: every arc2_train rollout first, the rest randomly from nvarc_b0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0, help="fraction of parent tasks held out (0 = train on everything)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N rollouts")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--correct-only", action="store_true",
                    help="only correct + train_pass + submitted rollouts")
    args = ap.parse_args()

    jobs = []
    for sub in SUBSETS:
        for line in open(RESULTS / f"{PREFIX}{sub}.jsonl"):
            row = json.loads(line)
            row["subset"] = sub
            if row.get("trace_path"):
                jobs.append((row, row["trace_path"], args.correct_only, args.max_tokens, args.overlong))
    if args.limit:
        jobs = jobs[:args.limit]

    reasons, kept = Counter(), {"train": [], "val": []}
    manifest = {"train": [], "val": []}
    cats_tot = {"train": Counter(), "val": Counter()}
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for row, res, why in ex.map(process, jobs, chunksize=8):
            if why:
                reasons[why.split(":")[0]] += 1
                if why.startswith(("mask_check", "render_error")):
                    print("WARN", row["subset"], Path(row["trace_path"]).name, why, file=sys.stderr)
                continue
            ids, w, cats = res
            assert len(ids) <= args.max_tokens
            split = "val" if is_val(parent_of(row), args.val_frac) else "train"
            kept[split].append((ids, w, cats))
            manifest[split].append({"subset": row["subset"], "task_id": row["task_id"], "pair": row["pair"],
                                    "trace": Path(row["trace_path"]).name, "tokens": int(len(ids)),
                                    "trained": int(w.sum()), "turns": row.get("turns"),
                                    "correct": row.get("correct") is True, "cut": row.get("_cut"),
                                    "truncated_turns": row.get("_truncated_turns")})

    if args.sample:
        rng = random.Random(args.seed)
        for split in kept:
            n = args.sample if split == "train" else len(kept[split])
            arc = [i for i, m in enumerate(manifest[split]) if m["subset"] == "arc2_train"]
            rest = [i for i, m in enumerate(manifest[split]) if m["subset"] != "arc2_train"]
            assert len(arc) <= n, f"{len(arc)} arc2_train rollouts exceed --sample {n}"
            pick = arc + rng.sample(rest, min(len(rest), n - len(arc)))
            kept[split] = [kept[split][i] for i in pick]
            manifest[split] = [manifest[split][i] for i in pick]
    for split in kept:
        for _, _, cats in kept[split]:
            cats_tot[split].update(cats)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"renderer": RENDERER, "tokenizer": TOKENIZER, "max_tokens": args.max_tokens,
              "overlong": args.overlong, "sample": args.sample, "seed": args.seed,
              "correct_only": args.correct_only,
              "input_rollouts": len(jobs), "rejected": dict(reasons.most_common()), "splits": {}}
    for split in ("train", "val"):
        rng = random.Random(0)
        order = list(range(len(kept[split])))
        rng.shuffle(order)  # stored shuffled; the trainer reshuffles per epoch
        seqs = [kept[split][i] for i in order]
        man = [manifest[split][i] for i in order]
        if seqs:
            save(args.out / f"{split}.npz", seqs)
        with open(args.out / f"{split}_manifest.jsonl", "w") as f:
            f.writelines(json.dumps(m) + "\n" for m in man)
        tot = sum(m["tokens"] for m in man)
        trn = sum(m["trained"] for m in man)
        lens = np.array([m["tokens"] for m in man] or [0])
        report["splits"][split] = {
            "rollouts": len(man), "by_subset": dict(Counter(m["subset"] for m in man)),
            "truncated_at_max_tokens": sum(bool(m["truncated_turns"]) for m in man),
            "correct": sum(m["correct"] for m in man), "cut_before_bad_turn": dict(Counter(m["cut"] for m in man if m["cut"])),
            "parents": len({m["task_id"].split("_")[0] for m in man}),
            "total_tokens": tot, "trained_tokens": trn, "masked_tokens": tot - trn,
            "token_categories": dict(cats_tot[split]),
            "seq_len": {"mean": float(lens.mean()), "p50": float(np.median(lens)),
                        "p90": float(np.percentile(lens, 90)), "max": int(lens.max())},
        }
    json.dump(report, open(args.out / "report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
