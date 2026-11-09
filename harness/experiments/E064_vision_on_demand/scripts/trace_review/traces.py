"""Trace stats per arm: how reasoning uses the images (E058/E061 vs text-only baselines)."""
import json, gzip, re, statistics as st, sys, collections
R = "/home/vu/Workspace/ARC-AGI-2/evals/results/"
ARMS = {
 "DS text": ["tools_persistent_grid_lean2sub_deepseek-flash_low_e046p_note_all.jsonl"],
 "DS img":  ["tools_persistent_grid_lean2sub_deepseek-flash_low_e058_pilot30.jsonl"],
 "Q text":  ["tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e048_qwenflash_sub80s40.jsonl",
             "tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e048_qwenflash_sub80s40_retry.jsonl"],
 "Q img":   ["tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e058q_pilot30.jsonl"],
 "Q best img": ["tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e061q_best_img.jsonl"],
}
IMG = re.compile(r"\b(images?|picture|pictures|visual(?:ly)?|drawn|drawing|rendered|the render)\b", re.I)
COL = re.compile(r"\b(black|blue|red|green|yellow|gr[ae]y|magenta|pink|orange|light ?blue|cyan|azure|maroon|brown)\b", re.I)
PRINTG = re.compile(r"print\(|for row in|\.join\(")
def load(fs):
    d = {}
    for f in fs:
        for r in map(json.loads, open(R + f)):
            if "task_id" in r and not r.get("error") and r.get("trace_path"):
                d[r["task_id"], r["pair"], r["attempt"]] = r
    return d
def stats(d):
    out = collections.defaultdict(list)
    for k, r in d.items():
        if r.get("multi_share"):
            continue
        try:
            m = json.load(gzip.open(r["trace_path"]))["messages"]
        except Exception:
            continue
        asst = [x for x in m if x["role"] == "assistant"]
        if not asst:
            continue
        rs = [(x.get("reasoning_content") or x.get("reasoning") or "") for x in asst]
        allr = " ".join(rs)
        out["img_any"].append(bool(IMG.search(allr)))
        out["img_t1"].append(len(IMG.findall(rs[0])))
        out["img_later"].append(sum(len(IMG.findall(x)) for x in rs[1:]))
        out["col_per_k"].append(1000 * len(COL.findall(allr)) / max(1, len(allr)))
        out["t1_chars"].append(len(rs[0]))
        out["all_chars"].append(len(allr))
        calls = [c for x in asst for c in (x.get("tool_calls") or [])]
        out["calls"].append(len(calls))
        out["print_calls"].append(sum(bool(PRINTG.search(c["function"]["arguments"])) for c in calls))
        # first tool call: turn index (0 = model coded immediately)
        out["first_call_turn"].append(next((i for i, x in enumerate(asst) if x.get("tool_calls")), len(asst)))
        out["prompt0"].append(r["usage"]["prompt"] / max(1, r["turns"]))
        out["correct"].append(bool(r["correct"]))
    return out
for name, fs in ARMS.items():
    s = stats(load(fs))
    n = len(s["correct"])
    print(f"{name:11} n={n:3d} acc={100*st.mean(s['correct']):5.1f}  attempts mentioning image={100*st.mean(s['img_any']):5.1f}%"
          f"  img refs t1/later={st.mean(s['img_t1']):.2f}/{st.mean(s['img_later']):.2f}  colour words/kchar={st.mean(s['col_per_k']):.2f}"
          f"  t1 reasoning={st.median(s['t1_chars'])/1e3:.1f}k chars  all={st.median(s['all_chars'])/1e3:.1f}k  calls={st.mean(s['calls']):.1f}"
          f"  print calls={st.mean(s['print_calls']):.1f}")
