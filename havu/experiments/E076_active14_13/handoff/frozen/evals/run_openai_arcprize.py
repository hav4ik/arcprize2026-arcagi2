"""No-tool ARC-AGI-2 eval against self-hosted OpenAI-compatible servers (vLLM / SGLang), ARC Prize prompt and parser.

Each attempt is pinned to one server by a hash of (task, pair, attempt), so several single-GPU servers act as data
parallel replicas. Results append to a JSONL file and re-running resumes (finished attempts are skipped).

    .venv/bin/python evals/run_openai_arcprize.py --base-url http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1 \
        --model minicpm --max-tokens 65536 --extra-body '{"temperature":1.0,"top_p":0.95}' --out evals/results/x.jsonl
"""
import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_harness.arcprize import TEMPLATE, build_prompt, parse  # noqa: E402
from arc_harness.grid_image import build_image_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="comma-separated OpenAI-compatible base URLs")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--extra-body", default="{}", help="JSON merged into each request (sampling, chat_template_kwargs)")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--challenges", default=str(ROOT / "data/arc-agi_evaluation_challenges.json"))
    ap.add_argument("--solutions", default=str(ROOT / "data/arc-agi_evaluation_solutions.json"))
    ap.add_argument("--pairs-file", help="optional file of task_id:pair lines to restrict the run")
    ap.add_argument("--slots-per-server", type=int, default=0,
                    help="if >0, send each attempt to a server with a free slot (least-loaded) instead of hashing; "
                         "workers = servers * slots")
    ap.add_argument("--grid-format", default="json", choices=["json", "image"],
                    help="E033: image = every grid is a PNG (ARC palette, --image-cell px per cell) instead of JSON")
    ap.add_argument("--image-cell", type=int, default=32, help="E033: pixels per cell (Qwen3.8: 32 = one LLM visual token per cell)")
    ap.add_argument("--save-reasoning", action="store_true", help="E033: store the full reasoning text in each record")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = json.load(open(args.challenges))
    sols = json.load(open(args.solutions)) if Path(args.solutions).exists() else {}
    keep = None
    if args.pairs_file:
        keep = {tuple(x.split(":")) for x in Path(args.pairs_file).read_text().split()}
        keep = {(t, int(p)) for t, p in keep}
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["task_id"], r["pair"], r["attempt"]))
    jobs = [(t, p, a) for a in range(args.attempts) for t in sorted(tasks) for p in range(len(tasks[t]["test"]))
            if (keep is None or (t, p) in keep) and (t, p, a) not in done]
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    clients = [OpenAI(api_key="none", base_url=u, timeout=7200, max_retries=0) for u in urls]
    extra = json.loads(args.extra_body)
    lock = threading.Lock()
    free = queue.Queue()  # server indices with a free slot (--slots-per-server)
    for _ in range(args.slots_per_server):
        for i in range(len(clients)):
            free.put(i)
    if args.slots_per_server:
        args.workers = len(clients) * args.slots_per_server
    print(f"{len(jobs)} attempts on {len(urls)} servers -> {out}", flush=True)

    def work(t, p, a):
        task = tasks[t]
        slot = free.get() if args.slots_per_server else None
        client = clients[slot if slot is not None else
                         int(hashlib.md5(f"{t}:{p}:{a}".encode()).hexdigest(), 16) % len(clients)]
        if args.grid_format == "image":
            prompt = build_image_prompt(TEMPLATE, task["train"], task["test"][p]["input"], args.image_cell)
        else:
            prompt = build_prompt(task["train"], task["test"][p]["input"])
        rec = {"task_id": t, "pair": p, "attempt": a, "model": args.model, "max_tokens": args.max_tokens,
               "extra_body": extra, "base_url": str(client.base_url), "grid_format": args.grid_format}
        if args.grid_format == "image":
            rec["image_cell"] = args.image_cell
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=args.max_tokens, extra_body=extra)
            msg = r.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning") or ""
            pred = parse(content) if content else None
            if args.save_reasoning:
                rec["reasoning"] = reasoning
            rec.update(content=content[-4000:], reasoning_chars=len(reasoning), finish_reason=r.choices[0].finish_reason,
                       usage={"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens},
                       pred=pred, seconds=round(time.time() - t0, 1))
            if t in sols:
                rec["correct"] = pred == sols[t][p]
        except Exception as exc:
            rec.update(error=f"{type(exc).__name__}: {exc}"[:500], seconds=round(time.time() - t0, 1))
            if slot is not None:  # a dead server answers instantly; wait so its slot does not burn the job list
                time.sleep(60)
        if slot is not None:
            free.put(slot)
        with lock, out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    n_ok = n_done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(work, *j) for j in jobs]):
            rec = fut.result()
            n_done += 1
            n_ok += bool(rec.get("correct"))
            print(f"[{n_done}/{len(jobs)}] {rec['task_id']}.{rec['pair']}#{rec['attempt']} "
                  f"{rec.get('error') or ('correct=' + str(rec.get('correct')) + ' tok=' + str(rec['usage']['completion']) + ' ' + str(rec['finish_reason']))}",
                  flush=True)
    print(f"done: {n_ok}/{n_done} correct this run", flush=True)


if __name__ == "__main__":
    main()
