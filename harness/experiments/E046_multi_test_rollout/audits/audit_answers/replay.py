import gzip, io, json, sys, tarfile
sys.path.insert(0, sys.argv[1])
import run_deepseek_tools as h
from sandbox_client import SandboxSession
CH = json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_evaluation_challenges.json"))
SOL = json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_evaluation_solutions.json"))
E = "/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/"
for tar, name in [(E + "kaggle_bench/dsv41_traces.tar.gz", "traces/da515329_p0_a0.json.gz"),
                  (E + "pods/runs/c36_heavy/out/dsv41_traces.tar.gz", "traces/de809cff_p0_a0.json.gz")]:
    tf = tarfile.open(tar)
    mem = [m for m in tf.getmembers() if m.name.endswith(name.split("/")[-1])][0]
    tr = json.load(gzip.open(io.BytesIO(tf.extractfile(mem).read()), "rt"))
    t = name.split("/")[-1].split("_")[0]
    codes = []
    for m in tr["messages"]:
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] in ("run_python", "run_ppython"):
                try:
                    codes.append(json.loads(c["function"]["arguments"])["code"])
                except Exception:
                    pass
    task = CH[t]
    with SandboxSession("http://172.18.0.6:6000", task["train"], task["test"][0]["input"], timeout=60) as s:
        for code in codes:
            s.execute(code)
        out1, g1 = h.submit_answer(s, "solve(test_input)")
        out2, g2 = h.submit_answer(s, "solve(np.array(test_input))")
    print(t, "cells", len(codes), "| solve(list):", g1 is not None, "| solve(np.array):", g2 is not None,
          "correct:", g2 == SOL[t][0] if g2 is not None else None, out2[:100])
