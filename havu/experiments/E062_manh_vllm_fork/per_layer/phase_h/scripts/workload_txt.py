"""workload.txt for a replica run (same content as manh_v12/data/workload.txt). usage: workload_txt.py RUN_LABEL START_UTC"""
import filecmp, hashlib, json, sys
label, start = sys.argv[1:3]
nbp = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
ch, so = "/root/data/evalx2t_s2026_challenges.json", "/root/data/evalx2t_s2026_solutions.json"
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
c, s = json.load(open(ch)), json.load(open(so))
print(f"Replica workload ({label}, rerun mode, pod 2, from {start} UTC)")
print(f"challenges as the notebook sees them: {nbp}")
print(f"  = copy of {ch} (byte-identical: {'yes' if filecmp.cmp(nbp, ch, shallow=False) else 'NO'})")
print(f"  sha256 {h(nbp)}")
print(f"solutions (scoring only, not visible to the notebook): {so}")
print(f"  sha256 {h(so)}")
print(f"tasks {len(c)} test pairs {sum(len(v['test']) for v in c.values())} | solutions tasks {len(s)} pairs {sum(len(v) for v in s.values())} | ids match {set(c) == set(s)}")
print("local repo copies: experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_{challenges,solutions}.json, sha256 83b249b1398d1b9c / 8ed7202a78eb3fcf")
