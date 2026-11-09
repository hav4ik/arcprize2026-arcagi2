"""sampler.py over a synthetic pod: fake /proc, cgroup v1, host-wide meminfo, fake nvidia-smi, real log formats."""
import json
import os
import shutil
import time
import unittest

from kit_helpers import (TmpCase, cgroup_v1, fake_nvidia_smi, fixture, make_proc, make_server_tree, read,
                         write_meminfo)
import sampler


class Sampler(TmpCase):
    def setUp(self):
        super().setUp()
        self.proc = self.path("proc")
        write_meminfo(self.proc)
        make_proc(self.proc, 100, ["python", "-m", "ipykernel_launcher", "-f", "k.json"])
        make_proc(self.proc, 200, ["python", "/tmp/dsv41/kit/solver_dsv41.py"], ppid=100)
        for i in range(3):
            make_proc(self.proc, 210 + i, ["python", "-c", "import ast, contextlib"], ppid=200, rss_gib=0.5 + i)
        make_server_tree(self.proc, dcp=True, parent=100)
        self.cg = self.path("cg")
        cgroup_v1(self.cg)
        self.work = self.path("working")
        logs = os.path.join(self.work, "dsv41_logs")
        os.makedirs(os.path.join(logs, "server"))
        shutil.copy(fixture("server_ansi.log"), os.path.join(logs, "server", "server.log"))
        self.solver = os.path.join(logs, "solver.log")
        with open(self.solver, "w") as f:
            f.write(read(fixture("solver.log")))
        with open(os.path.join(logs, "attempts.jsonl"), "w") as f:
            f.write(json.dumps({"task_id": "a", "error": None}) + "\n" + json.dumps({"task_id": "b", "error": "x"})
                    + "\n")
        with open(os.path.join(logs, "e066_supervisor.json"), "w") as f:
            json.dump({"start": 1, "state": "up", "pid": 19687, "up_s": 150, "history": ["big"] * 50}, f)
        self.timeline = self.path("run", "nb.timeline")
        with open(self.timeline, "w") as f:
            f.write("21:47:58 +      0s START cell 0: import os\n")
        self.smi = fake_nvidia_smi(self.path("bin"))
        self.out = self.path("mon.jsonl")
        self.args = ["--out", self.out, "--working", self.work, "--proc-root", self.proc, "--cgroup-root", self.cg,
                     "--nvidia-smi", self.smi, "--timeline", self.timeline, "--shm", self.tmp]

    def test_one_sample(self):
        s = sampler.Sampler(sampler.build_args(self.args))
        r = s.sample(time.time())
        self.assertEqual(r["kaggle_eq_source"], "cgroup_v1")
        self.assertAlmostEqual(r["kaggle_eq_gib"], 134.3, places=1)
        self.assertGreater(r["kaggle_eq_meminfo_gib"], r["kaggle_eq_gib"] + 11)
        self.assertEqual(r["engines"], 1)
        self.assertEqual(r["groups"]["tp_workers"]["n"], 4)
        self.assertEqual(r["sandboxes"]["n"], 3)
        self.assertAlmostEqual(r["sandboxes"]["max_gib"], 2.5, places=2)
        self.assertEqual(len(r["gpus"]), 4)
        self.assertEqual(r["gpu_apps"][3], [20828, 3, None])
        v = r["vllm"]
        self.assertEqual(sorted(v["ranks"]), ["TP0", "TP1", "TP2", "TP3"])
        self.assertAlmostEqual(v["worst"]["cap_minus_peak"], 20.37 - 19.98, places=3)
        self.assertEqual(v["worst"]["retries"], 1)
        self.assertEqual(v["counts"]["dead"], 2)
        self.assertEqual(r["solver"]["counts"]["stop_error"], 2)
        self.assertIsNotNone(r["solver"]["t0"])
        self.assertEqual(r["attempts"], {"records": 2, "error_records": 1})
        self.assertEqual(r["supervisor"]["state"], "up")
        self.assertNotIn("history", r["supervisor"])
        self.assertIsNone(r["notebook_end"])
        self.assertFalse(r["fast"])
        json.dumps(r)

    def test_incremental_fast_mode_and_notebook_end(self):
        s = sampler.Sampler(sampler.build_args(self.args))
        s.sample(time.time())
        with open(self.solver, "a") as f:
            f.write("[23:50:00 +1.98h] server died (code -9)\n")
        srv = os.path.join(self.work, "dsv41_logs", "server", "server.log")
        with open(srv, "a") as f:
            f.write("(Worker_TP3_DCP3 pid=20828) INFO 10-29 23:49:20 [gpu_worker.py:1184] GPU memory: 19.4 GiB "
                    "allocated (peak 20.25 GiB), 20.3 GiB reserved (0.27 GiB in split segments), 0.9 GiB free, 4 "
                    "allocator retries, 1 OOMs.\n")
        r = s.sample(time.time())
        self.assertTrue(r["fast"])
        self.assertEqual(r["solver"]["counts"]["server_died"], 2)
        self.assertEqual(r["vllm"]["ranks"]["TP3"]["retries"], 4)
        self.assertEqual(r["vllm"]["worst"]["ooms"], 1)
        with open(self.solver, "a") as f:
            f.write("[23:52:00 +2.01h] server up after 120s (external, start 2)\n")
        with open(self.timeline, "a") as f:
            f.write("NOTEBOOK_DONE ok +43000s 09:50:00\n")
        r = s.sample(time.time())
        self.assertFalse(r["fast"])
        self.assertTrue(r["notebook_end"].startswith("NOTEBOOK_DONE ok"))

    def test_fault_log_triggers_fast_mode(self):
        fl = self.path("fault.jsonl")
        with open(fl, "w") as f:
            f.write("{}\n")
        s = sampler.Sampler(sampler.build_args(self.args + ["--fault-log", fl]))
        self.assertTrue(s.sample(time.time())["fast"])
        self.assertFalse(s.sample(time.time() + 3600)["fast"])

    def test_main_once(self):
        self.assertEqual(sampler.main(self.args + ["--once"]), 0)
        lines = [json.loads(x) for x in read(self.out).splitlines()]
        self.assertEqual([x["kind"] for x in lines], ["start", "sample"])
        self.assertEqual(lines[0]["config"]["working"], self.work)


if __name__ == "__main__":
    unittest.main()
