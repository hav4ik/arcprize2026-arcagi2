"""fault_inject.py: exact targeting (fake /proc and real decoy processes), dry runs, GPU burst logic with a fake
libcudart, cpu-stress, and the recovery checks of `verify`. Signals go only to decoys this test spawned."""
import io
import json
import os
import subprocess
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from kit_helpers import (TmpCase, fake_nvidia_smi, make_proc, make_server_tree, proc_state, read, record,
                         spawn_decoy, write_records)
import fault_inject as fi
import recovery
import reli_common as rc


def run(argv):
    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
        code = fi.main(argv)
    return code, out.getvalue(), err.getvalue()


def log_lines(path):
    return [json.loads(x) for x in read(path).splitlines()] if os.path.exists(path) else []


class Titles(unittest.TestCase):
    def test_worker_rank(self):
        self.assertEqual(fi.worker_rank("VLLM::Worker_TP3"), 3)
        self.assertEqual(fi.worker_rank("VLLM::Worker_TP0_DCP0"), 0)
        self.assertEqual(fi.worker_rank("VLLM::Worker_PP1_TP12"), 12)
        for t in ("VLLM::Worker_TP3x", "VLLM::Worker", "VLLM::Worker_DP0", "xVLLM::Worker_TP3", None, ""):
            self.assertIsNone(fi.worker_rank(t), t)

    def test_engine_title(self):
        self.assertTrue(fi.ENGINE_TITLE.fullmatch("VLLM::EngineCore"))
        self.assertTrue(fi.ENGINE_TITLE.fullmatch("VLLM::EngineCore_DP0"))
        self.assertFalse(fi.ENGINE_TITLE.fullmatch("VLLM::EngineCoreX"))


class FakeProcTargeting(TmpCase):
    def setUp(self):
        super().setUp()
        self.proc = self.path("proc")
        make_server_tree(self.proc, dcp=True)
        self.log = self.path("fault.jsonl")

    def test_dry_run_picks_the_exact_worker(self):
        make_proc(self.proc, 7000, ["python", "VLLM::Worker_TP3_DCP3"], ppid=20496)  # two tokens: not a title
        code, out, err = run(["kill-tp", "3", "--proc-root", self.proc, "--log", self.log, "--dry-run"])
        self.assertEqual(code, 0, err)
        e = log_lines(self.log)[-1]
        self.assertEqual((e["phase"], e["pid"], e["target"], e["server_pid"]),
                         ("dry_run", 20828, "VLLM::Worker_TP3_DCP3", 19687))
        self.assertEqual(e["signal"], "SIGKILL")

    def test_fake_proc_root_cannot_signal(self):
        code, _, err = run(["kill-tp", "3", "--proc-root", self.proc, "--log", self.log])
        self.assertEqual(code, 2)
        self.assertIn("dry runs only", err)

    def test_two_servers_need_server_pid(self):
        make_server_tree(self.proc, api=29000, engine=29001, workers=(29002, 29003, 29004, 29005), parent=1)
        code, _, err = run(["kill-tp", "3", "--proc-root", self.proc, "--log", self.log, "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("found 2 vLLM API servers", err)
        code, _, err = run(["kill-tp", "3", "--proc-root", self.proc, "--log", self.log, "--dry-run",
                            "--server-pid", "29000"])
        self.assertEqual(code, 0, err)
        self.assertEqual(log_lines(self.log)[-1]["pid"], 29005)

    def test_duplicate_or_missing_titles_refuse(self):
        make_proc(self.proc, 7001, ["VLLM::Worker_TP3"], ppid=20496, pad=10)
        code, _, err = run(["kill-tp", "3", "--proc-root", self.proc, "--log", self.log, "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("found 2", err)
        code, _, err = run(["kill-tp", "7", "--proc-root", self.proc, "--log", self.log, "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("found 0", err)
        self.assertEqual(log_lines(self.log)[-1]["phase"], "refused")

    def test_server_pid_one_refused(self):
        code, _, err = run(["kill-tp", "3", "--proc-root", self.proc, "--dry-run", "--server-pid", "1",
                            "--log", self.log])
        self.assertEqual(code, 2)

    def test_engine_dry_run_and_find(self):
        code, _, err = run(["sigstop-engine", "--proc-root", self.proc, "--log", self.log, "--dry-run"])
        self.assertEqual(code, 0, err)
        self.assertEqual(log_lines(self.log)[-1]["pid"], 20496)
        code, out, _ = run(["find", "--proc-root", self.proc, "--nvidia-smi", "/nonexistent"])
        self.assertIn("<- kill-tp 3", out)
        self.assertIn("<- sigstop-engine", out)

    def test_refuses_in_a_kaggle_kernel(self):
        with mock.patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
            code, _, err = run(["kill-tp", "3", "--proc-root", self.proc, "--dry-run", "--log", self.log])
        self.assertEqual(code, 2)
        self.assertIn("Kaggle kernel", err)


class LiveDecoys(TmpCase):
    """Real processes titled like vLLM's, children of this test process (passed as --server-pid)."""

    def setUp(self):
        super().setUp()
        self.kids = []
        self.log = self.path("fault.jsonl")

    def tearDown(self):
        for p in self.kids:
            try:
                p.kill()
                p.wait(5)
            except OSError:
                pass
            if p.stdin:
                p.stdin.close()
        super().tearDown()

    def decoy(self, title):
        p = spawn_decoy(title)
        self.kids.append(p)
        deadline = time.time() + 5
        while time.time() < deadline and rc.read_argv(p.pid) != [title]:
            time.sleep(0.01)
        return p

    def test_kill_tp_hits_only_the_target(self):
        tp2, tp3 = self.decoy("VLLM::Worker_TP2"), self.decoy("VLLM::Worker_TP3")
        code, _, err = run(["kill-tp", "3", "--server-pid", str(os.getpid()), "--log", self.log])
        self.assertEqual(code, 0, err)
        self.assertEqual(tp3.wait(5), -9)
        self.assertIsNone(tp2.poll())
        phases = [e["phase"] for e in log_lines(self.log)]
        self.assertEqual(phases, ["about_to_signal", "signaled"])

    def test_multi_token_cmdline_is_never_a_target(self):
        p = subprocess.Popen(["VLLM::Worker_TP3", "-c", "sleep 30"], executable="/bin/sh")
        self.kids.append(p)
        time.sleep(0.1)
        code, _, err = run(["kill-tp", "3", "--server-pid", str(os.getpid()), "--log", self.log])
        self.assertEqual(code, 2)
        self.assertIsNone(p.poll())

    def test_sigstop_and_sigcont_engine(self):
        eng = self.decoy("VLLM::EngineCore")
        args = ["--server-pid", str(os.getpid()), "--log", self.log]
        self.assertEqual(run(["sigstop-engine"] + args)[0], 0)
        time.sleep(0.05)
        self.assertEqual(proc_state(eng.pid), "T")
        self.assertEqual(run(["sigcont-engine"] + args)[0], 0)
        time.sleep(0.05)
        self.assertNotEqual(proc_state(eng.pid), "T")
        self.assertEqual(run(["sigstop-engine", "--resume-after", "0.2"] + args)[0], 0)
        time.sleep(0.05)
        self.assertNotEqual(proc_state(eng.pid), "T")
        self.assertEqual([e["phase"] for e in log_lines(self.log)][-1], "resumed")

    def test_pidfd_recheck_refuses_a_changed_cmdline(self):
        d = self.decoy("VLLM::Worker_TP1")
        with self.assertRaises(fi.Refused):
            fi.send_exact(d.pid, "VLLM::Worker_TP9", 9)
        self.assertIsNone(d.poll())

    def test_cpu_stress(self):
        code, _, err = run(["cpu-stress", "--procs", "1", "--seconds", "0.3", "--log", self.log])
        self.assertEqual(code, 0, err)
        self.assertEqual([e["phase"] for e in log_lines(self.log)], ["signaled", "done"])
        self.assertEqual(run(["cpu-stress", "--procs", "0", "--seconds", "1", "--log", self.log])[0], 2)
        self.assertEqual(run(["cpu-stress", "--procs", "1", "--seconds", "99999", "--log", self.log])[0], 2)
        self.assertEqual(run(["cpu-stress", "--procs", "1", "--seconds", "5", "--dry-run", "--log", self.log])[0], 0)


class FakeCudart:
    """cudaMalloc succeeds until `budget_mib` is allocated."""

    def __init__(self, free_mib, budget_mib):
        self.free, self.budget, self.used, self.calls = free_mib << 20, budget_mib << 20, 0, 0

    def cudaFree(self, p):
        return 0

    def cudaMemGetInfo(self, f, t):
        f._obj.value, t._obj.value = self.free, 23034 << 20
        return 0

    def cudaMalloc(self, pp, size):
        self.calls += 1
        if self.used + size.value > self.budget:
            return 2
        self.used += size.value
        pp._obj.value = 0x1000 + self.calls
        return 0

    def cudaMemset(self, p, v, n):
        return 0


class GpuBurst(TmpCase):
    def test_burst_takes_what_it_can(self):
        got, free = fi.burst(FakeCudart(1000, 900), chunk_mib=256, hold_s=5, sleep=lambda s: None)
        self.assertEqual((got, free), (900, 1000))
        got, _ = fi.burst(FakeCudart(1000, 5000), mib=300, sleep=lambda s: None)
        self.assertEqual(got, 300)
        got, _ = fi.burst(FakeCudart(1000, 5000), leave_mib=100, sleep=lambda s: None)
        self.assertEqual(got, 900)

    def test_gpu_checks(self):
        proc = self.path("proc")
        make_server_tree(proc)
        log = self.path("fault.jsonl")
        smi = fake_nvidia_smi(self.path("bin"), apps_csv="999, GPU-aaaa, 300\n")
        base = ["--proc-root", proc, "--nvidia-smi", smi, "--log", log, "--dry-run"]
        code, _, err = run(["gpu-oom-burst", "0"] + base)
        self.assertEqual(code, 2)
        self.assertIn("hosts no vLLM worker", err)
        self.assertEqual(run(["gpu-oom-burst", "0", "--allow-any-gpu"] + base)[0], 0)
        self.assertEqual(run(["gpu-oom-burst", "7", "--allow-any-gpu"] + base)[0], 2)
        smi2 = fake_nvidia_smi(self.path("bin2"))  # the fixture puts TP0 (20827) on GPU 0
        code, _, err = run(["gpu-oom-burst", "0", "--proc-root", proc, "--nvidia-smi", smi2, "--log", log,
                            "--dry-run"])
        self.assertEqual(code, 0, err)
        e = log_lines(log)[-1]
        self.assertEqual((e["phase"], e["gpu"], e["free_mib"]), ("dry_run", 0, 1420))


T = 1793310000.0
T0 = T - 1800
OLD = (19687, 20496, 20827, 20829, 20830, 20828)
NEW = (27380, 27811, 28147, 28148, 28149, 28150)


def stamp(t, text):
    return "[%s +%.2fh] %s\n" % (time.strftime("%H:%M:%S", time.gmtime(t)), (t - T0) / 3600, text)


class Recovery(TmpCase):
    def scenario(self, up_after=174, coexist=False, stop_error=False, retry=False, action="kill-tp"):
        recs = [record(T - 60 + 5 * i, keq=140, t0=T0, pids=OLD) for i in range(12)]
        recs.append(record(T + 12, keq=120, t0=T0, pids=OLD[:5]))
        recs.append(record(T + 20, keq=20, t0=T0, engines=0, pids=()))
        if coexist:
            recs.append(record(T + 25, keq=150, t0=T0, engines=2, pids=(OLD[1], NEW[1])))
        recs += [record(T + 30 + 5 * i, keq=60 + i, t0=T0, pids=NEW, key="r1#0") for i in range(40)]
        recs += [record(T + up_after + 60 * i, keq=138, t0=T0, pids=NEW, key="r1#0") for i in range(20)]
        recs.sort(key=lambda r: r["t"])
        sp = self.path("mon.jsonl")
        write_records(sp, recs)
        logs = self.path("dsv41_logs", "x")[:-2]
        with open(os.path.join(logs, "solver.log"), "w") as f:
            f.write(stamp(T - 300, "server up after 0s (external, start 0)"))
            if retry:
                f.write(stamp(T + 11, "retry aaa.0#0: TurnError: APIError: The server had an error"))
            f.write(stamp(T + 13, "server died (code 1); tail: EngineDeadError"))
            f.write(stamp(T + up_after, "server up after %ds (external, start 1)" % (up_after - 14)))
            for j in ("aaa", "bbb", "ccc"):
                f.write(stamp(T + up_after, "re-queue %s.0#0 after a server-side error (1): APIConnectionError" % j))
            if stop_error:
                f.write(stamp(T + up_after + 1, "[5] aaa.0#0 pred=no stop=error turns=1 out=0 0.5s"))
        with open(os.path.join(logs, "attempts.jsonl"), "w") as f:
            for j in ("aaa", "bbb", "ccc"):
                f.write(json.dumps({"task_id": j, "pair": 0, "attempt": 0, "error": "APIConnectionError",
                                    "t_end_h": (T + up_after - T0) / 3600}) + "\n")
            for j in ("aaa", "bbb"):
                f.write(json.dumps({"task_id": j, "pair": 0, "attempt": 0, "error": None,
                                    "t_end_h": (T + up_after + 900 - T0) / 3600}) + "\n")
        fl = self.path("fault.jsonl")
        with open(fl, "w") as f:
            f.write(json.dumps({"phase": "about_to_signal", "action": action, "t": T, "pid": 20828}) + "\n")
            f.write(json.dumps({"phase": "signaled", "action": action, "t": T, "pid": 20828,
                                "target": "VLLM::Worker_TP3"}) + "\n")
        return sp, logs, fl

    def checks(self, res):
        return {c[0]: c[1] for c in res["checks"]}

    def analyze(self, **kw):
        sp, logs, fl = self.scenario(**kw)
        f = recovery.load_faults(fl)
        self.assertEqual(len(f), 1)
        return recovery.analyze(f[0], rc.load_records(sp), os.path.join(logs, "solver.log"),
                                os.path.join(logs, "attempts.jsonl"))

    def test_clean_restart_passes(self):
        res = self.analyze()
        c = self.checks(res)
        self.assertEqual(res["verdict"], "WARN")  # one job still pending at the end of the data
        for k in ("finalized_errors", "server_died", "requeued", "restart_time", "one_engine", "ram_limit",
                  "tries_spent"):
            self.assertEqual(c[k], "PASS", (k, res["checks"]))
        self.assertEqual(c["jobs"], "WARN")
        text = recovery.format_result(res)
        self.assertIn("all pre-fault vLLM processes gone", text)
        self.assertIn("server up 174 s after the fault", text)

    def test_failures(self):
        c = self.checks(self.analyze(coexist=True, stop_error=True, retry=True))
        self.assertEqual((c["one_engine"], c["finalized_errors"], c["tries_spent"]), ("FAIL", "FAIL", "WARN"))
        self.assertEqual(self.checks(self.analyze(up_after=1200))["restart_time"], "FAIL")

    def test_hang_gets_the_stall_allowance(self):
        res = self.analyze(up_after=1800, action="sigstop-engine")
        self.assertEqual(self.checks(res)["restart_time"], "PASS")

    def test_verify_command(self):
        sp, logs, fl = self.scenario(stop_error=True)
        code, out, _ = run(["verify", "--samples", sp, "--logs-dir", logs, "--log", fl])
        self.assertEqual(code, 1)
        self.assertIn("FAIL  finalized_errors", out)
        sp, logs, fl = self.scenario()
        self.assertEqual(run(["verify", "--samples", sp, "--logs-dir", logs, "--log", fl])[0], 0)


if __name__ == "__main__":
    unittest.main()
