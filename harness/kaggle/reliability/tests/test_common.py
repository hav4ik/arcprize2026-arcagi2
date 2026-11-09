"""reli_common: host memory (cgroup v1 / v2 vs host-wide meminfo), processes, nvidia-smi, vLLM and solver logs,
trend maths."""
import calendar
import os
import unittest

from kit_helpers import (GIB, TmpCase, cgroup_v1, cgroup_v2, fixture, make_proc, make_server_tree, read,
                         write_meminfo, write_self_cgroup)
import reli_common as rc


class HostMemory(TmpCase):
    def test_meminfo_in_bytes(self):
        write_meminfo(self.path("proc"))
        mi = rc.read_meminfo(self.path("proc"))
        self.assertAlmostEqual(mi["AnonPages"] / GIB, 131.70, places=1)
        self.assertNotIn("MemFree", mi)

    def test_cgroup_v1_total_rss_plus_total_shmem(self):
        cg = self.path("cg")
        cgroup_v1(cg)
        r = rc.read_cgroup_memory(cg, self.path("proc"))
        self.assertEqual(r["version"], 1)
        self.assertAlmostEqual(r["anon"] / GIB, 120.4, places=2)  # total_rss, not the non-hierarchical rss
        self.assertAlmostEqual(r["shmem"] / GIB, 6.9, places=2)
        self.assertIsNone(r["limit"])  # 9223372036854771712 = unlimited

    def test_cgroup_v1_limit_and_rss_fallback(self):
        cg = self.path("cg")
        os.makedirs(os.path.join(cg, "memory"))
        with open(os.path.join(cg, "memory", "memory.stat"), "w") as f:
            f.write("rss %d\nshmem %d\ncache 5\n" % (10 * GIB, 2 * GIB))
        with open(os.path.join(cg, "memory", "memory.limit_in_bytes"), "w") as f:
            f.write("%d\n" % (173 * GIB))
        r = rc.read_cgroup_memory(cg, self.path("proc"))
        self.assertEqual((r["version"], r["anon"], r["shmem"], r["limit"]), (1, 10 * GIB, 2 * GIB, 173 * GIB))

    def test_cgroup_v2_anon_plus_shmem(self):
        cg = self.path("cg")
        cgroup_v2(cg, memory_max=str(173 * GIB))
        r = rc.read_cgroup_memory(cg, self.path("proc"))
        self.assertEqual(r["version"], 2)
        self.assertAlmostEqual((r["anon"] + r["shmem"]) / GIB, 127.3, places=2)
        self.assertEqual(r["limit"], 173 * GIB)

    def test_cgroup_v2_nested_path_from_proc_self(self):
        cg, proc = self.path("cg"), self.path("proc")
        cgroup_v2(cg, sub="docker/abc", memory_max="max")
        write_self_cgroup(proc, "0::/docker/abc\n")
        r = rc.read_cgroup_memory(cg, proc)
        self.assertEqual(r["path"], os.path.join(cg, "docker/abc"))
        self.assertIsNone(r["limit"])

    def test_v1_is_preferred_on_hybrid_hosts(self):
        cg = self.path("cg")
        cgroup_v1(cg)
        with open(os.path.join(cg, "cgroup.controllers"), "w") as f:
            f.write("")
        self.assertEqual(rc.read_cgroup_memory(cg, self.path("proc"))["version"], 1)

    def test_kaggle_eq_uses_the_container_not_the_host(self):
        """Pod 2 at 03:56: meminfo AnonPages 131.7 GiB (host-wide), cgroup total_rss 120.4 + shmem 6.9 GiB."""
        write_meminfo(self.path("proc"))
        cgroup_v1(self.path("cg"))
        mi = rc.read_meminfo(self.path("proc"))
        cg = rc.read_cgroup_memory(self.path("cg"), self.path("proc"))
        keq, src = rc.kaggle_equivalent(cg, mi)
        self.assertEqual(src, "cgroup_v1")
        self.assertAlmostEqual(keq, 120.4 + 6.9 + 7.0, places=1)
        keq_mi, src_mi = rc.kaggle_equivalent({"version": None}, mi)
        self.assertEqual(src_mi, "meminfo_hostwide")
        self.assertGreater(keq_mi - keq, 11)  # the foreign anon the meminfo form would have counted

    def test_no_cgroup_no_meminfo(self):
        self.assertEqual(rc.kaggle_equivalent({"version": None}, {}), (None, None))


class Processes(TmpCase):
    def build(self):
        proc = self.path("proc")
        make_proc(proc, 100, ["/root/kenv/bin/python", "-m", "ipykernel_launcher", "-f", "k.json"])
        make_proc(proc, 200, ["python", "/tmp/dsv41/kit/solver_dsv41.py", "--x"], ppid=100, rss_gib=0.9)
        make_proc(proc, 201, ["python", "-c", "import ast, contextlib\n..."], ppid=200, rss_gib=0.2)
        make_proc(proc, 202, ["python", "-c", "worker"], ppid=200, rss_gib=0.3)
        make_proc(proc, 300, ["python", "/kaggle/input/datasets/u/src/dsv41_vllm.py", "--supervise", "3"], ppid=100)
        make_server_tree(proc, parent=300)
        make_proc(proc, 20900, ["python", "-c", "from multiprocessing.resource_tracker import main"], ppid=19687)
        make_proc(proc, 400, ["/usr/bin/python3", "/root/kit/kaggle/reliability/sampler.py", "--out", "m"])
        make_proc(proc, 401, ["sshd: root"])
        return proc

    def test_titles_are_exact_with_nul_padding(self):
        proc = self.build()
        self.assertEqual(rc.read_argv(20828, proc), ["VLLM::Worker_TP3"])
        self.assertEqual(rc.exact_title(rc.read_argv(20828, proc)), "VLLM::Worker_TP3")
        self.assertIsNone(rc.exact_title(rc.read_argv(200, proc)))

    def test_classify(self):
        proc = self.build()
        procs = rc.scan_procs(proc)
        g = rc.classify(procs)
        self.assertEqual(g[100], "kernel")
        self.assertEqual(g[200], "solver")
        self.assertEqual((g[201], g[202]), ("sandboxes", "sandboxes"))
        self.assertEqual(g[300], "supervisor")
        self.assertEqual(g[19687], "api_server")
        self.assertEqual(g[20496], "engine_core")
        self.assertEqual({g[p] for p in (20827, 20829, 20830, 20828)}, {"tp_workers"})
        self.assertEqual(g[20900], "vllm_other")
        self.assertEqual(g[400], "kit")
        self.assertEqual(g[401], "other")
        tot = rc.group_totals(procs, g)
        self.assertEqual(tot["tp_workers"]["n"], 4)
        self.assertAlmostEqual(tot["tp_workers"]["rss_gib"], 30 + 31 + 32 + 33, places=1)

    def test_a_server_started_by_the_solver_is_not_sandboxes(self):
        proc = self.path("proc")
        make_proc(proc, 200, ["python", "/tmp/dsv41/kit/solver_dsv41.py"], ppid=1)
        make_server_tree(proc, parent=200)
        make_proc(proc, 201, ["python", "-c", "sandbox"], ppid=200)
        g = rc.classify(rc.scan_procs(proc))
        self.assertEqual(g[19687], "api_server")
        self.assertEqual(g[20828], "tp_workers")
        self.assertEqual(g[201], "sandboxes")

    def test_descendants(self):
        procs = rc.scan_procs(self.build())
        self.assertEqual(set(rc.descendants(procs, 20496)), {20827, 20829, 20830, 20828})


class Gpu(unittest.TestCase):
    def test_parse_with_na_and_no_data(self):
        gpus = rc.parse_gpu_csv(read(fixture("nvidia_smi_gpus.csv")))
        self.assertEqual([g["index"] for g in gpus], [0, 1, 2, 3])
        self.assertIsNone(gpus[2]["power_w"])
        self.assertEqual(gpus[0]["free_mib"], 1420)
        apps = rc.parse_apps_csv(read(fixture("nvidia_smi_apps.csv")), gpus)
        self.assertEqual(apps[0], [20827, 0, 20820])
        self.assertEqual(apps[3], [20828, 3, None])  # "[No data]" right after a process exits

    def test_missing_binary_is_recorded(self):
        gpus, apps, err = rc.query_gpus("/nonexistent/nvidia-smi")
        self.assertEqual((gpus, apps), ([], []))
        self.assertIn("FileNotFoundError", err)


REF = calendar.timegm((2026, 10, 30, 0, 0, 0))


class VllmLogs(TmpCase):
    def feed(self, path):
        lg = rc.VllmLog(path)
        for line in read(path).splitlines():
            lg.feed(rc.ANSI.sub("", line), REF)
        return lg

    def test_ansi_dcp_caps_and_memory(self):
        lg = self.feed(fixture("server_ansi.log"))
        ranks = lg.ranks()
        self.assertEqual(sorted(ranks), ["TP0", "TP1", "TP2", "TP3"])
        tp3 = ranks["TP3"]
        self.assertEqual((tp3["cap"], tp3["cap_reserved0"]), (20.37, 19.71))
        self.assertAlmostEqual(tp3["cap_minus_reserved0"], 0.66)
        self.assertAlmostEqual(tp3["cap_minus_peak"], 20.37 - 19.98, places=3)
        self.assertAlmostEqual(tp3["cap_minus_reserved"], 20.37 - 20.20, places=3)
        self.assertEqual(tp3["retries"], 1)
        self.assertEqual(lg.kv_tokens, 1453240)
        self.assertEqual(lg.startup_t, calendar.timegm((2026, 10, 29, 21, 56, 20)))
        self.assertEqual(lg.counts["dead"], 2)
        self.assertEqual(lg.counts["traceback"], 1)
        self.assertEqual(lg.counts["oom"], 0)  # "0 OOMs" in GPU memory lines is not an OOM
        w = rc.worst_ranks(ranks)
        self.assertAlmostEqual(w["cap_minus_peak"], min(d["cap_minus_peak"] for d in ranks.values()))
        self.assertEqual(w["retries"], 1)
        dead_t = [e[0] for e in lg.events if e[1] == "dead"][0]
        self.assertEqual(dead_t, calendar.timegm((2026, 10, 29, 23, 40, 37)))

    def test_new_instance_in_the_same_file(self):
        p = self.path("server.log")
        with open(p, "w") as f:
            f.write("(Worker_TP0 pid=1) INFO 10-29 10:00:00 [w.py:1] Capped the PyTorch allocator at 20.0 GiB "
                    "(19.0 GiB reserved), keeping\n")
            f.write("(Worker_TP0 pid=1) INFO 10-29 10:01:00 [w.py:2] GPU memory: 18.0 GiB allocated (peak 19.5 GiB), "
                    "19.6 GiB reserved (0.2 GiB in split segments), 1.0 GiB free, 3 allocator retries, 0 OOMs.\n")
            f.write("(Worker_TP0 pid=2) INFO 10-29 11:00:00 [w.py:1] Capped the PyTorch allocator at 20.1 GiB "
                    "(19.0 GiB reserved), keeping\n")
        lg = self.feed(p)
        self.assertEqual(lg.instance, 1)
        self.assertNotIn("retries", lg.ranks()["TP0"])  # the old instance's memory line is gone
        self.assertTrue(lg.key.endswith("#1"))

    def test_memory_line_from_another_pid_gets_no_cap_difference(self):
        p = self.path("server.log")
        with open(p, "w") as f:
            f.write("(Worker_TP0 pid=1) INFO 10-29 10:00:00 Capped the PyTorch allocator at 20.0 GiB (19.0 GiB "
                    "reserved)\n(Worker_TP0 pid=9) INFO 10-29 10:01:00 GPU memory: 18.0 GiB allocated (peak 19.5 "
                    "GiB), 19.6 GiB reserved, 1.0 GiB free, 0 allocator retries, 0 OOMs.\n")
        r = self.feed(p).ranks()["TP0"]
        self.assertIsNone(r.get("cap_minus_peak"))
        self.assertIsNone(r["split"])  # the split part is optional

    def test_follower_incremental_partial_and_truncation(self):
        p = self.path("x.log")
        fo = rc.LogFollower()
        with open(p, "w") as f:
            f.write("a\nb\x1b[31mred\x1b[0m\npart")
        self.assertEqual(fo.read(p), (["a", "bred"], False))
        with open(p, "a") as f:
            f.write("ial\nc\n")
        self.assertEqual(fo.read(p), (["partial", "c"], False))
        self.assertEqual(fo.read(p), ([], False))
        with open(p, "w") as f:
            f.write("new\n")
        self.assertEqual(fo.read(p), (["new"], True))
        self.assertEqual(fo.read(self.path("missing.log")), ([], False))

    def test_server_logs_current_is_the_newest_with_ranks(self):
        a = self.path("dsv41_logs", "server", "server.log")
        b = self.path("dsv41_logs", "r1", "server", "server.log")
        with open(a, "w") as f:
            f.write(read(fixture("server_ansi.log")))
        with open(b, "w") as f:
            f.write("(Worker_TP0 pid=5) INFO 10-29 23:45:00 Capped the PyTorch allocator at 20.5 GiB (19.7 GiB "
                    "reserved)\n")
        os.utime(a, (REF - 100, REF - 100))
        s = rc.ServerLogs([self.path("dsv41_logs") + "/**/server*.log"]).update(REF)
        self.assertEqual(s["current_log"], b)
        self.assertEqual(s["logs"], 2)
        self.assertEqual(s["counts"]["dead"], 2)  # counted over every log
        self.assertEqual(list(s["ranks"]), ["TP0"])


class SolverLogs(TmpCase):
    def test_counts_t0_and_metrics(self):
        sl = rc.SolverLog(fixture("solver.log"))
        out = sl.update(calendar.timegm((2026, 10, 29, 23, 40, 0)))
        c = out["counts"]
        self.assertEqual((c["server_died"], c["server_up"], c["requeue"], c["retry"]), (1, 2, 2, 1))
        self.assertEqual(c["stop_error"], 2)  # finalized; the per-try "retry" line is not one
        self.assertEqual(c["timeout"], 1)
        self.assertEqual(c["kv_gate"], 1)
        self.assertEqual(out["t0"], calendar.timegm((2026, 10, 29, 21, 56, 29)) - 0.14 * 3600)
        self.assertEqual(out["metrics"]["gen_tok_s"], 482.0)
        self.assertEqual(out["metrics"]["waiting"], 6.0)
        self.assertFalse(out["down"])  # the death was followed by "server up after"

    def test_down_until_up_and_day_wrap(self):
        p = self.path("solver.log")
        with open(p, "w") as f:
            f.write("[23:59:50 +11.00h] server: 10.0 gen tok/s, prefix hits 0.1, uncached 1.0 tok/s, KV 0.5, "
                    "running 1.0, waiting 0.0, preemptions 0.0\n[00:00:10 +11.01h] server died (code -9)\n")
        sl = rc.SolverLog(p)
        out = sl.update(calendar.timegm((2026, 10, 30, 0, 1, 0)))
        self.assertTrue(out["down"])
        self.assertEqual(out["t0"], calendar.timegm((2026, 10, 29, 23, 59, 50)) - 11 * 3600)

    def test_stamp_helpers(self):
        ref = calendar.timegm((2026, 10, 30, 0, 5, 0))
        self.assertEqual(rc.stamp_epoch(23, 59, 0, ref), calendar.timegm((2026, 10, 29, 23, 59, 0)))
        self.assertEqual(rc.stamp_near(0, 10, 0, ref), calendar.timegm((2026, 10, 30, 0, 10, 0)))
        self.assertEqual(rc.stamp_near(23, 0, 0, ref), calendar.timegm((2026, 10, 29, 23, 0, 0)))


class Trends(unittest.TestCase):
    def test_linfit(self):
        self.assertAlmostEqual(rc.linfit([(0, 1), (1, 3), (2, 5), (3, 7)]), 2.0)
        self.assertIsNone(rc.linfit([(0, 1), (1, 2)]))
        self.assertIsNone(rc.linfit([(1, 1), (1, 2), (1, 3)]))

    def test_eta(self):
        self.assertAlmostEqual(rc.eta_hours(1.68, -0.193, 0.26, "down"), (1.68 - 0.26) / 0.193)
        self.assertIsNone(rc.eta_hours(1.68, 0.01, 0.26, "down"))
        self.assertEqual(rc.eta_hours(0.2, -0.1, 0.26, "down"), 0.0)
        self.assertAlmostEqual(rc.eta_hours(145.0, 2.0, 173.0, "up"), 14.0)

    def test_verdict_wording(self):
        st, txt = rc.trend_verdict("worst cap - peak", 1.68, -0.193, 0.26, "down", 2.0, 11.5)
        self.assertEqual(st, "RISK")
        self.assertIn("shrinking toward 0.26 GiB, ETA ~7.4 h", txt)
        st, txt = rc.trend_verdict("Kaggle-eq RAM", 134.3, 0.03, 173.0, "up", 2.0, 11.5)
        self.assertEqual(st, "OK")
        self.assertIn("stable", txt)
        st, txt = rc.trend_verdict("Kaggle-eq RAM", 150.0, 2.0, 173.0, "up", 2.0, 11.5)
        self.assertEqual(st, "WATCH")  # ETA 11.5 h > 9.5 h left
        st, txt = rc.trend_verdict("GPU memory", 0.5, -0.1, 0.26, "down", 3.0, 11.5, blew_at_h=4.2)
        self.assertEqual((st, txt), ("FAIL", "GPU memory blew up at 4.20 h"))
        self.assertEqual(rc.trend_verdict("x", 1.0, None, 0.26, "down", 1, 11.5)[0], "N/A")

    def test_segments_and_post_warmup(self):
        recs = [{"t": i * 60.0, "vllm": {"server_key": "a" if i < 30 else "b", "startup_t": None}} for i in range(60)]
        segs = rc.segments(recs)
        self.assertEqual([len(s) for s in segs], [30, 30])
        w = rc.post_warmup(segs[1], 20 * 60, None)
        self.assertEqual(w[0]["t"], (30 + 20) * 60.0)


if __name__ == "__main__":
    unittest.main()
