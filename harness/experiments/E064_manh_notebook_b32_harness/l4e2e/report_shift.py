"""E064b: replica_report.py with Kaggle's startup gap set explicitly (KAGGLE_SHIFT_S, default 600 s in the toolkit).
B32-M on Kaggle (V2/V3 commit runs): solver at +10-13 min, server up after 1105-1130 s, so first attempts at ~+30-32 min
against ~+9 min on the pod: a gap of ~1300-1400 s.
    python report_shift.py <shift_s> <replica_report.py args...>"""
import sys

sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit")
import replica_report  # noqa: E402

replica_report.KAGGLE_SHIFT_S = float(sys.argv[1])
sys.argv = [sys.argv[0]] + sys.argv[2:]
replica_report.main()
