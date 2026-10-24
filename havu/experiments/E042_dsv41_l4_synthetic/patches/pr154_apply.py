#!/usr/bin/env python3
"""E042: apply fork PR #154 (MXFP4 indexer: decode q once per row) to the installed wheel, plus an env gate.

Upstream enables the predecode on SM120 only. DSV41_INDEXER_Q_PREDECODE=1 also enables it on SM89, so the
same install can run with it off (default, original behaviour) or on. Idempotent.
Usage: pr154_apply.py <pr154_kernel.diff> [<site-packages>]   (default: the box venv)
"""

import pathlib
import subprocess
import sys

SP = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/root/dsv41/venv/lib/python3.12/site-packages")
F = SP / "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py"

s = F.read_text()
if "_mxfp4_indexer_q_decode_kernel" not in s:
    subprocess.run(["patch", "-p1", "-i", sys.argv[1]], cwd=SP, check=True)
    s = F.read_text()
if "DSV41_INDEXER_Q_PREDECODE" not in s:
    old = "    return major == 12\n"
    new = ("    # E042: DSV41_INDEXER_Q_PREDECODE=1 also enables it on SM89 (A/B on L4).\n"
           '    return major == 12 or os.environ.get("DSV41_INDEXER_Q_PREDECODE") == "1"\n')
    assert s.count(old) == 1 and s.count("import functools\n") == 1
    s = s.replace(old, new).replace("import functools\n", "import functools\nimport os\n")
    F.write_text(s)
print("pr154 applied, env gate present:", "DSV41_INDEXER_Q_PREDECODE" in F.read_text())
