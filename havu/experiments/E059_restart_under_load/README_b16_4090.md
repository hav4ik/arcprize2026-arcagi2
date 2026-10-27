# E059 B16 on 4x RTX 4090: restart under load at 16 GiB KV offload

Status (2026-10-27): **running** (results below are filled in when the run ends).

## Question

Candidate "B16" is the configuration of the Kaggle-smoked notebook: candidate B (venv_candb = kit overlay + E054 indexer
skip + E051 KV offload, pinned_candb solver `77fe80a3`), 56 workers, KV gate 0.85 with the queue hold, and 16 GiB of
private pinned KV offload. If the vLLM engine dies mid-run, does the solver's restart path hold up?
- The old server's Engram shared memory (~91 GiB) and pinned pool (16 GiB) must be released before the new server
  allocates its own. Otherwise the host briefly needs ~2x, which on Kaggle (173 GiB cgroup) would be fatal.
- The solver must detect the death, restart the server, and re-queue the lost attempts without using their job
  tries (S15).
- Throughput must recover.

Design: the twin of the L4 run `launch_b32kill.sh` (32 GiB) with `killer.sh`: SIGKILL of this run's own
`VLLM::Worker_TP1` (found among the solver's descendants) 1.5 h after "server up", with host memory sampled every 2 s
from 5 min before to 20 min after.

**Pass rule** (set before the run): the peak anon+shmem during the restart is <= ~1.1x the steady value before the kill.
