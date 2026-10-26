"""Mutation check: each mutant of the patched scheduler must fail some test."""
import os, shutil, subprocess, sys
K = sys.argv[1]
SRC = f"{K}/patched_src"
REL = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
MUTANTS = {
    "no_finished_cap": ("max(req.num_prompt_tokens, req.num_tokens - 1)", "req.num_tokens"),
    "no_alignment": ("        if hit_alignment > 1:\n            max_hit_size_tokens = round_down(max_hit_size_tokens, hit_alignment)\n", ""),
    "no_realign_rerun": ("            if realigned and not lookup_groups and not defer_lookup:", "            if False:"),
    "ckpt_desc_order": ("return targets, {g: sorted(c) for g, c in chunks.items()}", "return targets, {g: sorted(c, reverse=True) for g, c in chunks.items()}"),
    "ring_not_skipped_in_load": ("            if group_config.skip:\n                # [dsv41-offload] The scratch ring keeps", "            if False:\n                # [dsv41-offload] The scratch ring keeps"),
    "window_short": ("        window += int(group_config.is_eagle_group)", "        window += int(group_config.is_eagle_group) - 1"),
    "no_progress_protection": ("            req_status.group_states[group_idx].next_stored_chunk_idx = min(", "            req_status.group_states[group_idx].next_stored_chunk_idx = 10**9 + 0 * min("),
}
for name, (old, new) in MUTANTS.items():
    dst = f"{K}/mutant_src"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)
    p = f"{dst}/{REL}"
    s = open(p).read()
    assert s.count(old) == 1, name
    open(p, "w").write(s.replace(old, new))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{dst}:{K}/tests")
    r = subprocess.run([f"{K}/testvenv/bin/python", "-m", "unittest", "test_dsv41_kv_offload"],
                       cwd=f"{K}/tests", env=env, capture_output=True, text=True, timeout=900)
    tail = [l for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:", "Ran ", "OK", "FAILED"))]
    print(f"== {name}: rc={r.returncode}")
    for l in tail:
        print("   ", l)
shutil.rmtree(f"{K}/mutant_src", ignore_errors=True)
