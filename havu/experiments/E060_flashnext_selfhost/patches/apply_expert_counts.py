"""E060: install the expert-count hook into /root/e060/svenv's own vLLM 0.32.0 copy (idempotent; keeps a .orig)."""
import shutil
import sys
from pathlib import Path

SP = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/e060/svenv/lib/python3.12/site-packages/vllm")
shutil.copy(Path(__file__).with_name("e060_expert_counts.py"), SP / "e060_expert_counts.py")
f = SP / "v1/worker/gpu/model_runner.py"
s = f.read_text()
if "e060_counter" in s:
    print("already patched")
    sys.exit(0)
shutil.copy(f, f.with_suffix(".py.orig"))
a_old = """            if self.vllm_config.aux_output_config.enabled:
                self.aux_output_connector = get_aux_output_connector(
                    self.model, self.vllm_config, kv_cache_config
                )
"""
a_new = a_old + """            # E060: routed-expert selection counts (env E060_EXPERT_COUNTS=<dir>; vllm/e060_expert_counts.py)
            import os as _e060_os
            if _e060_os.environ.get("E060_EXPERT_COUNTS") and self.aux_output_connector is None:
                from vllm.e060_expert_counts import ExpertCounter
                self.e060_counter = ExpertCounter(
                    self.model, self.vllm_config, _e060_os.environ["E060_EXPERT_COUNTS"])
"""
b_old = """        pending_aux_output = None
        if self.aux_output_connector is not None:
            pending_aux_output = self.aux_output_connector.prepare_output(input_batch)
"""
b_new = b_old + """        if getattr(self, "e060_counter", None) is not None:
            self.e060_counter.step(input_batch)
"""
assert s.count(a_old) == 1 and s.count(b_old) == 1, "anchor not found exactly once"
f.write_text(s.replace(a_old, a_new).replace(b_old, b_new))
print("patched", f)
