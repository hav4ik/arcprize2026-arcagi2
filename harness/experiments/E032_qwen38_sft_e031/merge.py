"""E032: merge the exported Tinker LoRA adapter into Qwen/Qwen3.8-27B (BF16), on the box."""
import shutil, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from tinker_cookbook import weights
t = time.time()
weights.build_hf_model(base_model="/workspace/models/Qwen3.8-27B", adapter_path="/workspace/e032/adapter_ext",
                       output_path="/workspace/models/Qwen3.8-27B-e032-sft")
shutil.copy2("/workspace/models/Qwen3.8-27B/generation_config.json", "/workspace/models/Qwen3.8-27B-e032-sft/")
print("merged in", round(time.time() - t), "s", flush=True)
