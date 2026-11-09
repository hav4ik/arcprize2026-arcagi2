import time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from tinker_cookbook import weights
t = time.time()
weights.build_hf_model(base_model="/workspace/models/Qwen3.8-27B", adapter_path="/workspace/e028/adapter_ext",
                       output_path="/workspace/models/Qwen3.8-27B-e028-sft")
print("merged in", round(time.time() - t), "s", flush=True)
