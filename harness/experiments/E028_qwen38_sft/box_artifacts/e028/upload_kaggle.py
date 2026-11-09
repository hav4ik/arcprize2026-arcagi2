import os, time, kagglehub
os.environ["KAGGLE_API_TOKEN"] = open("/root/.e028_kaggle_token").read().strip()
notes = ("E028: Qwen3.8-27B LoRA r128 SFT (Tinker, Dimuon 3e-3, 1 epoch, 252 steps) on 2,500 DeepSeek-V4.1-Flash "
         "ARC tool-use rollouts. Parity (assistant-token NLL, 8 seqs): Tinker 0.5269 / merged BF16 0.5273 / ModelOpt FP8 0.5272. "
         "Inference must use the E025 harness interface (run_deepseek_tools.py lean2roles, task summary, helpers, submit_answer). "
         "Mirror of HF livobereshka/Qwen3.8-27B-e028-sft{,-FP8,-lora}.")
jobs = [("chankhavu/qwen3-8-27b-e028-sft/transformers/bf16", "/workspace/models/Qwen3.8-27B-e028-sft")]
for handle, d in jobs:
    t = time.time()
    print("uploading", handle, "from", d, flush=True)
    kagglehub.model_upload(handle, d, license_name="Apache 2.0", version_notes=notes, ignore_patterns=[".cache/*", ".cache/**"])
    print("uploaded", handle, "in", round(time.time() - t), "s", flush=True)
print("ALL DONE", flush=True)
