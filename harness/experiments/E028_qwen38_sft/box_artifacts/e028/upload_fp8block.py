import os, sys, time
which = sys.argv[1]
d = "/workspace/models/Qwen3.8-27B-e028-sft-FP8block"
t = time.time()
if which == "kaggle":
    os.environ["KAGGLE_API_TOKEN"] = open("/root/.e028_kaggle_token").read().strip()
    import kagglehub
    kagglehub.model_upload("chankhavu/qwen3-8-27b-e028-sft/transformers/fp8block", d, license_name="Apache 2.0",
        version_notes="E028 SFT re-quantized to the official Qwen3.8-27B-FP8 block layout (fp8 e4m3, 128x128, dynamic act); Tinker parity within +/-0.002 trained-token NLL.",
        ignore_patterns=[".cache/*", ".cache/**"])
else:
    from huggingface_hub import HfApi
    api = HfApi(token=open("/root/.e028_hf_token").read().strip())
    api.create_repo("livobereshka/Qwen3.8-27B-e028-sft-FP8block", private=True, exist_ok=True, repo_type="model")
    api.upload_large_folder(repo_id="livobereshka/Qwen3.8-27B-e028-sft-FP8block", folder_path=d, repo_type="model",
                            private=True, ignore_patterns=[".cache/**"], num_workers=16, print_report=False)
print("DONE", which, round(time.time() - t), "s", flush=True)
