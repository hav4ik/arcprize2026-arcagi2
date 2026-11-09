import os, time
from huggingface_hub import HfApi
tok = open("/root/.e028_hf_token").read().strip()
api = HfApi(token=tok)
print("whoami:", api.whoami().get("name"), flush=True)
jobs = [("livobereshka/Qwen3.8-27B-e028-sft-lora", "/workspace/e028/adapter_ext"),
        ("livobereshka/Qwen3.8-27B-e028-sft-FP8", "/workspace/models/Qwen3.8-27B-e028-sft-FP8"),
        ("livobereshka/Qwen3.8-27B-e028-sft", "/workspace/models/Qwen3.8-27B-e028-sft")]
for repo, folder in jobs:
    t = time.time()
    url = api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
    print("repo", url, flush=True)
    api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type="model", private=True,
                            ignore_patterns=[".cache/**"], num_workers=16, print_report=False)
    print("uploaded", repo, "in", round(time.time() - t), "s", flush=True)
print("ALL DONE", flush=True)
