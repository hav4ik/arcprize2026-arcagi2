#!/usr/bin/env python3
"""E043: upload artifacts / checkpoints to the user's Hugging Face account (private repos). Token from $HF_TOKEN.

  hf_upload.py --repo chankhavu/NAME --type dataset|model --path LOCAL [--in-repo DIR] [--large]
--large uses upload_large_folder (resumable, many files / large files); otherwise upload_folder or upload_file.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--in-repo", default="")
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--message", default="E043 upload")
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type=a.type, private=True, exist_ok=True)
    p = Path(a.path)
    if p.is_file():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=(a.in_repo.rstrip("/") + "/" if a.in_repo else "") + p.name,
                        repo_id=a.repo, repo_type=a.type, commit_message=a.message)
    elif a.large:
        assert not a.in_repo, "upload_large_folder mirrors the folder at the repo root"
        api.upload_large_folder(repo_id=a.repo, repo_type=a.type, folder_path=str(p), num_workers=16)
    else:
        api.upload_folder(repo_id=a.repo, repo_type=a.type, folder_path=str(p), path_in_repo=a.in_repo or None,
                          commit_message=a.message)
    print("uploaded", p, "->", a.repo, a.in_repo)


if __name__ == "__main__":
    main()
