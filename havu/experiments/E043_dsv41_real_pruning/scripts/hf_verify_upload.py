#!/usr/bin/env python3
"""E043: compare an uploaded HF repo with the local folder by file name and size (token in $HF_TOKEN).

  hf_verify_upload.py --repo chankhavu/NAME --path LOCAL [--type model]
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--type", default="model")
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.repo_info(a.repo, repo_type=a.type)
    remote = {f.path: f.size for f in api.list_repo_tree(a.repo, repo_type=a.type, recursive=True)
              if getattr(f, "size", None) is not None}
    root = Path(a.path)
    local = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*")
             if p.is_file() and ".cache" not in p.relative_to(root).parts}
    missing = sorted(set(local) - set(remote))
    wrong = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])
    extra = sorted(set(remote) - set(local) - {".gitattributes"})
    big = sorted(k for k, v in remote.items() if k.endswith(".safetensors") and v > 5_000_000_000)
    print(f"{a.repo}: private={info.private}, {len(remote)} remote files, {sum(remote.values()) / 1e9:.2f} GB; "
          f"local {len(local)} files")
    print(f"missing {missing[:5]} ({len(missing)}), size mismatch {wrong[:5]} ({len(wrong)}), extra {extra[:5]}, "
          f"safetensors above 5 GB {big}")
    if info.private and not missing and not wrong and not big:
        print("UPLOAD-VERIFIED")


if __name__ == "__main__":
    main()
