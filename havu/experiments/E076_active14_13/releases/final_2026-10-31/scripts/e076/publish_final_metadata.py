#!/usr/bin/env python3
"""Publish owner-authorized code/metadata to the existing private E076 model.

Use an explicit --publish flag. No model weights/configuration are changed.
All payload files are secret-scanned and remote hashes are verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/E076_active14_13'
REPO = 'chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    files = sorted(f for f in args.payload.rglob('*') if f.is_file())
    assert files
    names = [f.relative_to(args.payload).as_posix() for f in files]
    assert all(n == 'README.md' or n.startswith(('evaluation/e076_final_', 'reproducibility/')) for n in names)
    assert not any(n.endswith('.safetensors') for n in names)
    subprocess.run([sys.executable, str(ROOT / 'data_tools/secret_scan.py'), '-'],
                   input=''.join(str(f) + '\n' for f in files), text=True, check=True)
    if not args.publish:
        print(json.dumps({'repo': REPO, 'files': len(files), 'bytes': sum(f.stat().st_size for f in files), 'published': False}))
        return
    token = yaml.safe_load((ROOT / 'SECRETS.yaml').read_text())['huggingface_full_access_api_key']
    api = HfApi(token=token)
    before = api.model_info(REPO, files_metadata=True)
    assert before.private and before.sha == args.parent
    assert api.whoami()['name'] == REPO.split('/')[0]
    old = {f.rfilename: f for f in before.siblings}
    for name in names:
        assert name == 'README.md' or name not in old, 'Refusing to overwrite an existing versioned artifact: ' + name
    commit = api.create_commit(REPO, repo_type='model', parent_commit=before.sha,
        operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=f) for n, f in zip(names, files)],
        commit_message='E076 final image/text tools results, reproducible code and SFT metadata',
        commit_description='All required 120-task x2 Kaggle evaluations complete. Text pass@1 loss3.06pp; image loss0.76pp. Includes exact schedules, frozen code, calibration provenance, five prediction files, cache/throughput evidence and independent verification. Weights and serving configuration unchanged; repository remains private.',
        num_threads=4)
    after = api.model_info(REPO, revision=commit.oid, files_metadata=True)
    assert after.private
    remote = {f.rfilename: f for f in after.siblings}
    for name, previous in old.items():
        if name != 'README.md':
            assert remote[name].blob_id == previous.blob_id and remote[name].size == previous.size, name
            if previous.lfs:
                assert remote[name].lfs.sha256 == previous.lfs.sha256, name
    for name, local in zip(names, files):
        data = local.read_bytes()
        got = remote[name]
        assert got.size == len(data), name
        if got.lfs:
            assert got.lfs.sha256 == hashlib.sha256(data).hexdigest(), name
        else:
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert got.blob_id == blob, name
    for name in ('README.md', 'evaluation/e076_final_tools.json', 'reproducibility/E076_final_2026-10-31/metadata.json'):
        downloaded = Path(hf_hub_download(REPO, name, revision=commit.oid, token=token, cache_dir='/tmp/e076-final-verify-cache'))
        assert downloaded.read_bytes() == (args.payload / name).read_bytes(), name
    receipt = {'repo': REPO, 'private': after.private, 'parent_revision': before.sha,
        'revision': commit.oid, 'url': f'https://huggingface.co/{REPO}/commit/{commit.oid}',
        'verified_added_or_updated_files': len(files),
        'preserved_existing_files': len(old) - 1,
        'unchanged_weight_files': sum(n.endswith('.safetensors') for n in old),
        'verification': 'All payload sizes and LFS SHA256/Git blob hashes match; critical metadata downloaded and byte-compared; existing files except model card unchanged.'}
    (EXP / 'upload/final_metadata_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
