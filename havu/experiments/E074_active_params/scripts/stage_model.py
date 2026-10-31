#!/usr/bin/env python3
"""Stage the pinned causal E128; receive HF token over SSH stdin, never save it."""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/workspace/dsv4')
REPO = 'chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal'
REVISION = '209c810a747c043c2aca474843de8e357f920ef6'
ENGRAM_REPO = 'chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit'
ENGRAM_REVISION = '9bce98c5ecda0a9226edc6eab72a689089db1e37'


def main():
    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit('Missing token on stdin')
    api = HfApi(token=token)
    model = api.model_info(REPO, revision=REVISION, files_metadata=True)
    engram = api.model_info(ENGRAM_REPO, revision=ENGRAM_REVISION, files_metadata=True)
    assert model.sha == REVISION
    target = ROOT / 'models/causal-e128'
    target.mkdir(parents=True, exist_ok=True)
    print('Downloading pinned causal E128 and original Engram tables', flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=target, token=token, max_workers=4)
    source = ROOT / 'models/engram-source'
    patterns = ['model-engram-*.safetensors', 'model.safetensors.index.json', 'config.json']
    snapshot_download(ENGRAM_REPO, revision=engram.sha, local_dir=source, token=token,
                      allow_patterns=patterns, max_workers=4)
    tables = [f for f in engram.siblings if f.rfilename.startswith('model-engram-')]
    for base, entries in [(target, model.siblings), (source, tables)]:
        for entry in entries:
            path = base / entry.rfilename
            if not path.is_file() or path.stat().st_size != entry.size:
                raise RuntimeError('Missing or wrong size: ' + entry.rfilename)
    cfg = json.loads((target / 'config.json').read_text())['text_config']
    src_cfg = json.loads((source / 'config.json').read_text())['text_config']
    for key in set(cfg) | set(src_cfg):
        if key.startswith('engram_') and cfg.get(key) != src_cfg.get(key):
            raise RuntimeError('Engram geometry mismatch: ' + key)
    # E069 already validates these same compact tables in their 3-file packaging.
    # Retain causal attention/router/shared experts; replace only table index entries.
    index_path = target / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    source_index = json.loads((source / 'model.safetensors.index.json').read_text())
    weight_map = {k: v for k, v in index['weight_map'].items() if not v.startswith('model-engram-')}
    weight_map.update({k: v for k, v in source_index['weight_map'].items() if v.startswith('model-engram-')})
    for entry in tables:
        dst = target / entry.rfilename
        if not dst.exists():
            dst.symlink_to(source / entry.rfilename)
    (ROOT / 'causal_original_index.json').write_text(index_path.read_text())
    index['weight_map'] = weight_map
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    for filename in set(weight_map.values()):
        assert (target / filename).is_file(), filename
    required = list(model.siblings) + tables
    receipt = {'repo': REPO, 'revision': REVISION, 'engram_repo': ENGRAM_REPO,
               'engram_revision': engram.sha, 'files_size_verified': len(required),
               'bytes': sum(f.size for f in required)}
    (ROOT / 'model_ready.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
