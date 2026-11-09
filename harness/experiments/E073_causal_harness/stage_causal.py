#!/usr/bin/env python3
"""Run on the allocated pod, after the gold server exits. HF token on stdin only."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        out = super().redirect_request(req, fp, code, msg, headers, newurl)
        if out and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            out.remove_header('Authorization')
        return out


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    token = sys.stdin.readline().strip()
    assert token, 'token missing'
    manifest = json.load(open('/root/e073/frozen/model_manifest.json'))
    root = Path('/root/models/causal'); root.mkdir(exist_ok=True)
    old = Path('/root/models/e128')
    files = manifest['files']
    base = 'https://huggingface.co/' + manifest['repo_id'] + '/resolve/' + manifest['revision'] + '/'

    def fetch(name):
        size, sha = files[name]; dst = root / name
        if dst.exists() and dst.stat().st_size == size and digest(dst) == sha:
            return name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith('model-engram-'):
            origin = old / name
            assert origin.stat().st_size == size and digest(origin) == sha, name
            dst.symlink_to(origin)
            return name
        for attempt in range(3):
            try:
                req = urllib.request.Request(base + urllib.parse.quote(name, safe='/'),
                                             headers={'Authorization': 'Bearer ' + token})
                op = urllib.request.build_opener(Redirect())
                tmp = dst.with_name(dst.name + '.e073.part')
                with op.open(req, timeout=180) as response, open(tmp, 'wb') as f:
                    shutil.copyfileobj(response, f, 8 << 20)
                assert tmp.stat().st_size == size and digest(tmp) == sha, name
                tmp.replace(dst)
                return name
            except Exception:
                if attempt == 2:
                    raise RuntimeError('download or hash verification failed: ' + name) from None
                time.sleep(5 * (attempt + 1))

    fetch('config.json')  # verify access and revision before removing re-downloadable control weights
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 2048 for x in gpu.split()), 'GPU still in use; refusing model swap'
    # Preserve all Engram tables and small files. Only the old checkpoint's replaceable non-Engram weights go.
    obsolete = list((old / 'gptq').glob('*.safetensors'))
    obsolete += [old / 'model-00001-of-00002.safetensors', old / 'model-00002-of-00002.safetensors']
    released = sum(p.stat().st_size for p in obsolete if p.exists())
    for p in obsolete:
        p.unlink(missing_ok=True)
    print(f'Freed {released/1e9:.1f} GB of re-downloadable old weights; retained Engram', flush=True)
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        for i, name in enumerate(ex.map(fetch, files), 1):
            print(f'{i}/{len(files)} verified {name}', flush=True)
    cfg = json.load(open(root / 'config.json'))
    assert not cfg.get('is_mm_prefix_lm', False), 'bidirectional image attention configuration'
    assert all((root / name).stat().st_size == sz for name, (sz, sha) in files.items())
    Path('/root/e073/model_ready.json').write_text(json.dumps({
        'repo_id': manifest['repo_id'], 'revision': manifest['revision'], 'verified_files': len(files)}))


if __name__ == '__main__':
    main()
