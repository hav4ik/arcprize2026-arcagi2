#!/usr/bin/env python3
"""Freeze E073 controls, causal policy notebooks, and the existing E072 task split."""
import base64
import copy
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / 'kaggle/dsv41_tools'
sys.path.insert(0, str(D))
import input_paths as IP


def src(c):
    return c['source'] if isinstance(c['source'], str) else ''.join(c['source'])


def cell(nb, prefix):
    hits = [c for c in nb['cells'] if src(c).startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]


def overlay(nb):
    s = src(cell(nb, '# Cell 0b'))
    b = s.split('_MT = "', 1)[1].split('"', 1)[0]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b))) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf}


def set_overlay(nb, files):
    c = cell(nb, '# Cell 0b'); s = src(c)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in files.items():
            m = tarfile.TarInfo(name); m.size = len(data); m.mode = 0o644
            tf.addfile(m, io.BytesIO(data))
    old = s.split('_MT = "', 1)[1].split('"', 1)[0]
    s = s.replace(old, base64.b64encode(buf.getvalue()).decode(), 1)
    line = next(x for x in s.splitlines() if x.startswith('_MT_SHA = '))
    s = s.replace(line, '_MT_SHA = ' + json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in files.items()}))
    # Experimental controls fail loudly; a failed pin must never silently run a different harness.
    s = s.replace('        if not RERUN:\n', '        if True:  # E073: fail closed in experiments\n', 1)
    c['source'] = s


def policy_solver(text):
    assert 'E073_ATTEMPT_FORMATS' not in text
    anchor = 'ATTEMPTS = env_num("ARC_ATTEMPTS", 2, int)\n'
    settings = '''# E073: each rollout is fresh; formats never expose earlier predictions or thinking.
E073_ATTEMPT_FORMATS = tuple(E("ARC_ATTEMPT_FORMATS", "json,json,text+image").split(","))
if not E073_ATTEMPT_FORMATS or any(f not in ("json", "text+image") for f in E073_ATTEMPT_FORMATS):
    raise SystemExit("invalid ARC_ATTEMPT_FORMATS")
E073_EXTRA = E("ARC_EXTRA_IF_NO_TRAIN_PASS", "0") == "1"


def attempt_args(args, attempt):
    from copy import copy
    out = copy(args)  # shared Namespace is immutable across worker threads
    out.grid_format = E073_ATTEMPT_FORMATS[min(attempt, len(E073_ATTEMPT_FORMATS) - 1)]
    out.token_exact = True
    out.image_cell = 0
    return out


def needs_extra(records):
    final = {r["attempt"]: r for r in records if not r.get("provisional") and r.get("attempt") in (0, 1)}
    return (set(final) == {0, 1}
            and not any(r.get("train_pass") is True and valid_grid(r.get("pred")) for r in final.values()))

'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + settings)
    anchor = 'def run_attempt(args, task, t, p, a):\n'
    text = text.replace(anchor, anchor + '    args = attempt_args(args, a)\n', 1)
    text = text.replace('"cuts": client.stats, "t_end_h":', '"grid_format": args.grid_format, "cuts": client.stats, "t_end_h":', 1)
    anchor = '    held, free_starts = [], {}'
    text = text.replace(anchor, '    extra_queued = set()  # E073: at most one conditional third rollout per task/pair\n' + anchor, 1)
    anchor = '                            keep(rec, tasks[t])\n                        write_submission(all_tasks)'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''                            keep(rec, tasks[t])
                            if (E073_EXTRA and a == 1 and (t, p) not in extra_queued
                                    and needs_extra(RECORDS.get((t, p), []))):
                                extra_queued.add((t, p))
                                queue.append((t, p, 2, 0))
                                log(f"E073 conditional image retry queued: {t}.{p}")
                        write_submission(all_tasks)''', 1)
    compile(text, 'e073_solver', 'exec')
    return text


def write_nb(name, nb, meta):
    out = HERE / 'frozen'; out.mkdir(exist_ok=True)
    IP.finish(nb, meta)
    assert not IP.notebook_scans(nb)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            if not re.search(r'(?m)^\s*[!%]', src(c)):
                compile(src(c), f'{name}:cell', 'exec')
    p = out / f'{name}.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = HERE / 'frozen'; out.mkdir(exist_ok=True)
    gold_dir = D / 'out/manh_v12_363638159'
    meta = json.loads((gold_dir / 'kernel-metadata.json').read_text())
    raw = (gold_dir / meta['code_file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'c47be5a819ad63a09c6ee831bff7f9770e070cd91fe04a27865f6ddd6e7ec42f'
    gold = json.loads(raw); fs = overlay(gold)
    s = fs['solver_dsv41.py'].decode()
    s = s.replace('Path("/kaggle/input").rglob("e043_manifest.json")',
                  'Path("/kaggle/input").glob("models/*/*/*/*/*/e043_manifest.json")')
    assert not IP.input_scans(s)
    fs['solver_dsv41.py'] = s.encode(); set_overlay(gold, fs)
    receipt = {'gold': write_nb('gold', gold, meta)}
    te = json.loads((D / 'out/nb_v12_img_te/dsv41-tools-v12-img.ipynb').read_text())
    fs = overlay(te)
    fs['solver_dsv41.py'] = policy_solver(fs['solver_dsv41.py'].decode()).encode()
    set_overlay(te, fs)
    # Keep V12 serving flags: isolate harness/model changes from the unproven V12-fast changes.
    c = cell(te, '# Cell 2m'); s = src(c)
    s = s.replace('"--vllm-arg=--watermark-reload-only", ', '')
    lines = s.splitlines(keepends=True)
    s = ''.join(x for x in lines if 'decoder_replay_cudagraph_capture_sizes' not in x)
    c['source'] = s
    # Three development draws share a text first draw. Two-answer comparisons select explicit subsets.
    c = te['cells'][1]
    c['source'] = src(c) + '\nos.environ.setdefault("ARC_ATTEMPT_FORMATS", "json,json,text+image")\n'
    c['source'] += 'os.environ.setdefault("ARC_EXTRA_IF_NO_TRAIN_PASS", "0")\n'
    c['source'] += 'os.environ.setdefault("ARC_ATTEMPTS", "2")\n'
    c = cell(te, '# Cell 0b'); s = src(c)
    line = next(x for x in s.splitlines() if x.startswith('_MT_EXPECT = '))
    expected = json.loads(line.split(' = ', 1)[1])
    expected.update(ARC_ATTEMPTS='2', ARC_ATTEMPT_FORMATS='json,json,text+image', ARC_EXTRA_IF_NO_TRAIN_PASS='0')
    c['source'] = s.replace(line, '_MT_EXPECT = ' + json.dumps(expected))
    receipt['causal'] = write_nb('causal', te, json.loads((D / 'out/nb_v12_img_te/kernel-metadata.json').read_text()))
    (out / 'solver_causal.py').write_bytes(fs['solver_dsv41.py'])
    wl = HERE / 'workloads'; wl.mkdir(exist_ok=True)
    split = ROOT / 'experiments/E072_harness_loop/workloads'
    ch = json.loads((split / 'eval120_kaggle_challenges.json').read_text())
    sol = json.loads((split / 'eval120_kaggle_solutions.json').read_text())
    for part in ['all', 'dev', 'holdout']:
        ids = set(ch) if part == 'all' else set((split / f'eval120_kaggle_{part}_ids.txt').read_text().split())
        for kind, data in [('challenges', ch), ('solutions', sol)]:
            (wl / f'e073_{part}_{kind}.json').write_text(json.dumps({t: data[t] for t in sorted(ids)}))
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
