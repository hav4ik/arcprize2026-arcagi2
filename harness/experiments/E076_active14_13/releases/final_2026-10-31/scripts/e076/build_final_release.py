#!/usr/bin/env python3
"""Build a new final code/metadata release; preserve the historical image-only handoff."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/E076_active14_13'
OUT = EXP / 'releases/final_2026-10-31'
ARCHIVE = EXP / 'releases/E076_active14_final_code_metadata_2026-10-31.tar.gz'
PAPERS = ('moe_active_parameter_pruning', 'moe_reference_mass_normalization',
          'zeda_expert_self_distillation', 'matryoshka_moe', 'lexi_active_experts',
          'aset_expert_skipping', 'ace_expert_skipping', 'omp_moe', 'adapmoe')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or ARCHIVE.exists():
        raise SystemExit('Versioned release already exists; choose a new version instead of overwriting.')
    historical = EXP / 'handoff'
    for line in (historical / 'FILES.sha256').read_text().splitlines():
        h, name = line.split('  ', 1)
        assert sha(historical / name) == h, name
    shutil.copytree(historical, OUT)
    old_index = json.loads((OUT / 'FILE_INDEX.json').read_text())
    origins = {name: row['source'] for name, row in old_index.items()}

    def copy(src, dst):
        dest = OUT / dst
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        origins[dst] = str(src.relative_to(ROOT))

    for f in sorted((EXP / 'scripts').iterdir()):
        if f.is_file() and f.suffix in ('.py', '.sh', '.conf'):
            copy(f, 'scripts/e076/' + f.name)
    for f in sorted((EXP / 'configs').glob('*.json')):
        copy(f, 'configs/' + f.name)
    for name in ('completed_tools_report.json', 'historical_e128_scores_20261031.json',
                 'early_feasibility_signals_20261031.json',
                 'training_data_boundary_clarification_20261031.json'):
        copy(EXP / 'analysis' / name, 'evaluation/' + name)
    copy(EXP / 'analysis/summary_tables.md', 'evaluation/summary_tables.md')
    for name in PAPERS:
        copy(ROOT / 'prior_works' / (name + '.md'), 'literature/' + name + '.md')
    copy(EXP / 'configs/exploratory12_constraints.json', 'provenance/future12_constraints.json')
    copy(EXP / 'configs/training_data_policy.json', 'calibration/training_data_policy.json')
    folders = {'E128': ROOT / 'experiments/E074_active_params/results/baseline_arc',
               'A14p58': ROOT / 'experiments/E074_active_params/results/candidate_arc',
               'A14p02': EXP / 'results/preview14_img',
               'E128_text': EXP / 'results/baseline_txt',
               'A14p02_text': EXP / 'results/preview14_txt'}
    for name, folder in folders.items():
        for filename in ('arc.jsonl', 'complete.json', 'command.json', 'eval_command.json',
                         'schedule.json', 'cache_metrics.jsonl'):
            copy(folder / filename, 'evaluation/' + name + '/' + filename)
        dest = OUT / 'evaluation' / name / 'server.log.gz'
        dest.write_bytes(gzip.compress((folder / 'server.log').read_bytes(), mtime=0))
        origins[str(dest.relative_to(OUT))] = str((folder / 'server.log').relative_to(ROOT)) + ' (gzip)'

    report = json.loads((EXP / 'analysis/completed_tools_report.json').read_text())
    meta = json.loads((OUT / 'metadata.json').read_text())
    meta['schema_version'] = 2
    meta['release'] = 'final_2026-10-31'
    meta['created_local'] = report['snapshot_local']
    meta['historical_handoff_sha256'] = sha(EXP / 'E076_active14_SFT_handoff_2026-10-31.tar.gz')
    meta['evaluation'].update({'image_status': 'complete', 'text_status': 'complete',
        'primary_modality': 'text+tools', 'skipped_arms': ['14.58B text'],
        'acceptance_margin_pp': 2, 'observed_text_pass1_delta_pp': -3.0555555555555554,
        'acceptance': 'Point estimate exceeds the 2pp loss target; noninferiority is not established.',
        'final_report': 'evaluation/completed_tools_report.json'})
    for name in ('E128_text', 'A14p02_text'):
        meta['evaluation']['runs'][name] = {'path': 'evaluation/' + name, 'source': str(folders[name].relative_to(ROOT))}
    meta['training_data_policy'] = json.loads((EXP / 'configs/training_data_policy.json').read_text())
    meta['publication_scope'] = 'Code, metadata, calibration provenance and prepared inputs, predictions and metrics. No model weights, credentials, or evaluation solution file.'
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    readme = (historical / 'README.md').read_text()
    start = readme.index('The schedule is transferable')
    prefix = '''# E076 final active-expert pruning code and metadata

Final release, 2026-10-31. Both image+tools and text+tools evaluations are complete:
120 Kaggle ARC-AGI-2 tasks x two draws per model. Text is the primary acceptance
modality. The 14.58B text reference was explicitly skipped.

| Modality | E128 ~16B pass@1 / pass@2 | E074 ~14.58B | E076 ~14.02B |
|---|---:|---:|---:|
| Text + tools | 72.29% / 81.81% | Skipped | 69.24% / 79.44% |
| Image + tools | 72.43% / 83.47% | 70.83% / 81.81% | 71.67% / 83.89% |

Text pass@1 falls 3.06 percentage points, exceeding the requested two-point
loss target on this run; paired-task 95% interval [-8.68,+2.78]pp. Noninferiority
is not established. Text matched-window generation throughput improves 2.61%,
but total eval time increases 1.13% because output tokens increase 2.74%.
The current schedule is a poor tradeoff for this text harness. Image pass@1
falls 0.76pp versus E128, with 4.39% less total eval time.

See [complete usage tables](evaluation/summary_tables.md) and
[machine-readable final results](evaluation/completed_tools_report.json).
Usage is deduplicated by task/attempt, including missing answers on any input.
Image and text use the same causal image-recalibrated E128 weights. This is a
routing-only experiment; no recovery training was performed. SFT transfer and
Manh's active-top-k port remain unvalidated.

The final release supersedes the image-only handoff, which remains unchanged.
To verify all bundle hashes and logical-attempt counters:

```bash
python scripts/e076/verify_release.py .
python scripts/e076/verify_release.py . --solutions /path/to/arc-agi_evaluation_solutions.json
```

The optional second command rescoring requires the exact Kaggle solution file;
its SHA256 is checked against metadata. Solutions are not included. Five full
prediction files and compressed serving logs are included for independent audit.
Original runner scripts retain workspace paths; adapt them before using a new
checkpoint. No script is automatically executed by this bundle.
The historical search optimizer does not enforce the later protected-k6 policy;
add that constraint before any future search. Size targets remain approximate.

'''
    readme = prefix + readme[start:]
    readme = readme.replace('| Attempts missing the pair-zero answer | 9 | 4 | 6 |', '| Attempts missing any test answer | 9 | 4 | 7 |')
    readme = readme.replace('It is a frozen\nimage-result snapshot, not an automatic feed of the unfinished text evaluations.', 'It is a frozen final image-and-text result snapshot.')
    readme = readme.replace('All parent IDs belong to ARC training;', 'Calibration mixes original ARC tasks from the 1,000-task training pool with NVARC\ntraining-parent rollouts (E065 image/text twins). Keep each original and its\nsynthetic descendants in the same family split. All parent IDs belong to ARC training;')
    (OUT / 'README.md').write_text(readme)
    usage = json.loads((OUT / 'evaluation/full_usage.json').read_text())
    for name, model in [('E128', 'E128'), ('A14p58', '14.58B'), ('A14p02', '14.02B')]:
        usage[name]['attempts_any_missing_answer'] = report['image'][model]['attempts_any_missing_answer']
    for name, model in [('E128_text', 'E128'), ('A14p02_text', '14.02B')]:
        usage[name] = report['text_matched']['models'][model]
    (OUT / 'evaluation/full_usage.json').write_text(json.dumps(usage, indent=2) + '\n')
    index = {str(f.relative_to(OUT)): {'bytes': f.stat().st_size, 'sha256': sha(f),
             'source': origins.get(str(f.relative_to(OUT)), 'generated final release metadata')}
             for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ('FILES.sha256', 'FILE_INDEX.json')}
    (OUT / 'FILE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    files = [f for f in sorted(OUT.rglob('*')) if f.is_file() and f.name != 'FILES.sha256']
    (OUT / 'FILES.sha256').write_text(''.join(sha(f) + '  ' + str(f.relative_to(OUT)) + '\n' for f in files))
    with ARCHIVE.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():
                info = tar.gettarinfo(str(f), arcname='E076_final_2026-10-31/' + str(f.relative_to(OUT)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                with f.open('rb') as data:
                    tar.addfile(info, data)
    ARCHIVE.with_suffix(ARCHIVE.suffix + '.sha256').write_text(sha(ARCHIVE) + '  ' + ARCHIVE.name + '\n')
    print(json.dumps({'release': str(OUT), 'files': len(files) + 1, 'archive': str(ARCHIVE),
                      'archive_bytes': ARCHIVE.stat().st_size, 'sha256': sha(ARCHIVE)}, indent=2))


if __name__ == '__main__':
    main()
