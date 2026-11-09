#!/usr/bin/env python3
"""Score saved results locally; evaluation answer keys never leave the workstation."""
import importlib.util
import json
from pathlib import Path
import time

from search import compare, nll

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E076_active14_13'


def scoring():
    path=ROOT/'experiments/E066_dsv41_image_pruning/scripts/score_b32_engram.py'
    spec=importlib.util.spec_from_file_location('e076_scorer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    s=scoring()
    result={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'version':'Kaggle ARC-AGI-2 evaluation','task_count':120,'attempts_per_task':2,
        'active_targets':['14b'],'cancelled_targets':['13b'],
        'weighting':'Mean within each task over test inputs; pass@1 averaged over two draws',
        'reference_models':{'baseline':'E128, ~16B active, k6',
                            'baseline1458':'E128, ~14.58B active, E074 mean5_profiled'},
        'arc':{},'comparisons':{},'training_diagnostics':{},'training_nll':{}}
    policy_path=EXP/'configs/eval_queue_policy.json'
    if policy_path.exists():
        result['eval_queue_policy']=json.loads(policy_path.read_text())
    data={}
    for mode in ('img','txt'):
        for target in ('baseline','baseline1458','preview14','14b'):
            tag=f'{target}_{mode}';folder=EXP/'results'/tag;path=folder/'arc.jsonl'
            reused=False
            if mode=='img' and target in ('baseline','baseline1458') and not path.exists() and (EXP/'configs/tools_first_manifest.json').exists():
                reference='baseline_arc' if target=='baseline' else 'candidate_arc'
                folder=ROOT/'experiments/E074_active_params/results'/reference
                path=folder/'arc.jsonl';reused=True
            if not path.exists():continue
            try:d=s.load_run(path)
            except json.JSONDecodeError:continue
            data[tag]=d
            done=sum(s.done(d,t,a) for t in s.TASKS for a in (0,1))
            summary={'completed_task_attempts':done,'expected_task_attempts':240,'complete':done==240,
                     'reused_reference':reused,'source':str(folder.relative_to(ROOT))}
            if done==240:
                summary.update(zip(('pass_at_1','pass_at_2'),s.score(d,s.TASKS)))
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                summary['errors']=sum(bool(r.get('error')) for r in rows)
                summary['output_tokens']=sum(r.get('usage',{}).get('completion',0) for r in rows)
                if (folder/'complete.json').exists():
                    timer=json.loads((folder/'complete.json').read_text())
                    summary['eval_seconds']=timer['eval_elapsed_s']
                    summary['output_tokens_per_second']=summary['output_tokens']/summary['eval_seconds']
            result['arc'][tag]=summary
        for target in ('preview14','14b'):
            tag=f'{target}_{mode}'
            if not result['arc'].get(tag,{}).get('complete'):continue
            for reference in ('baseline','baseline1458'):
                baseline=f'{reference}_{mode}'
                if not result['arc'].get(baseline,{}).get('complete'):continue
                d,lo,hi,better,worse=s.paired(data[tag],data[baseline],s.TASKS)
                candidate,control=result['arc'][tag],result['arc'][baseline]
                comparison={'candidate':tag,'baseline':baseline,
                    'pass_at_1_delta_pp':d,
                    'pass_at_2_delta_pp':candidate['pass_at_2']-control['pass_at_2'],
                    'paired_task_bootstrap_95ci_pp':[lo,hi],'tasks_better':better,'tasks_worse':worse,
                    'margin_pp':2,'conclusion':('supports <=2pp loss' if lo>=-2 else
                        'loss exceeds 2pp with this interval' if hi < -2 else 'inconclusive for <=2pp loss')}
                if all('eval_seconds' in x for x in (candidate,control)):
                    comparison.update(
                        output_throughput_gain_pct=100*(candidate['output_tokens_per_second']/control['output_tokens_per_second']-1),
                        eval_wall_time_reduction_pct=100*(1-candidate['eval_seconds']/control['eval_seconds']))
                result['comparisons'][f'{tag}_vs_{reference}']=comparison
    s.SOL=json.loads((ROOT/'data/arc-agi_training_solutions.json').read_text())
    diag=json.loads((EXP/'data/training_diagnostic_challenges.json').read_text())
    for p in sorted((EXP/'results').glob('*/diagnostic_*_complete.json')):
        mode=p.name.split('_')[1];tag=p.parent.name
        d=s.load_run(p.parent/f'diagnostic_{mode}.jsonl')
        if not all(s.done(d,t,0) for t in diag):continue
        result['training_diagnostics'][tag+'_'+mode]={'version':'Kaggle ARC training originals',
            'tasks':len(diag),'attempts':1,'pass_at_1':s.score(d,list(diag),(0,))[0],
            'note':'Training-only short diagnostic, not evaluation accuracy'}
    for pool in ('screen','select','audit'):
        try:b=nll(EXP,'baseline_calibration',pool)
        except (OSError,AssertionError,json.JSONDecodeError):continue
        for p in sorted((EXP/'results').glob(f'*/{pool}_img.jsonl')):
            if p.parent.name=='baseline_calibration':continue
            try:result['training_nll'][p.parent.name+'_'+pool]=compare(nll(EXP,p.parent.name,pool),b)
            except (OSError,AssertionError,json.JSONDecodeError):continue
    path=EXP/'results/report.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k!='training_nll'},indent=2))


if __name__=='__main__':main()
