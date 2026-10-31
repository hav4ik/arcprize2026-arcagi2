#!/usr/bin/env python3
"""Score complete paired runs locally; evaluation solutions never go to the GPU box."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E074_active_params'
path=ROOT/'experiments/E066_dsv41_image_pruning/scripts/score_b32_engram.py'
spec=importlib.util.spec_from_file_location('arc_scoring',path)
scoring=importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def main():
    arms={}
    details={}
    for name in ('baseline_arc','candidate_arc'):
        folder=EXP/'results'/name
        rows=[json.loads(line) for line in (folder/'arc.jsonl').read_text().splitlines()]
        data=scoring.load_run(folder/'arc.jsonl')
        assert all(scoring.done(data,t,a) for t in scoring.TASKS for a in (0,1)), 'Incomplete arm: '+name
        arms[name]=data
        timing=json.loads((folder/'complete.json').read_text())
        p1,p2=scoring.score(data,scoring.TASKS)
        tokens=sum(r.get('usage',{}).get('completion',0) for r in rows)
        details[name]={'pass_at_1':p1,'pass_at_2':p2,'output_tokens':tokens,
            'tokens_per_task_attempt':tokens/240,'eval_seconds':timing.get('eval_elapsed_s'),
            'server_plus_eval_seconds':timing['elapsed_s']}
    delta,low,high,better,worse=scoring.paired(arms['candidate_arc'],arms['baseline_arc'],scoring.TASKS)
    report={'version':'Kaggle ARC-AGI-2 evaluation','task_count':120,'attempts_per_task':2,
        'weighting':'mean within each task over test inputs; pass@1 averaged over two independent draws',
        'arms':details,'pass_at_1_difference_pp':delta,'paired_task_bootstrap_95ci_pp':[low,high],
        'tasks_better':better,'tasks_worse':worse,'noninferiority_margin_pp':2,
        'conclusion':('supports the 2pp noninferiority target' if low>=-2 else
                      'loss exceeds 2pp with this confidence interval' if high < -2 else
                      'inconclusive for a 2pp noninferiority margin')}
    for name,stats in details.items():
        seconds=stats['eval_seconds']
        if seconds:
            stats['end_to_end_output_tokens_per_second']=stats['output_tokens']/seconds
    (EXP/'results/arc_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
