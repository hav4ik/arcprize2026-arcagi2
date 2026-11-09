#!/usr/bin/env python3
"""Sync the bounded remote batch and score complete ARC results on this workstation."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E074_active_params'
SSH='ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/tmp/arc2-vast-58489667-known_hosts -p 33408'
REMOTE='root@70.69.192.6:/workspace/dsv4/repo/experiments/E074_active_params/'


def update_registry(status,result):
    path=ROOT/'experiments/experiments.tsv'
    lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('E074\t'):
            row=line.split('\t');row[3]=status;row[7]=result;row[10]=time.strftime('%Y-%m-%d',time.gmtime())
            lines[i]='\t'.join(row)
            break
    path.write_text('\n'.join(lines)+'\n')


def main():
    deadline=time.monotonic()+16*3600
    while time.monotonic()<deadline:
        try:
            result=subprocess.run(['rsync','-az','--exclude','__pycache__/','-e',SSH,
                                   REMOTE+'results/',str(EXP/'results')+'/'],
                                  capture_output=True,text=True,timeout=120)
            if result.returncode:
                print('Result sync unavailable; will retry',flush=True)
                time.sleep(60)
                continue
            state_path=EXP/'results/batch_status.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else {}
            print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),state,flush=True)
            if state.get('state')=='stopped':
                update_registry('stopped: inspect E074 batch status',state.get('error','Unknown failure'))
                return
            if state.get('state')=='complete':
                if state.get('arc_launched'):
                    subprocess.run([sys.executable,str(EXP/'scripts/score_arc.py')],check=True,cwd=ROOT)
                    report=json.loads((EXP/'results/arc_report.json').read_text())
                    arms=report['arms']; b=arms['baseline_arc']; c=arms['candidate_arc']
                    summary=(f"Kaggle 120 tasks x2: baseline pass@1/pass@2 {b['pass_at_1']:.2f}/{b['pass_at_2']:.2f}; "
                             f"candidate {c['pass_at_1']:.2f}/{c['pass_at_2']:.2f}; "
                             f"pass@1 delta {report['pass_at_1_difference_pp']:+.2f}pp; {report['conclusion']}")
                    update_registry('done: first mean-five comparison',summary)
                    with (EXP/'README.md').open('a') as f:
                        f.write('\n## First-batch result\n\n'+summary+'\n\nSee `results/arc_report.json` for paired intervals, token cost and timing.\n')
                else:
                    update_registry('screening done; ARC not launched','See results/selection.json; no ARC accuracy conclusion')
                return
        except Exception as error:
            print('Monitor error:',type(error).__name__,str(error),flush=True)
        time.sleep(60)
    update_registry('monitor expired; check remote batch','16h monitor bound reached; no automatic rental action')


if __name__=='__main__':
    main()
