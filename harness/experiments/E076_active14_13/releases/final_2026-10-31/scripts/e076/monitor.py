#!/usr/bin/env python3
"""Independent local sync/report watcher; run as a transient user systemd service."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E076_active14_13'
SSH='ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/tmp/arc2-vast-58489667-known_hosts -p 33408'
REMOTE='root@70.69.192.6:/workspace/dsv4/repo/experiments/E076_active14_13/'


def registry(state,report):
    path=ROOT/'experiments/experiments.tsv';lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('E076\t'):
            row=line.split('\t')
            row[3]=state['state']+': '+state.get('phase',state.get('error','bounded batch'))
            comparisons=report.get('comparisons',{})
            if comparisons:
                row[7]='; '.join(f"{k}: pass@1 delta {v['pass_at_1_delta_pp']:+.2f}pp, {v['conclusion']}" for k,v in comparisons.items())
            else:
                row[7]='Full tools eval pending; compare against E128 ~16B and E074 ~14.58B references'
            row[10]=time.strftime('%Y-%m-%d',time.gmtime())
            lines[i]='\t'.join(row);break
    else:return
    path.write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');args=ap.parse_args()
    deadline=time.monotonic()+50*3600
    while time.monotonic()<deadline:
        try:
            for name in ('results','configs'):
                subprocess.run(['rsync','-az','--exclude','failed/','-e',SSH,REMOTE+name+'/',str(EXP/name)+'/'],
                               check=True,capture_output=True,text=True,timeout=240)
            state=json.loads((EXP/'results/batch_status.json').read_text())
            subprocess.run([sys.executable,str(EXP/'scripts/report.py')],check=True,capture_output=True,text=True,timeout=90)
            report=json.loads((EXP/'results/report.json').read_text())
            registry(state,report)
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),json.dumps(state),flush=True)
            extension_path=EXP/'results/dual_baseline_extension.json'
            extension=json.loads(extension_path.read_text()) if extension_path.exists() else {}
            if state['state']=='stopped':return
            if state['state']=='complete' and extension.get('state') not in ('waiting','running'):return
        except Exception as e:
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),type(e).__name__,str(e),flush=True)
        if args.once:return
        time.sleep(90)
    print('Watcher reached 50-hour bound; remote instance unchanged',flush=True)


if __name__=='__main__':main()
