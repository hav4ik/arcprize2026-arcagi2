#!/usr/bin/env python3
"""Append the requested second text reference without interrupting the active queue."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/dsv4')
EXP=ROOT/'repo/experiments/E076_active14_13'
RECEIPT=EXP/'results/dual_baseline_extension.json'


def status(state, **extra):
    value={'state':state,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'reference':'E074 mean5_profiled, ~14.58B','arm':'baseline1458_txt',**extra}
    tmp=RECEIPT.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(RECEIPT)


def main():
    def skipped():
        policy=EXP/'configs/eval_queue_policy.json'
        return policy.exists() and 'baseline1458_txt' in json.loads(policy.read_text()).get('skipped_eval_arms',[])
    if skipped():
        status('skipped',note='User cancelled this text reference; preserve existing image reference')
        return
    clock=json.loads((EXP/'results/batch_clock.json').read_text())
    deadline=clock['started_unix']+clock['max_hours']*3600
    status('waiting',note='Wait for the existing tools queue; preserve original batch deadline')
    try:
        while time.time()<deadline:
            if skipped():
                status('skipped',note='User cancelled this text reference; preserve existing image reference')
                return
            state=json.loads((EXP/'results/batch_status.json').read_text())
            if state['state']=='stopped':
                raise RuntimeError('Initial tools queue stopped; inspect it before launching reference')
            if state['state']=='complete':
                with (ROOT/'e076.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:break
            time.sleep(10)
        else:
            raise TimeoutError('Original batch deadline reached while waiting')
        if skipped():
            status('skipped',note='User cancelled this text reference; preserve existing image reference')
            return
        status('running')
        subprocess.run([sys.executable,str(EXP/'scripts/batch.py'),'--hours',str(clock['max_hours']),
                        '--reference-only'],check=True)
        assert (EXP/'results/baseline1458_txt/complete.json').exists()
        status('complete')
    except Exception as error:
        status('stopped',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
