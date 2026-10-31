#!/usr/bin/env python3
"""Finite mean-top-k-five search. Run on Vast only; never manage the rental itself."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import signal
import sys
import time
from budgets import describe

ROOT=Path('/workspace/dsv4')
EXP=ROOT/'repo/experiments/E074_active_params'


def nll(tag, count=None):
    scores=[]
    for mode in ('img','txt'):
        rows=[json.loads(line) for line in (EXP/'results'/tag/f'{mode}.jsonl').read_text().splitlines()]
        if count is not None:
            assert len(rows)==count, (tag,mode,len(rows))
        gen=sum(r['nll_gen'] for r in rows)/sum(r['n_gen'] for r in rows)
        block=sum(r['nll_blk'] for r in rows)/max(1,sum(r['n_blk'] for r in rows))
        assert math.isfinite(gen) and math.isfinite(block)
        scores.append({'modality':mode,'generated_nll':gen,'tool_block_nll':block})
    return {'generated_nll':sum(x['generated_nll'] for x in scores)/2,
            'tool_block_nll':sum(x['tool_block_nll'] for x in scores)/2,'modalities':scores}


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}; stopping owned arm')
    signal.signal(signal.SIGTERM, interrupted)
    ap=argparse.ArgumentParser()
    ap.add_argument('--hours',type=float,default=14)
    ap.add_argument('--skip-arc',action='store_true')
    args=ap.parse_args()
    start=time.monotonic()
    deadline=start+args.hours*3600
    (EXP/'results').mkdir(exist_ok=True)
    status=EXP/'results/batch_status.json'
    def arm(schedule,tag,n=4,arc=False):
        if (EXP/'results'/tag/'complete.json').exists():
            return
        remaining=deadline-time.monotonic()
        limit=10800 if arc else 2100
        if remaining < limit:
            raise TimeoutError(f'Not enough batch budget to start {tag}')
        status.write_text(json.dumps({'phase':tag,'state':'running','elapsed_s':time.monotonic()-start})+'\n')
        cmd=[sys.executable,str(EXP/'scripts/smoke.py'),'--schedule',schedule,'--tag',tag,
             '--n-seq',str(n),'--max-seconds',str(limit)]
        if arc: cmd.append('--arc')
        print('START',tag,flush=True)
        proc=subprocess.Popen(cmd)
        try:
            code=proc.wait(timeout=limit+90)
            if code:
                raise subprocess.CalledProcessError(code,cmd)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        print('DONE',tag,flush=True)
    try:
        arm('uniform6','baseline_screen')
        baseline=nll('baseline_screen',4)
        arm('uniform5','uniform5_screen')
        costs=[]
        for layer in range(40):
            tag=f'layer{layer:02d}_k4'
            ks=[6]*40; ks[layer]=4
            (EXP/'configs'/f'{tag}.json').write_text(json.dumps(describe(ks),indent=2)+'\n')
            arm(tag,tag)
            scores=nll(tag,4)
            costs.append({'layer':layer,'delta':scores['generated_nll']-baseline['generated_nll'],**scores})
            (EXP/'results/sensitivity.json').write_text(json.dumps(costs,indent=2)+'\n')
        # One measured 6->4 point per layer: 20 reduced layers + 20 protected = mean five.
        cheapest=sorted(costs,key=lambda row:(row['delta'],row['layer']))[:20]
        ks=[6]*40
        for row in cheapest: ks[row['layer']]=4
        assert sum(ks)==200
        (EXP/'configs/mean5_profiled.json').write_text(json.dumps(describe(ks),indent=2)+'\n')
        for schedule in ('uniform6','uniform5','mean5_profiled'):
            arm(schedule,schedule+'_verify',n=32)
        scores={name:nll(name+'_verify',32) for name in ('uniform6','uniform5','mean5_profiled')}
        chosen=min(('uniform5','mean5_profiled'),key=lambda name:scores[name]['generated_nll'])
        selection={'chosen':chosen,'scores':scores,'criterion':'lowest image/text average generated NLL on 32 training families',
                   'caution':'4 families informed the sweep; 28 are fresh for joint verification; ARC accuracy unmeasured'}
        (EXP/'results/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        if not args.skip_arc:
            arm('uniform6','baseline_arc',arc=True)
            arm(chosen,'candidate_arc',arc=True)
        status.write_text(json.dumps({'state':'complete','elapsed_s':time.monotonic()-start,
                                     'chosen':chosen,'arc_launched':not args.skip_arc})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'state':'stopped','error_type':type(error).__name__,
                                     'error':str(error),'elapsed_s':time.monotonic()-start})+'\n')
        raise


if __name__=='__main__':
    main()
