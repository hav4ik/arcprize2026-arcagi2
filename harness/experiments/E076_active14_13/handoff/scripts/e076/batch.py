#!/usr/bin/env python3
"""Bounded, resumable E065-only architecture search followed by matched ARC runs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from budgets import describe
from search import allocate_exact, compare, nll, swap_neighbors

ROOT=Path('/workspace/dsv4')
EXP=ROOT/'repo/experiments/E076_active14_13'
# User narrowed the experiment to ~14B on 2026-10-31. Historical 13B results stay on disk.
TARGETS={'14b':184}


def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--hours',type=float,default=48)
    ap.add_argument('--skip-arc',action='store_true')
    ap.add_argument('--tools-first',action='store_true',help='Evaluate frozen seed14 before any further architecture search')
    ap.add_argument('--reference-only',action='store_true',help='Add the E074 mean5 text+tools reference after the initial tools queue')
    args=ap.parse_args()
    lock=(ROOT/'e076.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):
        raise InterruptedError(f'Received signal {signum}; stopping only owned arm')
    signal.signal(signal.SIGTERM,interrupted)
    # Restarts preserve the original compute deadline instead of resetting its 48-hour bound.
    clock_path=EXP/'results/batch_clock.json'
    if clock_path.exists():
        clock=json.loads(clock_path.read_text())
    else:
        clock={'started_unix':time.time(),'max_hours':args.hours}
        write(clock_path,clock)
    hours=min(args.hours,clock['max_hours'])
    start=time.monotonic()-max(0,time.time()-clock['started_unix'])
    deadline=start+hours*3600
    for name,digest in json.loads((EXP/'frozen_manifest.json').read_text()).items():
        assert hashlib.sha256((EXP/'frozen'/name).read_bytes()).hexdigest()==digest, name
    def status(**kw):
        write(EXP/'results/batch_status.json',{'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'elapsed_s':time.monotonic()-start,'max_hours':hours,
              'workflow':'dual-baseline-extension' if args.reference_only else ('tools-first' if args.tools_first else 'calibration-search'),
              'targets':TARGETS,'cancelled_targets':['13b'],**kw})
    def schedule(name,ks):
        p=EXP/'configs'/f'{name}.json';value=describe(ks)
        if p.exists():
            assert json.loads(p.read_text())==value, 'Schedule changed: '+name
        else:
            write(p,value)
        return name
    def arm(name,tag,pools=('screen',),diagnostic=False,mode=None):
        out=EXP/'results'/tag
        if (out/'complete.json').exists():
            assert json.loads((out/'schedule.json').read_text())==json.loads((EXP/'configs'/f'{name}.json').read_text())
            return
        limit=16200 if mode else (5400 if diagnostic else 2700)
        if deadline-time.monotonic()<limit:
            raise TimeoutError(f'Insufficient remaining batch time for {tag}')
        # At most one automatic retry per arm, preserving all failed artifacts.
        for attempt in range(2):
            if out.exists():
                archive=EXP/'results/failed'/f'{tag}_{time.time_ns()}'
                archive.parent.mkdir(exist_ok=True);out.rename(archive)
            status(state='running',phase=tag,schedule=name,mode=mode,pools=pools,retry=attempt)
            cmd=[sys.executable,str(EXP/'scripts/arm.py'),'--schedule',name,'--tag',tag,'--max-seconds',str(limit)]
            if mode:
                cmd+=['--arc-mode',mode]
                if args.tools_first and mode=='img':
                    cmd+=['--reference-arm',str(ROOT/'repo/experiments/E074_active_params/results/baseline_arc')]
            else:
                cmd+=['--pools',*pools]
            if diagnostic:cmd.append('--training-diagnostic')
            print('START',tag,flush=True)
            proc=subprocess.Popen(cmd)
            try:
                code=proc.wait(timeout=limit+90)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
            if code==0:
                print('DONE',tag,flush=True);return
            if attempt or deadline-time.monotonic()<limit:
                raise subprocess.CalledProcessError(code,cmd)
            print('RETRY',tag,'after failure; prior output archived',flush=True)
        raise AssertionError('Unreachable')
    try:
        if args.reference_only:
            assert not args.skip_arc and not args.tools_first
            for tag in ('preview14_img','baseline_txt','preview14_txt'):
                assert (EXP/'results'/tag/'complete.json').exists(), 'Initial tools queue must finish first'
            reference=json.loads((ROOT/'repo/experiments/E074_active_params/results/candidate_arc/schedule.json').read_text())
            name='baseline1458'
            assert json.loads((EXP/'configs'/f'{name}.json').read_text())==reference
            arm(name,'baseline1458_txt',mode='txt')
            status(state='complete',phase='dual_baseline_evaluations_complete',
                   selected={'14b':'seed14'},arc_launched=True,search_deferred=True,
                   references=['E128 k6 ~16B','E074 mean5 ~14.58B'])
            return
        if args.tools_first:
            assert not args.skip_arc
            plan=json.loads((EXP/'configs/tools_first_manifest.json').read_text())
            chosen=plan['candidate_schedule']
            assert hashlib.sha256((EXP/'configs'/f'{chosen}.json').read_bytes()).hexdigest()==plan['schedule_sha256']
            write(EXP/'results/tools_first_selection.json',plan)
            # Fixed before viewing any ARC outcomes; the ongoing search is deferred.
            arm(chosen,'preview14_img',mode='img')
            arm('uniform6','baseline_txt',mode='txt')
            arm(chosen,'preview14_txt',mode='txt')
            status(state='complete',phase='tools_first_evaluations_complete',selected={'14b':chosen},
                   arc_launched=True,search_deferred=True,
                   note='Tool-harness priority block complete; image k6 reference reused from E074. Remaining E065 architecture search is preserved and deferred.')
            return
        arm('uniform6','baseline_calibration',('screen','select','audit'),True)
        base={p:nll(EXP,'baseline_calibration',p) for p in ('screen','select','audit')}
        # Initial targets use the earlier training-only ranking, providing prompt feasibility checks.
        prior=json.loads((ROOT/'repo/experiments/E074_active_params/results/sensitivity.json').read_text())
        ks=[6]*40
        for row in sorted(prior,key=lambda r:(r['delta'],r['layer']))[:28]:ks[row['layer']]=4
        seeds={'14b':schedule('seed14',ks)}
        for target,name in seeds.items():
            arm(name,name+'_select',('select',),True)
        curves=[{6:{m:0. for m in ('gen','tool','balanced','robust','stderr','joint_score')}} for _ in range(40)]
        for k in (4,5,3):
            for layer in range(40):
                name=f'layer{layer:02d}_k{k}';ks=[6]*40;ks[layer]=k
                schedule(name,ks);arm(name,name)
                curves[layer][k]=compare(nll(EXP,name,'screen'),base['screen'])
                write(EXP/'results/sensitivity.json',curves)
        selected={}
        for target,slots in TARGETS.items():
            candidates={}
            def consider(name):
                arm(name,name+'_select',('select',))
                result=compare(nll(EXP,name+'_select','select'),base['select'])
                candidates[name]={'schedule':json.loads((EXP/'configs'/f'{name}.json').read_text()),'loss':result}
                write(EXP/'results'/f'{target}_candidates.json',candidates)
            consider(seeds[target])
            known={tuple(candidates[seeds[target]]['schedule']['top_k'])}
            for metric in ('gen','balanced','robust'):
                loss,ks=allocate_exact(curves,slots,metric)
                if tuple(ks) in known:continue
                known.add(tuple(ks));name=schedule(f'{target}_{metric}',ks);consider(name)
            # Joint local refinement: single-layer marginal costs do not determine combined loss.
            for step in range(2):
                incumbent=min(candidates,key=lambda n:candidates[n]['loss']['joint_score'])
                ks=candidates[incumbent]['schedule']['top_k']
                fresh=[x for x in swap_neighbors(ks,curves,limit=100) if tuple(x) not in known][:3]
                if not fresh:break
                for j,ks in enumerate(fresh):
                    known.add(tuple(ks));consider(schedule(f'{target}_refine{step}_{j}',ks))
                best=min(candidates,key=lambda n:candidates[n]['loss']['joint_score'])
                if best==incumbent:break
            name=min(candidates,key=lambda n:candidates[n]['loss']['joint_score'])
            selected[target]={'name':name,**candidates[name]}
        # This file is frozen BEFORE independent audit and ARC outcomes are available.
        write(EXP/'results/selection.json',{'selected':selected,
            'criterion':'E065 select24: mean relative gen NLL + 0.25 tool NLL + 0.25 worst-modality gen NLL',
            'audit_used_for_selection':False,'arc_used_for_selection':False})
        for target,row in selected.items():
            arm(row['name'],target+'_audit',('audit',),True)
            write(EXP/'results'/f'{target}_audit_summary.json',compare(nll(EXP,target+'_audit','audit'),base['audit']))
        if not args.skip_arc:
            for mode in ('img','txt'):
                arm('uniform6',f'baseline_{mode}',mode=mode)
                for target,row in selected.items():
                    arm(row['name'],f'{target}_{mode}',mode=mode)
        status(state='complete',selected={k:v['name'] for k,v in selected.items()},arc_launched=not args.skip_arc,
               note='Unrepaired schedule search/evaluation complete. Recovery training remains a separate decision based on measured losses.')
    except Exception as error:
        status(state='stopped',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
