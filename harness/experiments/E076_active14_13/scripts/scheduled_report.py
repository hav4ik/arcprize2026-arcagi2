#!/usr/bin/env python3
"""Request the user's 30-minute eval update in the existing Codex conversation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

EXP=Path(__file__).resolve().parents[1]
CONFIG=EXP/'monitor/report_schedule.json'
STATE=EXP/'monitor/report_delivery.json'
TIMER='arc-e076-report.timer'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--probe',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    cfg=json.loads(CONFIG.read_text())
    with (EXP/'monitor/report_delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        policy_path=EXP/'configs/eval_queue_policy.json'
        policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
        required=policy.get('required_new_eval_arms',
                            ['preview14_img','baseline_txt','preview14_txt','baseline1458_txt'])
        skipped=policy.get('skipped_eval_arms',[])
        done=all((EXP/'results'/tag/'complete.json').exists() for tag in required)
        expired=time.time()>=cfg['stop_after_unix']
        if state.get('terminal_queued'):
            if not args.dry_run:
                subprocess.run(['systemctl','--user','stop',TIMER],check=False)
            return
        prefix=('[E076 scheduled-report delivery test, explicitly requested by the user] '
                if args.probe else '[E076 30-minute eval update, explicitly requested by the user] ')
        message=prefix+(
            'Read /home/vu/Workspace/ARC-AGI-2/experiments/E076_active14_13/README.md and the latest '
            'synced results, refreshing live status if needed. Report progress and intermediate or final '
            'results to the user here. Compare the ~14.02B candidate with E128 k6 (~16B), and with '
            'E074 mean5 (~14.58B) wherever its matching reference exists, matching modality, Kaggle version, task subset and pass@k. '
            'For partial accuracy explicitly state completed-subset bias and task count. Report throughput '
            'against available matching references using comparable elapsed-time windows, plus errors/cache health and ETA. '
            'Include matched-task output-token use, tool calls, tool error/rejection rates after auto-repair, '
            'cap stops and missing answers when useful; distinguish these from server/harness errors and '
            'deduplicate multi-test pair records by task/attempt. '
            'Use America/Los_Angeles for user-facing times. Do not change eval settings or launch new jobs '
            'for this report. The existing GPU queues and sync watcher continue independently. '
            'Read configs/eval_queue_policy.json for the current required/skipped arms. '
            'If all required eval arms are complete, give the final comparison and stop only '
            'arc-e076-report.timer; never stop the GPU rental. '
            'If this is a delivery test received during setup, record that delivery worked and finish '
            'setting up the timer instead of producing a duplicate update.')
        if 'baseline1458_txt' in skipped:
            message+=(' The user explicitly skipped the 14.58B text-only eval. Compare images with both baselines, '
                      'but text with E128 only; label 14.58B text skipped, not queued. '
                      'Finish the 14.02B evals before starting the authorized approximate 12-13B pruning work. '
                      'Do not launch pruning during this scheduled report.')
        if cfg.get('primary_modality') == 'text+tools':
            message+=(' The user primarily cares about text-only+tools: lead with its matched results and '
                      'use text task-weighted pass@1 as the primary acceptance criterion. '
                      'Image results are secondary context and do not establish text retention.')
        if expired:message+=' The reporting safety bound has expired; give a final status and explain any unfinished work.'
        cmd=[cfg['codex_path'],'queue','--thread',cfg['thread_id'],'--message',message,'--resume']
        if args.dry_run:
            print(json.dumps({'thread_id':cfg['thread_id'],'interval_minutes':30,
                              'all_eval_arms_complete':done,'expired':expired,'message':message},indent=2))
            return
        env=dict(os.environ)
        env['PATH']=str(Path(cfg['codex_path']).parent)+os.pathsep+env.get('PATH','/usr/bin:/bin')
        result=subprocess.run(cmd,cwd=EXP.parents[1],env=env,capture_output=True,text=True,timeout=45)
        output=result.stdout+result.stderr
        receipt={'queued_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'returncode':result.returncode,'probe':args.probe,'output':output,
                 'terminal_queued':bool((done or expired) and result.returncode==0)}
        STATE.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        result.check_returncode()
        if 'failed to resume' in output.lower() or 'resume failed' in output.lower():
            raise RuntimeError('Message queued but automatic resume failed; inspect delivery receipt')
        if receipt['terminal_queued']:
            subprocess.run(['systemctl','--user','stop',TIMER],check=False)


if __name__=='__main__':main()
