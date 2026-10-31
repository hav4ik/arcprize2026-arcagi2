#!/usr/bin/env python3
"""Bounded full-model baseline smoke. Own and clean up only the server we start."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import threading
import time
import psutil
import requests
from process_utils import terminate_owned

ROOT = Path('/workspace/dsv4')
EXP = ROOT / 'repo/experiments/E074_active_params'
VENV = ROOT / 'venv'


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}; cleaning up owned server')
    signal.signal(signal.SIGTERM, interrupted)
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', default='uniform6')
    ap.add_argument('--n-seq', type=int, default=2)
    ap.add_argument('--max-seconds', type=int, default=1500)
    ap.add_argument('--tag', help='Separate result name, allowing fresh full-set verification')
    ap.add_argument('--arc', action='store_true', help='120 Kaggle task versions, two attempts, local tool executor')
    a = ap.parse_args()
    assert (ROOT/'model_ready.json').exists()
    tag = a.tag or a.schedule
    out = EXP/'results'/tag
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'complete.json').exists(), 'Completed arm already exists; choose a fresh tag'
    for mode in ('txt', 'img'):
        assert not (out/f'{mode}.jsonl').exists(), 'Partial arm exists; archive it before retrying'
    control = out/'control.json'
    control.write_text(json.dumps({'engram_keep':None,'expert_keep':None,'observe':None}))
    env = dict(os.environ)
    for k in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','NCCL_P2P_LEVEL'):
        env.pop(k, None)
    env.update(PATH=str(VENV/'bin')+':/usr/local/cuda/bin:'+env.get('PATH',''),
        VLLM_USE_V2_MODEL_RUNNER='1', FLASHINFER_CUDA_ARCH_LIST='12.0', TORCH_CUDA_ARCH_LIST='12.0',
        TRITON_PTXAS_BLACKWELL_PATH=str(VENV/'lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell'),
        CUDA_VISIBLE_DEVICES='0,1,2,3', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        E043_CTRL=str(control), E043_GPTQ_DIR=str(ROOT/'models/causal-e128/gptq'),
        E074_TOPK_FILE=str(EXP/'configs'/f'{a.schedule}.json'),
        DSV41_EXPERT_HUMMING=json.dumps({'quant_method':'humming','weight_dtype':'uint2',
            'weight_scale_dtype':'bfloat16','group_size':128,'has_zero_point':True}),
        DSV41_EXPERT_HUMMING_W2='{"group_size":64}')
    cmd=[str(VENV/'bin/vllm'),'serve',str(ROOT/'models/causal-e128'),'--served-model-name','dsv41',
         '--host','127.0.0.1','--port','18074','--trust-remote-code','--tensor-parallel-size','4',
         '--distributed-executor-backend','mp','--kv-cache-dtype','fp8','--block-size','128',
         '--max-model-len', '163840' if a.arc else '65536',
         '--max-num-seqs', '64' if a.arc else '16','--max-num-batched-tokens','4096',
         '--gpu-memory-utilization','0.80','--enable-prefix-caching','--engram-config','{"cpu_offload":true}',
         '--tokenizer-mode','deepseek_v41','--reasoning-parser','deepseek_v41','--tool-call-parser','deepseek_v41',
         '--enable-auto-tool-choice','--limit-mm-per-prompt','{"image":32}',
         '--hf-overrides','{"is_mm_prefix_lm":false}','--disable-chunked-mm-input',
         '--compilation-config',json.dumps({'cudagraph_mode':'FULL',
             'max_cudagraph_capture_size':64 if a.arc else 16})]
    (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    (out/'schedule.json').write_text(Path(env['E074_TOPK_FILE']).read_text())
    started=time.monotonic()
    def remaining():
        return max(1, a.max_seconds-(time.monotonic()-started))
    with (out/'server.log').open('w') as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner=psutil.Process(process.pid)
        monitor_stop=threading.Event()
        def monitor():
            with (out/'cache_metrics.jsonl').open('a') as handle:
                while not monitor_stop.wait(10):
                    try:
                        raw=requests.get('http://127.0.0.1:18074/metrics',timeout=3).text
                        lines=[line for line in raw.splitlines() if not line.startswith('#') and
                               any(key in line for key in ('cache','preempt','running','waiting'))]
                        handle.write(json.dumps({'time':time.time(),'metrics':lines})+'\n'); handle.flush()
                    except requests.RequestException:
                        pass
        monitor_thread=threading.Thread(target=monitor,daemon=True)
        monitor_thread.start()
        (out/'server.pid').write_text(str(process.pid)+'\n')
        try:
            while time.monotonic()-started < min(900,a.max_seconds):
                if process.poll() is not None:
                    raise RuntimeError('Server exited; see server.log')
                try:
                    if requests.get('http://127.0.0.1:18074/health',timeout=3).status_code==200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError('Server startup limit')
            print('SERVER_READY', a.schedule, round(time.monotonic()-started,1), flush=True)
            if a.arc:
                command=[str(ROOT/'tvenv/bin/python'),str(ROOT/'repo/experiments/E066_dsv41_image_pruning/scripts/b32_local_eval.py'),
                    '--mode','grid','--effort','low','--prompt','lean2sub','--answer-tool','--final-mode','session',
                    '--lenient-parse','--max-tokens','64000','--max-total-output','80000','--budget-soft','40000',
                    '--budget-notes','0.01','--budget-warn','0.75','--effort-header','minimal','--turns-note','5',
                    '--train-check','--rescue-code','--model','dsv41','--fallback-solve','--multi-test','--auto-repair',
                    '--multi-budget','0.5','--context-limit','163840','--max-turns','40','--grid-format','text+image',
                    '--image-cell','0','--base-url','http://127.0.0.1:18074/v1',
                    '--extra-body',json.dumps({'chat_template_kwargs':{'thinking':True,'reasoning_effort':50},
                        'temperature':1.0,'top_p':1.0,'cache_salt':'e074-'+tag}),
                    '--sandbox-url','local','--attempts','2','--workers','64',
                    '--challenges',str(ROOT/'repo/data/arc-agi_evaluation_challenges.json'),
                    '--out',str(out/'arc.jsonl')]
                (out/'eval_command.json').write_text(json.dumps(command,indent=2)+'\n')
                eval_started=time.monotonic()
                with (out/'eval.log').open('w') as elog:
                    subprocess.run(command,cwd=ROOT/'repo/evals',env=env,stdout=elog,
                                   stderr=subprocess.STDOUT,check=True,timeout=remaining())
                rows=[json.loads(line) for line in (out/'arc.jsonl').read_text().splitlines()]
                good={(r['task_id'],r['attempt']) for r in rows if not r.get('error')}
                assert len(good)==240, f'Incomplete ARC arm: {len(good)}/240 task attempts'
                (out/'complete.json').write_text(json.dumps({'schedule':a.schedule,'elapsed_s':time.monotonic()-started,
                    'eval_elapsed_s':time.monotonic()-eval_started,
                    'version':'Kaggle ARC-AGI-2 evaluation','tasks':120,'attempts':2})+'\n')
                return
            for mode in ('txt','img'):
                runner=ROOT/'repo/experiments/E066_dsv41_image_pruning/scripts/nll_eval_mm.py'
                command=[str(ROOT/'tvenv/bin/python'),str(runner),'--tok',str(EXP/'data'/mode),
                    '--parquet',str(EXP/'data'/f'{mode}.parquet'),'--per-task','--n-seq',str(a.n_seq),
                    '--workers','2','--server','http://127.0.0.1:18074','--tag',a.schedule+'_'+mode,
                    '--out',str(out/f'{mode}.jsonl')]
                subprocess.run(command,check=True,timeout=remaining(),env=env)
            response=requests.post('http://127.0.0.1:18074/v1/chat/completions',json={
                'model':'dsv41','messages':[{'role':'user','content':'Write a Python function that reverses a list. Be brief.'}],
                'max_tokens':128,'temperature':0,'cache_salt':'e074-'+a.schedule},timeout=min(remaining(),90))
            response.raise_for_status()
            (out/'generation_smoke.json').write_text(json.dumps(response.json(),indent=2)+'\n')
            metrics=requests.get('http://127.0.0.1:18074/metrics',timeout=10)
            (out/'metrics.txt').write_text(metrics.text)
            (out/'complete.json').write_text(json.dumps({'schedule':a.schedule,'elapsed_s':time.monotonic()-started,
                'note':'Compatibility/NLL pilot only; not ARC accuracy or a throughput comparison'})+'\n')
        finally:
            monitor_stop.set()
            try:
                children=owner.children(recursive=True)
            except psutil.NoSuchProcess:
                children=[]
            terminate_owned(children+[owner])
            process.wait(timeout=20)


if __name__=='__main__':
    main()
