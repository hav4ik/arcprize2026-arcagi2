"""Generated tool checks on 12 training originals, never evaluation tasks."""
import json
import subprocess
import time


def run(root, exp, out, tag, env, remaining):
    for mode in ('img', 'txt'):
        cmd = [str(root/'tvenv/bin/python'),
            str(root/'repo/experiments/E066_dsv41_image_pruning/scripts/b32_local_eval.py'),
            '--mode','grid','--effort','low','--prompt','lean2sub','--answer-tool',
            '--final-mode','session','--lenient-parse','--max-tokens','24000',
            '--max-total-output','24000','--budget-soft','12000','--budget-notes','0.01',
            '--budget-warn','0.75','--effort-header','minimal','--turns-note','5',
            '--train-check','--rescue-code','--model','dsv41','--fallback-solve','--multi-test',
            '--auto-repair','--multi-budget','0.5','--context-limit','65536','--max-turns','20',
            '--grid-format','text+image' if mode == 'img' else 'json','--image-cell','0',
            '--base-url','http://127.0.0.1:18076/v1',
            '--extra-body',json.dumps({'chat_template_kwargs':{'thinking':True,'reasoning_effort':50},
                'temperature':1.0,'top_p':1.0,'cache_salt':'e076-train-'+tag+'-'+mode}),
            '--sandbox-url','local','--attempts','1','--workers','12',
            '--challenges',str(exp/'data/training_diagnostic_challenges.json'),
            '--out',str(out/f'diagnostic_{mode}.jsonl')]
        if mode == 'txt':
            cmd.append('--token-exact')
        (out/f'diagnostic_{mode}_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        start = time.monotonic()
        with (out/f'diagnostic_{mode}.log').open('w') as log:
            subprocess.run(cmd,cwd=exp/'frozen/evals',env=env,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=remaining())
        rows = [json.loads(s) for s in (out/f'diagnostic_{mode}.jsonl').read_text().splitlines()]
        expected = set(json.loads((exp/'data/training_diagnostic_challenges.json').read_text()))
        assert {r['task_id'] for r in rows} == expected
        assert not any(r.get('error') for r in rows), 'Diagnostic request failure'
        (out/f'diagnostic_{mode}_complete.json').write_text(json.dumps({
            'tasks':len(expected),'attempts':1,'seconds':time.monotonic()-start,
            'source':'Kaggle ARC training originals; not an evaluation score'})+'\n')
