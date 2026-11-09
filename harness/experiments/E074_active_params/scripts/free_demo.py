#!/usr/bin/env python3
"""Stop only this template's identified Qwen demo and its captured descendants."""
import subprocess
import psutil
import requests
from process_utils import terminate_owned

models = requests.get('http://127.0.0.1:18000/v1/models', timeout=10).json()['data']
assert len(models) == 1 and models[0]['id'] == 'Qwen/Qwen3.5-9B', 'Unexpected server; leave it alone'
pid = int(subprocess.check_output(['supervisorctl', 'pid', 'vllm'], text=True).strip())
assert pid > 1
parent = psutil.Process(pid)
assert any('vllm' in arg for arg in parent.cmdline()), 'Unexpected supervisor process'
children = parent.children(recursive=True)
subprocess.run(['supervisorctl', 'stop', 'vllm'], check=True)
terminate_owned(children, grace=10)
print('Stopped Qwen template demo and its captured descendants:', len(children))
