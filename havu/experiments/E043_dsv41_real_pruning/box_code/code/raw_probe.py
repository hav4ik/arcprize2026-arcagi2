import json, os, sys, time, requests
sys.path.insert(0, '/workspace/dsv4/code/scripts')
from tokenize_rollouts import harness_tools
S = 'http://127.0.0.1:18001'
ctrl = '/dev/shm/e043/ctrlB.json'
def setc(exp):
    json.dump({"engram_keep": "/workspace/dsv4/data/engram/masks/engram_top64gib.npz", "expert_keep": exp, "observe": None}, open(ctrl + '.tmp', 'w')); os.replace(ctrl + '.tmp', ctrl)
    requests.post(f'{S}/v1/completions', json={"model": "dsv41", "prompt": [0, 5, 6, 7], "max_tokens": 1})
msgs = [{"role": "system", "content": "Reasoning Effort: Minimal. Keep your thinking short."}, {"role": "user", "content": "What is 17*23? Use the python tool to check."}]
tok = requests.post(f'{S}/tokenize', json={"model": "dsv41", "messages": msgs, "tools": [harness_tools()[0]], "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}}).json()["tokens"]
for tag, exp in [("e72", "/workspace/dsv4/data/reap/keep_all/keep_contrib_dec_e72.npy"), ("e96", "/workspace/dsv4/data/reap/keep_all/keep_contrib_dec_e96.npy")]:
    setc(exp)
    g = requests.post(f'{S}/v1/completions', json={"model": "dsv41", "prompt": tok, "max_tokens": 160, "temperature": 0.0, "skip_special_tokens": False, "cache_salt": tag + str(time.time()), "logprobs": 5}).json()["choices"][0]
    print("==", tag, repr(g["text"][:500]))
    lp = g["logprobs"]
    for i, (t, top) in enumerate(zip(lp["tokens"], lp["top_logprobs"])):
        if i < 40 and any(s in (t or '') for s in ["think", "DSML", "\n"]):
            print("  ", i, repr(t), {k: round(v, 2) for k, v in list(top.items())[:5]})
setc(None)
