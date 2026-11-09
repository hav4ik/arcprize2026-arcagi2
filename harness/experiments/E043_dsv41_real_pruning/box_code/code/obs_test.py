import json, glob, os, time, numpy as np, requests
ctrl = '/dev/shm/e043/ctrl.json'; od = '/dev/shm/e043/obs_test'
os.makedirs(od, exist_ok=True); [os.remove(f) for f in glob.glob(od + '/*')]
fx = np.load('/workspace/dsv4/data/hash_parity_cpu.npz')
ids = fx['ids'].tolist()
def post(p): return requests.post('http://127.0.0.1:18000/v1/completions', json={"model": "dsv41", "prompt": p, "max_tokens": 1, "temperature": 0, "cache_salt": str(time.time())}, timeout=600).json()
json.dump({"observe": od, "dump_hashes": 100000}, open(ctrl + '.tmp', 'w')); os.replace(ctrl + '.tmp', ctrl)
post([0, 5, 6, 7]); post(ids)
json.dump({"observe": None}, open(ctrl + '.tmp', 'w')); os.replace(ctrl + '.tmp', ctrl)
post([0, 5, 6, 7])
steps = sorted(glob.glob(od + '/step_*.npz'))
print('steps', len(steps))
inp = []; hs = {0: [], 1: []}; recs = []
for s in steps:
    z = np.load(s); inp.append(z['input_ids']); recs.append(z)
    for l in (0, 1):
        if f'hash{l}' in z.files: hs[l].append(z[f'hash{l}'])
allin = np.concatenate(inp)
print('recorded tokens', len(allin), 'expected', 4 + len(ids), 'match', np.array_equal(allin[4:], np.array(ids)))
for l in (0, 1):
    h = np.concatenate(hs[l])[4:]
    print('layer', l, 'hash shape', h.shape, 'parity', np.array_equal(h.astype(np.int64), fx['hashes'][:, l, :]))
z = recs[-1]
ids_ = np.concatenate([r['ids'] for r in recs])[4:]; amp = np.concatenate([r['amp'] for r in recs])[4:].astype(np.float32); w = np.concatenate([r['w'] for r in recs])[4:].astype(np.float32)
print('ids range', ids_.min(), ids_.max(), 'any -1', (ids_ < 0).any(), 'amp>0 frac', (amp > 0).mean(), 'amp mean', amp.mean(), 'w sum per token (should be 1.5)', w.sum(-1).mean(), w.sum(-1).std())
print('distinct experts per layer (first 5 layers)', [len(np.unique(ids_[:, li])) for li in range(5)])
print('amp per layer mean (every 8th)', [round(float(amp[:, li].mean()), 4) for li in range(0, 40, 8)])
zero = amp == 0
print('zero frac per layer', [round(float(zero[:, li].mean()), 3) for li in range(0, 40, 4)])
rk = ids_ // 48
print('zero frac per EP rank', [round(float(zero[rk == r].mean()), 3) for r in range(8)])
print('nan/inf', np.isnan(amp).sum(), np.isinf(amp).sum(), 'max', np.nanmax(amp), 'p99.9', np.nanpercentile(amp, 99.9))
print('zero frac by slot', [round(float(zero[:, :, k].mean()), 3) for k in range(6)])
