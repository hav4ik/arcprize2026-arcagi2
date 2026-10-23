"""E043: runtime expert masks, Engram row masks and REAP observation for DeepSeek-V4.1 in the vLLM fork.

Installed as vllm/e043_obs.py and called from four patched spots (see apply_e043_patches.py):
  DeepseekV4Model.forward start      -> step_begin(input_ids, positions)
  DeepseekV4MoE._forward_fused_moe   -> begin_moe(prefix)
  V41 router (deepseek_v41_topk)     -> masked_bias(correction_bias)
  MarlinExperts.apply / moe_sum      -> set_topk(...) / observe(moe_output, topk_ids, expert_map)
  Engram.prepare_embeddings          -> engram_ids(layer_hash_index, hash_ids)

Everything is driven by a JSON control file named by $E043_CTRL, re-read when its mtime changes (checked at
the start of every eager forward; CUDA-graph replays keep using the same tensors, which are updated in place):
  {"expert_keep": "/path/keep.npy" | null,     # bool [40, 384]; pruned experts get bias -1e9 (never selected,
                                               #  the native top-6 refills from kept ids, weights renormalized)
   "engram_keep": "/path/keep.npz" | null,     # per Engram layer: packbits(bool[num_embeddings], 'little')
                                               #  -> dropped rows read as zero before Wkv (the declared miss rule)
   "observe": "/path/outdir" | null,           # record per-token router ids/weights and native expert amplitudes
   "dump_hashes": 0}                           # also record the first N tokens' Engram hash ids
Observation needs expert parallel (each routed expert output complete on one rank). Rank 0 writes one .npz per
forward step: input_ids [T], positions [T], ids/w/amp [T, 40, 6] (amp = ||native weighted expert output||_2,
the reap_native_amp observation, all-reduced over the EP group so every (token, slot) has its owner's value).
"""
import json
import os

import numpy as np
import torch

N_LAYERS = 40
N_EXPERTS = 384
TOPK = 6
_CTRL = os.environ.get("E043_CTRL")

_s = {
    "mtime": None, "layer": -1, "bias_add": None, "eng_bits": {}, "obs_dir": None, "dump_hashes": 0,
    "rec": None, "topk_w": None, "step_no": 0, "rank": None, "tp_group": None, "hash_rec": {},
}


def _rank():
    if _s["rank"] is None:
        from vllm.distributed import get_tensor_model_parallel_rank
        _s["rank"] = get_tensor_model_parallel_rank()
    return _s["rank"]


def _capturing():
    return torch.cuda.is_current_stream_capturing()


ENGRAM_ROWS = (384006168, 384016682)


def _ensure_tensors(device):
    """Allocate every mask tensor before CUDA-graph capture; later control loads only copy in place."""
    if _s["bias_add"] is None:
        _s["bias_add"] = torch.zeros(N_LAYERS + 8, N_EXPERTS, dtype=torch.float32, device=device)
        for li, rows in enumerate(ENGRAM_ROWS):
            _s["eng_bits"][li] = torch.full(((rows + 7) // 8,), 255, dtype=torch.uint8, device=device)


def _load_ctrl(device):
    if not _CTRL or not os.path.exists(_CTRL):
        return
    mtime = os.stat(_CTRL).st_mtime_ns
    if mtime == _s["mtime"]:
        return
    _s["mtime"] = mtime
    ctrl = json.load(open(_CTRL))
    _ensure_tensors(device)
    keep = ctrl.get("expert_keep")
    if keep:
        k = torch.from_numpy(np.load(keep).astype(bool))
        assert k.shape == (N_LAYERS, N_EXPERTS), k.shape
        add = torch.where(k, 0.0, -1e9).to(torch.float32)
        _s["bias_add"][:N_LAYERS].copy_(add.to(device))
    else:
        _s["bias_add"].zero_()
    ek = ctrl.get("engram_keep")
    if ek:
        z = np.load(ek)
        for li in range(2):
            bits = torch.from_numpy(z[f"layer{li}"])
            assert bits.shape == _s["eng_bits"][li].shape, (bits.shape, _s["eng_bits"][li].shape)
            _s["eng_bits"][li].copy_(bits.to(device))
    else:
        for b in _s["eng_bits"].values():
            b.fill_(255)
    _s["obs_dir"] = ctrl.get("observe")
    _s["dump_hashes"] = int(ctrl.get("dump_hashes", 0))
    if _s["obs_dir"] and _rank() == 0:
        os.makedirs(_s["obs_dir"], exist_ok=True)
    if _rank() == 0:
        print(f"[E043] ctrl loaded: experts={keep} engram={ek} observe={_s['obs_dir']}", flush=True)


def step_begin(input_ids, positions):
    """Start of a model forward: flush the previous step's records, reload the control file."""
    if _capturing() or input_ids is None:
        return
    _flush()
    _load_ctrl(input_ids.device)
    _ensure_tensors(input_ids.device)
    if _s["obs_dir"]:
        T = input_ids.shape[0]
        _s["rec"] = {"input_ids": input_ids.detach().to(torch.int32).clone(),
                     "positions": positions.detach().to(torch.int32).clone(),
                     "ids": torch.full((T, N_LAYERS, TOPK), -1, dtype=torch.int16, device=input_ids.device),
                     "w": torch.zeros((T, N_LAYERS, TOPK), dtype=torch.float32, device=input_ids.device),
                     "amp": torch.zeros((T, N_LAYERS, TOPK), dtype=torch.float32, device=input_ids.device),
                     "seen": torch.zeros(N_LAYERS, dtype=torch.int32)}
    else:
        _s["rec"] = None


def begin_moe(prefix):
    try:
        _s["layer"] = int(prefix.split("layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        _s["layer"] = -1


def masked_bias(correction_bias, li):
    """Selection bias of layer li with pruned experts pushed to -1e9."""
    if correction_bias is None or _s["bias_add"] is None or li < 0 or li >= N_LAYERS or correction_bias.shape[0] != N_EXPERTS:
        return correction_bias
    return correction_bias + _s["bias_add"][li]


def engram_ids(layer_hash_index, hash_ids):
    """Rewrite dropped Engram row ids to -1 (not owned by any rank -> zero row before Wkv)."""
    rec = _s["rec"]
    if rec is not None and _s["dump_hashes"] and not _capturing() and _rank() == 0:
        n = _s["dump_hashes"]
        _s["hash_rec"].setdefault(layer_hash_index, []).append(hash_ids[:n].detach().to("cpu"))
    bits = _s["eng_bits"].get(layer_hash_index)
    if bits is None:
        return hash_ids
    idx = hash_ids.to(torch.int64)
    keep = ((bits[(idx >> 3).clamp_min(0)] >> (idx & 7).to(torch.uint8)) & 1).bool() & (idx >= 0)
    return torch.where(keep, hash_ids, torch.full_like(hash_ids, -1))


def set_topk(topk_weights, topk_ids):
    _s["topk_w"] = topk_weights


def observe(moe_output, topk_ids, expert_map):
    """moe_output: [M, topk, K] native weighted expert outputs (before moe_sum)."""
    rec = _s["rec"]
    li = _s["layer"]
    if rec is None or _capturing() or li < 0 or li >= N_LAYERS:
        return
    M = moe_output.shape[0]
    if M != rec["ids"].shape[0]:
        raise RuntimeError(f"E043 observe: {M} MoE tokens vs {rec['ids'].shape[0]} step tokens (padding/SP?)")
    amp = moe_output.float().norm(dim=-1)
    if expert_map is not None:
        # Rows of experts owned by other EP ranks are uninitialized workspace (may hold NaN/Inf): select, don't
        # multiply, then sum the owners' values over the EP group.
        local = expert_map[topk_ids.long()] >= 0
        amp = torch.where(local, amp, torch.zeros_like(amp))
        from vllm.distributed import get_ep_group
        amp = get_ep_group().all_reduce(amp)
    else:
        from vllm.distributed import get_tensor_model_parallel_world_size
        if get_tensor_model_parallel_world_size() > 1:
            raise RuntimeError("E043 observe needs --enable-expert-parallel (TP partial outputs)")
    rec["ids"][:, li] = topk_ids.to(torch.int16)
    rec["amp"][:, li] = amp
    if _s["topk_w"] is not None and _s["topk_w"].shape[0] == M:
        rec["w"][:, li] = _s["topk_w"].float()
    rec["seen"][li] += 1


def _flush():
    rec = _s["rec"]
    if rec is None or _rank() != 0:
        _s["rec"] = None
        return
    _s["rec"] = None
    if int(rec["seen"].min()) != 1 or int(rec["seen"].max()) != 1:
        print(f"[E043] WARNING step {_s['step_no']}: per-layer observe counts {rec['seen'].tolist()}", flush=True)
    out = {k: v.cpu().numpy() for k, v in rec.items() if k != "seen"}
    out["ids"] = out["ids"].astype(np.int16)
    out["w"] = out["w"].astype(np.float16)
    if _s["hash_rec"]:
        for li, chunks in _s["hash_rec"].items():
            out[f"hash{li}"] = torch.cat(chunks).numpy()
        _s["hash_rec"] = {}
    path = os.path.join(_s["obs_dir"], f"step_{_s['step_no']:07d}.npz")
    np.savez(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    _s["step_no"] += 1


def flush_now():
    _flush()
