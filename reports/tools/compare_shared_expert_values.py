"""Decode FP8 (e4m3fn) + UE8M0 block-32 shared-expert weights in two shards and compare VALUES (not bytes)."""
import json
import struct
import sys

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_u8(path, base, info):
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        return np.frombuffer(f.read(e - s), dtype=np.uint8).reshape(info["shape"])


# e4m3fn lookup table (256 entries)
def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** (-6)
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = s * v
    return t


LUT = e4m3_table()


def decode(w_u8, s_u8):
    w = LUT[w_u8]  # [N, K]
    sc = np.exp2(s_u8.astype(np.float32) - 127.0)  # [N/32, K/32]
    N, K = w.shape
    sc_full = np.repeat(np.repeat(sc, 32, axis=0), 32, axis=1)[:N, :K]
    return w * sc_full


def main(base_path, exp_path, max_tensors=12):
    hb, ob = read_header(base_path)
    he, oe = read_header(exp_path)
    keys = sorted(k for k in hb if ".shared_experts." in k and k.endswith(".weight") and hb[k]["dtype"] == "F8_E4M3")
    print(f"shared-expert FP8 weights in shard: {len(keys)}")
    agg_changed = agg_n = 0
    rel_rms = []
    for k in keys[:max_tensors] + keys[-3:]:
        ks = k[: -len(".weight")] + ".scale"
        wb = decode(read_u8(base_path, ob, hb[k]), read_u8(base_path, ob, hb[ks]))
        we = decode(read_u8(exp_path, oe, he[k]), read_u8(exp_path, oe, he[ks]))
        d = we - wb
        changed = int((d != 0).sum())
        n = d.size
        rms_w = float(np.sqrt((wb * wb).mean()))
        rms_d = float(np.sqrt((d * d).mean()))
        agg_changed += changed
        agg_n += n
        rel_rms.append(rms_d / rms_w)
        nz = d[d != 0]
        print(f"  {k}: values changed {100*changed/n:.3f}%  rms(W)={rms_w:.4e} rms(dW)={rms_d:.3e} rel={rms_d/rms_w:.3e}  "
              f"median|dW| (changed)={np.median(np.abs(nz)) if nz.size else 0:.3e}  max|dW|={np.abs(d).max():.3e}  "
              f"scale bytes changed={int((read_u8(base_path, ob, hb[ks]) != read_u8(exp_path, oe, he[ks])).sum())}/{hb[ks]['shape'][0]*hb[ks]['shape'][1]}")
    print(f"\nAGGREGATE over {len(rel_rms)} tensors: values changed {100*agg_changed/agg_n:.3f}% ; mean rel RMS change {np.mean(rel_rms):.3e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
