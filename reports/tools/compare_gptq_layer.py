"""Compare 2-bit Humming GPTQ expert tensors between the base checkpoint and an SFT export (one layer).

Pure numpy safetensors reader (no torch / safetensors needed).
"""
import json
import struct
import sys

import numpy as np

DT = {"I32": np.int32, "BF16": np.uint16, "U8": np.uint8, "F32": np.float32, "I8": np.int8, "F8_E4M3": np.uint8}


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            buf = f.read(e - s)
            out[k] = (np.frombuffer(buf, dtype=DT[v["dtype"]]).reshape(v["shape"]), v["dtype"])
    return out


def unpack_u2(packed_i32):
    x = packed_i32.astype(np.uint32)
    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    return ((x[..., None] >> shifts) & 3).astype(np.uint8).reshape(*x.shape[:-1], x.shape[-1] * 16)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequant(q, s, z, group):
    qu = unpack_u2(q).astype(np.float32)                     # [N, K]
    zu = unpack_u2(np.ascontiguousarray(z.T)).T.astype(np.float32)  # [N, K//group]
    sf = bf16_to_f32(s)                                      # [N, K//group]
    N, K = qu.shape
    g = K // group
    w = (qu.reshape(N, g, group) - zu[:, :, None]) * sf[:, :, None]
    return w.reshape(N, K), qu, zu, sf


def main(base_path, exp_path, n_experts=128):
    b = read_safetensors(base_path)
    e = read_safetensors(exp_path)
    print(f"base tensors={len(b)} export tensors={len(e)} same keys={set(b) == set(e)}")
    tot = {"w1": [0, 0, 0, 0, 0, 0], "w3": [0, 0, 0, 0, 0, 0], "w2": [0, 0, 0, 0, 0, 0]}
    wdelta = {"w1": [], "w3": [], "w2": []}
    wnorm = {"w1": [], "w3": [], "w2": []}
    nbytes_identical = 0
    nbytes_total = 0
    for ei in range(n_experts):
        for proj, group in (("w1", 128), ("w3", 128), ("w2", 64)):
            kq, ks, kz = f"e{ei}.{proj}.q", f"e{ei}.{proj}.s", f"e{ei}.{proj}.z"
            for k in (kq, ks, kz):
                assert b[k][1] == e[k][1] and b[k][0].shape == e[k][0].shape, (k, b[k][1], e[k][1], b[k][0].shape, e[k][0].shape)
                nbytes_total += b[k][0].nbytes
                if np.array_equal(b[k][0], e[k][0]):
                    nbytes_identical += b[k][0].nbytes
            wb, qb, zb, sb = dequant(b[kq][0], b[ks][0], b[kz][0], group)
            we, qe, ze, se = dequant(e[kq][0], e[ks][0], e[kz][0], group)
            t = tot[proj]
            t[0] += int((qb != qe).sum()); t[1] += qb.size
            t[2] += int((sb != se).sum()); t[3] += sb.size
            t[4] += int((zb != ze).sum()); t[5] += zb.size
            d = we - wb
            wdelta[proj].append(float(np.sqrt((d * d).mean())))
            wnorm[proj].append(float(np.sqrt((wb * wb).mean())))
            if ei < 2:
                print(f"  expert {ei} {proj}: q changed {100*(qb!=qe).mean():.3f}%  s changed {100*(sb!=se).mean():.3f}%  z changed {100*(zb!=ze).mean():.3f}%  "
                      f"rms(dW)={np.sqrt((d*d).mean()):.3e} rms(W)={np.sqrt((wb*wb).mean()):.3e} max|dW|={np.abs(d).max():.3e}  mean s={sb.mean():.4f}")
    print(f"\nbyte-identical tensors: {100*nbytes_identical/nbytes_total:.2f}% of bytes")
    for proj in ("w1", "w3", "w2"):
        t = tot[proj]
        print(f"{proj}: codes q changed {100*t[0]/t[1]:.4f}% ({t[0]}/{t[1]}) | scales s changed {100*t[2]/t[3]:.4f}% | zero-points z changed {100*t[4]/t[5]:.4f}% | "
              f"rms(dW)/rms(W) = {np.mean(wdelta[proj])/np.mean(wnorm[proj]):.4e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
