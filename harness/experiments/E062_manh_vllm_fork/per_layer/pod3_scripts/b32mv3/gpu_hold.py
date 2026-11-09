"""Hold a fixed amount of GPU memory on every GPU for the whole run, so the pod's free VRAM after vLLM's startup matches
Kaggle's (the memory auditor: Kaggle's allocator cap minus reserved was 0.67-0.73 GiB vs the pod's 1.16-1.22, from
~0.38 GiB more non-torch memory on Kaggle, e.g. CUDA graphs 0.65 vs 0.34 GiB). A CUDA context per GPU (libcudart via
ctypes, no torch) plus one cudaMalloc, sized so this process's nvidia-smi compute-app memory reaches <MiB> per GPU.
Usage: gpu_hold.py <MiB per GPU> <log>"""
import ctypes, os, subprocess, sys, time

target, logf = int(sys.argv[1]), sys.argv[2]
cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")


def log(msg):
    with open(logf, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}\n")


def used_by_me():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    idx = {u.strip(): int(i) for i, u in (l.split(",") for l in uu.strip().splitlines())}
    m = {}
    for l in out.strip().splitlines():
        pid, uuid, mib = [x.strip() for x in l.split(",")]
        if int(pid) == os.getpid():
            m[idx[uuid]] = int(mib)
    return m


n = ctypes.c_int()
assert cudart.cudaGetDeviceCount(ctypes.byref(n)) == 0
ptrs = []
for d in range(n.value):
    assert cudart.cudaSetDevice(d) == 0
    assert cudart.cudaFree(ctypes.c_void_p(0)) == 0  # create the context
time.sleep(1)
ctx = used_by_me()
for d in range(n.value):
    assert cudart.cudaSetDevice(d) == 0
    extra = max(0, target - ctx.get(d, 0)) << 20
    p = ctypes.c_void_p()
    rc = cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(extra))
    assert rc == 0, rc
    assert cudart.cudaMemset(p, 0, ctypes.c_size_t(extra)) == 0  # touch it
    ptrs.append(p)
time.sleep(1)
log(f"HOLD pid {os.getpid()}: target {target} MiB/GPU; context {ctx} MiB; now {used_by_me()} MiB per GPU (nvidia-smi compute apps)")
while True:
    time.sleep(3600)
