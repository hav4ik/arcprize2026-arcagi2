"""Sequential, version-pinned kagglehub downloads of whole datasets into /root/kin/<owner>/<slug>/<version>."""
import kagglehub, os, shutil, sys, time
for h in sys.argv[1:]:
    t = time.time()
    p = kagglehub.dataset_download(h)
    print("GOT", h, "->", p, "%.0fs" % (time.time() - t), flush=True)
    time.sleep(2)
print("DL_KIN_DONE", flush=True)
