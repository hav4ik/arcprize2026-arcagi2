"""Stream one file of an HF repo to disk (mode file) or straight into a directory (mode untar: a .tar.gz extracted with
tarfile's 'data' filter, never stored whole), checking the bytes against the expected LFS sha256 on the fly.
The token is read from stdin (first line); it is sent only to huggingface.co, never to the CDN the resolve redirects to.
usage: hf_stream.py <model|dataset> REPO REV FILENAME SHA256 <file|untar|peek> OUT [tar compression: * (auto) | gz | ""]"""
import hashlib, http.client, os, sys, tarfile, time, urllib.parse

kind, repo, rev, fname, want, mode, out = sys.argv[1:8]
comp = sys.argv[8] if len(sys.argv) > 8 else "*"
token = sys.stdin.readline().strip()
prefix = "datasets/" if kind == "dataset" else ""
url = f"https://huggingface.co/{prefix}{repo}/resolve/{rev}/{urllib.parse.quote(fname)}"


def open_url(url, hops=0):
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPSConnection(u.netloc, timeout=120)
    headers = {"User-Agent": "hf_stream/1"}
    if u.hostname == "huggingface.co" and token:
        headers["Authorization"] = "Bearer " + token
    conn.request("GET", u.path + ("?" + u.query if u.query else ""), headers=headers)
    r = conn.getresponse()
    if r.status in (301, 302, 303, 307, 308):
        loc = urllib.parse.urljoin(url, r.getheader("Location"))
        r.read(); conn.close()
        if hops > 5:
            raise RuntimeError("too many redirects")
        return open_url(loc, hops + 1)
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status} from {u.hostname}")
    return r


class Hashing:
    def __init__(self, r):
        self.r, self.h, self.n, self.t0, self.last = r, hashlib.sha256(), 0, time.time(), time.time()

    def read(self, n=-1):
        b = self.r.read(n if n and n > 0 else 1 << 20)
        self.h.update(b); self.n += len(b)
        if time.time() - self.last > 30:
            self.last = time.time()
            print(f"  {self.n / 1e9:.2f} GB, {self.n / 1e6 / (time.time() - self.t0):.0f} MB/s", flush=True)
        return b


r = open_url(url)
total = int(r.getheader("Content-Length") or 0)
print(f"GET {kind} {repo}@{rev[:12]} {fname}: {total} bytes", flush=True)
src = Hashing(r)
if mode == "file":
    tmp = out + ".partial"
    with open(tmp, "wb") as f:
        while True:
            b = src.read(16 << 20)
            if not b:
                break
            f.write(b)
elif mode == "peek":
    b = src.read(4096)
    print("first bytes:", b[:16].hex(), "| ustar at 257:", b[257:263], flush=True)
    sys.exit(0)
else:
    os.makedirs(out, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=src, mode="r|" + comp) as tf:
        for m in tf:
            tf.extract(m, out, filter="data")
            n += 1
    while src.read(16 << 20):
        pass
    print(f"extracted {n} members into {out}", flush=True)
got = src.h.hexdigest()
ok = got == want and (not total or src.n == total)
print(f"{'SHA_OK' if ok else 'SHA_FAIL'} {src.n} bytes sha256 {got[:16]} (want {want[:16]}) in {time.time() - src.t0:.0f}s", flush=True)
if mode == "file":
    if ok:
        os.replace(tmp, out)
    else:
        os.remove(tmp)
sys.exit(0 if ok else 1)
