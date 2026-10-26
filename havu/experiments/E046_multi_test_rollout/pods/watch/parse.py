"""Turn poll.sh output into timeline rows (markdown) + a TSV for analysis. Usage: poll.sh | python parse.py TIMELINE TSV"""
import re, sys
tl, tsv = sys.argv[1], sys.argv[2]
blocks, cur = [], None
for line in sys.stdin:
    line = line.rstrip("\n")
    if line.startswith("=== "):
        cur = {"hdr": line[4:].split()}
        blocks.append(cur)
    elif cur is not None and " " in line:
        k, v = line.split(" ", 1)
        cur[k] = v
    elif cur is not None:
        cur[line] = ""
rows, trows = [], []
for b in blocks:
    pid, run, t = b["hdr"][:3]
    if "SSH_FAIL" in b or "SERVER" not in b:
        rows.append(f"| {t} | {run} | UNREACHABLE | | | | | | | | | | | | {b.get('SSH_FAIL', '')} |")
        continue
    s = b.get("SERVER", "")
    m = re.search(r"\[(\S+) \+([\d.]+)h\] server: ([\d.]+) gen tok/s, prefix hits ([\d.]+), uncached ([\d.]+) tok/s, KV ([\d.]+), running ([\d.]+), waiting ([\d.]+), preemptions ([\d.]+).*?; (\d+) attempts done", s)
    last = b.get("LAST", "")
    ma = re.search(r"\+([\d.]+)h\]", last)
    age = ma.group(1) if ma else ""
    nums = b.get("NSTART", "")
    ng = re.search(r"NGATE (\d+)", "NSTART " + nums)
    gate = ng.group(1) if ng else ""
    nd = re.search(r"NDIED (\d+)", "NSTART " + nums); ns = re.match(r"(\d+)", nums)
    mem = b.get("MEM", ""); mr = re.search(r"rss_gib=([\d.]+)", mem); mx = re.search(r"maxrss=([\d.]+)", mem)
    cg = b.get("CG", ""); mc = re.search(r"rss\+shmem_gib=([\d.]+) limit_gib=([\d.]+)", cg)
    notes = []
    if b.get("SOLVER", "none").strip() in ("", "none"):
        notes.append("NO SOLVER PROC")
    if nd and nd.group(1) != "0":
        notes.append(f"server died x{nd.group(1)}")
    if ns and ns.group(1) != "1":
        notes.append(f"server starts {ns.group(1)}")
    if "DONE 1" in nums:
        notes.append("DONE")
    if b.get("EVENTS", "").strip():
        notes.append(b["EVENTS"][:150])
    if m:
        st, sage, tps, hits, unc, kv, rn, wt, pre, att = m.groups()
    else:
        st, sage, tps, hits, unc, kv, rn, wt, pre, att = [""] * 10
    att_now = (b.get("ATT") or "").strip()
    memtxt = f"{mr.group(1) if mr else ''} (max {mx.group(1) if mx else ''})"
    cgtxt = f"{mc.group(1)}/{mc.group(2)}" if mc else ""
    rows.append(f"| {t} | {run} | {age}h (srv {st}) | {tps} | {hits} | {unc} | {kv} | {rn} | {wt} | {pre} | {gate} | {att_now} | {memtxt} | {cgtxt} | {'; '.join(notes)} |")
    trows.append("\t".join([t, run, age, st, sage, tps, hits, unc, kv, rn, wt, pre, gate, att_now, mr.group(1) if mr else "", mc.group(1) if mc else "", "; ".join(notes)]))
with open(tl, "a") as f:
    f.write("\n".join(rows) + "\n")
with open(tsv, "a") as f:
    f.write("\n".join(trows) + "\n")
print("\n".join(rows))
