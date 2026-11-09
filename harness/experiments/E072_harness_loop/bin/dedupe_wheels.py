"""dedupe_wheels.py <keep dir> <dedupe dir>: replace files in <dedupe dir> that are byte-identical to the same-named file in <keep dir> by hardlinks (saves disk)."""
import os, sys, filecmp
a, b = sys.argv[1:3]; saved = n = 0
for r, _, fs in os.walk(b):
    for f in fs:
        pb = os.path.join(r, f); pa = os.path.join(a, os.path.relpath(pb, b))
        if os.path.isfile(pa) and not os.path.samefile(pa, pb) and os.path.getsize(pa) == os.path.getsize(pb) and filecmp.cmp(pa, pb, shallow=False):
            os.unlink(pb); os.link(pa, pb); saved += os.path.getsize(pa); n += 1
print(f"hardlinked {n} files, saved {saved / 1e9:.2f} GB")
