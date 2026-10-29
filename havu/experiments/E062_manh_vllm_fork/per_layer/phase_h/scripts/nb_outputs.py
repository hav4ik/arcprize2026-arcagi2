"""Dump an executed notebook's cell outputs as plain text, in the order Kaggle's log shows them: one header per cell
(index, first source line, execution timing from nbclient's record_timing), then its stream / result / error text.
Usage: nb_outputs.py <executed.ipynb> > outputs.txt"""
import re, sys
import nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
ansi = re.compile(r"\x1b\[[0-9;]*m")
for i, c in enumerate(nb.cells):
    first = (c.source.splitlines() or [""])[0][:120]
    t = c.get("metadata", {}).get("execution", {})
    print(f"\n===== cell {i} [{c.cell_type}] {first}")
    if t:
        print(f"      start {t.get('iopub.execute_input', '?')} end {t.get('shell.execute_reply', '?')}")
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            sys.stdout.write(ansi.sub("", o.text))
        elif o.output_type in ("execute_result", "display_data"):
            sys.stdout.write(ansi.sub("", o.get("data", {}).get("text/plain", "")) + "\n")
        elif o.output_type == "error":
            sys.stdout.write(f"ERROR {o.ename}: {ansi.sub('', o.evalue)}\n" + ansi.sub("", "\n".join(o.traceback)) + "\n")
