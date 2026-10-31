"""E081: build probe notebook N (slug chankhavu/e081-l4-probe-NN) with V12's mounts; usage: build.py N OUTDIR."""
import base64, json, sys
from pathlib import Path
HERE = Path(__file__).parent
n, out = int(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
enc = lambda p: base64.b64encode((HERE / p).read_bytes()).decode()
cells = [
    "import base64\n"
    f"open('/tmp/gpucheck.py','wb').write(base64.b64decode('{enc('gpucheck_manh.py')}'))\n"
    f"open('/tmp/probe.py','wb').write(base64.b64decode('{enc('probe.py')}'))",
    "!nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.width.max --format=csv",
    "!python3 -u /tmp/probe.py",
]
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "probe.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"chankhavu/e081-l4-probe-{n:02d}", "title": f"e081-l4-probe-{n:02d}", "code_file": "probe.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaL4",
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1"],
        "dataset_sources": ["nguyennguyen599/vllm-wheels", "nguyennguyen599/cu13-install", "chankhavu/dsv41-tools-kit",
                            "nguyennguyen599/arc2-2026-src", "nguyennguyen599/unsloth-main"],
        "kernel_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(meta["id"])
