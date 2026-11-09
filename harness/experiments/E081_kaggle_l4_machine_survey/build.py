"""E081: build probe notebook N (slug chankhavu/e081-l4-probe-NN) with V12's mounts; usage: build.py N OUTDIR [b].
Variant b (E081b): slug e081-l4-probeb-NN, both E128 models attached (V12's text model and arazzd's causal image
model), read in alternating order (odd N: causal image model first)."""
import base64, json, sys
from pathlib import Path
HERE = Path(__file__).parent
n, out = int(sys.argv[1]), Path(sys.argv[2])
B = len(sys.argv) > 3 and sys.argv[3] == 'b'
TXT = '/kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1'
IMG = '/kaggle/input/models/arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-causal/pytorch/default/1'
WD = '/kaggle/input/datasets/nguyennguyen599/vllm-wheels,/kaggle/input/datasets/nguyennguyen599/cu13-install'
ORDER = (f'{IMG},{TXT},{WD}' if n % 2 else f'{TXT},{IMG},{WD}') if B else ''
out.mkdir(parents=True, exist_ok=True)
enc = lambda p: base64.b64encode((HERE / p).read_bytes()).decode()
cells = [
    "import base64\n"
    f"open('/tmp/gpucheck.py','wb').write(base64.b64decode('{enc('gpucheck_manh.py')}'))\n"
    f"open('/tmp/probe.py','wb').write(base64.b64decode('{enc('probe.py')}'))",
    "!nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.width.max --format=csv",
    (f"!E081_ROOTS={ORDER} python3 -u /tmp/probe.py" if B else "!python3 -u /tmp/probe.py"),
]
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "probe.ipynb").write_text(json.dumps(nb, indent=1))
slug = f"e081-l4-probe{'b' if B else ''}-{n:02d}"
meta = {"id": f"chankhavu/{slug}", "title": slug, "code_file": "probe.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaL4",
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1"]
        + (["arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-causal/PyTorch/default/1"] if B else []),
        "dataset_sources": ["nguyennguyen599/vllm-wheels", "nguyennguyen599/cu13-install", "chankhavu/dsv41-tools-kit",
                            "nguyennguyen599/arc2-2026-src", "nguyennguyen599/unsloth-main"],
        "kernel_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(meta["id"])
