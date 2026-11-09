"""Export the E032 LoRA adapter from Tinker as signed download URLs (the API key never leaves this host).

  python -I export_adapter.py OUT.json [STATE_PATH [EXTERNAL_PATH]]
Writes {"path", "expires", "urls": {filename: signed_url}}. Pass only the URLs to the box.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from train import tinker_key  # noqa: E402

STATE = "tinker://99c4c413-7a89-5c0b-9d9a-30bcfa9a9e1d:train:0/weights/final"
out = sys.argv[1]
state = sys.argv[2] if len(sys.argv) > 2 else STATE
tinker_key()
import tinker  # noqa: E402
t0 = time.time()
sc = tinker.ServiceClient()
tc = sc.create_training_client_from_state(state)
print("client loaded", round(time.time() - t0), "s", flush=True)
if len(sys.argv) > 3:  # already saved: reuse the external path, only fetch fresh URLs
    res = type("R", (), {"path": sys.argv[3]})
else:
    res = tc.save_weights_external("e032_final_external", ttl_seconds=7 * 86400).result()
print("saved external", res.path, round(time.time() - t0), "s", flush=True)
urls = sc.create_rest_client().get_external_weights_urls(res.path).result()
d = {"path": res.path, "expires": str(urls.expires), "urls": dict(urls.urls)}
json.dump(d, open(out, "w"), indent=1)
print("files", list(d["urls"]), "expires", d["expires"], round(time.time() - t0), "s")
