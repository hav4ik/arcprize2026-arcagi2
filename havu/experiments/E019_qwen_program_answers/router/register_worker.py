"""Register / remove / list SGLang workers on the local SGLang Model Gateway (smg, E019). Tokens are never printed.

    R=experiments/E019_qwen_program_answers/router
    .venv/bin/python $R/register_worker.py vastai-1x6000-us-qwen27b      # health-check, tunnel, register
    .venv/bin/python $R/register_worker.py --remove vastai-1x6000-us-qwen27b
    .venv/bin/python $R/register_worker.py --list
    .venv/bin/python $R/register_worker.py --restore                     # re-register all (after a gateway restart)

Register (default, mode "tunnel"):
  1. reads `addr` and `token` of the ENDPOINTS.yaml entry at runtime;
  2. health-checks the worker directly through Vast's Caddy edge: GET /v1/models + a 4-token completion;
  3. finds the box's SSH port with `vastai show instances` (matched on the public IP + mapped 8000/tcp port, since boxes can share an IP; Vast key from SECRETS.yaml)
     unless --ssh HOST:PORT is given, adds a line to tunnels.conf and starts its restart loop (tunnels.sh), which
     forwards 127.0.0.1:<local port> to the box's internal SGLang port 127.0.0.1:18000;
  4. POSTs {"url": "http://127.0.0.1:<local port>"} to the gateway's /workers and waits until it is healthy.
  The name -> local port / ssh mapping is kept in workers.json (no secrets) so --restore can re-register.

Mode "--direct": POSTs {"url": "http://<addr>", "api_key": <token>} instead. The gateway sends that key on
completions and /server_info, /model_info, but its connection-detection and health probes hit /health with NO
key, so Vast's Caddy answers 401 and the worker never registers (sglang-router 0.3.2 and smg 1.11.0). Use it
only for workers whose /health is reachable without auth.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DIR = Path(__file__).resolve().parent
REG = DIR / "workers.json"
CONF = DIR / "tunnels.conf"
FIRST_PORT = 30101


def http(url, body=None, method=None, token=None, timeout=30):
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw.strip().startswith(("{", "[")) else raw


def entry(name):
    for e in yaml.safe_load((ROOT / "ENDPOINTS.yaml").read_text()):
        if e.get("name") == name:
            return f"http://{e['addr']}", e.get("token") or ""
    raise SystemExit(f"{name}: not in ENDPOINTS.yaml")


def load_reg():
    return json.loads(REG.read_text()) if REG.exists() else {}


def save_reg(reg):
    REG.write_text(json.dumps(reg, indent=1) + "\n")
    lines = ["# name local_port ssh_host ssh_port (written by register_worker.py)"]
    lines += [f"{n} {w['local_port']} {w['ssh_host']} {w['ssh_port']}" for n, w in reg.items() if w["mode"] == "tunnel"]
    CONF.write_text("\n".join(lines) + "\n")


def direct_check(url, token):
    t = time.time()
    models = http(url + "/v1/models", token=token)
    model = models["data"][0]["id"]
    r = http(url + "/v1/completions", {"model": model, "prompt": "1+1=", "max_tokens": 4, "temperature": 0},
             token=token, timeout=120)
    assert r["choices"][0]["text"] is not None
    print(f"  direct check ok: model={model} max_model_len={models['data'][0].get('max_model_len')} "
          f"({time.time() - t:.1f}s)", flush=True)
    return model


def pick_instance(instances, ip, web_port=None):
    """Pick the Vast instance for a box. Several instances can share one public IP (different host ports), so match
    on the mapped 8000/tcp host port from ENDPOINTS as well; an IP-only match is accepted only if unambiguous."""
    same_ip = [i for i in instances if (i.get("public_ipaddr") or "").strip() == ip]
    if web_port is not None:
        hit = [i for i in same_ip
               if any(int(m["HostPort"]) == int(web_port) for m in ((i.get("ports") or {}).get("8000/tcp") or []))]
        if len(hit) == 1:
            return hit[0]
        raise SystemExit(f"{len(hit)} Vast instances match {ip}:{web_port} (8000/tcp); use --ssh HOST:PORT")
    if len(same_ip) == 1:
        return same_ip[0]
    raise SystemExit(f"{len(same_ip)} Vast instances share public IP {ip}; use --ssh HOST:PORT")


def vast_ssh(ip, web_port=None):
    sec = (ROOT / "SECRETS.yaml").read_text()
    key = next(line.split(":", 1)[1].strip().strip("\"' ") for line in sec.splitlines()
               if line.startswith("vast_ai_api_key:"))
    out = subprocess.run([str(ROOT / ".venv/bin/vastai"), "show", "instances", "--raw", "--api-key", key],
                         capture_output=True, text=True, check=True).stdout
    i = pick_instance(json.loads(out), ip, web_port)
    p = (i.get("ports") or {}).get("22/tcp")
    if p:
        return ip, int(p[0]["HostPort"]), i["id"]
    return i["ssh_host"], int(i["ssh_port"]), i["id"]


def gw_workers(gw):
    return http(gw + "/workers")["workers"]


def wait_healthy(gw, url, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for w in gw_workers(gw):
            if w["url"] == url and w["is_healthy"]:
                return w
        time.sleep(3)
    return None


def add_to_gateway(gw, url, api_key=None, timeout=120):
    if any(w["url"] == url for w in gw_workers(gw)):
        print(f"  already registered: {url}")
        return wait_healthy(gw, url, 5)
    body = {"url": url}
    if api_key:
        body["api_key"] = api_key
    r = http(gw + "/workers", body)
    print(f"  gateway: {r.get('status')} ({r.get('message', '')})", flush=True)
    return wait_healthy(gw, url, timeout)


def tunnel_up(lport, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{lport}/health", timeout=10)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return False


def register(args, name):
    url, token = entry(name)
    reg = load_reg()
    direct_check(url, token)
    if args.direct:
        reg[name] = {"mode": "direct", "gateway_url": url}
        w = add_to_gateway(args.gateway, url, api_key=token)
    else:
        if args.ssh:
            host, port = args.ssh.rsplit(":", 1)
            port, iid = int(port), None
        else:
            ipp = url.split("//")[1].split("/")[0]
            host, port, iid = vast_ssh(ipp.split(":")[0], ipp.split(":")[1] if ":" in ipp else None)
        used = {w.get("local_port") for n, w in reg.items() if n != name}
        lport = reg.get(name, {}).get("local_port") or next(p for p in range(FIRST_PORT, FIRST_PORT + 100)
                                                             if p not in used)
        reg[name] = {"mode": "tunnel", "local_port": lport, "ssh_host": host, "ssh_port": port,
                     "instance_id": iid, "gateway_url": f"http://127.0.0.1:{lport}"}
        save_reg(reg)
        subprocess.run([str(DIR / "tunnels.sh"), "start"], check=True)
        if not tunnel_up(lport):
            raise SystemExit(f"  tunnel 127.0.0.1:{lport} -> {host}:{port} not answering /health; see tunnel_*.log")
        print(f"  tunnel ok: 127.0.0.1:{lport} -> {host}:{port} (instance {iid})", flush=True)
        w = add_to_gateway(args.gateway, reg[name]["gateway_url"])
    save_reg(reg)
    print(f"  {name}: " + (f"registered, healthy, id={w['id']}" if w else "NOT healthy in gateway (check router.log)"))


def remove(args, name):
    reg = load_reg()
    w = reg.get(name)
    if not w:
        raise SystemExit(f"{name}: not in workers.json")
    for gw in gw_workers(args.gateway):
        if gw["url"] == w["gateway_url"]:
            http(f"{args.gateway}/workers/{gw['id']}", method="DELETE")
            print(f"  removed from gateway: {gw['url']}")
    if w["mode"] == "tunnel":
        pid = DIR / f"tunnel_{name}.pid"
        if pid.exists():
            p = pid.read_text().strip()
            subprocess.run(["pkill", "-P", p], check=False)
            subprocess.run(["kill", p], check=False)
            pid.unlink()
            print(f"  tunnel loop stopped (127.0.0.1:{w['local_port']})")
    del reg[name]
    save_reg(reg)


def list_workers(args):
    reg = load_reg()
    by_url = {w["gateway_url"]: n for n, w in reg.items()}
    try:
        ws = gw_workers(args.gateway)
    except Exception as e:  # noqa: BLE001
        print(f"gateway {args.gateway} not reachable: {type(e).__name__}")
        ws = []
    print(f"{'name':28s} {'gateway url':24s} {'healthy':7s} {'load':4s} id")
    seen = set()
    for w in ws:
        n = by_url.get(w["url"], "?")
        seen.add(n)
        print(f"{n:28s} {w['url']:24s} {str(w['is_healthy']):7s} {w['load']:<4} {w['id']}")
    for n, w in reg.items():
        if n not in seen:
            print(f"{n:28s} {w['gateway_url']:24s} NOT REGISTERED in gateway (run --restore)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="ENDPOINTS.yaml entry names")
    ap.add_argument("--gateway", default="http://127.0.0.1:30100")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", action="store_true", help="re-register every worker in workers.json")
    ap.add_argument("--direct", action="store_true", help="register the Caddy URL with api_key (see docstring)")
    ap.add_argument("--ssh", help="HOST:PORT for the tunnel (default: looked up with vastai)")
    args = ap.parse_args()
    if args.list:
        return list_workers(args)
    if args.restore:
        reg = load_reg()
        if any(w["mode"] == "tunnel" for w in reg.values()):
            subprocess.run([str(DIR / "tunnels.sh"), "start"], check=True)
        for n, w in reg.items():
            print(n, flush=True)
            if w["mode"] == "tunnel":
                tunnel_up(w["local_port"])
            add_to_gateway(args.gateway, w["gateway_url"], api_key=entry(n)[1] if w["mode"] == "direct" else None)
        return list_workers(args)
    if not args.names:
        ap.error("give at least one ENDPOINTS.yaml name, or --list / --restore")
    for n in args.names:
        print(n, flush=True)
        try:
            (remove if args.remove else register)(args, n)
        except (urllib.error.URLError, OSError, KeyError, AssertionError) as e:
            print(f"  {n}: FAILED {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
