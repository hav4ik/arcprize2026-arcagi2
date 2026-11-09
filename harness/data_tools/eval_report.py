#!/usr/bin/env python3
"""Regenerate reports/eval_report.md and reports/eval_runs.csv: one report of ALL ARC-AGI-2 evaluation runs.

    .venv/bin/python data_tools/eval_report.py [--hf]

Walks every *.jsonl under the repo (not data/, .venv, .git) whose first record has `task_id` and a `pred`/`correct`
field, scores each run against the answer key of ITS OWN task version (Kaggle vs GitHub), and writes the markdown
report and a one-row-per-run CSV. Read-only on all inputs. `--hf` additionally downloads the serving-lab replica
`attempts.jsonl` files (read-only HF key from SECRETS.yaml, never printed) into a cache dir and scores them too.
Definitions are printed at the top of the report (see DEFS below).
"""
import argparse, csv, glob, hashlib, json, os, re, statistics as st, sys, datetime
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = "https://github.com/hav4ik/arcprize2026-arcagi2/blob/main/havu/"
HF_CACHE = os.environ.get("EVAL_REPORT_HF_CACHE", "/tmp/eval_report_hf")
SKIP_DIRS = (".venv", "data", ".git", "node_modules", "__pycache__")
DIFF6 = ["4a21e3da", "abc82100", "b6f77b65", "f560132c", "faa9f03d", "d8e07eb2"]


def P(*a):
    return os.path.join(ROOT, *a)


# ------------------------------------------------------------------ answer keys
def load_keys():
    gh = {}
    for f in sorted(glob.glob(P("data/arc-agi-2-github/data/evaluation/*.json"))):
        d = json.load(open(f))
        gh[os.path.basename(f)[:-5]] = [t["output"] for t in d["test"]]
    kg_sol = json.load(open(P("data/arc-agi_evaluation_solutions.json")))
    kg = {t: v for t, v in kg_sol.items()}
    return gh, kg


GH, KG = load_keys()
EVAL = set(KG)
IDENT = sorted(t for t in EVAL if t not in DIFF6)  # 114 tasks identical in both versions


def load_workloads():
    """workload name -> dict(sol, idmap). Found next to idmap files named *_idmap.json."""
    out = {}
    for idm in glob.glob(P("experiments/**/*idmap.json"), recursive=True):
        base = idm[: -len("idmap.json")]
        sol = base + "solutions.json"
        if not os.path.exists(sol):
            continue
        name = os.path.basename(base).rstrip("_") or "eval_x2"
        try:
            out[name] = dict(sol=json.load(open(sol)), idmap=json.load(open(idm)), path=os.path.relpath(idm, ROOT))
        except Exception:
            pass
    return out


WORK = load_workloads()
WORK_IDS = {}
for _n, _w in WORK.items():
    for _i in _w["sol"]:
        WORK_IDS.setdefault(_i, []).append(_n)


# ------------------------------------------------------------------ experiments metadata
def load_tsv():
    rows = list(csv.DictReader(open(P("experiments/experiments.tsv")), delimiter="\t"))
    by_file = {}
    for r in rows:
        txt = " ".join(str(v) for v in r.values())
        for m in re.findall(r"evals/results/([\w.\-]+?)(?:\.jsonl|\b)", txt):
            by_file.setdefault(m.replace(".jsonl", ""), r)
    return rows, by_file


TSV, TSV_BY_FILE = load_tsv()
TSV_BY_ID = {r["id"]: r for r in TSV}


def exp_of(rel):
    stem = os.path.basename(rel)[:-6]
    if rel.startswith("evals/results/"):
        r = TSV_BY_FILE.get(stem)
        if r:
            return r["id"], r["name"], r.get("model", ""), r.get("started", "")
        m = re.search(r"(?<![a-z])e(\d{3})", stem)
        if m and ("E" + m.group(1)) in TSV_BY_ID:
            r = TSV_BY_ID["E" + m.group(1)]
            return r["id"], r["name"], r.get("model", ""), r.get("started", "")
        m = re.search(r"(?<![a-z])e(\d{3})", stem)
        if m:
            return "E" + m.group(1), "", "", ""
        return "n/a", "", "", ""
    m = re.search(r"/(E\d{3}[a-z]?)_", "/" + rel)
    if m:
        eid = m.group(1)
        r = TSV_BY_ID.get(eid) or next((x for x in TSV if x["id"].startswith(eid[:4])), None)
        return eid, (r or {}).get("name", ""), (r or {}).get("model", ""), (r or {}).get("started", "")
    if rel.startswith("kaggle/runs/"):
        return "Kaggle runs", "", "", ""
    return "n/a", "", "", ""


HF_RUNS = {}
for _r in csv.DictReader(open(P("reports/hf_trace_runs.csv"))):
    for _s in _r["source_files"].split(";"):
        HF_RUNS.setdefault(_s.strip(), _r)
SERVING_LAB = {"manh_v12", "sft_025ep_v12", "v12_fast", "v12_img_final", "v12_img_r0", "b32m_v3_kaggle_matched", "v1_qual"}


_READMES = None


def readme_exp(stem):
    global _READMES
    if _READMES is None:
        _READMES = {}
        for f in glob.glob(P("experiments/E*/README.md")):
            _READMES[os.path.basename(os.path.dirname(f)).split("_")[0]] = open(f, errors="ignore").read()
    for eid, txt in sorted(_READMES.items()):
        if stem in txt:
            return eid
    return None


PRIV = "https://huggingface.co/datasets/livobereshka/arc-agi2-eval-traces-private/tree/main/"
LAB = "https://huggingface.co/datasets/livobereshka/arc-agi2-vllm-serving-lab/tree/main/replicas/"
PRIV_BY_PATH = {}
if os.path.exists(P("reports/hf_private_index.csv")):
    for _r in csv.DictReader(open(P("reports/hf_private_index.csv"))):
        if _r["source_record_file"]:
            PRIV_BY_PATH[_r["source_record_file"]] = _r["path"]


def safe_id(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def hf_url(rel, aliases=()):
    """Deep link for a run: private traces repo, or the serving-lab replica directory; None if nothing was uploaded."""
    for x in [rel] + list(aliases):
        r = HF_RUNS.get(x)
        if r:
            return PRIV + "runs/pub_" + safe_id(r["run_id"])
    for x in [rel] + list(aliases):
        parts = x.split("/")
        for p in parts:
            if p in SERVING_LAB and ("phase_h" in parts or x.startswith("hf:")):
                return LAB + p
        if "candBf_x2t" in parts and "E055_kaggle_replica" in parts:
            return "https://huggingface.co/datasets/livobereshka/arc-agi2-dsv41-e128-candb32-replica/tree/main"
        if x in PRIV_BY_PATH:
            return PRIV + PRIV_BY_PATH[x]
    return None


def harness_similarity(rel, run_id, flags):
    """Closeness to the current best harness (Manh's V12 = B32 pinned harness 882c817f + auto-repair + multi-test 0.5 +
    token-exact + context 163840 + lean2sub + effort-header minimal, old tool E128). Rule-based, see DEFS."""
    t = (rel + " " + run_id).lower()
    fl = flags or ""
    img = "img" in t.replace("image_pruning", "") or "text+image" in fl
    v12_flags = all(x in fl for x in ("lean2sub", "effort_header=minimal", "multi_budget=0.5", "context_limit=163840"))
    if any(x in t for x in ("v12img", "v12_img")) or (v12_flags and "text+image" in fl):
        return "V12 + images"
    if any(x in t for x in ("manh_v12", "v12_m12", "sft_025ep_v12", "v12_fast", "b1_noise_v12", "s025_rerun")) or v12_flags:
        return "exact V12"
    if any(x in t for x in ("b32", "candb", "e052_next_code", "e055_kaggle_replica", "e057b", "e059_restart")):
        return "B32-family"
    return "other"


def hf_link(rel, recs_have_trace):
    r = HF_RUNS.get(rel)
    if r:
        return "HF livobereshka/arc-agi2-eval-traces (config %s, run_id %s)" % (r["config"], r["run_id"])
    parts = rel.split("/")
    if "candBf_x2t" in parts and "E055_kaggle_replica" in parts:
        return "HF livobereshka/arc-agi2-dsv41-e128-candb32-replica (run/attempts.jsonl + dsv41_traces.tar.gz)"
    for p in parts:
        if p in SERVING_LAB and "phase_h" in parts:
            return "HF livobereshka/arc-agi2-vllm-serving-lab (replicas/%s: dsv41_traces.tar.gz%s)" % (
                p, ", attempts.jsonl" if p in ("manh_v12", "v12_fast") else "")
    if rel.startswith("hf:"):
        return "HF " + rel[3:]
    return "local only: " + rel


# ------------------------------------------------------------------ record normalisation
def tokens(rec):
    """-> (completion, prompt_total, cache_hit) ints or None. Documented in DEFS."""
    u = rec.get("usage")
    comp = prompt = cache = None
    if isinstance(u, dict) and (u.get("completion_reported_cut") or (u.get("completion") or 0) > 5e8):
        return None  # server cut: completion count is a sentinel (1e9), not a real count
    if isinstance(u, dict):
        if "completion_tokens" in u:  # one chat call (OpenAI / DeepSeek / OpenRouter usage)
            comp = u.get("completion_tokens")
            prompt = u.get("prompt_tokens")
            cache = u.get("prompt_cache_hit_tokens")
            if cache is None:
                cache = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        elif "completion" in u:  # tool harness: prompt/completion summed over turns
            comp = u.get("completion")
            prompt = u.get("prompt", u.get("prompt_total"))
            cache = u.get("cache_hit", u.get("cached"))
    if comp is None and isinstance(rec.get("completion_tokens"), (int, float)):
        comp, prompt = rec["completion_tokens"], rec.get("prompt_tokens")
    if comp is None and isinstance(rec.get("tokens"), (int, float)) and not isinstance(rec.get("tokens"), bool):
        comp = rec["tokens"]  # Kaggle qwen solver: output tokens only
    if comp is None:
        return None
    return comp, prompt, cache


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None, None, None
    k = lambda q: xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
    return st.mean(xs), k(0.5), k(0.9)


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def read_recs(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


derived = []


def discover():
    found = []
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        dn[:] = [d for d in dn if d not in SKIP_DIRS or rel != "."] if rel == "." else [d for d in dn if d not in (".git", "__pycache__")]
        if rel == ".":
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if not f.endswith(".jsonl"):
                continue
            p = os.path.join(dp, f)
            try:
                with open(p) as fh:
                    first = json.loads(fh.readline())
            except Exception:
                continue
            if re.search(r"features|train_labels|continuations|nll_|/index\.jsonl|samp\.jsonl|metrics\.jsonl", os.path.relpath(p, ROOT)):
                derived.append(os.path.relpath(p, ROOT))
                continue
            if isinstance(first, dict) and "task_id" in first and ("pred" in first or "correct" in first) and (
                    "pair" in first or "attempt" in first):
                found.append(p)
    return sorted(found)


# ------------------------------------------------------------------ scoring
def eq(a, b):
    return isinstance(a, list) and a == b


def run_version(recs, rel, cfg):
    """-> (version, basis). Evidence order: explicit challenges path, workload, pair indices, recorded correct."""
    ch = (cfg or {}).get("challenges") or ""
    if "arc-agi-2-github" in ch:
        return "github", "config.challenges"
    if "arc-agi_evaluation_challenges" in ch:
        return "kaggle", "config.challenges"
    if any(r["task_id"] in WORK_IDS and r["task_id"] not in EVAL for r in recs):
        return "kaggle", "workload built from Kaggle files"
    idx = defaultdict(set)
    for r in recs:
        idx[r["task_id"]].add(r.get("pair", 0))
    for t in DIFF6:
        if t in idx and max(idx[t]) >= len(GH[t]):
            return "kaggle", "records have a pair index beyond GitHub's test count"
    # agreement of recorded `correct` with each version's key on the 6 differing tasks
    ag = {"github": [0, 0], "kaggle": [0, 0]}
    for r in recs:
        t = r["task_id"]
        if t in DIFF6 and r.get("pred") is not None and isinstance(r.get("correct"), bool):
            for v, key in (("github", GH), ("kaggle", KG)):
                p = r.get("pair", 0)
                if p < len(key[t]):
                    ag[v][1] += 1
                    ag[v][0] += int(eq(r["pred"], key[t][p]) == r["correct"])
    if ag["github"][1] and ag["github"][0] != ag["kaggle"][0]:
        v = "github" if ag["github"][0] > ag["kaggle"][0] else "kaggle"
        return v, "recorded `correct` agrees better with %s key on the 6 tasks" % v
    has6 = any(t in idx for t in DIFF6)
    if not rel.startswith("evals/results/"):
        return "kaggle", "default for pod/solver runs (no evidence either way)" + ("" if has6 else "; 6 tasks not covered")
    return "github", "harness default --challenges (no evidence either way)" + ("" if has6 else "; 6 tasks not covered")


def expand_multi(recs):
    """Multi-test solver records: one record per task attempt, multi_test = n, pred = list of n grids -> one record per pair.
    Usage stays on pair 0 only (as the harness writes it)."""
    out = []
    perpair = {(r["task_id"], r.get("attempt")) for r in recs if (r.get("pair") or 0) > 0}
    for r in recs:
        n = r.get("multi_test")
        pr = r.get("pred")
        if isinstance(n, int) and not isinstance(n, bool) and n >= 2 and isinstance(pr, list) and len(pr) == n and all(
                x is None or (isinstance(x, list) and x and isinstance(x[0], list)) for x in pr):
            for q in range(n):
                d = dict(r, pair=q, pred=pr[q])
                if q:
                    for k in ("usage", "tokens", "prompt_tokens"):
                        d.pop(k, None)
                out.append(d)
        elif isinstance(n, int) and not isinstance(n, bool) and n >= 2 and pr is None and (r["task_id"], r.get("attempt")) not in perpair \
                and (r.get("pair") or 0) == 0 and "multi_share" not in r and r.get("agent") is None and "trace_file" in r:  # solver record, no answer: n no-answer pairs
            for q in range(n):
                d = dict(r, pair=q, pred=None)
                if q:
                    for k in ("usage", "tokens", "prompt_tokens"):
                        d.pop(k, None)
                out.append(d)
        else:
            out.append(r)
    return out


def analyse(path):
    rel = os.path.relpath(path, ROOT)
    recs = expand_multi(read_recs(path))
    cfg = next((r["config"] for r in recs if isinstance(r.get("config"), dict)), {})
    ids = {r["task_id"] for r in recs}
    n_eval = len(ids & EVAL)
    n_work = len({i for i in ids if i in WORK_IDS})
    info = dict(path=rel, n_records=len(recs), n_task_ids=len(ids))
    if len(ids & EVAL) + n_work < 0.5 * len(ids):
        info["kind"] = "noneval"
        return info
    # workload
    wl = None
    if any(i in WORK_IDS and i not in EVAL for i in ids):
        c = Counter(w for i in ids for w in WORK_IDS.get(i, []))
        wl = c.most_common(1)[0][0]
        # prefer the workload that contains all ids
        for w in c:
            if all(i in WORK[w]["sol"] for i in ids):
                wl = w
                break
    info["workload"] = wl
    ver, basis = run_version(recs, rel, cfg)
    if wl:
        ver, basis = "kaggle", "workload %s built from Kaggle files" % wl
    key = GH if ver == "github" else KG
    sol_of = lambda t, p: (WORK[wl]["sol"][t][p] if wl and t in WORK[wl]["sol"] else
                           (key[t][p] if t in key and p < len(key[t]) else None))
    npairs = lambda t: (len(WORK[wl]["sol"][t]) if wl and t in WORK[wl]["sol"] else len(key.get(t, [])))
    # dedup (task, pair, attempt) last wins; attempts renumbered by rank
    att_vals = sorted({r.get("attempt", 0) if r.get("attempt") is not None else 0 for r in recs})
    amap = {a: i for i, a in enumerate(att_vals)}
    latest = {}
    for r in recs:
        a = amap[r.get("attempt", 0) if r.get("attempt") is not None else 0]
        latest[(r["task_id"], r.get("pair", 0) or 0, a)] = r
    keys = [k for k in latest if k[0] in (WORK[wl]["sol"] if wl else key)]
    if not keys:
        info["kind"] = "noneval"
        return info
    tasks = sorted({k[0] for k in keys})
    from_pred = from_rec = n_noans = n_err = n_over = 0
    corr = {}
    for (t, p, a) in keys:
        r = latest[(t, p, a)]
        if p >= npairs(t):
            n_over += 1
            continue
        pr = r.get("pred")
        if isinstance(pr, list):
            c = eq(pr, sol_of(t, p))
            from_pred += 1
        elif isinstance(r.get("correct"), bool) and pr is None and not (r.get("error")):
            c = r["correct"]
            from_rec += 1
            if not c:
                n_noans += 1
        else:
            c = False
            n_noans += 1
        if r.get("error"):
            n_err += 1
        corr[(t, p, a)] = bool(c)
    nat = len(att_vals)
    ta = defaultdict(set)
    for (t, p, a) in keys:
        ta[t].add(a)

    def metrics(taskset):
        taskset = [t for t in tasks if t in taskset]
        if not taskset:
            return dict(n=0, p1=None, p1a0=None, p2=None, a1_frac=None)
        p1 = []; p1a0 = []; p2 = []; a1 = 0
        for t in taskset:
            n = npairs(t)
            sc = {a: sum(corr.get((t, p, a), False) for p in range(n)) / n for a in ta[t]}
            p1.append(st.mean(sc.values()))
            p1a0.append(sc.get(0, 0.0))
            has1 = 1 in ta[t]
            a1 += has1
            p2.append(sum(corr.get((t, p, 0), False) or corr.get((t, p, 1), False) for p in range(n)) / n)
        return dict(n=len(taskset), p1=100 * st.mean(p1), p1a0=100 * st.mean(p1a0),
                    p2=100 * st.mean(p2) if a1 else None, a1_frac=a1 / len(taskset))

    allm = metrics(set(tasks))
    ident = metrics(set(IDENT) & set(tasks))
    split = {}
    if wl:
        idm = WORK[wl]["idmap"]
        def _is_orig(t):
            v = idm.get(t)
            if isinstance(v, dict):
                return v.get("kind", "orig") == "orig"
            return t in EVAL
        orig = {t for t in tasks if _is_orig(t)}
        split["orig"] = metrics(orig)
        split["copy"] = metrics(set(tasks) - orig)
        split["orig114"] = metrics(orig & set(IDENT))
        tot_tasks = len(WORK[wl]["sol"])
    else:
        tot_tasks = 120
    full = len(tasks) == tot_tasks and all(
        0 in ta[t] and 1 in ta[t] and all((t, p, 0) in corr and (t, p, 1) in corr for p in range(npairs(t))) for t in tasks)
    # tokens: per record with usage; multi-test runs put usage on pair 0 only, so sum over pairs per (task, attempt)
    multi = bool(cfg.get("multi_test")) or any((r.get("multi_test") or 0) >= 2 for r in recs)
    tk = []
    if multi:
        grp = defaultdict(lambda: [None, None, None])
        for (t, p, a), r in latest.items():
            x = tokens(r)
            if x:
                g = grp[(t, a)]
                for i in range(3):
                    if x[i] is not None:
                        g[i] = (g[i] or 0) + x[i]
        tk = list(grp.values())
    else:
        tk = [tokens(r) for r in latest.values()]
        tk = [x for x in tk if x]
    comp = [x[0] for x in tk]
    tot = [(x[0] + x[1]) if x[1] is not None else None for x in tk]
    cache = [x[2] / x[1] for x in tk if x[2] is not None and x[1]]
    # per-attempt n/a note
    pr_total_na = sum(1 for x in tk if x[1] is None)
    # extras
    models = Counter(r.get("model") or cfg.get("model") for r in latest.values())
    served = Counter(r.get("served_model") for r in latest.values() if r.get("served_model"))
    base_url = next((r.get("base_url") for r in latest.values() if r.get("base_url")), None) or cfg.get("base_url")
    hv = Counter(r.get("harness_version") or cfg.get("harness_version") for r in latest.values())
    info.update(kind="eval", version=ver, version_basis=basis, tasks=len(tasks), tot_tasks=tot_tasks, full=full,
                attempts_idx=nat, att_per_task=st.mean(len(ta[t]) for t in tasks), n_records_dedup=len(latest),
                n_pairs=len({(t, p) for (t, p, a) in keys}), n_noans=n_noans, n_err=n_err, n_over=n_over,
                scored_from_pred=from_pred, scored_from_recorded=from_rec, multi=multi,
                m_all=allm, m_ident=ident, split=split,
                comp=stats(comp), tot=stats(tot), cache=st.mean(cache) if cache else None, n_tok=len(tk),
                tok_prompt_na=pr_total_na, model=models.most_common(1)[0][0] if models else None,
                served=served.most_common(1)[0][0] if served else None, base_url=base_url,
                harness_version=hv.most_common(1)[0][0] if hv else None, cfg=cfg, ids_in_eval=n_eval)
    return info


# ------------------------------------------------------------------ stack / harness description
KEYFLAGS = ["prompt", "mode", "effort", "effort_header", "grid_format", "image_cell", "multi_test", "multi_budget",
            "context_limit", "max_tokens", "max_total_output", "budget_soft", "max_turns",
            "answer_tool", "final_mode", "train_check", "helpers", "task_summary", "answer_mode", "tier", "temperature", "openrouter_provider",
            "show_tool", "alt_answer", "prior_mode", "turn_think_cap", "feedback", "image_in", "image_layout"]


def flags(cfg):
    out = []
    for k in KEYFLAGS:
        v = cfg.get(k)
        if v not in (None, False, "", 0) or k in ("effort",) and v is not None:
            out.append("%s=%s" % (k, v))
    return " ".join(out)


def harness_file(rel, info):
    name = os.path.basename(rel)
    if "dsv41_logs/attempts" in rel or "/arc_logs/" in rel or name.startswith("attempts"):
        if "/arc_logs/" in rel:
            return "kaggle/build_notebook.py"
        return "kaggle/dsv41_tools/solver_dsv41.py"
    if name.startswith("qwen_tools") or name.startswith("e019") or "qwen-tools" in (info.get("harness_version") or ""):
        return "evals/run_qwen_tools.py"
    if name.startswith("qwen_nvfp4") or (info.get("harness_version") or "").startswith("qwen-gen"):
        return "evals/run_qwen_generate.py"
    if name.startswith("tools_"):
        return "evals/run_deepseek_tools.py"
    if name.startswith(("arcprize_", "e044", "e043nt", "e039", "e033", "e024")):
        return "evals/run_deepseek_arcprize.py"
    if name.startswith("arc2_"):
        return "evals/run_deepseek_arc2.py"
    return "evals/run_deepseek_tools.py"


def stack_of(rel, info):
    m = info.get("model") or ""
    bu = info.get("base_url") or ""
    name = os.path.basename(rel)
    cfg = info.get("cfg") or {}
    if "dsv41_logs/attempts" in rel or "arc_logs" in rel:
        segs = [x for x in rel.split("/")[1:] if x not in ("runs", "results", "out", "dsv41_logs", "data", "working", "raw") and not x.startswith("attempts")]
        s = "pod/box run: solver_dsv41 + vLLM fork (dir %s)" % "/".join(segs[:3])
        if "arc_logs" in rel:
            s = "Kaggle notebook run (Qwen solver, vLLM on Kaggle GPUs)"
        return s
    if cfg.get("openrouter_provider") or "openrouter" in name or "deepinfra" in name or "or_" in name or "inferencenet" in name:
        return "OpenRouter/DeepInfra API (%s)" % (cfg.get("openrouter_provider") or "provider per file name")
    if bu and ("127.0.0.1" in bu or "localhost" in bu or re.match(r"https?://\d", bu)):
        return "self-hosted OpenAI-compatible server (%s), model %s" % (re.sub(r"//[^:/]+", "//<host>", bu), m)
    if m.startswith("deepseek"):
        return "DeepSeek API"
    if "qwen" in m.lower() or name.startswith("qwen"):
        return "Qwen served via SGLang/vLLM (base_url not recorded)" if not bu else "Qwen self-hosted"
    return "n/a"


def pod_meta(path):
    d = os.path.dirname(path)
    for _ in range(5):
        rj = os.path.join(d, "replica.json")
        if os.path.exists(rj):
            try:
                j = json.load(open(rj))
                env = j.get("env", {})
                keep = {k: v for k, v in env.items() if k in ("ARC_KIT", "ARC_GPU_UTIL", "ARC_MAX_NUM_SEQS", "ARC_MULTI_TEST",
                                                              "ARC_MULTI_BUDGET", "ARC_CONTEXT_LIMIT", "ARC_KV_ADMIT_MAX")}
                gpu = next((x.split(",", 1)[1].strip()[:60] for x in j.get("preflight", []) if isinstance(x, str) and x.startswith("gpu:")), "")
                return dict(date=(j.get("launched_utc") or "")[:10], detail="replica.json name=%s workload=%s gpu=%s %s" % (
                    j.get("name"), j.get("workload"), gpu or "n/a", " ".join("%s=%s" % kv for kv in keep.items())))
            except Exception:
                pass
        d = os.path.dirname(d)
    return {}


# ------------------------------------------------------------------ report
DEFS = """## Definitions

- **Task weighting.** Each task counts once. A task's score for one attempt is the mean over its test pairs of
  "predicted grid == answer grid" (Kaggle-style). Scores are percentages over the tasks the run covers
  (tasks with at least one record); a pair with no record for an attempt counts as wrong.
- **pass@1 (all attempts)** = mean over tasks of the mean over that task's attempts of the per-attempt task score.
  **pass@1 (a0)** = attempt 0 only. **pass@2** = per pair, correct if attempt 0 or attempt 1 is right, then the task mean.
  Runs with more than 2 attempts use attempts {0,1} for pass@2 (column `att/task` gives the mean number of attempts per task);
  their pass@1 averages all attempts. Attempts are renumbered by rank within a file (an `attempt` that starts at 1 becomes 0).
  Runs with one attempt per task show pass@2 = n/a. `a1%` = share of covered tasks that have an attempt 1; pass@2 of a run with a
  low `a1%` is a lower bound.
- **Resume duplicates.** Several records for the same (task, pair, attempt): the last one wins.
- **Scoring.** The recorded `pred` grid is re-scored against the answer key of the run's task version. If a record has no `pred`
  (some Qwen harness files keep only `correct`), the recorded `correct` flag is used (`scored_from_recorded` in the CSV).
  A missing/None pred is wrong ("no-answer"); `error` records are counted separately.
- **Task versions.** Kaggle = `data/arc-agi_evaluation_{challenges,solutions}.json` (172 test pairs); GitHub =
  `data/arc-agi-2-github/data/evaluation/*.json` (167 pairs), the harness default `--challenges`. They differ on 6 tasks (@@DIFF6@@);
  114 tasks are identical. Version is detected, in order, from the config `challenges` path, a replica workload (always Kaggle-built),
  a pair index beyond GitHub's test count on a differing task, and the agreement of the recorded `correct` with each key on the 6
  tasks; otherwise a default (GitHub for `evals/` harness files, Kaggle for pod/solver files) is used and the basis column says so.
  Each run is scored ONLY against its own version. **p1/p2 (114)** repeat the scores on the 114 identical tasks, so runs compare
  across versions.
- **Full vs partial.** Full = all 120 tasks (all 240 for replica workloads), every test pair of the run's version has attempts 0 and 1.
  Everything else is partial (task count given). Smoke runs (<= 10 tasks) and non-eval sets are appendices.
- **Token fields** (per attempt record; multi-test runs: summed over the pairs of one task attempt, since usage is on pair 0 only):
  - `comp` = generated/completion tokens. API/OpenRouter no-tool records: `usage.completion_tokens`. Tool harness: `usage.completion`
    (summed over turns). Qwen solver without usage: top-level `tokens`.
  - `total` = prompt tokens summed over all turns + completion. API: `prompt_tokens + completion_tokens`; tool harness:
    `usage.prompt + usage.completion` (`prompt` already sums every turn); Qwen harness: `prompt_total + completion`.
    n/a when the record has no summed prompt count (old Qwen `max_prompt`-only usage, Kaggle solver logs).
  - `cache` = mean prompt-cache hit share (`cache_hit/prompt`, or `prompt_cache_hit_tokens`, `cached_tokens`); n/a when unrecorded.
  - Statistics are mean / p50 / p90 (nearest rank) over attempts that have the field.
""".replace("@@DIFF6@@", ", ".join(DIFF6))


def f1(x, d=1):
    return "n/a" if x is None else ("%.*f" % (d, x))


def fk(x):
    return "n/a" if x is None else "%.1fk" % (x / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true", help="also score serving-lab replica attempts.jsonl from HF")
    args = ap.parse_args()
    files = discover()
    extra_hf = []
    if args.hf:
        extra_hf = fetch_hf()
    seen = {}
    runs = []
    noneval = []
    for path in files + extra_hf:
        h = sha_of(path)
        if h in seen:
            seen[h]["aliases"].append(os.path.relpath(path, ROOT) if path.startswith(ROOT) else path)
            continue
        info = analyse(path)
        info["sha256"] = h
        info["aliases"] = []
        if path.startswith(HF_CACHE):
            info["path"] = "hf:arc-agi2-vllm-serving-lab/" + os.path.relpath(path, HF_CACHE)
        seen[h] = info
        (noneval if info["kind"] == "noneval" else runs).append(info)
    rows = []
    for info in runs:
        rel = info["path"]
        eid, ename, mtsv, started = exp_of(rel)
        if eid == "n/a":
            hr0 = HF_RUNS.get(rel)
            eid = (hr0["experiment"] if hr0 else None) or readme_exp(os.path.basename(rel)[:-6]) or "n/a"
            r0 = TSV_BY_ID.get(eid) or next((x for x in TSV if x["id"].startswith(eid)), None)
            if r0:
                ename, mtsv, started = r0["name"], r0.get("model", ""), r0.get("started", "")
        pm = pod_meta(P(rel)) if not rel.startswith("hf:") else {}
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(P(rel))).strftime("%Y-%m-%d") if not rel.startswith("hf:") else "n/a"
        hr = HF_RUNS.get(rel)
        if rel.startswith("evals/results/"):
            run_id = (hr["run_id"] if hr else None) or os.path.basename(rel)[:-6]
        else:
            segs = [x for x in rel.split("/") if x not in ("out", "dsv41_logs", "data", "working", "raw", "arc_logs", "runs", "results", "experiments")]
            segs[-1] = segs[-1].replace(".jsonl", "")
            if segs[-1].startswith("attempts") and len(segs) > 1:
                segs = segs[:-1]
            run_id = "/".join(segs[0:1] + segs[-2:]) if len(segs) > 2 else "/".join(segs)
        E072COPY = "experiments/E072_harness_loop/data_in/v12_m12_attempts.jsonl"
        v12rep = rel == E072COPY or any(a.endswith("replicas/manh_v12/attempts.jsonl") for a in info["aliases"])
        if v12rep:  # E072's file is a copy of the E062 phase_h/manh_v12 replica: report it once, as the E062 run
            run_id = "Manh V12 replica (E062)"
            eid, ename = "E062", ename or "Manh vLLM fork"
        hf = hf_url(rel, info["aliases"] + ["hf:arc-agi2-vllm-serving-lab/replicas/manh_v12"] * v12rep) or "local only (nothing to upload: no trace folder or records file on disk): " + rel
        hfile = harness_file(rel, info)
        m = info["m_all"]; mi = info["m_ident"]
        wl = info["workload"]
        sm = info["split"]
        cfg = info["cfg"]
        row = dict(
            section="", run_id=run_id, path=rel, aliases=";".join(info["aliases"]), date_file_mtime=mt,
            date_launched=pm.get("date", ""), experiment=eid, experiment_name=ename, model_record=info["model"] or "",
            model_experiment_tsv=mtsv, stack=stack_of(rel, info), stack_detail=pm.get("detail", ""),
            harness_file=hfile, harness_link=REPO + hfile, harness_version=info["harness_version"] or "n/a",
            key_flags=flags(cfg) if cfg else "n/a (records carry no config)",
            run_type="",
            harness_similarity=harness_similarity(rel, run_id, flags(info["cfg"]) if info.get("cfg") else ""),
            hf_traces=hf, version=info["version"], version_basis=info["version_basis"], workload=wl or "",
            full_partial="full" if info["full"] else "partial", tasks=info["tasks"], tasks_total=info["tot_tasks"],
            attempts_per_task=round(info["att_per_task"], 2), attempt_ranks=info["attempts_idx"], pairs=info["n_pairs"],
            records=info["n_records_dedup"], no_answer=info["n_noans"], errors=info["n_err"],
            pairs_beyond_key=info["n_over"], scored_from_pred=info["scored_from_pred"],
            scored_from_recorded=info["scored_from_recorded"], multi_test=info["multi"],
            pass1_all=m["p1"], pass1_a0=m["p1a0"], pass2=m["p2"], a1_task_frac=m["a1_frac"],
            n114=mi["n"], pass1_all_114=mi["p1"], pass1_a0_114=mi["p1a0"], pass2_114=mi["p2"],
            comp_mean=info["comp"][0], comp_p50=info["comp"][1], comp_p90=info["comp"][2],
            total_mean=info["tot"][0], total_p50=info["tot"][1], total_p90=info["tot"][2],
            cache_hit_share=info["cache"], tokens_records=info["n_tok"], sha256=info["sha256"][:16])
        for k in ("orig", "copy", "orig114"):
            s = sm.get(k)
            row["wl_%s_tasks" % k] = s["n"] if s else ""
            row["wl_%s_pass1_all" % k] = s["p1"] if s else ""
            row["wl_%s_pass1_a0" % k] = s["p1a0"] if s else ""
            row["wl_%s_pass2" % k] = s["p2"] if s else ""
        if info["tasks"] <= 10:
            sec = "smoke"
        elif wl:
            sec = "replica"
        else:
            sec = ("full_" if info["full"] else "partial_") + info["version"]
        row["section"] = sec
        if v12rep:
            row["hf_traces"] = LAB + "manh_v12"
            row["path"] = "hf:arc-agi2-vllm-serving-lab/replicas/manh_v12/attempts.jsonl (E062 phase_h/manh_v12; E072 data_in copy skipped)"
        row["run_type"] = ("smoke" if sec == "smoke" else "Kaggle-replica" if (sec == "replica" or "E072_harness_loop/runs" in rel or row["run_id"] == "Manh V12 replica (E062)") else "benchmark")
        rows.append(row)
    for d in derived:
        noneval.append(dict(path=d, n_records=sum(1 for _ in open(P(d))), n_task_ids=0, note="derived analysis file, not a run"))
    write_csv(rows, noneval)
    write_md(rows, noneval)
    print("runs:", Counter(r["section"] for r in rows), "noneval:", len(noneval))


def fetch_hf():
    try:
        from huggingface_hub import hf_hub_download
        key = re.search(r"^huggingface_readonly_api_key:\s*(\S+)", open(P("SECRETS.yaml")).read(), re.M).group(1).strip("\"'")
    except Exception as e:
        print("hf fetch skipped:", type(e).__name__)
        return []
    out = []
    for run in ("manh_v12", "v12_fast"):
        try:
            p = hf_hub_download("livobereshka/arc-agi2-vllm-serving-lab", "replicas/%s/attempts.jsonl" % run, repo_type="dataset",
                                token=key, local_dir=HF_CACHE)
            out.append(p)
        except Exception as e:
            print("hf", run, type(e).__name__)
    return out


def write_csv(rows, noneval):
    cols = list(rows[0].keys())
    with open(P("reports/eval_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    with open(P("reports/eval_nonEval_files.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "records", "task_ids"])
        for i in noneval:
            w.writerow([i["path"], i["n_records"], i["n_task_ids"]])


def srt(rows):
    return sorted(rows, key=lambda r: (-(r["pass2"] if r["pass2"] is not None else -1), -(r["pass1_all"] or 0)))


def link(r):
    return "[%s](%s)" % (r["harness_file"].split("/")[-1], r["harness_link"])


def star(r):
    return ("★ " if r["harness_similarity"] != "other" else "") + r["run_id"]


HOWTO = """## How to study the reasoning traces

Every run below links to its traces in the private HF dataset `livobereshka/arc-agi2-eval-traces-private`
(or, for pod replicas, to `livobereshka/arc-agi2-vllm-serving-lab` `replicas/<run>`).

- **Access**: membership of the `livobereshka` HF org plus your own read token (`huggingface-cli login`, or `HF_TOKEN`). Never share tokens.
- **One run** (the directory name is the `traces` link; `data_tools/show_trace.py` resolves it from the run_id):

  ```bash
  huggingface-cli download livobereshka/arc-agi2-eval-traces-private --repo-type dataset --include "runs/<run_dir>/*" --local-dir traces
  ```

  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("livobereshka/arc-agi2-eval-traces-private", repo_type="dataset",
                    allow_patterns=["runs/<run_dir>/*"], local_dir="traces")
  ```

  One task of a run: use `--include "runs/<run_dir>/*<task_id>_p0_a0*"` (the records file `runs/<run_dir>/<run>.jsonl` is small; fetch it too).
  `runs/pub_<run_id>/traces.parquet` holds the runs that used to be in the public `arc-agi2-eval-traces` (same columns:
  `task_id, pair, attempt, correct, predicted_output, messages, trace_text, ...`). `index.csv` maps run_id to directory, files and bytes.
- **Files**: `<task>_p<pair>_a<attempt>[_<hash>].reasoning.txt.gz` is the model's thinking for a no-tool call (text);
  `<task>_p<pair>_a<attempt>_<hash>.messages.json.gz` is `{"messages": [...], "events": [...]}` for a tool run (system/user/assistant
  turns with `reasoning_content`, `tool_calls`, tool results, and per-turn usage events). The records file `<run>.jsonl` has one JSON
  object per attempt: `task_id, pair, attempt, pred` (final grid), `content` (final text), `usage` (prompt, cache_hit, completion),
  `turns, tool_calls, tool_errors, stop_reason, train_pass, train_frac, seconds, config, correct` (as recorded by the harness at the time).
- **Correct or wrong**: compare `pred` with the right answer key for the run's `version` column in `eval_runs.csv`: GitHub key
  `data/arc-agi-2-github/data/evaluation/<task>.json` (`test[pair].output`) for G, `data/arc-agi_evaluation_solutions.json[task][pair]` for K.
  The recorded `correct` can be stale for the 6 tasks whose answers differ between the two versions.
- **Viewer**: `.venv/bin/python data_tools/show_trace.py <run_id> --task <task_id> --attempt 0` prints reasoning, tool calls and results and
  the final answer marked CORRECT/WRONG against the right key; `--wrong-only` lists the failed attempts of a run; `--download` fetches the
  files from the private repo first.

"""


def table(rows, replica=False):
    if replica:
        h = ("| run | exp | date | model / stack | harness | traces | wl tasks | att/task | a1% | p1 all (240) | p1 a0 | p2 | "
             "orig p1/a0/p2 | copies p1/a0/p2 | orig-114 p1/p2 | comp mean/p50/p90 | total mean/p50/p90 | noans/err |\n"
             "|" + "---|" * 18 + "\n")
    else:
        h = ("| run | exp | date | model / stack | harness + flags | traces | tasks | att/task | a1% | p1 all | p1 a0 | p2 | "
             "p1/p2 (114) | comp mean/p50/p90 | total mean/p50/p90 | noans/err | ver |\n"
             "|" + "---|" * 17 + "\n")
    out = h
    for r in srt(rows):
        tk = lambda a, b, c: "%s/%s/%s" % (fk(r[a]), fk(r[b]), fk(r[c]))
        a1 = "n/a" if r["a1_task_frac"] in (None, "") else "%d" % round(100 * r["a1_task_frac"])
        ms = "%s %s" % (r["model_record"] or r["model_experiment_tsv"] or "n/a", r["stack"])
        hs = "%s `%s` (%s)" % (link(r), r["key_flags"][:140], r["harness_version"])
        tr = "[traces](%s)" % r["hf_traces"] if r["hf_traces"].startswith("http") else "local only"
        d = r["date_launched"] or r["date_file_mtime"]
        if replica:
            tri = lambda p: "%s/%s/%s" % (f1(r["wl_%s_pass1_all" % p] or None), f1(r["wl_%s_pass1_a0" % p] or None), f1(r["wl_%s_pass2" % p] or None))
            out += "| %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s/%s |\n" % (
                star(r), r["experiment"], d, ms[:120], link(r), tr, r["tasks"], r["tasks_total"], r["attempts_per_task"], a1,
                f1(r["pass1_all"]), f1(r["pass1_a0"]), f1(r["pass2"]), tri("orig"), tri("copy"),
                f1(r["wl_orig114_pass1_all"] or None), f1(r["wl_orig114_pass2"] or None),
                tk("comp_mean", "comp_p50", "comp_p90"), tk("total_mean", "total_p50", "total_p90"), r["no_answer"], r["errors"])
        else:
            out += "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s/%s | %s |\n" % (
                star(r), r["experiment"], d, ms[:120], hs, tr, r["tasks"] if r["full_partial"] == "partial" else "120 (full)",
                r["attempts_per_task"], a1, f1(r["pass1_all"]), f1(r["pass1_a0"]), f1(r["pass2"]), f1(r["pass1_all_114"]),
                f1(r["pass2_114"]), tk("comp_mean", "comp_p50", "comp_p90"), tk("total_mean", "total_p50", "total_p90"),
                r["no_answer"], r["errors"], r["version"][0].upper() + ("?" if "default" in r["version_basis"] or "no evidence" in r["version_basis"] else ""))
    return out


def v12_section(rows):
    sel = [r for r in rows if r["harness_similarity"] != "other" and r["section"] != "smoke"]
    sel.sort(key=lambda r: (["exact V12", "V12 + images", "B32-family"].index(r["harness_similarity"]), -(r["pass2"] if r["pass2"] is not None else -1)))
    out = ("## Closest to the current best (V12) harness\n\n"
           "V12 = Manh's run of B32's pinned harness 882c817f + auto-repair + multi-test (budget 0.5) + token-exact + context 163840 + "
           "lean2sub + effort-header minimal, on the old tool E128. Runs marked ★ in the tables below. `harness_similarity` (CSV) is "
           "rule-based (path names and recorded flags; see Definitions): **exact V12** = the V12 replicas (manh_v12, v12_fast, SFT 025ep, "
           "E072 V12 runs) and records whose flags match V12 (E066 b32_fp8/fp4, E069); **V12 + images** = the same with images "
           "(`grid_format=text+image`, v12img runs); **B32-family** = B32 harness replicas, E052 and E055/E057/E059. Smoke runs are left out here. "
           "Click the traces link for the reasoning.\n\n"
           "| run | similarity | section | tasks | p1 all | p1 a0 | p2 | traces |\n|---|---|---|---|---|---|---|---|\n")
    for r in sel:
        tr = "[traces](%s)" % r["hf_traces"] if r["hf_traces"].startswith("http") else "local only"
        out += "| %s | %s | %s | %s | %s | %s | %s | %s |\n" % (r["run_id"], r["harness_similarity"], r["section"], r["tasks"], f1(r["pass1_all"]), f1(r["pass1_a0"]), f1(r["pass2"]), tr)
    return out + "\n"


def write_md(rows, noneval):
    by = defaultdict(list)
    for r in rows:
        by[r["section"]].append(r)
    L = ["# ARC-AGI-2 evaluation runs: one report\n",
         "Generated by `data_tools/eval_report.py` on %s from every result file in the workspace. Per-run detail (every field) is in "
         "`reports/eval_runs.csv`; non-eval files are listed in `reports/eval_nonEval_files.csv`. All scores are percentages. "
         "Rows within a table are sorted by pass@2 (n/a last). In the `ver` column K = Kaggle, G = GitHub, `?` = assumed from a default "
         "(basis in the CSV).\n" % datetime.date.today().isoformat(),
         HOWTO,
         v12_section(rows),
         DEFS,
         "## Counts\n"]
    names = [("full_github", "1. Full runs, GitHub task version (run type: benchmark)"), ("full_kaggle", "2. Full runs, Kaggle task version (run type: benchmark)"),
             ("partial_github", "3. Partial runs, GitHub task version (run type: benchmark)"), ("partial_kaggle", "4. Partial runs, Kaggle task version (run type: benchmark; E072 loop jobs are Kaggle-replica, see run_type)"),
             ("replica", "5. Replica workloads (run type: Kaggle-replica; pod replicas on 240-task workloads: 120 Kaggle originals + copies)"),
             ("smoke", "A. Appendix: smoke runs (run type: smoke, <= 10 tasks)")]
    L.append("| section | runs |\n|---|---|\n" + "".join("| %s | %d |\n" % (n, len(by[k])) for k, n in names) +
             "| B. Appendix: non-eval record files (training pools, NVARC, probes) | %d |\n\n" % len(noneval))
    for k, n in names:
        L.append("## %s\n" % n)
        if k == "replica":
            L.append("Workload `evalx2t_s2026` = the 120 Kaggle originals + 120 rotated/flipped, colour-permuted copies; other workloads "
                     "(`eval_x2`: identical copies; `evalt2_s2027`) are listed with the same columns (`workload` column in the CSV). "
                     "p1/a0/p2 are given for the whole workload, originals only, copies only and originals on the 114 identical tasks. "
                     "Most replicas finish about one attempt per task in the time budget, so pass@2 is n/a or a lower bound; read `a1%`.\n")
        if k == "smoke":
            L.append("These cover 10 or fewer tasks; their scores are noise-level and shown only for completeness.\n")
        L.append(table(by[k], replica=(k == "replica")) if by[k] else "_none_\n")
        L.append("\n")
    L.append("## B. Appendix: non-eval sets (excluded from the tables)\n\nFiles whose task ids are mostly not ARC-AGI-2 evaluation tasks "
             "(training pools, NVARC puzzles, ...). One line each in `reports/eval_nonEval_files.csv` (%d files).\n\n" % len(noneval))
    byd = Counter(("evals/results" if i["path"].startswith("evals/results") else "/".join(i["path"].split("/")[:2])) for i in noneval)
    L.append("".join("- `%s`: %d files, %d records\n" % (d, c, sum(i["n_records"] for i in noneval if i["path"].startswith(d))) for d, c in byd.most_common()))
    L.append(CAVEATS)
    open(P("reports/eval_report.md"), "w").write("\n".join(L))


CAVEATS = """
## Caveats

- **One run = one result file.** A run split over several files (for example `_s2`, `_s3` seed shards, or `_rest` completions) appears
  as separate rows; files with identical content are merged (aliases in the CSV). Experiments that merged shards in their README
  may report different (combined) numbers.
- **Scores here are recomputed from `pred`**, so a run's number can differ from its README: READMEs often report pairs/attempt-weighted scores,
  a different task version, the Kaggle-clock correction (replicas) or a pass@2 over a specific attempt subset. Where a record has only
  `correct` (some Qwen files), the recorded flag was used and is against the GitHub key, whatever the run's version.
- **Version detection is evidence-based but not always conclusive**: when the six differing tasks are not in the run, or no record
  disambiguates, a default is used (`?` in `ver`, `version_basis` in the CSV). The p1/p2 (114) columns are immune to this.
- **Partial runs** are scored over the tasks they cover; their numbers are not comparable with full runs. The easy tasks of a
  "smallest-first" ordering (replicas) bias partial replica scores upward.
- **Replicas on pods run faster than Kaggle** (see experiments/E055_kaggle_replica/README.md); the scores here are raw, not Kaggle-clock
  corrected.
- **Harness links** point at the file on `main` (the mirror is a snapshot, not pinned per run). The records carry only `harness_version`
  (and, for tool runs, a `config`); pinned code snapshots exist on HF for a few runs (for example E030 full_api `2d1a627a`
  and E043 e128 `2345e933` in `arc-agi2-tool-traces-dsv41-full-vs-e128`, solver/harness hashes in replica READMEs).
  Pod-replica `attempts.jsonl` records carry no config; flags come from the replica's `replica.json` where one exists.
- **Dates** are the result file's modification time unless `date_launched` (replica.json) exists; the experiment start date is in the TSV.
- **HF traces**: every run with records or trace files has a deep link into the private `arc-agi2-eval-traces-private` (`runs/<run_id>`;
  `runs/pub_<run_id>` for runs copied from the former public dataset, index in `reports/hf_private_index.csv`) or into the serving-lab
  `replicas/<run>` / the candb32 replica repo. "local only" would mean nothing was uploaded.
- **run_type**: **Kaggle-replica** = the exact Kaggle notebook run end-to-end on a 4xL4 pod in scored-rerun mode with Kaggle's deadlines
  (E055/E062/E072 jobs; E072 is the harness-improvement loop, pod job queues on the 120 Kaggle-version tasks, b1_noise_v12_pod2 in progress);
  **benchmark** = a harness eval run untimed through `evals/run_*.py` against the API or a self-hosted server; **smoke** = a run on <= 10 tasks.
  Sections 1-4 are benchmark runs (a few E072/E052 pod rows carry their own `run_type` in the CSV), section 5 is Kaggle-replica, appendix A is smoke.
- **harness_similarity**: rule-based label for closeness to V12 (see the V12 section): "exact V12", "V12 + images", "B32-family", "other". Serving-lab `replicas/<run>` entries without `attempts.jsonl` (traces and submission only) could not
  be scored from the files here.
- **Server-cut attempts** (pod records with `usage.completion_reported_cut`, completion = 1e9 sentinel) have no usable token count and are left out of the token statistics (`tokens_records` < records).
- **Tokens**: totals count prompt tokens as billed per turn (so cached prefixes are counted every turn). `n/a` where the harness did not log a summed prompt count.
"""

if __name__ == "__main__":
    main()
