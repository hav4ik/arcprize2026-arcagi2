"""Read-only vLLM live reporting, using accepted generation counters, not SSE chunks.

The model-scoped /metrics counter is server-wide: another client using the same
served model contributes to its TPS. Wall-time rates include prefill/stalls/idle
time in the measured interval. They are not pure-kernel or per-request speeds.
"""

from datetime import datetime, timezone
import json
import math
import re
import time

from dsv4_llama import LiveView, clock, generated


GENERATION = "vllm:generation_tokens_total"
GAUGES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
          "vllm:kv_cache_usage_perc": "kv_usage"}
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+.*)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def metric_values(text, model):
    """Ignore histograms, created timestamps, draft counters and other models."""
    samples = {name: {} for name in (GENERATION, *GAUGES)}
    for line in (text or "").splitlines():
        match = SAMPLE.fullmatch(line.strip())
        if match is None or match[1] not in samples:
            continue
        try:
            labels = {key: json.loads('"' + value + '"') for key, value in LABEL.findall(match[2] or "")}
            value = float(match[3])
        except (ValueError, TypeError):
            continue
        if labels.get("model_name", model) != model or not math.isfinite(value) or value < 0:
            continue
        # Label order in the exposition may change. A counter is per engine,
        # not per TP worker; do not multiply by the tensor-parallel degree.
        samples[match[1]][tuple(sorted(labels.items()))] = value
    values = {name: (sum(samples[key].values()) if samples[key] else None) for key, name in GAUGES.items()}
    kv = samples["vllm:kv_cache_usage_perc"]
    values["kv_usage"] = sum(kv.values()) / len(kv) if kv else None
    return samples[GENERATION], values


def number(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


class VllmLiveView(LiveView):
    """Keep the existing result/accuracy view, with native vLLM TPS and KV data."""

    def __init__(self, *args, model, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.previous = None
        self.baseline_time = self.sample_time = None
        self.total = 0.0
        self.resets = 0

    def measure(self, metrics, now=None):
        now = time.monotonic() if now is None else now
        counters, values = metric_values(metrics, self.model)
        values.update(tokens_per_s=None, tokens_per_s_avg=None, decoded_tokens=None,
                      window_s=None, rate_status="unavailable", counter_resets=self.resets)
        if not counters:
            # Do not report zero TPS for a failed scrape. A recovered scrape
            # measures the whole gap since the previous valid sample.
            return values
        reset = self.previous is not None and (
            counters.keys() != self.previous.keys()
            or any(value < self.previous[key] for key, value in counters.items()))
        if self.previous is None or reset:
            self.total = 0.0
            self.baseline_time = self.sample_time = now
            self.previous = counters
            self.resets += bool(reset)
            values.update(decoded_tokens=0, rate_status="counter-reset" if reset else "baseline",
                          counter_resets=self.resets)
            return values
        window = now - self.sample_time
        if window <= 0:
            return values
        delta = sum(value - self.previous[key] for key, value in counters.items())
        self.total += delta
        values.update(tokens_per_s=delta / window, tokens_per_s_avg=self.total / (now - self.baseline_time),
                      decoded_tokens=int(self.total), window_s=window, rate_status="ok")
        self.sample_time, self.previous = now, counters
        return values

    def result_line(self, row, results):
        # vLLM's reasoning parser does not supply llama.cpp's think_closed flag.
        verdict = ("correct" if row.get("correct") else "wrong" if row.get("valid") else "no-grid") \
            if self.labelled else ("grid" if row.get("valid") else "no-grid")
        line = (f"RESULT {row['task']}_{row['test_index']}: {verdict} finish={row.get('finish_reason')} "
                f"gen={generated(row)} t/s={number(row.get('decode_tokens_per_s_estimate'))} "
                f"{clock(row.get('elapsed_s') or 0)} | done {len(results)}/{self.planned}")
        if self.labelled:
            correct, score = self.accuracy(results)
            line += f" correct {correct} score {score:.2f}/{self.routed_tasks}"
        if row.get("error") is not None:
            line += f" error={row['error'][:120]}"
        return line

    async def tick(self, results, progress, queued, *, metrics=None):
        now = time.monotonic()
        measured = self.measure(metrics, now)
        correct, score = self.accuracy(results)
        finishes = [row.get("finish_reason") for row in results]
        snap = dict(elapsed_s=now - self.started, queued=queued, done=len(results), planned=self.planned,
                    client_active_outputs=len(progress), valid=sum(bool(row.get("valid")) for row in results),
                    correct=correct if self.labelled else None, routed_score=score if self.labelled else None,
                    natural_stop=finishes.count("stop"), length=finishes.count("length"),
                    loop=finishes.count("loop_detected"), errors=sum(row.get("error") is not None for row in results),
                    tps_source=GENERATION, tps_scope="served-model/server-wide", **measured)
        accuracy = (f"correct {correct}/{len(results)} score {score:.2f}/{self.routed_tasks}" if self.labelled
                    else f"valid {snap['valid']}/{len(results)}")
        kv = None if snap["kv_usage"] is None else 100 * snap["kv_usage"]
        print(f"LIVE {clock(snap['elapsed_s'])} | run {number(snap['running'], 0)} wait {number(snap['waiting'], 0)}"
              f" queue {queued} done {len(results)}/{self.planned} | {accuracy}"
              f" | server TPS {number(snap['tokens_per_s'])} now {number(snap['tokens_per_s_avg'])} avg"
              f" (window {number(snap['window_s'])}s) | KV {number(kv)}%"
              f" | stop {snap['natural_stop']} len {snap['length']} loop {snap['loop']} err {snap['errors']}"
              + (f" | metrics {snap['rate_status']}" if snap["rate_status"] != "ok" else ""), flush=True)
        self.write_markdown(snap, results, progress)
        temp = self.output_root / "live.json.tmp"
        temp.write_text(json.dumps(snap, indent=2, allow_nan=False) + "\n")
        temp.replace(self.output_root / "live.json")
        return snap

    def write_markdown(self, snap, results, progress):
        kv = None if snap["kv_usage"] is None else 100 * snap["kv_usage"]
        lines = [f"# {self.title} live ({datetime.now(timezone.utc).isoformat(timespec='seconds')})", "",
                 f"Elapsed {clock(snap['elapsed_s'])}; server running {number(snap['running'], 0)}, "
                 f"waiting {number(snap['waiting'], 0)}; pipeline queued {snap['queued']}, "
                 f"active outputs {snap['client_active_outputs']}, done {snap['done']}/{snap['planned']}.", "",
                 f"**Server TPS: {number(snap['tokens_per_s'])}** over {number(snap['window_s'])} s; "
                 f"average {number(snap['tokens_per_s_avg'])}; KV cache {number(kv)}%.", "",
                 f"Counter status: {snap['rate_status']}; resets: {snap['counter_resets']}. "
                 f"Generated since baseline: {snap['decoded_tokens']}. "
                 "Rates use accepted generation tokens, not draft tokens, and include wall-time stalls/prefill. "
                 "With a shared server, other requests to this model also contribute. "
                 "Per-request TPS and live context depths are not inferred from these aggregate metrics.", ""]
        if self.labelled:
            lines += [f"Correct outputs: {snap['correct']}; routed score: "
                      f"{snap['routed_score']:.3f}/{self.routed_tasks} (unfinished outputs remain unknown).", ""]
        lines += ["| Output | State | Prompt tokens | Generated tokens | Finish | Correct |",
                  "|---|---|---:|---:|---|---|"]
        for label, item in sorted(progress.items()):
            lines.append(f"| {label} | active | {item.get('prompt_tokens', '')} | | | |")
        for row in reversed(results):
            tokens = generated(row)
            label = f"{row['task']}_{row['test_index']}"
            token_text = str(tokens) if tokens is not None else "n/a"
            if row.get("usage_shared_task"):
                token_text += " (shared task)"
            lines.append(f"| {label} | {'error' if row.get('error') else 'done'} | {row.get('prompt_tokens', '')} | "
                         f"{token_text} | {row.get('finish_reason') or ''} | {row.get('correct', '')} |")
        temp = self.output_root / "live.md.tmp"
        temp.write_text("\n".join(lines) + "\n")
        temp.replace(self.output_root / "live.md")
