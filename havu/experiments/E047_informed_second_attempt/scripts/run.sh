#!/bin/bash
# E047: later attempts informed by attempt 0's answer (--prior-mode; never its thinking), all 120 eval tasks, Flash-1017.
# Attempt 0 is reused from E046 e046p_note_all (DeepSeek API; seeded into the new file); attempts 1..ATTEMPTS-1 run.
# Usage (repo root): [PROVIDER=DeepInfra] [ATTEMPTS=3] run.sh <ctrl|verify|verifycode|hybrid|differ>
#   PROVIDER unset = DeepSeek API (tag e047_<mode>); else OpenRouter, that provider pinned (tag e047or_<mode>).
set -e
M=$1; R=evals/results; P=$R/tools_persistent_grid_lean2sub_
SRC=${P}deepseek-flash_low_e046p_note_all.jsonl
if [ -n "$PROVIDER" ]; then
  MODEL=deepseek/deepseek-v4.1-flash-1017; TAG=e047or_$M; PROV=(--openrouter-provider "$PROVIDER")
else
  MODEL=deepseek-flash; TAG=e047_$M; PROV=(--auto-peak-price)
  export DEEPSEEK_API_KEY="$(sed -n 's/^deepseek_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
fi
OUT=${P}deepseek-flash_low_$TAG.jsonl
PRIOR=(); [ "$M" != ctrl ] && PRIOR=(--prior-mode $M --prior-file $SRC)
ARGS=(--challenges data/arc-agi-2-github/data/evaluation \
  --model $MODEL --effort low --prompt lean2sub --answer-tool --final-mode session --budget-notes 0.01 \
  --budget-warn 0.75 --lenient-parse --budget-soft 40000 --effort-header minimal --max-tokens 64000 \
  --max-total-output 80000 --max-turns 40 --attempts ${ATTEMPTS:-2} --multi-test --multi-budget 0.5 --fallback-solve \
  --train-check --turns-note 5 "${PRIOR[@]}" "${PROV[@]}" --tag $TAG --out $OUT --sandbox-url http://172.18.0.2:6000 \
  --workers ${WORKERS:-40} --max-balance-stop ${BALANCE_STOP:-6} --allow-expensive-run)
if [ ! -s "$OUT" ]; then  # seed attempt 0 with this run's config, so resume runs only the later attempts
  .venv/bin/python - "$SRC" "$OUT" "$M" "$MODEL" "${PROVIDER:-}" <<'PY'
import json, sys
src, out, m, model, prov = sys.argv[1:]
with open(out, "w") as f:
    for r in map(json.loads, open(src)):
        if r.get("attempt") == 0:
            c = dict(r["config"], model=model)
            if m != "ctrl":
                c["prior_mode"] = m
            if prov:
                c["openrouter_provider"] = prov
            r.update(config=c, seeded_from="e046p_note_all")
            f.write(json.dumps(r) + "\n")
PY
fi
.venv/bin/python evals/run_deepseek_tools.py "${ARGS[@]}" > $R/log_$TAG.txt 2>&1
