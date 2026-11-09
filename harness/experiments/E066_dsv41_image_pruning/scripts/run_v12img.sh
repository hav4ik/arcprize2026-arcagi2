#!/bin/bash
# V12 + images on node 5 (user 2026-10-30: "make the latest submission Manh V12 use images ... same harness, just a
# surgical change"): V12's pinned harness flags (solver 11bfaaaa HARNESS + its additions; dsv4-infer) minus
# --token-exact, plus --grid-format text+image --image-cell 0 (pinned_img harness 1d1a281e, grid_image 18ed78fb,
# train_check 8fe36175, auto_repair 2292369c/571f47eb), tool code in the local executor as on Kaggle, against the
# causal-fixed c1fx server (the recalibrated image model). No solutions on the box. Two passes (the harness resumes).
D=/workspace/dsv4; TAG=${TAG:-v12img_c1f}
cd $D/imgkit/repo/evals
for pass in 1 2; do
  $D/tvenv/bin/python b32_local_eval.py --mode grid --effort low --prompt lean2sub --answer-tool --final-mode session \
    --lenient-parse --max-tokens 64000 --max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 --budget-warn 0.75 \
    --effort-header minimal --turns-note 5 --train-check --rescue-code --model dsv41 --fallback-solve \
    --multi-test --auto-repair --multi-budget 0.5 --context-limit 163840 --max-turns 40 \
    --grid-format text+image --image-cell 0 \
    --base-url http://127.0.0.1:18001/v1 \
    --extra-body "{\"chat_template_kwargs\": {\"thinking\": true, \"reasoning_effort\": 50}, \"temperature\": 1.0, \"top_p\": 1.0, \"cache_salt\": \"$TAG\"}" \
    --sandbox-url local --attempts 2 --workers ${WORKERS:-160} \
    --challenges $D/data/arc-agi_evaluation_challenges.json --out $D/results/$TAG.jsonl >> $D/logs/eval_$TAG.log 2>&1
  n=$($D/tvenv/bin/python -c "import json; print(len({(r['task_id'], r['attempt']) for r in map(json.loads, open('$D/results/$TAG.jsonl')) if 'error' not in r}))" 2>/dev/null || echo 0)
  echo "$(date -u +%H:%M:%S) $TAG pass $pass: $n/240 attempts"
  [ "$n" -ge 240 ] && break
done
echo "$(date -u +%H:%M:%S) EVAL_DONE $TAG"
