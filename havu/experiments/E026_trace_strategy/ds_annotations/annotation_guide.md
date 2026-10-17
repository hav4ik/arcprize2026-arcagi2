# Annotation guide (E026, DeepSeek-Flash trace strategy)

You annotate saved reasoning traces of DeepSeek V4.1-Flash solving ARC-AGI-2 eval tasks. Read-only; do not run any model or eval; spend no API money; never print secrets. Rendered traces are in
`/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/343f3b40-c572-4831-9a69-fc2997169a0c/scratchpad/render/<task>_p<pair>_<COND>.txt`.
The task JSON (train pairs, test input AND test output) is at `/home/vu/Workspace/ARC-AGI-2/data/arc-agi-2-github/data/evaluation/<task>.json` — look at it so you understand the true rule.

Three conditions per task (same test pair):
- **NT**: no tools, ARC Prize prompt, max effort; one long reasoning text. Grids were given as JSON lists.
- **S80**: tool harness (Python `run_python` tool with `train`/`test_input` preloaded, `submit_answer(expr)`), low effort, 80k output-token budget, NO computed summary. Each assistant turn shows its output-token count and cumulative tokens.
- **V1**: identical to S80, but the first user message also carried a host-computed "task summary" (shown at the top of the V1 file: per-pair sizes/bg/changed cells/colour counts/whole-grid relations, 8-connected single-colour object tables with size/bbox/holes/border/shape letters, multi-colour objects, in->out object matching, and a test "Outside train range" line). The prompt said "trust it over reading the grids by hand, and use code for any count".

Files are large (up to 350k chars). Read them in chunks (Read with offset/limit, or grep for markers). You do not need to read every hand-transcribed grid row; skim those, but do measure them.

## For EACH trace produce one JSON object (one line) with these fields
```
{"task": "...", "pair": 0, "cond": "NT|S80|V1", "correct": true/false, "tokens": N,
 "phases": [ {"phase": "<name>", "start_pct": 0, "end_pct": 12, "note": "short"} , ...],   # in order; phases may repeat
 "token_share_pct": {"perception_hand": x, "perception_code": x, "hypothesis_prose": x, "code_solve_debug": x,
                     "verification": x, "test_application": x, "post_answer_doubt": x, "other": x},   # sums to ~100
 "idea_pct": <where the true core idea first appears, % of trace, or null>,
 "final_rule_pct": <where the rule it finally submits is settled, %>,
 "self_stats": {   # summary-like statistics the model computed ITSELF; value is one of
                   # "hand" (in reasoning text), "code" (in Python), "both", "summary" (read off the provided summary only), "none"
     "dims": "...", "background": "...", "colour_counts": "...", "cell_diff_in_out": "...",
     "object_enumeration_bbox_size": "...", "holes": "...", "d8_symmetry": "...", "crop_scale_tile": "...",
     "in_out_object_matching": "...", "test_vs_train_range": "..."},
 "hand_transcription": "none|light|heavy",   # rows of digits / coordinate lists copied into reasoning
 "perception_tokens_est": <your estimate of output tokens spent on perception (hand + code), absolute>,
 "first_code_pct": <% of tokens before the first tool call; null for NT>,
 "uncertainty_handling": "short phrase: how it deals with contradictions / ambiguity (explains away, tests in code, picks simplest, asks test grid, gives up 'given time')",
 "perception_errors": "short list or none; were they caught?",
 # V1 only:
 "summary_used_items": ["header", "colour_counts", "objects", "shape_letters", "in_out_matching", "test_range", ...],
 "summary_rederived": "none|partial|full; which items it recomputed anyway and WHY (distrust? needed per-cell detail? didn't notice?)",
 "summary_misleading": "did any summary item mislead it or was it misread/distrusted when correct? e.g. 8- vs 4-connectivity, merged multi-colour object, bg choice; did the model trust a wrong/misleading item? quote",
 "excerpts": ["<=3 verbatim quotes, each <=300 chars, that best show its strategy/perception behaviour"],
 "notes": "2-4 sentences"}
```
Then, per task, one extra line: `{"task": "...", "pair": 0, "cond": "COMPARE", "notes": "what changed between S80 and V1 (and vs NT) in perception work and strategy; did the summary remove work, shift it, or get duplicated?"}`

Write all lines to `/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/343f3b40-c572-4831-9a69-fc2997169a0c/scratchpad/annot/<yourname>.jsonl` (create the folder if needed) and also return them in your final report, followed by a 5-line summary of patterns you saw. Be concrete and honest; estimates are fine but say when you are unsure.
