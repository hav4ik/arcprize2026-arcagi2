# havu: Vu's DSV4.1 tool harness (B32-M), as source

`qwen38_pipeline.py --qwen38-variant dsv4-vllm --dsv41-harness havu` runs `kaggle/dsv41_tools/solver_dsv41.py` from
this folder on our supervised dsv41 server (`dsv41_havu.py`). Edit the code here. Each run records the sha256 of every
file in `havu_summary.json` (`code`, `code_sha256`). `--havu-dir <checkout>/havu` runs another copy instead, which must
contain S26, S27 and E066.

Paths match his repo's `havu/` folder (hav4ik/arcprize2026-arcagi2), so the two diff directly:
`meld kaggle_submit/havu arcprize2026-arcagi2/havu`.

## Origin (2026-10-30; "his main" = 18fd6be93, 2026-10-29; images from his 0c838db34 `pinned_img`)

The B32-M submission as his `dsv41-tools-submit-b32m.ipynb` runs it: his `khavu/ds-toolkit` `dsv41_kit.tgz`
(`repo/evals`, `repo/notes`) with his `pinned_candb` files on top. Only the files the solver imports are copied.

| file | version |
| --- | --- |
| `kaggle/dsv41_tools/solver_dsv41.py` | `pinned_candb` (B32, sha 77fe80a3) + S25/S26 + E066/S28 (our fork branch `havu-s24-supervised-server`) + S27 + his `pinned_img` image switch and `APITimeoutError` re-queue + S29 (ours, 2026-10-31): our pipeline's LIVE line every `ARC_LIVE_EVERY_S` (30; 0 = off) and at the end |
| `evals/run_deepseek_tools.py` | `pinned_img` (1d1a281e = his main's; was `pinned_candb` 882c817f): E058 `--grid-format text+image`, E064 `show_grid`/`check_image`, the chat-path `--context-limit` (`chat_context_cap`), `offpeak_wait` |
| `evals/arc_harness/grid_image.py` | `pinned_img` (18ed78fb = his main's): the E058 token-aligned image sizes (`aligned_cell`) |
| `evals/train_check.py`, `auto_repair.py`, `auto_repair_names.json` | `pinned_candb` (same as his main) |
| other `evals/` files, `notes/prompts/arcprize_official_system_prompt.txt` | kit (same as his main) |

Our edits:
- S25: the solver can use an external server.
- S26: the solver follows that server's restart supervisor.
- S27: the solver finds `evals/` in this layout as well as in his kit's `repo/evals`.
- E066 (his `dsv41-tools-submit-b32m-fix.ipynb`, solver 1ea2b25d): the 500 a dying vLLM engine answers in-flight
  requests with ("The server had an error while processing your request") is a free re-queue, not one of an attempt's
  2 tries; after a restart the solver waits for the supervisor's `up`.
- S28: with a status file, 12 free re-queues per attempt (`ARC_CONN_TRIES`), as his E066 notebook sets.
- From his `pinned_img` solver (64ef0a0f): the E066 image switch (`ARC_GRID_FORMAT=text+image`: drop `--token-exact`,
  add `--grid-format text+image --image-cell 0`), and `APITimeoutError` / "Request timed out" as free re-queues.

Images: `--input-mode text-image --vision-max-images 32` on a checkpoint with the vision tower. `dsv41_havu.py` then
sets `ARC_GRID_FORMAT=text+image` and our server keeps its vision encoder; in havu mode it always parses tool calls
(`--tool-call-parser deepseek_v41`), as his `server_cmd()`. His E068 causal-attention flags are not needed: our vLLM
fork has the upstream fix (#57152), so DeepSeek-V4.1 image spans are causal.

S25, S26 and S28 are in the solver docstring; S27 is the comment at `EVALS =`; E066 is at `CONN_ERRORS` and in
`start_external()`. On the server side, `dsv41_havu.py` also restarts a server that is up but makes no token for
`--havu-stall-s` (1500 s, as his E066 supervisor cell).

Not copied, because the solver never imports them: `arc_harness/{agent_qwen,audit,digit_grid_parse,model,prompts,
selection,slots}.py` and `worked_example.txt`.

## Using his modules from our code

His `evals/` modules import each other by flat name, so put `evals/` on `sys.path`:

```python
sys.path.insert(0, str(Path(__file__).with_name("havu") / "evals"))
from train_check import ranked_answers, train_check
from auto_repair import propose
```

`run_deepseek_arc2.api_key` and `run_deepseek_tools.openrouter_key` read `havu/SECRETS.yaml`, but only for the hosted
APIs, which the Kaggle path never uses. Never copy his `SECRETS.yaml` or `ENDPOINTS.yaml` here; `.gitignore` excludes
both.
