# Experiments

`experiments.tsv` is the registry: one row per experiment or run. Columns:

| Column | Meaning |
|---|---|
| `id` | `E###`, with a letter suffix for variants (`E001a`) |
| `name` | Short description |
| `path` | Folder holding the experiment's README, code pointers and analysis |
| `status` | `planned` / `running` / `done` / `failed` / `abandoned` |
| `model` | Model(s) used |
| `eval_set` | What it was scored on |
| `metric` | `pass@1`, `pass@2`, … |
| `result` | Headline number |
| `cost_usd` | API or compute cost |
| `started`, `updated` | YYYY-MM-DD |
| `notes` | Results file, caveats, links |

Conventions:
- **Folders:** each experiment gets `experiments/E###_short_name/README.md` with these sections: Hypothesis, Setup, Results, Conclusions, Next.
- **Code:** reusable code lives outside (e.g. `evals/`); the README points to the exact command and results file.
- **Updating:** update the TSV row whenever status or results change.
- **Eval-set comparability:** always say which eval set a number comes from. NVIDIA-val ARC-AGI-2 (120 rows, one test input each), the official public eval (pass@2 over 167 test pairs) and ARC Prize semi-private are not directly comparable.
