# ARC-AGI-3 agent harnesses: a survey for ARC-AGI-2

These are the ARC-AGI-3 entries on the ARC Prize community leaderboard (Public Demo, 25 games), checked on 2026-10-01. Scores are mean RHAE (relative human action efficiency). Cost is the submitted USD total for all 25 games, unless a note says otherwise; accounting methods differ. Models are taken from the repos and from Retrodict's `docs/arc-agi-3-harness-comparison.md`.

## Summary

| Name | Score | Cost (25 games) | Model | Core idea | License | Transferable idea |
|---|---:|---:|---|---|---|---|
| Tycho | 100 | $2,986 (API-equiv.) | Opus 5 (GPT-5.6 Sol also 100) | Actor plus an optional builder of an executable `world_model.py`, checked by replay | Apache-2.0 | Replay verifier that reports the first divergence as a diff |
| Retrodict | 99.86 | $654 (API list) | GPT-5.6 Sol max | Log-as-context; test hypotheses against recorded history before acting; playbook memory | none found | Check each hypothesis in code against all evidence |
| baseline1 | 98.97 | $400 (subscription); $2,722 API-equiv. | GPT-5.6 Sol via Codex | Executable world model, simplified, verified by exact replay | MIT | Simplify after verifying |
| Human Intelligence Harness | 95.35 | — | humans | Replays human play | — | none (reference ceiling) |
| NOOA | 85.13 | $332 ($13.28/game) | GPT-5.6 Sol xhigh | CodeAct REPL agent; reusable NumPy helpers; memory/Markdown knowledge | Apache-2.0 | Persistent helper library; agent-as-Python-object |
| OPINE-World | 78.37 | $1,040 | Opus 4.8 high | Two agents in a CEGIS loop; NL + Bayesian + program world model; adversarial tests | none found | Failing train pair = counterexample; NL and code hypotheses kept together |
| Vision-CL v1 | 63.15 | $4,788 | Vispark Vision Large | Opaque "continual-learning weights"; trained on the test games | MIT | none (contaminated, opaque) |
| Read-Grep-Bash (now PRO-LONG) | 50.2 (one game visible) | — | Opus 4.6 | Append-only log; agent greps/parses it in Python | MIT | Raw data in files, queried with code |
| TELL | 43.9 | $1,406 | Opus 4.6 | One conversation, 3 tools, a compounding `MEMORY.md` | MIT (README; no LICENSE file) | Cleaned memory carried across tasks (−29% actions) |
| DreamTeam | 38.06 | $18,000 | Opus 4.6 + GPT-5.5 | 6 fixed roles around a shared executable world model | Apache-2.0 (NVIDIA) | Synthetic tasks as harness dev set |
| Continual Harness | 20.54 | $774 | Gemini 3.1 Pro | Refiner rewrites prompt/skills/subagents/memory mid-run | MIT | Self-edited prompts and skills (weak evidence) |
| Polyphony | 19.8 | $115 (self-hosted; basis unstated) | Qwen3.6-27B (vLLM) | Per-game Python state/engine/planner/verifier, gated by exact replay | MIT | Exact-replay gate works on a 27B model |
| a-evolve MAS | 12.3 | $5,300 | Opus 4.6 high | Evolved multi-agent orchestrator; 9 skills mined from logs | MIT | Mining skills from logs offline |
| OpenClaw | 5.2 | $2,912 | Opus 4.7 | General assistant with memory + code tools | MIT (template) | none (control) |

## Entries

**Tycho.** The actor sees frames and keeps notes. It can call a builder that writes `world_model.py` (`State`, `init_state`, `transition`, `render`, `outcome`). A verifier replays every recorded transition and reports exact accuracy, known-cell accuracy (`-1` marks an unknown cell) and the first divergence, as observed vs. predicted run-length diffs plus PNGs. Opus 4.8 ablations: no model 79.1, actor writes the model 85.4, actor calls the builder 88.5, harness forces a rebuild on falsification 83.1. Code helps most when the agent chooses when to formalize.

**Retrodict.** Every frame goes into `log.txt`; after one vision-model priming note, the model never sees images. It works through a `python` tool with an `arclog` library (parsed boards, diffs, connected components). A hypothesis must replay without contradiction over the recorded history before it earns a live action. Each committed action carries its expected cells, and the runner halts on the first mismatch and returns the diff. `playbook.md` is all that survives context resets; a simulator is built only after 300 stuck actions. It scored higher than baseline1 with 5.5x fewer tokens.

**baseline1** (Rodionov; 2 arXiv papers). Codex works in a workspace with empty engine and planner stubs plus replay verifiers. The paper ablates four variants: a textual model; an executable model; executable plus simplification; and executable plus simplification plus verification (`ewma_sv`). `ewma_sv` scores 98.97 at about $2.7k; the textual model scores 95.97 at $918. Exact replay and simplification buy the last points at about 3x the tokens.

**Human Intelligence Harness.** A reference entry from the ARC Prize Foundation. Per Retrodict's comparison doc, it replays human solutions; the link was rate-limited, so we did not verify this. It is not a method.

**NOOA** (NVIDIA). An agent is a Python class: docstrings are prompts, and `...` bodies become LLM loops that act by writing code in a REPL. The ARC agent analyzes grids with NumPy, keeps reusable helpers in `world_model.py`, and stores what it learns in vector memory or Markdown files. A 2-hour cap and an effort ladder (xhigh, dropping to medium after 600 s) keep it cheap.

**OPINE-World.** One agent acts while a second rewrites an executable engine through CEGIS: every prediction mismatch becomes a counterexample, and the engine is tested adversarially. The world model combines an NL ontology with open hypotheses, Bayesian object uncertainty, and Python transition and reward code. Rate limits cut 5 games short, so the authors estimate the true score at 78–92.

**Vision-CL v1.** A proprietary API returns a "weights" blob that is fed back into later calls. Phase 1 plays the 25 public games repeatedly, and phase 2 replays the same games for the score. That is train-on-test, so the score is uninformative.

**Read-Grep-Bash / PRO-LONG.** The repo has been renamed. A Claude Code or Codex agent gets `logs.txt` (all boards plus its own analyses) and Read/Grep/Bash, with a ~30-line prompt: parse the log in code, because reading 64×64 grids directly is error-prone. The paper reports +18 pp over the same agents without the log, and 97.4% best@2 with Fable 5. Retrodict builds on this design.

**TELL** (rednote). One Claude conversation per game with three tools, one of which writes `MEMORY.md`; context is compacted at 80%. Confirmed rules accumulate across levels. In the "prior memory" experiment, an LLM strips level-specific details from each final memory (delete only) and feeds it to a fresh run. That cut actions by about 29% on levels completed in both runs.

**DreamTeam** (NVIDIA). Six roles share a file workspace and an executable world model. A Game Creator generates validated synthetic games, used for "workspace optimization". At $18k for 38%, fixed role decomposition looks dominated by a single coding agent.

**Continual Harness.** A Gemini 3.1 Pro orchestrator plays each game. A meta-call triggered by events or stagnation rewrites the prompt, skills, subagents and memory mid-run. A score of 20.5% does not show that this pays off.

**Polyphony.** baseline1's recipe on open-weight Qwen3.6-27B, served by self-hosted vLLM. The agent grows a per-game Python state/engine/planner/verifier, accepted only when it reproduces the observed transitions exactly. The low score probably reflects the smaller model, not the recipe.

**a-evolve.** A generic framework that evolves workspace files. It evolved an ARC multi-agent setup (theorist, explorer, solver, wiki) from 10% to 12.3%, which is weak evidence.

**OpenClaw.** A general assistant with memory and code tools, run behind a thin shim. It scored 5.2% for $2.9k, a control showing that a strong model fails without verification structure.

## Ideas worth porting to ARC-AGI-2

1. **Gate on reproducing the evidence, and return a structured diff.** On ARC-AGI-2, a candidate `transform()` must reproduce every train pair. On failure, return the first failing pair and the differing cells (expected vs. predicted, run-length), not just pass/fail. This is the shared core of the top 3, and of Imbue's evolver.
2. **Python sandbox with grid helpers; data in files, not the prompt.** Retrodict and RGB/PRO-LONG show that computing over grids (components, diffs, counts) beats reading them. Put the task JSON and a small helper library in the workspace.
3. **Known vs. unknown cells.** Following Tycho, score partial programs on the cells they commit to. This gives search a gradient below 100% train accuracy, and flags test cells a program cannot decide.
4. **Simplify after passing** (baseline1). Ask for a shorter or more general program that still passes train. This fights train-overfit and matches Imbue's simplicity term.
5. **NL first; escalate to heavy code search only on failures.** In Tycho, letting the agent choose when to build a model beat forcing it, and Retrodict builds a simulator only when stuck.
6. **Cross-task memory.** TELL's cleaned memories and NOOA's helper library suggest a library of verified primitives and concept notes mined from solved training tasks. Transfer is less certain than between levels of one game, so test it on held-out tasks.
7. **Synthetic tasks as the harness dev set** (DreamTeam). Tune on NVARC/RE-ARC puzzles, never on eval (cf. Vision-CL).
8. **Low priority:** fixed multi-agent role teams and online self-rewriting of the harness. Both are expensive, and neither has beaten a single verified coding agent.
