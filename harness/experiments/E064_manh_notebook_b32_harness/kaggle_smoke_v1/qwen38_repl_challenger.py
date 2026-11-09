"""Opt-in fresh-context falsification using only unused task tokens.

No test labels, learned routing or candidate-quality oracle. Source extraction is
best effort, NOT a Python program slicer: a sandbox replay must reproduce every
saved train/test prediction before the model is asked anything. Otherwise skip.
"""
import ast
import asyncio
from copy import copy, deepcopy
import hashlib
import json
import symtable
import time


GUIDANCE = """
# Fresh-context rule challenger
Another session found the candidate below. It fits the demonstrations, but its
test predictions are UNVERIFIED. Your task is falsification, not reassurance.
You have only the explicitly stated remaining budget; the primary answer is safe.

Read the original grids/images and executable candidate. Identify a consequential
assumption not uniquely established by the demonstrations: for example an object
selection predicate, boundary condition, overlap precedence, arbitrary enumeration
order/first match, or a training-pattern lookup. These are possibilities, not rules
to impose on every puzzle. Look for a single coherent transformation explaining ALL
demonstrations. Do not assume color/rotation invariance without evidence.

Use Python early to test a small family of meaningful alternatives against EVERY
training pair. Compare their test predictions. Do not call a prediction generated
by the same rule independent evidence, and do not invent synthetic ground truth.
Do not patch individual test cells, branch on test identity, or add a lookup of
training answers. Training fit is necessary but is not proof of the test rule.

The reconstructed candidate and its dependencies are already loaded in Python.
Define a separate alternative function, use check(function), and submit it only
if it passes every training example and makes a different test prediction. The
harness rechecks these conditions and preserves the original attempt_1. One valid
distinct alternative ends this challenger. If no defensible alternative exists,
finish with ordinary text; a duplicate or forced difference is not required.
"""


def _root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _writes(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or a.name.split('.')[0] for a in node.names}
    writes = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    # Retain mutations building a required table/list/array, including loop bodies.
    for n in ast.walk(node):
        if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, ast.Store):
            writes.add(_root(n))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            writes.add(_root(n.func.value))
    return writes - {None}


def _reads(node):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # Local scratch names like x/g/objects should not drag earlier exploration
    # into the prompt. Include definition-time dependencies and nested globals.
    table = symtable.symtable(ast.unparse(node), '<candidate>', 'exec')
    needed = {s.get_name() for s in table.get_symbols() if s.is_referenced()}
    def visit(scope):
        needed.update(s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced())
        for child in scope.get_children(): visit(child)
    for child in table.get_children(): visit(child)
    return needed


def candidate_cells(journal, function, *, max_chars=50000):
    """Conservative dependency closure, ordered as original cells; verified by execution later.

    Retain all definitions/assignments of required names (not just the latest:
    aliases and late-bound globals can depend on earlier ones). Module-level
    temporary names may over-retain code. Untracked dynamic side effects may under-retain
    it; exact replay validation below fails closed in that case.
    """
    nodes = []
    for cell, source in enumerate(journal):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # An originally syntax-invalid cell cannot have changed state.
        for node in tree.body:
            nodes.append((cell, node, _writes(node), _reads(node)))
    needed, selected = {function}, set()
    while True:
        old = len(selected)
        for i, (_, _, writes, reads) in enumerate(nodes):
            if writes & needed:
                selected.add(i)
                needed.update(reads)
        if len(selected) == old:
            break
    cells = {}
    for i, (cell, node, _, _) in enumerate(nodes):
        if i in selected:
            cells.setdefault(cell, []).append(ast.unparse(node))
    result = ['\n\n'.join(parts) for parts in cells.values()]
    if not result:
        raise ValueError('No reconstructable candidate definition')
    if sum(map(len, result)) > max_chars:
        raise ValueError('Candidate dependency bundle exceeds 50000 characters')
    return result


def allowance(args, used):
    budget = getattr(args, 'task_token_budget', None)
    cap = getattr(args, 'tir_repl_challenger_tokens', 0)
    if not cap or budget is None:
        return 0
    left = min(cap, max(0, budget - used))
    return left if left >= 2000 else 0


def challenger_args(args, tokens):
    result = copy(args)
    result.tir_repl_budget = tokens
    result.tir_repl_challenger_tokens = 0  # No recursive challengers.
    result.tir_repl_progress_notes = 'optional'
    result.tir_repl_review = False
    result.tir_repl_targeted_review = False
    result.tir_repl_candidate_pool = False
    result.tir_repl_review_token_budget = 0
    result.tir_repl_solver_by_tokens = 0
    result.tir_repl_solver_max_turns = 0
    result.tir_repl_structure_tools = True
    result.tir_repl_finalize_reserve = min(3000, tokens)
    result.tir_repl_finalize_turn_tokens = min(4000, tokens)
    result.tir_repl_finalize_think_tokens = min(1024, tokens - 512)
    result.tir_repl_turn_tokens = min(6000, tokens)
    result.tir_repl_think_tokens = min(2048, tokens - 512)
    result.tir_repl_min_turn = min(1000, tokens)
    result.tir_repl_wrap_up = 0  # Dedicated challenger/finalization guidance instead.
    result.tir_repl_max_turns = min(16, args.tir_repl_max_turns)
    return result


def merge_alternative(top, alt, picks, record):
    """Append only, never rerank or replace a primary/existing distinct second answer."""
    alt, picks = deepcopy(alt), deepcopy(picks)
    changed = []
    if record.get('accepted') is None:
        return alt, picks, changed
    entry = record['submissions'][record['accepted']]
    if not entry.get('accepted'):
        return alt, picks, changed
    for i, (primary, proposal) in enumerate(zip(top, entry['test'])):
        if (primary is not None and proposal is not None and proposal != primary
                and (alt[i] is None or alt[i] == primary)):
            alt[i] = deepcopy(proposal)
            picks[i] = picks[i][:1] + [dict(grid=deepcopy(proposal), votes=0, accepted=1,
                best=(1.0, 1.0), partial=0, fresh_challenger=True)]
            changed.append(i)
    return alt, picks, changed


async def run(ctx, task, primary, journal, top, folder, write_json, *, used_tokens):
    """Return a challenger summary and diagnostic. Primary snapshots are never mutated."""
    import qwen38_repl as qr
    task = qr.public(task)
    args = ctx['args']
    tokens = allowance(args, used_tokens)
    diagnostic = dict(status='skipped', tokens_allowed=tokens, primary_tokens=used_tokens,
                      test_answers_used=False)
    if not tokens:
        return None, dict(diagnostic, reason='insufficient-unused-budget')
    if ctx['deadline'] - time.monotonic() < max(120, args.min_request_seconds):
        return None, dict(diagnostic, reason='deadline')
    if not isinstance(ctx['prefix'], qr.CheckpointPrefix):
        return None, dict(diagnostic, reason='native-template-required')
    cargs = challenger_args(args, tokens)
    # Isolate state as well as conversation. No previous reasoning/feedback is sent.
    trajectory = qr.Trajectory(dict(ctx, args=cargs), 1, task, folder)
    trajectory.record.update(fresh_challenger=True, primary_predictions=deepcopy(top))
    started = time.monotonic()
    async def within_deadline(awaitable):
        return await asyncio.wait_for(awaitable, max(0.001, ctx['deadline']-time.monotonic()))
    try:
        cells = candidate_cells(journal[:primary.get('journal_length', len(journal))], primary['function'])
        for cell in cells:
            await within_deadline(trajectory.session.cell(cell))
        result = await within_deadline(trajectory.session.evaluate(primary['function']))
        if (result.get('error') or len(result.get('train', [])) != len(task['train'])
                or len(result.get('test', [])) != len(top)
                or any(c.get('error') or c.get('grid') != p['output']
                       for c, p in zip(result.get('train', []), task['train']))
                or any(c.get('error') or c.get('grid') != g
                       for c, g in zip(result.get('test', []), primary['test']))):
            return None, dict(diagnostic, reason='candidate-reconstruction-mismatch',
                              reconstruction_seconds=time.monotonic()-started)
        source = '\n\n# --- next original cell ---\n\n'.join(cells)
        diagnostic.update(reconstruction_seconds=time.monotonic()-started,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(), source_chars=len(source))
        request = deepcopy(ctx['prefix'].request)
        # Native prefix is only system + original task/images, not a running transcript.
        if [m['role'] for m in request['messages']] != ['system', 'user']:
            raise ValueError('Expected original system/task prefix only')
        request['messages'][0]['content'] = qr.experiment_prompt(qr.TEMPLATE.read_text(), cargs) + GUIDANCE
        request['tools'] = qr.tool_schemas(cargs)
        ids = await within_deadline(ctx['server'].render_tokens(request))
        text = (await within_deadline(ctx['server'].post('/detokenize', dict(tokens=ids))))['prompt']
        qr.native_generation_prompt(text)
        trajectory.ctx = dict(trajectory.ctx, prefix=qr.CheckpointPrefix(request, text), prefix_ids=ids)
        trajectory.messages = [dict(role='user', content=(
            f'Fresh challenger allowance: {tokens} generated tokens including reasoning, code and tools. '
            f'The candidate {primary["function"]} was independently replayed and reproduces '
            f'all {len(task["train"])} training outputs and its saved test predictions. '
            'This confirms reconstruction ONLY, not test correctness. Dependencies are loaded. '
            'The original answer is protected. Test a meaningful competing rule; submit only a '
            'training-passing, different alternative. You may also conclude no alternative.\n\n'
            f'```python\n{source}\n```'))]
        folder.parent.mkdir(parents=True, exist_ok=True)
        write_json(folder.parent / (folder.name + '.bootstrap.json'), dict(
            **diagnostic, candidate=primary['function'], cells=cells,
            training_passed=len(task['train']), predictions_reproduced=True,
            prefix_request=request, initial_messages=trajectory.messages))
        record = await within_deadline(trajectory.run(write_json))
        diagnostic.update(status='cancelled' if record.get('cancelled') else
                          'error' if record.get('error') else 'completed', stop=record['stop'],
                          tokens_used=record['completion_tokens'], error=record.get('error'))
        return trajectory.summary(), diagnostic
    except TimeoutError:
        return None, dict(diagnostic, reason='deadline-during-reconstruction')
    except Exception as exc:
        # Setup/optional-stage failure must not discard a completed primary answer.
        return None, dict(diagnostic, reason='challenger-setup-error',
                          error=f'{type(exc).__name__}: {str(exc)[:500]}')
