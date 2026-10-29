"""Prefix-only ARC16 coordinates shared by LoRA decoding and vLLM.

No torch/vLLM imports: request planning must not initialize CUDA. Unlike the
strict pre-SFT document validator, decoding accepts unfinished/nonrectangular
grids. Search can propose such prefixes; coordinates never inspect their future.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArcPositionState:
    message: int = 0
    header: int = 0
    column: int = 0
    row: int = 0


def advance_positions(tokens, state=None, *, allow_padding=False):
    """Return xyz and a NEW immutable state (safe for DFS cache branches)."""
    state = state or ArcPositionState()
    m, header, x, y = state.message, state.header, state.column, state.row
    positions = []
    for raw_token in tokens:
        token = int(raw_token)
        if token == 13 and allow_padding:
            positions.append((0, 0, 0))
            continue
        if header < 3:
            expected = (14, 11 if m % 2 == 0 else 12, 10)[header]
            if token != expected:
                raise ValueError(
                    f"3D RoPE expected ARC16 header token {expected}, got {token}"
                )
            positions.append((0, 0, 2 * m))
            header += 1
        elif 0 <= token <= 9:
            positions.append((x, y, 2 * m + 1))
            x += 1
        elif token == 10:
            positions.append((x, y, 2 * m + 1))
            x, y = 0, y + 1
        elif token == 15:
            positions.append((0, 0, 2 * m + 2))
            m, header, x, y = m + 1, 0, 0, 0
        else:
            raise ValueError(f"Invalid ARC16 grid token for 3D RoPE: {token}")
    return positions, ArcPositionState(m, header, x, y)


def scheduled_positions(requests, starts, lengths):
    """Build flattened xyz for complete known prefixes at scheduled offsets.

    Rebuild from authoritative token IDs, not a mutable per-request counter.
    This handles prefix-cache hits, chunked prefill, preemption and DFS request
    duplication without depending on the order of requests in a batch.
    """
    if not (len(requests) == len(starts) == len(lengths)):
        raise ValueError("3D RoPE request/offset counts differ")
    result = []
    for tokens, start, length in zip(requests, starts, lengths):
        start, length = int(start), int(length)
        if start < 0 or length <= 0 or start + length > len(tokens):
            raise ValueError("3D RoPE scheduled tokens exceed the known request prefix")
        xyz, _ = advance_positions(tokens[:start + length])
        result.extend(xyz[start:start + length])
    return result


class ArcRequestPositionCache:
    """Bounded live-request memoization; token equality remains authoritative.

    Compare the already-parsed token prefix before reuse, including after
    rollback, in-place token edits, preemption or request-ID reuse. Only new
    suffix tokens need Python grid parsing. No model/KV state is cached here.
    """

    def __init__(self):
        self._entries = {}

    def retain(self, request_ids):
        live = set(request_ids)
        for key in list(self._entries):
            if key not in live:
                del self._entries[key]

    def positions(self, request_id, prompt, output, start, length):
        start, length = int(start), int(length)
        end = start + length
        if start < 0 or length <= 0 or end > len(prompt) + len(output):
            raise ValueError("3D RoPE scheduled tokens exceed the known request prefix")
        if end <= len(prompt):
            tokens = tuple(prompt[:end])
        else:
            tokens = tuple(prompt) + tuple(output[:end - len(prompt)])
        entry = self._entries.get(request_id)
        if entry is not None:
            old_tokens, coordinates, state = entry
            common = min(end, len(old_tokens))
            if tokens[:common] != old_tokens[:common]:
                entry = None
        if entry is None:
            coordinates, state = advance_positions(tokens)
            self._entries[request_id] = tokens, coordinates, state
        elif end > len(old_tokens):
            suffix, state = advance_positions(tokens[len(old_tokens):], state)
            coordinates.extend(suffix)
            self._entries[request_id] = tokens, coordinates, state
        return coordinates[start:end]
