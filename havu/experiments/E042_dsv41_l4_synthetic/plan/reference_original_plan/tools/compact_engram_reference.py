"""CPU reference for compact, row-pruned DeepSeek-V4.1 Engram storage.

This is a tested mapping/byte-gather reference, not a vLLM integration or a
performance claim. It preserves original hash IDs, quantized retained bytes,
and logical output slots. Dropped rows return a shared exact-zero row;
invalid original IDs are rejected rather than treated as pruning misses.

Run the synthetic boundary/parity checks with:
    python compact_engram_reference.py

For real tables, stream the export into memmaps; do not call from_arrays on
the full original 203 GB of data. Native BF16 dequantization belongs in the
runtime adapter: preserve the pinned SM89 Engram gather/dequant semantics.
The native pinned-host UVA path and the optional CPU-dequant/prestage path
are different integration choices; neither is implemented by this reference.
"""

from dataclasses import dataclass
import numpy as np


def _validate_shard_range(vocab_start, num_original_rows, shard_rows):
    for name, value in (
        ("vocab_start", vocab_start),
        ("num_original_rows", num_original_rows),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(
            value, (bool, np.bool_)
        ):
            raise ValueError(f"{name} must be an integer")
    start, total = int(vocab_start), int(num_original_rows)
    if not 0 < total <= np.iinfo(np.int64).max:
        raise ValueError("original module row count must fit positive int64")
    if not 0 <= start <= start + shard_rows <= total:
        raise ValueError("shard range outside original module")


def _checked_integer_ids(values, limit, name):
    ids = np.asarray(values)
    # Empty Python sequences default to float64 but contain no IDs to coerce.
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers")
    if np.any((ids < 0) | (ids >= limit)):
        raise ValueError(f"{name} out of bounds")
    return ids.astype(np.int64, copy=False)


@dataclass
class CompactEngramShard:
    """One module's contiguous TP-owned hash-head range.

    id_map[i] = 0 for a dropped original local row, otherwise 1 + the row's
    physical index. The physical arrays include sentinel row 0. The native
    hash ID remains a module-global original ID; vocab_start is subtracted
    only for accessing this rank's remapping array. num_original_rows is
    the entire original module's row count, including other ranks' rows.
    """

    vocab_start: int
    num_original_rows: int
    id_map: np.ndarray       # uint32 [original rows in this rank's range]
    weights: np.ndarray      # uint8  [kept rows + 1, 256], FP8 E4M3FN bytes
    scales: np.ndarray       # uint8  [kept rows + 1, 8], UE8M0 bytes

    def __post_init__(self):
        if self.id_map.dtype != np.uint32 or self.id_map.ndim != 1:
            raise ValueError("id_map must be a one-dimensional uint32 array")
        _validate_shard_range(
            self.vocab_start, self.num_original_rows, len(self.id_map)
        )
        self.vocab_start = int(self.vocab_start)
        self.num_original_rows = int(self.num_original_rows)
        if self.weights.dtype != np.uint8 or self.weights.ndim != 2:
            raise ValueError("weights must be a two-dimensional byte array")
        if self.weights.shape[1] != 256:
            raise ValueError("preserve the native 256-dimensional row width")
        if self.scales.dtype != np.uint8 or self.scales.shape != (
            self.weights.shape[0], 8
        ):
            raise ValueError("expected eight original UE8M0 scales per row")
        if self.weights.shape[0] < 1:
            raise ValueError("a shared zero row is required even if none survive")
        if np.any(self.weights[0] != 0) or np.any(self.scales[0] != 127):
            raise ValueError("sentinel must be FP8 zero with neutral scales")
        if self.id_map.size and self.id_map.max() >= self.weights.shape[0]:
            raise ValueError("compact map points outside retained storage")

    @classmethod
    def from_arrays(
        cls, weights, scales, kept_local_ids, *, num_original_rows, vocab_start=0
    ):
        """Synthetic/reference exporter; use chunked file export at scale."""
        if weights.dtype != np.uint8 or scales.dtype != np.uint8:
            raise ValueError("export raw quantized bytes without requantizing")
        if weights.ndim != 2 or weights.shape[1] != 256:
            raise ValueError("source weights must have shape [rows, 256]")
        n = len(weights)
        if scales.shape != (n, 8):
            raise ValueError("source scales must have shape [rows, 8]")
        _validate_shard_range(vocab_start, num_original_rows, n)
        kept = _checked_integer_ids(kept_local_ids, n, "retained row IDs")
        if kept.ndim != 1:
            raise ValueError("retained row IDs must be one-dimensional")
        if np.unique(kept).size != kept.size:
            raise ValueError("duplicate retained row ID")
        kept = np.sort(kept)
        if kept.size >= np.iinfo(np.uint32).max:
            raise ValueError("uint32 compact ID overflow")
        mapping = np.zeros(n, dtype=np.uint32)
        mapping[kept] = np.arange(1, kept.size + 1, dtype=np.uint32)
        packed_w = np.zeros((kept.size + 1, 256), dtype=np.uint8)
        packed_s = np.full((kept.size + 1, 8), 127, dtype=np.uint8)
        packed_w[1:] = weights[kept]
        packed_s[1:] = scales[kept]
        return cls(
            vocab_start=vocab_start,
            num_original_rows=num_original_rows,
            id_map=mapping,
            weights=packed_w,
            scales=packed_s,
        )

    def compact_ids(self, original_ids, owned=None):
        """Preserve arbitrary input shape, including [tokens, local heads].

        Valid IDs owned by another rank return local zero. Globally invalid
        IDs always fail, even with owned=False. Padding uses a valid dummy
        original ID with owned=False, never an invalid address sentinel.
        """
        original = _checked_integer_ids(
            original_ids, self.num_original_rows, "original row IDs"
        )
        valid = (original >= self.vocab_start) & (
            original < self.vocab_start + len(self.id_map)
        )
        if owned is not None:
            owned = np.asarray(owned, dtype=bool)
            if owned.shape != original.shape:
                raise ValueError("ownership mask must match IDs")
            valid &= owned
        physical = np.zeros(original.shape, dtype=np.uint32)
        physical[valid] = self.id_map[original[valid] - self.vocab_start]
        return physical

    def gather_packed(self, original_ids, owned=None):
        """Return quantized bytes, scales and retained-hit mask.

        The output has exactly the same logical row slots as original_ids.
        No per-token Python loop or Python dict is used. Compare direct
        gathers with unique/inverse deduplication on the actual ARC trace
        before adding the latter: sorting can cost more at small batches.
        """
        physical = self.compact_ids(original_ids, owned)
        return self.weights[physical], self.scales[physical], physical != 0


def self_test():
    rng = np.random.default_rng(41)
    w = rng.integers(0, 256, size=(19, 256), dtype=np.uint8)
    s = rng.integers(1, 255, size=(19, 8), dtype=np.uint8)

    # Includes original local row 0; its real compact ID must not be sentinel 0.
    table = CompactEngramShard.from_arrays(
        w, s, [0, 3, 8, 18], num_original_rows=200, vocab_start=101
    )
    # 0, 100 and 120 are valid global IDs owned by other ranks.
    query = np.array([[101, 104, 102, 119], [100, 120, 0, 104]])
    pw, ps, hits = table.gather_packed(query)
    assert pw.shape == (2, 4, 256) and ps.shape == (2, 4, 8)
    np.testing.assert_array_equal(hits, [[1, 1, 0, 1], [0, 0, 0, 1]])
    np.testing.assert_array_equal(pw[0, 0], w[0])
    np.testing.assert_array_equal(ps[0, 0], s[0])
    np.testing.assert_array_equal(pw[0, 1], w[3])
    np.testing.assert_array_equal(pw[1, 3], w[3])
    assert np.all(pw[~hits] == 0) and np.all(ps[~hits] == 127)

    # Padding uses a valid dummy ID and remains zero even if that row is kept.
    owned = np.ones_like(query, dtype=bool)
    owned[0, 0] = False
    pw, ps, hits = table.gather_packed(query, owned)
    assert not hits[0, 0] and not pw[0, 0].any()

    # Invalid addresses must not silently become dropped rows, even when unowned.
    for bad in (
        [-1], [200], [101.5], ["101"], [True],
        np.array([2**64 - 1], dtype=np.uint64),
    ):
        with np.testing.assert_raises(ValueError):
            table.gather_packed(bad)
    with np.testing.assert_raises(ValueError):
        table.gather_packed([-1], owned=[False])

    # Retention IDs must be in-range integers, not silently truncated floats.
    for bad in ([-1], [19], [0.5], [True], [0, 0]):
        with np.testing.assert_raises(ValueError):
            CompactEngramShard.from_arrays(
                w, s, bad, num_original_rows=200, vocab_start=101
            )
    with np.testing.assert_raises(TypeError):
        CompactEngramShard.from_arrays(w, s, [0])
    with np.testing.assert_raises(ValueError):
        CompactEngramShard.from_arrays(
            w, s, [0], num_original_rows=119, vocab_start=101
        )
    with np.testing.assert_raises(ValueError):
        CompactEngramShard.from_arrays(
            w, s, [0], num_original_rows=200, vocab_start=101.5
        )
    bad_map = table.id_map.copy()
    bad_map[0] = len(table.weights)
    with np.testing.assert_raises(ValueError):
        CompactEngramShard(
            vocab_start=table.vocab_start,
            num_original_rows=table.num_original_rows,
            id_map=bad_map,
            weights=table.weights,
            scales=table.scales,
        )

    # All-keep is an exact-byte reference; no normalization/requantization.
    full = CompactEngramShard.from_arrays(
        w, s, np.arange(19), num_original_rows=512, vocab_start=307
    )
    fw, fs, fh = full.gather_packed(np.arange(307, 326))
    np.testing.assert_array_equal(fw, w)
    np.testing.assert_array_equal(fs, s)
    assert fh.all()

    empty = CompactEngramShard.from_arrays(
        w, s, [], num_original_rows=200, vocab_start=101
    )
    ew, es, eh = empty.gather_packed(query)
    assert not eh.any() and not ew.any() and np.all(es == 127)

    # Model the two-layer/TP4 layout, six complete hash heads per rank. Unequal
    # toy bank sizes exercise offsets; reassembly preserves all 24 head slots.
    for layer in range(2):
        sizes = np.arange(5 + layer, 29 + layer)
        offsets = np.r_[0, np.cumsum(sizes)]
        rows = int(offsets[-1])
        lw = rng.integers(0, 256, size=(rows, 256), dtype=np.uint8)
        ls = rng.integers(1, 255, size=(rows, 8), dtype=np.uint8)
        head_ids = offsets[:-1] + (sizes // 2)
        parts = []
        for rank in range(4):
            first, last = int(offsets[rank * 6]), int(offsets[(rank + 1) * 6])
            shard = CompactEngramShard.from_arrays(
                lw[first:last], ls[first:last], np.arange(last - first),
                num_original_rows=rows, vocab_start=first,
            )
            got_w, got_s, got_h = shard.gather_packed(
                head_ids[rank * 6:(rank + 1) * 6][None, :]
            )
            assert got_h.all()
            parts.append(got_w)
            np.testing.assert_array_equal(
                got_s[0], ls[head_ids[rank * 6:(rank + 1) * 6]]
            )
        np.testing.assert_array_equal(np.concatenate(parts, axis=1)[0], lw[head_ids])
    print(
        "PASS: byte parity, zero misses, ownership, original-row-0, offsets, "
        "TP4 slots, module bounds, integer IDs, retained-map bounds"
    )


if __name__ == "__main__":
    self_test()
