"""Opt-in streaming gzip heuristic; compressibility is NOT proof of a loop.

Use actual vLLM delta token IDs and an offline copy of the target tokenizer, or (llama.cpp
/completion) the server's own text for each streamed token.
Never compress integer IDs, include the prompt, or approximate tokens by chars.
"""

from collections import deque
import gzip
import hashlib
import math
from pathlib import Path


def load_tokenizer(model, override=None):
    """Load tokenizer.json only: no Torch, model weights, or network access."""
    path = Path(override or model)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise ValueError(f"Gzip guard requires a local tokenizer.json: {path}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    return tokenizer, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class GzipLoopGuard:
    def __init__(self, tokenizer, *, window_tokens=1000, ratio=8.0,
                 check_tokens=128, confirmations=1):
        if any(type(n) is not int or n <= 0 for n in (window_tokens, check_tokens, confirmations)):
            raise ValueError("Loop window/check/confirmation counts must be positive integers")
        if not math.isfinite(ratio) or ratio <= 1:
            raise ValueError("Loop gzip ratio must be finite and greater than 1")
        self.tokenizer = tokenizer
        self.window_tokens, self.ratio = window_tokens, ratio
        self.check_tokens, self.confirmations = check_tokens, confirmations
        self.ids = deque(maxlen=window_tokens)
        self.total = self.checks = self.streak = 0
        self.next_check = window_tokens
        self.max_ratio = 0.0
        self.first_hit = None
        self.disabled_reason = None

    def feed(self, token_ids):
        """Return the first confirmed hit once, independent of SSE chunk sizes."""
        if self.disabled_reason:
            return None
        if not isinstance(token_ids, list) or any(type(i) is not int or i < 0 for i in token_ids):
            raise ValueError("Expected non-negative vLLM delta token IDs")
        return self._extend(token_ids)

    def _extend(self, token_ids):
        hit = None
        offset = 0
        while offset < len(token_ids):
            take = min(len(token_ids)-offset, self.next_check-self.total)
            self.ids.extend(token_ids[offset:offset+take])
            offset += take
            self.total += take
            if self.total != self.next_check:
                continue
            data = self.tokenizer.decode(list(self.ids), skip_special_tokens=False).encode("utf-8")
            compressed = gzip.compress(data, compresslevel=6, mtime=0)
            ratio = len(data) / len(compressed)
            self.checks += 1
            self.max_ratio = max(self.max_ratio, ratio)
            self.streak = self.streak+1 if ratio > self.ratio else 0
            if self.streak >= self.confirmations and self.first_hit is None:
                hit = self.first_hit = dict(observed_completion_tokens=self.total,
                    window_tokens=self.window_tokens, raw_bytes=len(data), gzip_bytes=len(compressed),
                    ratio=ratio, threshold=self.ratio, consecutive_windows=self.streak)
            self.next_check += self.check_tokens
        return hit

    def snapshot(self):
        return dict(window_tokens=self.window_tokens, threshold=self.ratio,
            check_tokens=self.check_tokens, confirmations=self.confirmations,
            observed_completion_tokens=self.total, checked_windows=self.checks,
            max_ratio=self.max_ratio, first_hit=self.first_hit, disabled_reason=self.disabled_reason)


class _Pieces:
    @staticmethod
    def decode(pieces, skip_special_tokens=False):
        return "".join(pieces)


class StreamTextLoopGuard(GzipLoopGuard):
    """The same token windows over a llama.cpp /completion stream: the window holds each token's text.

    An event's text belongs to its first token (llama-server holds back an incomplete UTF-8 piece and
    sends it with a later token); control tokens it does not render add no text.
    """

    def __init__(self, **kwargs):
        super().__init__(_Pieces, **kwargs)

    def feed_event(self, token_ids, text):
        if self.disabled_reason or not token_ids:
            return None
        if not isinstance(token_ids, list) or not isinstance(text, str):
            raise ValueError("Expected llama.cpp stream token ids and text")
        return self._extend([text] + [""] * (len(token_ids) - 1))
