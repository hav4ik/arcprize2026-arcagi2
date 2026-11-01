"""E066: chat requests that make the vLLM fork consume exactly a tokenized image rollout.

The token stream of tokenize_image_rollouts.py cannot be sent as token ids (images need pixels), so requests go to
/v1/chat/completions with the rollout's messages cut after the last assistant turn, the exact harness tools and
reasoning effort 50. The fork's DeepSeek-V4.1 tokenizer ignores add_generation_prompt and renders the messages
with the same encode_messages (tools present => earlier reasoning kept), and its processor expands each image
placeholder with the same sizing rule, so the server consumes the whole offline stream, final EOS included.
The observer's input_ids are checked against the stream by class_stats.py aggregate.
"""
import json
import uuid
from pathlib import Path

import pyarrow.parquet as pq

KEY = ("batch", "task_id", "pair", "attempt")


def load_rows(parquets):
    """(batch, task_id, pair, attempt) -> {messages, tools} (JSON strings) for every rollout in the parquets."""
    rows = {}
    for p in parquets:
        t = pq.read_table(p, columns=list(KEY) + ["messages", "tools"])
        for r in t.to_pylist():
            rows[tuple(r[k] for k in KEY)] = r
    return rows


def load_index(tokdirs, max_len=None):
    recs = []
    for d in tokdirs:
        for line in open(Path(d) / "index.jsonl"):
            r = json.loads(line)
            r["dir"] = str(d)
            if max_len is None or r["length"] <= max_len:
                recs.append(r)
    return recs


def chat_body(row, model, effort=50, **extra):
    messages = json.loads(row["messages"])
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    body = {"model": model, "messages": messages[: last + 1], "tools": json.loads(row["tools"]), "max_tokens": 1,
            "temperature": 0.0, "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
            "cache_salt": uuid.uuid4().hex}
    body.update(extra)
    return body
