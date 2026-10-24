"""Model access: one streamed assistant turn over an OpenAI-compatible async client, plus token accounting.

Works with the DeepSeek API and with any OpenAI-compatible server (e.g. vLLM on Kaggle) via base_url.
"""

# USD / million tokens: cache hit, cache miss, output (off-peak list prices).
PRICES = {"deepseek-flash": (0.003, 0.15, 0.60), "deepseek-v4-pro": (0.022, 0.66, 1.98)}


class TurnError(RuntimeError):
    """A turn that did not complete; carries whatever was streamed so the trace and usage stay accurate."""

    def __init__(self, message, assistant, usage, finish):
        super().__init__(message)
        self.assistant, self.usage, self.finish = assistant, usage, finish


async def stream_turn(client, *, model, effort, messages, tools, max_tokens, final=False):
    """Stream one assistant turn; assemble content, reasoning and indexed, fragmented tool calls.

    Returns (assistant_message, usage, finish_reason). Raises TurnError if the stream is incomplete.
    `tools` stays present on the final turn (tool_choice="none") so DeepSeek keeps reading earlier
    reasoning_content, which it requires whenever a request carries tools.
    """
    content, reasoning, calls, usage, finish = [], [], {}, None, None
    stream, failure = None, None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="none" if final else "auto", max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"reasoning_effort": effort, "thinking": {"type": "enabled"}},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                content.append(delta.content or "")
                reasoning.append(getattr(delta, "reasoning_content", None) or "")
                for tc in delta.tool_calls or []:
                    call = calls.setdefault(tc.index, {"id": "", "type": "function",
                                                      "function": {"name": "", "arguments": ""}})
                    call["id"] += tc.id or ""
                    if tc.function:
                        call["function"]["name"] += tc.function.name or ""
                        call["function"]["arguments"] += tc.function.arguments or ""
            finish = choice.finish_reason or finish
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            await stream.close()
    assistant = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        assistant["tool_calls"] = [calls[i] for i in sorted(calls)]
    if failure or finish is None or usage is None:
        raise TurnError(failure or "Incomplete stream: missing finish reason or usage", assistant, usage, finish)
    if any(not c["id"] or not c["function"]["name"] for c in calls.values()):
        raise TurnError("Incomplete streamed tool call", assistant, usage, finish)
    return assistant, usage, finish


def new_usage():
    return {"prompt": 0, "cache_hit": 0, "completion": 0}


def add_usage(total, usage):
    total["prompt"] += usage.get("prompt_tokens", 0) or 0
    total["cache_hit"] += usage.get("prompt_cache_hit_tokens", 0) or 0
    total["completion"] += usage.get("completion_tokens", 0) or 0


def estimated_cost(total, model, multiplier):
    hit, miss, output = PRICES.get(model, (0.0, 0.0, 0.0))
    return multiplier * (total["cache_hit"] * hit + max(0, total["prompt"] - total["cache_hit"]) * miss
                         + total["completion"] * output) / 1e6
