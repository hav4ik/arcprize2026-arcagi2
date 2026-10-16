"""LLM backends. The agent only needs one call: complete a batch of chats.

`VLLMBackend` runs offline on the Kaggle GPUs. `MockBackend` replays scripted
programs so the whole harness can be exercised on CPU with no model.
"""
import re
import time

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def extract_code(text):
    """The last fenced block that defines transform(); falls back to any block
    containing it, then to a bare `def transform` in unfenced text."""
    if not text:
        return None
    body = THINK_RE.sub("", text)
    if "<think>" in body and "</think>" not in body:
        return None                      # truncated inside the reasoning
    blocks = [b for b in CODE_RE.findall(body) if "def transform" in b]
    if blocks:
        return blocks[-1].strip()
    i = body.rfind("def transform")
    if i >= 0:                           # unfenced: take the function to the end
        return "import numpy as np\n" + body[i:].strip()
    return None


class VLLMBackend:
    """Offline vLLM chat. One engine over all GPUs (tensor parallel)."""

    def __init__(self, model, tp=4, max_model_len=32768, gpu_mem=0.90,
                 max_tokens=16384, temperature=0.8, top_p=0.95, seed=0,
                 chat_kwargs=None, quantization=None, enforce_eager=False):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem, seed=seed, trust_remote_code=True,
                       quantization=quantization, enforce_eager=enforce_eager)
        self.max_tokens, self.temperature, self.top_p = max_tokens, temperature, top_p
        self.chat_kwargs = chat_kwargs or {}
        self.seed = seed
        self.tokenizer = self.llm.get_tokenizer()

    def prompt_tokens(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 **self.chat_kwargs)
        return len(ids)

    def chat(self, conversations, max_tokens=None, metas=None):
        """Returns [(text, n_generated_tokens, finish_reason)] in order."""
        params = [self.SamplingParams(temperature=self.temperature, top_p=self.top_p,
                                      max_tokens=max_tokens or self.max_tokens,
                                      seed=self.seed + i)
                  for i in range(len(conversations))]
        self.seed += len(conversations)
        outs = self.llm.chat(conversations, params, use_tqdm=False,
                             chat_template_kwargs=self.chat_kwargs)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append((c.text, len(c.token_ids), c.finish_reason))
        return res


class MockBackend:
    """Replays `script(task_id, round, attempt, conversation) -> text`.

    `metas` (task_id, attempt) travel beside the conversations, never inside
    them, so the vLLM path sees plain chat messages.
    """

    def __init__(self, script, delay=0.0):
        self.script = script
        self.delay = delay
        self.calls = 0

    def prompt_tokens(self, messages):
        return sum(len(m["content"]) for m in messages) // 3

    def chat(self, conversations, max_tokens=None, metas=None):
        self.calls += 1
        time.sleep(self.delay)
        out = []
        for conv, meta in zip(conversations, metas or [{}] * len(conversations)):
            rnd = sum(m["role"] == "assistant" for m in conv)
            text = self.script(meta.get("task_id"), rnd, meta.get("attempt", 0), conv)
            out.append((text, len(text) // 3, "stop"))
        return out
