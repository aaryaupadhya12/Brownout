import json
import time
from openai import OpenAI
from src.contracts import ModelClient, ModelReply, ToolCall, Usage
from openai import RateLimitError



def parse_response(resp, latency_ms):
    msg = resp.choices[0].message

    calls = []
    for i, tc in enumerate(msg.tool_calls or []):
        raw = tc.function.arguments
        try:
            args = json.loads(raw)
        except Exception:
            args = {}
        calls.append(ToolCall(
            id=tc.id or f"call_{i}",
            name=tc.function.name,
            args=args,
            raw_args=raw,
            extra=getattr(tc, "extra_content", None) or {},
        ))

    u = resp.usage
    usage = Usage(
        input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens,
        reasoning_tokens=u.total_tokens - u.prompt_tokens - u.completion_tokens,
    )

    return ModelReply(text=msg.content or "", tool_calls=calls, usage=usage, latency_ms=latency_ms)


class OpenAICompatClient(ModelClient):
    def __init__(self, model_id, base_url, api_key, min_gap_s=0.0, max_retries=3):
        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.min_gap_s = min_gap_s
        self.max_retries = max_retries
        self.last_call_end = 0.0

    def complete(self, messages, tools, temperature=0.0):
        # stay under the requests-per-minute limit by spacing calls out
        wait = self.min_gap_s - (time.time() - self.last_call_end)
        if wait > 0:
            time.sleep(wait)

        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id, messages=messages, tools=tools, temperature=temperature)
                break
            except RateLimitError:
                attempt = attempt + 1
                if attempt > self.max_retries:
                    raise
                print(f"rate limited, waiting 45 s (retry {attempt} of {self.max_retries})")
                time.sleep(45)

        latency_ms = (time.perf_counter() - start) * 1000
        self.last_call_end = time.time()
        return parse_response(resp, latency_ms)