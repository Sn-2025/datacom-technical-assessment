"""One provider adapter for streaming, tools and validated structured responses."""
from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Iterator

import httpx
from openai import OpenAI

from .config import Connection
from .telemetry import Telemetry, cost_usd


class LLM:
    def __init__(self, connection: Connection, telemetry: Telemetry, client=None):
        self.connection = connection.model_copy(deep=True)
        self.telemetry = telemetry
        if client is None and not connection.api_key.get_secret_value():
            raise ValueError("Configure an API key before making model requests")
        # No invisible SDK retries: every attempted request has its own telemetry.
        self.client = client or OpenAI(api_key=connection.api_key.get_secret_value(),
            base_url=connection.base_url, timeout=connection.timeout_s, max_retries=0,
            http_client=httpx.Client(follow_redirects=False, timeout=connection.timeout_s))

    def _stats(self, run_id, started, usage, model, request_id, status, ttft=None):
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        return self.telemetry.record(run_id, "llm_request", status=status,
            prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached,
            cost_usd=cost_usd(prompt, completion, cached, self.connection.pricing),
            cost_estimated=True, latency_ms=(time.perf_counter()-started)*1000,
            ttft_ms=ttft, returned_model=model, request_id=request_id,
            config=self.connection.public_snapshot())

    def stream(self, messages: list[dict], run_id: str | None = None) -> Iterator[dict]:
        run_id = run_id or uuid.uuid4().hex
        start = time.perf_counter()
        usage = None
        model = request_id = ttft = None
        status = "interrupted"
        stream = None
        try:
            stream = self.client.chat.completions.create(model=self.connection.model, messages=messages,
                stream=True, stream_options={"include_usage": True},
                max_completion_tokens=self.connection.max_output_tokens)
            for chunk in stream:
                usage = chunk.usage or usage
                model = chunk.model or model
                request_id = getattr(chunk, "id", request_id)
                for choice in chunk.choices:
                    if choice.delta.content:
                        if ttft is None:
                            ttft = (time.perf_counter()-start)*1000
                        yield {"type": "delta", "text": choice.delta.content, "run_id": run_id}
            status = "success"
        except Exception as exc:
            status = "error"
            yield {"type": "error", "error_type": type(exc).__name__, "run_id": run_id,
                   "message": "Provider request failed; inspect connection settings and request telemetry."}
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()
            stats = self._stats(run_id, start, usage, model, request_id, status, ttft)
        yield {"type": "stats", **stats}

    def complete(self, messages: list[dict], *, run_id: str, tools=None, schema=None) -> dict:
        start = time.perf_counter()
        response = None
        status = "error"
        try:
            params = dict(model=self.connection.model, messages=messages,
                          max_completion_tokens=self.connection.max_output_tokens)
            if tools:
                params["tools"] = tools
            if schema:
                params["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "result", "strict": True, "schema": schema}}
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            if message.refusal:
                raise ValueError("The model declined this request")
            status = "success"
            return message.model_dump(exclude_none=True)
        finally:
            self._stats(run_id, start, getattr(response, "usage", None),
                getattr(response, "model", None), getattr(response, "_request_id", None), status)

    def structured(self, messages, output_type, *, run_id):
        result = self.complete(messages, run_id=run_id, schema=output_type.model_json_schema())
        return output_type.model_validate_json(result.get("content") or "")


class ChatSession:
    def __init__(self):
        self.history = deque(maxlen=10)

    def turn(self, text: str, llm: LLM):
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": "You are a concise, helpful technical assistant."},
                    *self.history]
        answer = ""
        success = False
        for event in llm.stream(messages):
            if event["type"] == "delta":
                answer += event["text"]
            if event["type"] == "stats":
                success = event["status"] == "success"
            yield event
        if success:
            self.history.append({"role": "assistant", "content": answer})
        elif self.history and self.history[-1]["role"] == "user":
            self.history.pop()


def format_stats(event: dict) -> str:
    cost = event.get("cost_usd")
    return (f"[stats] prompt={event.get('prompt_tokens')} completion={event.get('completion_tokens')} "
            f"cost={'unknown' if cost is None else f'${cost:.8f} (estimate)'} "
            f"latency={event['latency_ms']:.0f} ms")
