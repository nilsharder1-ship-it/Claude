"""Thin client for any OpenAI-compatible chat endpoint (LM Studio, Ollama, llama.cpp, vLLM).

LM Studio's local server: Developer tab -> Start Server -> http://localhost:1234/v1
Only the standard library is used so the harness has no hard dependency on the openai package.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)
    finish_reason: str | None = None


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str | None = None,
                 api_key: str = "lm-studio", temperature: float = 0.2, max_tokens: int = 2048,
                 timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model = model or self._first_loaded_model()

    # -- HTTP -----------------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST" if data else "GET",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM server error {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach {self.base_url}. Is the LM Studio server running? ({e.reason})") from e

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._request("/models").get("data", [])]

    def _first_loaded_model(self) -> str:
        models = [m for m in self.list_models() if "embed" not in m.lower()]
        if not models:
            raise RuntimeError("No model available. Load a model in LM Studio first.")
        return models[0]

    # -- Chat -----------------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._request("/chat/completions", payload)
        choice = data["choices"][0]
        msg = choice["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(id=tc.get("id") or f"call_{len(calls)}", name=fn.get("name", ""),
                                  arguments=args))
        return LLMResponse(content=msg.get("content") or "", tool_calls=calls, raw_message=msg,
                           finish_reason=choice.get("finish_reason"))
