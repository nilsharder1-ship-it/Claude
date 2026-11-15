"""The agent loop: model -> tool calls -> permission check -> execute -> feed results back.

Two tool-calling strategies:
  native - uses the OpenAI `tools` API. LM Studio supports this for models with a tool-capable
           chat template (Qwen 2.5/3, Llama 3.1+, Mistral, Gemma 3, GPT-OSS ...).
  prompt - tools are described in the system prompt and the model writes
           <tool_call>{"name": ..., "arguments": {...}}</tool_call>. Works with almost any model.
  auto   - native, but also parse <tool_call> blocks from plain text (common with local models
           whose template LM Studio doesn't fully parse).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .llm import LLMClient, LLMResponse, ToolCall
from .permissions import Permissions
from .tools.base import ToolRegistry, ToolResult

DEFAULT_SYSTEM = """You are an autonomous agent running on the user's computer with access to tools.
Work step by step. Before acting, briefly think about what you need to do. Call one tool at a time
when actions depend on each other, then inspect the result before continuing.
For computer use: always take a screenshot first, act, then verify the result from the new screenshot.
Coordinates refer to the pixels of the latest screenshot. Prefer keyboard shortcuts when reliable.
When the task is complete, reply with a final answer and no tool call."""

PROMPT_TOOLS_TEMPLATE = """

# Tools
You can call these tools:
{tools}

To call a tool, reply with exactly one block in this format and then stop:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>
The result will be sent back to you. When you are done, answer without any <tool_call> block."""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.S)


def parse_text_tool_calls(text: str, known: set[str]) -> list[ToolCall]:
    """Extract tool calls a model wrote as text instead of using the native API."""
    blobs = TOOL_CALL_RE.findall(text) or FENCE_RE.findall(text)
    calls = []
    for blob in blobs:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # tolerate trailing junk: cut at the last closing brace
            try:
                obj = json.loads(blob[:blob.rfind("}") + 1])
            except json.JSONDecodeError:
                continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name in known:
            calls.append(ToolCall(id=f"call_{len(calls)}", name=name, arguments=args))
    return calls


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class AgentConfig:
    tool_mode: str = "auto"          # native | prompt | auto
    max_steps: int = 40
    keep_images: int = 2             # only the newest N screenshots stay in context
    context_chars: int = 60_000      # rough budget (~4 chars/token); trims old turns beyond it
    system_prompt: str = DEFAULT_SYSTEM
    strip_thinking: bool = True      # drop <think> blocks from history to save context


@dataclass
class Agent:
    llm: LLMClient
    tools: ToolRegistry
    permissions: Permissions = field(default_factory=Permissions)
    config: AgentConfig = field(default_factory=AgentConfig)
    on_event: Callable[[str, dict], None] = lambda kind, data: None
    messages: list[dict] = field(default_factory=list)

    def __post_init__(self):
        system = self.config.system_prompt
        if self.config.tool_mode == "prompt":
            listing = "\n".join(json.dumps(s["function"], ensure_ascii=False) for s in self.tools.schemas())
            system += PROMPT_TOOLS_TEMPLATE.format(tools=listing)
        self.messages = [{"role": "system", "content": system}]

    # -- public ---------------------------------------------------------------
    def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        recent: list[str] = []
        for step in range(1, self.config.max_steps + 1):
            self._manage_context()
            resp = self._call_model()
            content = THINK_RE.sub("", resp.content).strip() if self.config.strip_thinking else resp.content
            calls = resp.tool_calls
            if not calls and self.config.tool_mode in ("auto", "prompt"):
                calls = parse_text_tool_calls(content, set(self.tools.tools))
            native = bool(resp.tool_calls) and self.config.tool_mode != "prompt"

            self.on_event("assistant", {"step": step, "content": content, "calls": calls})
            self._record_assistant(content, calls, native)
            if not calls:
                return content

            for call in calls:
                sig = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                recent = (recent + [sig])[-6:]
                result = self._execute(call)
                if recent.count(sig) >= 3:
                    result.text += ("\n[harness] You repeated the same call 3 times. It is not working; "
                                    "try a different approach.")
                self._record_result(call, result, native)
        return "[harness] Stopped: reached max_steps without a final answer."

    # -- internals ------------------------------------------------------------
    def _call_model(self) -> LLMResponse:
        tools = self.tools.schemas() if self.config.tool_mode != "prompt" else None
        return self.llm.chat(self.messages, tools=tools)

    def _execute(self, call: ToolCall) -> ToolResult:
        t = self.tools.get(call.name)
        if t is None:
            return ToolResult(f"Unknown tool '{call.name}'. Available: {', '.join(self.tools.tools)}",
                              is_error=True)
        if "_raw" in call.arguments:
            return ToolResult(f"Could not parse arguments as JSON: {call.arguments['_raw']}", is_error=True)
        ok, feedback = self.permissions.check(call.name, t.risk, call.arguments)
        if not ok:
            self.on_event("denied", {"call": call, "feedback": feedback})
            return ToolResult(feedback, is_error=True)
        self.on_event("tool_start", {"call": call})
        result = t.run(**call.arguments)
        self.on_event("tool_end", {"call": call, "result": result})
        return result

    def _record_assistant(self, content: str, calls: list[ToolCall], native: bool):
        msg: dict = {"role": "assistant", "content": content}
        if native and calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                 for c in calls]
        self.messages.append(msg)

    def _record_result(self, call: ToolCall, result: ToolResult, native: bool):
        text = ("ERROR: " if result.is_error else "") + result.text
        if native:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        else:
            self.messages.append({"role": "user", "content": f"<tool_result name=\"{call.name}\">\n{text}\n</tool_result>"})
        # Tool messages can't carry images in the OpenAI format, so attach them as a user message.
        if result.images:
            parts = [{"type": "text", "text": f"[screenshot from {call.name}]"}]
            parts += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                      for b64 in result.images]
            self.messages.append({"role": "user", "content": parts})

    def _manage_context(self):
        # 1) keep only the newest N images
        seen = 0
        for msg in reversed(self.messages):
            if isinstance(msg["content"], list):
                imgs = [p for p in msg["content"] if p.get("type") == "image_url"]
                if imgs:
                    seen += len(imgs)
                    if seen > self.config.keep_images:
                        msg["content"] = "[old screenshot removed to save context]"
        # 2) drop oldest turns when over budget (keep system + first user task)
        def size(m):
            c = m["content"]
            return len(c) if isinstance(c, str) else 3000 * sum(p.get("type") == "image_url" for p in c)
        while sum(map(size, self.messages)) > self.config.context_chars:
            # A "turn" = one assistant message + the tool results after it. Drop the oldest whole turn
            # (never the latest one), so a tool result never loses its assistant tool_calls message.
            starts = [i for i, m in enumerate(self.messages) if i >= 2 and m["role"] == "assistant"]
            if len(starts) < 2:
                break
            del self.messages[starts[0]:starts[1]]
            marker = "\n\n[harness: earlier steps were trimmed to save context]"
            if isinstance(self.messages[1]["content"], str) and marker not in self.messages[1]["content"]:
                self.messages[1]["content"] += marker
