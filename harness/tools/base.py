"""Tool registry. A tool is a Python function plus a JSON schema the model sees."""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class ToolResult:
    """What a tool returns. `images` are base64 PNG/JPEG strings sent to vision models."""
    text: str
    images: list[str] = field(default_factory=list)
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict
    risk: str = "safe"  # "safe" | "write" | "dangerous" -> drives permission prompts

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def run(self, **kwargs) -> ToolResult:
        try:
            out = self.func(**kwargs)
        except TypeError as e:
            return ToolResult(f"Invalid arguments for {self.name}: {e}", is_error=True)
        except Exception as e:  # tools must never crash the loop
            return ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        if isinstance(out, ToolResult):
            return out
        return ToolResult(out if isinstance(out, str) else json.dumps(out, default=str))


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def extend(self, tools: list[Tool]):
        for t in tools:
            self.register(t)


def tool(risk: str = "safe", name: str | None = None, params: dict[str, str] | None = None):
    """Decorator: build a Tool from a function's signature, type hints and docstring.

    `params` maps argument name -> description shown to the model.
    """
    params = params or {}

    def wrap(fn: Callable) -> Tool:
        hints = get_type_hints(fn)
        props, required = {}, []
        for pname, p in inspect.signature(fn).parameters.items():
            prop = {"type": _PY_TO_JSON.get(hints.get(pname, str), "string")}
            if pname in params:
                prop["description"] = params[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                prop["default"] = p.default
            props[pname] = prop
        return Tool(name=name or fn.__name__, description=inspect.cleandoc(fn.__doc__ or ""), func=fn,
                    parameters={"type": "object", "properties": props, "required": required}, risk=risk)

    return wrap
