"""Agent harness for local LLMs served through an OpenAI-compatible API (LM Studio, Ollama, llama.cpp)."""
from .agent import Agent, AgentConfig
from .llm import LLMClient
from .permissions import Permissions
from .tools import SYSTEM_TOOLS, Tool, ToolRegistry, ToolResult, tool

__all__ = ["Agent", "AgentConfig", "LLMClient", "Permissions", "SYSTEM_TOOLS", "Tool", "ToolRegistry",
           "ToolResult", "tool"]
