"""Shell and filesystem tools."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import ToolResult, tool

MAX_OUTPUT = 12_000  # chars; local models have small context windows


def _truncate(s: str, limit: int = MAX_OUTPUT) -> str:
    if len(s) <= limit:
        return s
    half = limit // 2
    return f"{s[:half]}\n... [{len(s) - limit} chars truncated] ...\n{s[-half:]}"


@tool(risk="dangerous", params={"command": "Shell command to run", "timeout": "Seconds before the command is killed"})
def run_shell(command: str, timeout: int = 60) -> ToolResult:
    """Run a shell command and return stdout, stderr and the exit code."""
    try:
        p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout,
                           cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        return ToolResult(f"Command timed out after {timeout}s", is_error=True)
    out = f"exit_code: {p.returncode}\n"
    if p.stdout:
        out += f"stdout:\n{p.stdout}\n"
    if p.stderr:
        out += f"stderr:\n{p.stderr}\n"
    return ToolResult(_truncate(out), is_error=p.returncode != 0)


@tool(params={"path": "File path", "offset": "First line (1-based)", "limit": "Max lines to read"})
def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
    """Read a text file. Returns numbered lines."""
    lines = Path(path).expanduser().read_text(errors="replace").splitlines()
    start = max(offset, 1) - 1
    chunk = lines[start:start + limit]
    body = "\n".join(f"{i + start + 1:>5}| {line}" for i, line in enumerate(chunk))
    more = f"\n[{len(lines) - start - len(chunk)} more lines]" if start + len(chunk) < len(lines) else ""
    return _truncate(body + more)


@tool(risk="write", params={"path": "File path", "content": "Full new file content"})
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {p}"


@tool(risk="write", params={"path": "File path", "old": "Exact text to replace (must be unique)",
                            "new": "Replacement text"})
def edit_file(path: str, old: str, new: str) -> ToolResult:
    """Replace one exact, unique occurrence of `old` with `new` in a file."""
    p = Path(path).expanduser()
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        return ToolResult(f"`old` found {n} times; it must occur exactly once. Add more context.", is_error=True)
    p.write_text(text.replace(old, new, 1))
    return ToolResult(f"Edited {p}")


@tool(params={"path": "Directory", "pattern": "Glob pattern, e.g. **/*.py"})
def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files in a directory matching a glob pattern (max 300 entries)."""
    items = sorted(Path(path).expanduser().glob(pattern))[:300]
    return "\n".join(f"{'d' if i.is_dir() else 'f'} {i}" for i in items) or "(no matches)"


SYSTEM_TOOLS = [run_shell, read_file, write_file, edit_file, list_files]
