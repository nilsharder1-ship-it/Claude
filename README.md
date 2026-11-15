# Local Agent Harness

A small, readable agent harness for **local models** (LM Studio, Ollama, llama.cpp, vLLM). It gives
a model tools (shell, files, and optionally **computer use**: screenshot, mouse and keyboard) and
runs the agent loop: model → tool call → permission check → run it → send the result back → repeat.

Only the Python standard library is needed for the core. Computer use needs `pyautogui`, `pillow`
and `mss`.

## Quick start

1. **LM Studio:** load a model, then go to *Developer* → *Start Server* (default `http://localhost:1234`).
   Set the model's **context length** high enough (16k+; 32k is better for computer use).
2. Install and run:

```bash
pip install -e .                 # core (no dependencies)
pip install -e ".[computer]"     # + computer use

python -m harness --list-models
python -m harness                                    # REPL with shell + file tools
python -m harness --task "How much free disk space do I have?"
python -m harness --computer --permissions auto      # computer use (needs a vision model)
```

Other servers: `--base-url http://localhost:11434/v1` (Ollama), `--base-url http://localhost:8080/v1` (llama.cpp).

## Choosing a model

| Use | What works well locally |
|---|---|
| Tool use (shell/files) | Qwen3 (8B+), Qwen2.5-Coder, GPT-OSS 20B, Mistral Small 3.x, Llama 3.1/3.3 |
| Computer use (needs vision) | Qwen2.5-VL / Qwen3-VL (7B+), Gemma 3 (12B+), Mistral Small 3.1+ (vision); UI-TARS for GUI-only tasks |

LM Studio shows a hammer icon on models that support native tool calling. If a model has no
hammer, or produces tool calls as text, use `--tool-mode prompt`.

## Tool-calling modes (`--tool-mode`)

- `auto` *(default)*: uses the native OpenAI `tools` API, and also accepts
  `<tool_call>{...}</tool_call>` or fenced JSON in plain text, which many local models emit.
- `native`: only the native API.
- `prompt`: tools are described in the system prompt and the harness parses the model's text.
  Works with almost any instruct model.

## Permissions (`--permissions`)

- `ask` *(default)*: confirm every file write, shell command and mouse/keyboard action. At the
  prompt, `y` allows, `a` always allows that tool, `n` denies, or type feedback, which is sent to
  the model.
- `auto`: file reads/writes run freely; shell and mouse/keyboard still ask.
- `yolo`: never asks. **Use only in a VM or sandbox.** A local model clicking around your real
  desktop can delete files, send messages or buy things.

Emergency stop for computer use: move the mouse hard into a **screen corner** (pyautogui failsafe),
or press Ctrl+C in the terminal.

## How computer use works

- Screenshots are downscaled to `--screen-width` (default 1280px) to save tokens. The model
  gives coordinates in screenshot pixels; the harness maps them back to the real screen,
  including HiDPI/Retina scaling.
- A labeled red grid (every 100px) and a blue cursor circle are drawn on each screenshot. This
  helps small VLMs aim much better. Disable it with `--no-grid` for models trained on clean
  screenshots (e.g. UI-TARS).
- After each action the harness waits briefly and sends a new screenshot, so the model can check
  what happened.
- Only the last `--keep-images` (default 2) screenshots stay in context; older ones are replaced
  with a placeholder. Old turns are trimmed when the history exceeds `--context-chars`.
- If the model repeats the same call 3 times, the harness tells it to try something else.

macOS: give your terminal **Screen Recording** and **Accessibility** permissions (System Settings →
Privacy & Security). Linux: needs an X11 session (Wayland support in pyautogui is limited).

## Using it from your own code

```python
from harness import Agent, AgentConfig, LLMClient, Permissions, ToolRegistry, SYSTEM_TOOLS, tool

@tool(params={"city": "City name"})
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 21°C in {city}"

reg = ToolRegistry()
reg.extend(SYSTEM_TOOLS)
reg.register(get_weather)

agent = Agent(llm=LLMClient(model="qwen3-8b"), tools=reg,
              permissions=Permissions("auto"), config=AgentConfig(max_steps=20))
print(agent.run("What's the weather in Hamburg? Save it to weather.txt"))
```

`@tool` builds the JSON schema from the type hints and docstring. Set `risk="write"` or
`risk="dangerous"` so the permission gate asks before it runs. Return a `ToolResult(text,
images=[b64])` to send images back to the model. Pass `on_event=` to `Agent` to stream steps to
your own UI.

## Layout

```
harness/
  llm.py          OpenAI-compatible client (stdlib only)
  agent.py        agent loop, text tool-call parser, context management
  permissions.py  ask / auto / yolo gate
  tools/base.py   @tool decorator, registry, ToolResult
  tools/system.py run_shell, read_file, write_file, edit_file, list_files
  tools/computer.py screenshot, click, move_mouse, drag, scroll, type_text, press_keys, wait
  cli.py          REPL / one-shot CLI
tests/            runs against a fake OpenAI server; no model needed
```

Run the tests with `pip install pytest && pytest`.

## Tips for local models

- Keep temperature low (0.1–0.3) for tool use.
- Small models do better with **fewer tools**. Remove tools you don't need from the registry.
- For long computer-use tasks, give the model a clear, step-by-step task description.
- Reasoning models (Qwen3, GPT-OSS): `<think>` blocks are removed from history to save context.
