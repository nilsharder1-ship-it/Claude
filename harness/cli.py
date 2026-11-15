"""Interactive CLI.

  python -m harness                       # chat REPL with shell + file tools
  python -m harness --computer            # add screenshot/mouse/keyboard tools (needs a vision model)
  python -m harness --task "..."          # one-shot
  python -m harness --list-models
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent, AgentConfig
from .llm import LLMClient
from .permissions import Permissions
from .tools import SYSTEM_TOOLS, ToolRegistry

DIM, BOLD, CYAN, YEL, RED, RST = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def printer(kind: str, d: dict):
    if kind == "assistant":
        if d["content"]:
            print(f"\n{BOLD}{CYAN}assistant{RST} {DIM}(step {d['step']}){RST}\n{d['content']}")
    elif kind == "tool_start":
        c = d["call"]
        args = json.dumps(c.arguments, ensure_ascii=False)
        print(f"{YEL}-> {c.name}{RST} {DIM}{args[:300]}{RST}")
    elif kind == "tool_end":
        r = d["result"]
        text = r.text if len(r.text) < 800 else r.text[:800] + " ..."
        color = RED if r.is_error else DIM
        print(f"{color}{text}{RST}" + (f" {DIM}[+{len(r.images)} image]{RST}" if r.images else ""))
    elif kind == "denied":
        print(f"{RED}denied: {d['feedback']}{RST}")


def build_agent(args) -> Agent:
    llm = LLMClient(base_url=args.base_url, model=args.model, temperature=args.temperature,
                    max_tokens=args.max_tokens)
    reg = ToolRegistry()
    reg.extend(SYSTEM_TOOLS)
    if args.computer:
        from .tools.computer import Screen, make_computer_tools
        reg.extend(make_computer_tools(Screen(target_width=args.screen_width, grid=not args.no_grid)))
    cfg = AgentConfig(tool_mode=args.tool_mode, max_steps=args.max_steps, context_chars=args.context_chars,
                      keep_images=args.keep_images)
    print(f"{DIM}model: {llm.model} | tools: {', '.join(reg.tools)} | mode: {args.tool_mode} | "
          f"permissions: {args.permissions}{RST}")
    return Agent(llm=llm, tools=reg, permissions=Permissions(args.permissions), config=cfg, on_event=printer)


def main(argv=None):
    p = argparse.ArgumentParser(prog="harness", description="Agent harness for local LLMs (LM Studio etc.)")
    p.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible endpoint")
    p.add_argument("--model", help="Model id (default: first loaded model)")
    p.add_argument("--task", help="Run a single task and exit")
    p.add_argument("--computer", action="store_true", help="Enable computer-use tools")
    p.add_argument("--tool-mode", choices=["auto", "native", "prompt"], default="auto")
    p.add_argument("--permissions", choices=["ask", "auto", "yolo"], default="ask")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--context-chars", type=int, default=60_000,
                   help="~4x your model's context length in tokens, minus room for the reply")
    p.add_argument("--keep-images", type=int, default=2)
    p.add_argument("--screen-width", type=int, default=1280, help="Width screenshots are scaled to")
    p.add_argument("--no-grid", action="store_true", help="Don't draw coordinate grid on screenshots")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args(argv)

    try:
        if args.list_models:
            print("\n".join(LLMClient(base_url=args.base_url, model="-").list_models()))
            return
        agent = build_agent(args)
    except RuntimeError as e:
        sys.exit(f"{RED}{e}{RST}")

    if args.task:
        agent.run(args.task)
        return
    print(f"{DIM}Type a task. Commands: /reset, /quit{RST}")
    while True:
        try:
            task = input(f"\n{BOLD}you>{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        if task in ("/quit", "/exit"):
            break
        if task == "/reset":
            agent = build_agent(args)
            continue
        try:
            agent.run(task)
        except KeyboardInterrupt:
            print(f"\n{RED}interrupted{RST}")
        except RuntimeError as e:
            print(f"{RED}{e}{RST}")


if __name__ == "__main__":
    main()
