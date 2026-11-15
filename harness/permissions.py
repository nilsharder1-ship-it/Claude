"""Permission gate between the model and your machine.

Modes:
  ask   - prompt before every write/dangerous tool (default)
  auto  - run safe+write tools freely, prompt for dangerous ones (shell, mouse, keyboard)
  yolo  - never prompt. Only use in a VM / sandbox.
"""
from __future__ import annotations

import json

RISK_LEVEL = {"safe": 0, "write": 1, "dangerous": 2}
MODE_THRESHOLD = {"ask": 1, "auto": 2, "yolo": 99}


class Permissions:
    def __init__(self, mode: str = "ask", prompt_fn=input):
        if mode not in MODE_THRESHOLD:
            raise ValueError(f"mode must be one of {list(MODE_THRESHOLD)}")
        self.mode = mode
        self.prompt_fn = prompt_fn
        self.always_allow: set[str] = set()

    def check(self, tool_name: str, risk: str, args: dict) -> tuple[bool, str]:
        """Returns (allowed, feedback). Feedback is passed back to the model on denial."""
        if RISK_LEVEL.get(risk, 2) < MODE_THRESHOLD[self.mode] or tool_name in self.always_allow:
            return True, ""
        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 400:
            preview = preview[:400] + "..."
        ans = self.prompt_fn(f"\n  Allow {tool_name} {preview}?  [y]es / [n]o / [a]lways / or type feedback: ").strip()
        if ans.lower() in ("y", "yes", ""):
            return True, ""
        if ans.lower() in ("a", "always"):
            self.always_allow.add(tool_name)
            return True, ""
        if ans.lower() in ("n", "no"):
            return False, "The user denied this action."
        return False, f"The user denied this action and said: {ans}"
