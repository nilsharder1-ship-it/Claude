"""Computer-use tools: screenshot, mouse, keyboard.

Requires: pip install pyautogui pillow mss
The model only ever sees a downscaled screenshot (default 1280px wide). It gives coordinates in
that screenshot's pixel space; we scale them back to the real screen. This keeps image tokens low
(important for local VLMs) and avoids HiDPI/Retina confusion.

Safety: pyautogui's FAILSAFE is on, so slamming the mouse into a screen corner aborts any action.
"""
from __future__ import annotations

import base64
import io
import time

from .base import Tool, ToolResult, tool


class Screen:
    def __init__(self, target_width: int = 1280, grid: bool = True, settle: float = 0.6):
        import pyautogui  # imported lazily so the harness runs without a display

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        self.pg = pyautogui
        self.target_width = target_width
        self.grid = grid
        self.settle = settle
        self.scale = 1.0  # screenshot px -> logical screen px

    # -- capture --------------------------------------------------------------
    def _grab(self):
        from PIL import Image
        try:
            import mss
            with mss.mss() as sct:
                raw = sct.grab(sct.monitors[1])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except ImportError:
            img = self.pg.screenshot()
        # Logical size (what pyautogui clicks in) may differ from pixel size on HiDPI displays.
        logical_w, logical_h = self.pg.size()
        if img.size != (logical_w, logical_h):
            img = img.resize((logical_w, logical_h))
        return img

    def capture(self) -> tuple[str, tuple[int, int]]:
        from PIL import Image, ImageDraw
        img = self._grab().convert("RGB")
        w, h = img.size
        self.scale = w / self.target_width if w > self.target_width else 1.0
        if self.scale != 1.0:
            img = img.resize((round(w / self.scale), round(h / self.scale)), Image.LANCZOS)
        if self.grid:
            d = ImageDraw.Draw(img)
            step = 100
            for x in range(step, img.width, step):
                d.line([(x, 0), (x, img.height)], fill=(255, 0, 0), width=1)
                d.text((x + 2, 2), str(x), fill=(255, 0, 0))
            for y in range(step, img.height, step):
                d.line([(0, y), (img.width, y)], fill=(255, 0, 0), width=1)
                d.text((2, y + 2), str(y), fill=(255, 0, 0))
        mx, my = self.pg.position()
        d = ImageDraw.Draw(img)
        cx, cy = mx / self.scale, my / self.scale
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(0, 120, 255), width=3)  # show cursor
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode(), img.size

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        return round(x * self.scale), round(y * self.scale)

    def shot(self, note: str) -> ToolResult:
        time.sleep(self.settle)
        b64, (w, h) = self.capture()
        return ToolResult(f"{note}\nScreenshot attached ({w}x{h}). Red grid lines every 100px are labeled "
                          f"with coordinates; the blue circle is the mouse cursor.", images=[b64])


def make_computer_tools(screen: Screen, screenshot_after_action: bool = True) -> list[Tool]:
    pg = screen.pg

    def after(note: str) -> ToolResult:
        return screen.shot(note) if screenshot_after_action else ToolResult(note)

    @tool(name="screenshot")
    def screenshot() -> ToolResult:
        """Take a screenshot of the screen. Use it to see the current state before acting."""
        return screen.shot("Current screen.")

    @tool(name="click", risk="dangerous", params={
        "x": "X in screenshot pixels", "y": "Y in screenshot pixels",
        "button": "left | right | middle", "clicks": "1 for single, 2 for double click"})
    def click(x: int, y: int, button: str = "left", clicks: int = 1) -> ToolResult:
        """Move the mouse to (x, y) in screenshot coordinates and click."""
        sx, sy = screen.to_screen(x, y)
        pg.click(sx, sy, clicks=clicks, button=button, interval=0.1)
        return after(f"Clicked {button} x{clicks} at ({x},{y}).")

    @tool(name="move_mouse", risk="dangerous", params={"x": "X in screenshot pixels", "y": "Y in screenshot pixels"})
    def move_mouse(x: int, y: int) -> ToolResult:
        """Move the mouse without clicking (e.g. to hover)."""
        pg.moveTo(*screen.to_screen(x, y), duration=0.2)
        return after(f"Moved mouse to ({x},{y}).")

    @tool(name="drag", risk="dangerous", params={"x1": "Start X", "y1": "Start Y", "x2": "End X", "y2": "End Y"})
    def drag(x1: int, y1: int, x2: int, y2: int) -> ToolResult:
        """Drag with the left mouse button from (x1,y1) to (x2,y2) in screenshot coordinates."""
        pg.moveTo(*screen.to_screen(x1, y1))
        pg.dragTo(*screen.to_screen(x2, y2), duration=0.5, button="left")
        return after(f"Dragged ({x1},{y1}) -> ({x2},{y2}).")

    @tool(name="scroll", risk="dangerous", params={
        "amount": "Positive scrolls up, negative scrolls down (wheel clicks)",
        "x": "Optional X to scroll at", "y": "Optional Y to scroll at"})
    def scroll(amount: int, x: int = -1, y: int = -1) -> ToolResult:
        """Scroll the mouse wheel, optionally at a position."""
        if x >= 0 and y >= 0:
            pg.moveTo(*screen.to_screen(x, y))
        pg.scroll(amount)
        return after(f"Scrolled {amount}.")

    @tool(name="type_text", risk="dangerous", params={"text": "Text to type"})
    def type_text(text: str) -> ToolResult:
        """Type text at the current keyboard focus. Click the target field first."""
        try:
            pg.write(text, interval=0.01)
        except Exception:
            # pg.write only handles ASCII; fall back to clipboard paste for Unicode (umlauts etc.)
            import pyperclip
            pyperclip.copy(text)
            pg.hotkey("command" if _is_mac() else "ctrl", "v")
        return after(f"Typed {len(text)} chars.")

    @tool(name="press_keys", risk="dangerous", params={
        "keys": "Key or combo joined by '+', e.g. 'enter', 'ctrl+c', 'cmd+space', 'alt+tab'"})
    def press_keys(keys: str) -> ToolResult:
        """Press a key or key combination."""
        parts = [k.strip().lower() for k in keys.split("+")]
        parts = ["command" if k in ("cmd", "super", "win") and _is_mac() else
                 "win" if k in ("cmd", "super") else k for k in parts]
        pg.hotkey(*parts) if len(parts) > 1 else pg.press(parts[0])
        return after(f"Pressed {keys}.")

    @tool(name="wait", params={"seconds": "Seconds to wait"})
    def wait(seconds: float = 1.0) -> ToolResult:
        """Wait for the UI to load, then take a screenshot."""
        time.sleep(min(seconds, 30))
        return screen.shot(f"Waited {seconds}s.")

    return [screenshot, click, move_mouse, drag, scroll, type_text, press_keys, wait]


def _is_mac() -> bool:
    import sys
    return sys.platform == "darwin"
