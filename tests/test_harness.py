"""Tests run against a fake OpenAI-compatible server (no LM Studio needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from harness import Agent, AgentConfig, LLMClient, Permissions, ToolRegistry, SYSTEM_TOOLS, tool
from harness.agent import parse_text_tool_calls
from harness.tools.base import ToolResult


class FakeServer:
    """Replays scripted assistant messages and records the requests it received."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"data": [{"id": "text-embedding-x"}, {"id": "qwen3-8b"}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                msg = outer.script.pop(0)
                self._send({"choices": [{"message": msg, "finish_reason": "stop"}]})

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"


def make_agent(server, mode="auto", perms="yolo", extra=()):
    reg = ToolRegistry()
    reg.extend(SYSTEM_TOOLS)
    reg.extend(extra)
    return Agent(llm=LLMClient(base_url=server.url), tools=reg, permissions=Permissions(perms),
                 config=AgentConfig(tool_mode=mode))


def native_call(name, args, cid="c1"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_picks_first_non_embedding_model():
    s = FakeServer([])
    assert LLMClient(base_url=s.url).model == "qwen3-8b"


def test_native_tool_loop(tmp_path):
    f = tmp_path / "hello.txt"
    s = FakeServer([native_call("write_file", {"path": str(f), "content": "hi"}),
                    {"role": "assistant", "content": "Done."}])
    out = make_agent(s).run("write a file")
    assert out == "Done." and f.read_text() == "hi"
    second = s.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool" and "Wrote 2 chars" in second[-1]["content"]


def test_text_tool_call_fallback_in_auto_mode():
    s = FakeServer([{"role": "assistant", "content":
                     '<think>hmm</think>Let me check.\n<tool_call>{"name": "run_shell", '
                     '"arguments": {"command": "echo 42"}}</tool_call>'},
                    {"role": "assistant", "content": "It printed 42."}])
    assert make_agent(s).run("run echo") == "It printed 42."
    last = s.requests[1]["messages"][-1]
    assert last["role"] == "user" and "42" in last["content"]
    assert "<think>" not in s.requests[1]["messages"][-2]["content"]


def test_prompt_mode_puts_tools_in_system_prompt():
    s = FakeServer([{"role": "assistant", "content": "ok"}])
    make_agent(s, mode="prompt").run("hi")
    req = s.requests[0]
    assert "tools" not in req and "run_shell" in req["messages"][0]["content"]


def test_permission_denial_feedback_reaches_model(tmp_path):
    s = FakeServer([native_call("write_file", {"path": str(tmp_path / "x"), "content": "y"}),
                    {"role": "assistant", "content": "ok, I won't."}])
    agent = make_agent(s, perms="ask")
    agent.permissions.prompt_fn = lambda _: "use another folder"
    agent.run("write")
    assert not (tmp_path / "x").exists()
    assert "use another folder" in s.requests[1]["messages"][-1]["content"]


def test_images_attached_and_pruned():
    @tool()
    def snap() -> ToolResult:
        """fake screenshot"""
        return ToolResult("shot", images=["AAAA"])

    s = FakeServer([native_call("snap", {}, f"c{i}") for i in range(4)] + [{"role": "assistant", "content": "done"}])
    make_agent(s, extra=[snap]).run("look")
    msgs = s.requests[-1]["messages"]
    with_img = [m for m in msgs if isinstance(m["content"], list)]
    assert len(with_img) == 2  # keep_images default
    assert any(m["content"] == "[old screenshot removed to save context]" for m in msgs)


def test_context_trimming_keeps_tool_pairs_valid():
    s = FakeServer([native_call("run_shell", {"command": "python3 -c \"print('x'*5000)\""}, f"c{i}")
                    for i in range(6)] + [{"role": "assistant", "content": "done"}])
    agent = make_agent(s)
    agent.config.context_chars = 12_000
    agent.run("spam")
    msgs = s.requests[-1]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            j = i - 1
            while msgs[j]["role"] == "tool":
                j -= 1
            assert msgs[j]["role"] == "assistant" and msgs[j].get("tool_calls")


def test_unknown_tool_and_bad_args_are_reported():
    s = FakeServer([native_call("nope", {}), native_call("read_file", {"wrong": 1}),
                    {"role": "assistant", "content": "x"}])
    make_agent(s).run("t")
    assert "Unknown tool" in s.requests[1]["messages"][-1]["content"]
    assert "Invalid arguments" in s.requests[2]["messages"][-1]["content"]


def test_parse_variants():
    known = {"click"}
    assert parse_text_tool_calls('```json\n{"name": "click", "arguments": {"x": 1, "y": 2}}\n```', known)[0].arguments == {"x": 1, "y": 2}
    assert parse_text_tool_calls('<tool_call>{"name": "click", "parameters": {"x": 1}}', known)[0].name == "click"
    assert parse_text_tool_calls('<tool_call>{"name": "other"}</tool_call>', known) == []
