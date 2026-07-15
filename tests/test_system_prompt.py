from __future__ import annotations

from py_agent.utils.system_prompt import build_system_prompt
from py_agent.utils.types import AgentTool


def _tool(name: str, description: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update):
        raise NotImplementedError

    return AgentTool(name=name, description=description, parameters={"type": "object"}, execute=execute)


def test_no_leading_indentation():
    prompt = build_system_prompt("/home/project", [])
    for line in prompt.splitlines():
        if line.strip():
            assert line[0] != " " and line[0] != "\t", f"line has leading whitespace: {line!r}"


def test_contains_tools():
    tools = [_tool("read", "Read a file"), _tool("grep", "Search files")]
    prompt = build_system_prompt("/home/project", tools)
    assert "- read: Read a file" in prompt
    assert "- grep: Search files" in prompt


def test_contains_sections():
    prompt = build_system_prompt("/home/project", [])
    assert "Available tools:" in prompt
    assert "Guidelines:" in prompt
    assert "todo tool" in prompt
    assert "verify" in prompt.lower()


def test_cwd_normalized():
    prompt = build_system_prompt("C:\\Users\\dev\\project", [])
    assert "Working directory: C:/Users/dev/project" in prompt
    assert "\\" not in prompt
