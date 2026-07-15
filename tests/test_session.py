from __future__ import annotations

from py_agent.utils.session import load_session, save_session
from py_agent.utils.types import (
    EMPTY_USAGE,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    UserMessage,
)


def test_roundtrip_all_roles(tmp_path):
    messages = [
        UserMessage(content=[TextContent(text="do it")]),
        AssistantMessage(
            content=[TextContent(text="okay"), ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
            model="faux", provider="faux", usage=EMPTY_USAGE, stop_reason="tool_use",
        ),
        ToolResultMessage(tool_call_id="c1", content=[ToolResultContent(text="out")], is_error=False),
    ]
    path = str(tmp_path / "session.jsonl")
    save_session(path, messages)
    loaded = load_session(path)

    assert len(loaded) == 3
    assert loaded[0].role == "user"
    assert loaded[1].role == "assistant"
    assert isinstance(loaded[1].content[1], ToolCall)
    assert loaded[1].content[1].arguments == {"command": "ls"}
    assert loaded[2].role == "tool_result"
    assert loaded[2].tool_call_id == "c1"
