from __future__ import annotations

from conftest import make_assistant_message, make_faux_stream_fn

from py_agent.tools.builtin import create_task_tool
from py_agent.utils.types import TextContent


async def test_task_returns_final_text(tmp_path):
    stream_fn = make_faux_stream_fn(
        [make_assistant_message([TextContent(text="X is defined in a.py")], stop_reason="end_turn")]
    )
    tool = create_task_tool(str(tmp_path), "faux", stream_fn, api_key=None)
    result = await tool.execute("c1", {"task": "find X"}, None, None)
    text = "\n".join(c.text for c in result.content)
    assert "X is defined in a.py" in text
    assert result.is_error is False


def test_task_tool_metadata(tmp_path):
    stream_fn = make_faux_stream_fn([make_assistant_message([TextContent(text="ok")])])
    tool = create_task_tool(str(tmp_path), "faux", stream_fn, api_key=None)
    assert tool.name == "task"
    assert tool.execution_mode == "sequential"
