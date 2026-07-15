from __future__ import annotations

from py_agent.tools.builtin import create_coding_tools, create_todo_tool


def _text(result):
    return "\n".join(c.text for c in result.content)


async def test_todo_write_sets_state():
    tool = create_todo_tool(".")
    result = await tool.execute(
        "c1",
        {"todos": [{"content": "step one", "status": "in_progress"}, {"content": "step two", "status": "pending"}]},
        None,
        None,
    )
    text = _text(result)
    assert "[~] step one" in text
    assert "[ ] step two" in text


async def test_todo_write_replaces():
    tool = create_todo_tool(".")
    await tool.execute("c1", {"todos": [{"content": "old", "status": "pending"}]}, None, None)
    result = await tool.execute("c2", {"todos": [{"content": "new", "status": "completed"}]}, None, None)
    text = _text(result)
    assert "old" not in text
    assert "[x] new" in text


async def test_todo_rejects_multiple_in_progress():
    tool = create_todo_tool(".")
    result = await tool.execute(
        "c1",
        {"todos": [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}]},
        None,
        None,
    )
    assert result.is_error


async def test_todo_invalid_status_rejected():
    tool = create_todo_tool(".")
    result = await tool.execute("c1", {"todos": [{"content": "a", "status": "bogus"}]}, None, None)
    assert result.is_error


def test_todo_registered():
    tools = create_coding_tools(".")
    todo = next((t for t in tools if t.name == "todo_write"), None)
    assert todo is not None
    assert todo.execution_mode == "sequential"
