from __future__ import annotations

from ...utils.types import AgentTool, AgentToolResult, TodoItem
from ._helpers import err, ok

PARAMETERS = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": "The full todo list. Replaces the previous list on every call.",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What the step does"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "Step status",
                    },
                },
                "required": ["content", "status"],
            },
        },
    },
    "required": ["todos"],
}

_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def _render(items: list[TodoItem]) -> str:
    if not items:
        return "(todo list is empty)"
    return "\n".join(f"{_MARKERS[item.status]} {item.content}" for item in items)


def create_todo_tool(cwd: str) -> AgentTool:
    state: list[TodoItem] = []

    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        raw = args.get("todos", [])
        in_progress = [t for t in raw if t.get("status") == "in_progress"]
        if len(in_progress) > 1:
            return err("Keep at most one todo in_progress at a time.")

        try:
            items = [TodoItem(content=t["content"], status=t["status"]) for t in raw]
        except Exception as exc:
            return err(f"Invalid todo item: {exc}")

        state.clear()
        state.extend(items)
        return ok(_render(state))

    return AgentTool(
        name="todo_write",
        description=(
            "Record and update the plan for multi-step work. Pass the full todo list every call; it replaces "
            "the previous list. Keep exactly one item in_progress, and mark items completed as you finish them. "
            "Use this to track progress on any task with more than a couple of steps."
        ),
        parameters=PARAMETERS,
        execution_mode="sequential",
        execute=execute,
    )
