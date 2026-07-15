from __future__ import annotations

import os

from ...utils.types import AgentTool, AgentToolResult
from ._helpers import err, ok, resolve_to_cwd

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit"},
        "old_string": {
            "type": "string",
            "description": "Exact text to replace. Must be unique in the file.",
        },
        "new_string": {"type": "string", "description": "Replacement text"},
    },
    "required": ["path", "old_string", "new_string"],
}


def create_edit_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        path = resolve_to_cwd(args["path"], cwd)
        old_string = args["old_string"]
        new_string = args["new_string"]

        if not os.path.isfile(path):
            return err(f"File not found: {args['path']}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return err("old_string not found in file")
        if count > 1:
            return err(f"old_string not unique ({count} matches) — add more surrounding context")

        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        snippet = new_string if len(new_string) <= 200 else new_string[:200] + "..."
        return ok(f"Edited {args['path']}. New text:\n{snippet}")

    return AgentTool(
        name="edit",
        description="Replace exact text in a file. old_string must match exactly once in the file.",
        parameters=PARAMETERS,
        execute=execute,
    )
