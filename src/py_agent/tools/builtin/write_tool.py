from __future__ import annotations

import os

from ...utils.types import AgentTool, AgentToolResult
from ._helpers import ok, resolve_to_cwd

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def create_write_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        path = resolve_to_cwd(args["path"], cwd)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        content = args["content"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return ok(f"Wrote {len(content.encode('utf-8'))} bytes to {args['path']}")

    return AgentTool(
        name="write",
        description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        parameters=PARAMETERS,
        execute=execute,
    )
