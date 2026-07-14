from __future__ import annotations

import os

from ...types import AgentTool, AgentToolResult
from ._helpers import err, ok, resolve_to_cwd, truncate_head

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


def create_read_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        path = resolve_to_cwd(args["path"], cwd)

        if not os.path.isfile(path):
            return err(f"File not found: {args['path']}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        offset = args.get("offset", 1)
        limit = args.get("limit")
        start = max(offset - 1, 0)
        end = start + limit if limit is not None else len(lines)
        selected = lines[start:end]

        text, truncated = truncate_head("".join(selected))
        if truncated:
            next_offset = start + len(text.splitlines())
            text += f"\n[truncated — use offset={next_offset + 1} to continue]"

        return ok(text)

    return AgentTool(
        name="read",
        description="Read the contents of a file. Supports offset/limit for large files. Output truncated to 2000 lines or 50KB.",
        parameters=PARAMETERS,
        execute=execute,
    )
