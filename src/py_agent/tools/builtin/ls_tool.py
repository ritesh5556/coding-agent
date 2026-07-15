from __future__ import annotations

import os

from ...utils.types import AgentTool, AgentToolResult
from ._helpers import err, ok, resolve_to_cwd, truncate_head

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (default: working directory)"},
    },
    "required": [],
}

DEFAULT_LIMIT = 500


def create_ls_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        target = resolve_to_cwd(args.get("path", "."), cwd)

        if not os.path.exists(target):
            return err(f"Path not found: {args.get('path', '.')}")
        if not os.path.isdir(target):
            return err(f"Not a directory: {args.get('path', '.')}")

        entries: list[str] = []
        with os.scandir(target) as it:
            for entry in it:
                entries.append(entry.name + "/" if entry.is_dir() else entry.name)

        if not entries:
            return ok("(empty directory)")

        entries.sort()
        truncated_note = ""
        if len(entries) > DEFAULT_LIMIT:
            entries = entries[:DEFAULT_LIMIT]
            truncated_note = f"\n[showing first {DEFAULT_LIMIT} entries]"

        text, truncated = truncate_head("\n".join(entries))
        if truncated:
            truncated_note = f"\n[showing first {DEFAULT_LIMIT} entries]"
        return ok(text + truncated_note)

    return AgentTool(
        name="ls",
        description=(
            "List the contents of a directory, sorted alphabetically. Directories are suffixed with '/'. "
            "Prefer this over 'bash ls' for listing files."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
