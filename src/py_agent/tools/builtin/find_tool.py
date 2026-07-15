from __future__ import annotations

import os
from pathlib import Path

from ...utils.types import AgentTool, AgentToolResult
from ._helpers import err, ok, resolve_to_cwd, truncate_head

PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern to match file names, e.g. '**/*.py' or 'test_*.py'"},
        "path": {"type": "string", "description": "Directory to search from (default: working directory)"},
        "limit": {"type": "integer", "description": "Maximum number of paths to return (default 1000)"},
    },
    "required": ["pattern"],
}

DEFAULT_LIMIT = 1000
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def create_find_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        pattern = args["pattern"]
        root = resolve_to_cwd(args.get("path", "."), cwd)
        limit = args.get("limit", DEFAULT_LIMIT)

        if not os.path.isdir(root):
            return err(f"Directory not found: {args.get('path', '.')}")

        results: list[str] = []
        for match in Path(root).glob(pattern):
            if not match.is_file():
                continue
            parts = set(match.relative_to(root).parts)
            if parts & _SKIP_DIRS:
                continue
            results.append(os.path.relpath(str(match), cwd).replace("\\", "/"))
            if len(results) >= limit:
                break

        if not results:
            return ok("No files found")

        results.sort()
        text, truncated = truncate_head("\n".join(results))
        if truncated or len(results) >= limit:
            text += f"\n[showing up to {limit} paths]"
        return ok(text)

    return AgentTool(
        name="find",
        description=(
            "Find files by glob pattern (e.g. '**/*.py'). Returns file paths relative to the working directory. "
            "Skips .git, node_modules, and other vendor dirs. Prefer this over 'bash find' for locating files."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
