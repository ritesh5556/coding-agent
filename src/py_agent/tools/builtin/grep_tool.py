from __future__ import annotations

import fnmatch
import os
import re

from ...utils.types import AgentTool, AgentToolResult
from ._helpers import err, ok, resolve_to_cwd, truncate_head

PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regex (or literal, if literal=true) to search for"},
        "path": {"type": "string", "description": "Directory or file to search (default: working directory)"},
        "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'"},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default false)"},
        "literal": {"type": "boolean", "description": "Treat pattern as a literal string, not regex (default false)"},
        "context": {"type": "integer", "description": "Lines of context before and after each match (default 0)"},
        "limit": {"type": "integer", "description": "Maximum number of matches to return (default 100)"},
    },
    "required": ["pattern"],
}

DEFAULT_LIMIT = 100
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def _iter_files(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            yield os.path.join(dirpath, name)


def create_grep_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        pattern = args["pattern"]
        root = resolve_to_cwd(args.get("path", "."), cwd)
        glob = args.get("glob")
        ignore_case = args.get("ignore_case", False)
        literal = args.get("literal", False)
        context = max(args.get("context", 0), 0)
        limit = args.get("limit", DEFAULT_LIMIT)

        if not os.path.exists(root):
            return err(f"Path not found: {args.get('path', '.')}")

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(re.escape(pattern) if literal else pattern, flags)
        except re.error as exc:
            return err(f"Invalid regex: {exc}")

        matches: list[str] = []
        for filepath in _iter_files(root):
            if glob and not fnmatch.fnmatch(os.path.basename(filepath), glob):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except (OSError, UnicodeError):
                continue

            rel = os.path.relpath(filepath, cwd).replace("\\", "/")
            for i, line in enumerate(lines):
                if regex.search(line):
                    if context:
                        start = max(i - context, 0)
                        end = min(i + context + 1, len(lines))
                        for j in range(start, end):
                            matches.append(f"{rel}:{j + 1}:{lines[j].rstrip()}")
                    else:
                        matches.append(f"{rel}:{i + 1}:{line.rstrip()}")
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break

        if not matches:
            return ok("No matches found")

        text, truncated = truncate_head("\n".join(matches))
        if truncated or len(matches) >= limit:
            text += f"\n[showing up to {limit} matches]"
        return ok(text)

    return AgentTool(
        name="grep",
        description=(
            "Search file contents for a regex pattern across a directory tree. "
            "Returns matching lines as path:line:text. Skips .git, node_modules, and other vendor dirs. "
            "Prefer this over 'bash grep' for searching code."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
