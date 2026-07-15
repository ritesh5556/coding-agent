from __future__ import annotations

import os

from ...utils.types import AgentToolResult, ToolResultContent

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024


def resolve_to_cwd(path: str, cwd: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(cwd, expanded))


def truncate_head(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    else:
        truncated = False

    result = "".join(lines)
    encoded = result.encode("utf-8")
    if len(encoded) > max_bytes:
        result = encoded[:max_bytes].decode("utf-8", errors="ignore")
        last_newline = result.rfind("\n")
        if last_newline != -1:
            result = result[: last_newline + 1]
        truncated = True

    return result, truncated


def ok(text: str) -> AgentToolResult:
    return AgentToolResult(content=[ToolResultContent(text=text)])


def err(text: str) -> AgentToolResult:
    return AgentToolResult(content=[ToolResultContent(text=text)], is_error=True)
