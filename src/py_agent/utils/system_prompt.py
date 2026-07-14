from __future__ import annotations

import datetime

from .types import AgentTool


def build_system_prompt(cwd: str, tools: list[AgentTool]) -> str:
    tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    today = datetime.date.today().isoformat()

    return f"""You are a coding agent. You help by reading files, running commands, editing and writing code.

Available tools:
{tool_list}

Guidelines:
- Use tools to inspect the project before making changes.
- Make minimal, targeted edits. Prefer edit over rewriting whole files.
- After changes, verify with bash when appropriate.

Working directory: {cwd}
Today: {today}"""
