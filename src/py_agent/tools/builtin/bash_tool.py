from __future__ import annotations

import asyncio

from ...utils.types import AgentTool, AgentToolResult
from ._helpers import err, ok, truncate_head

PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (optional, default 120)"},
    },
    "required": ["command"],
}

DEFAULT_TIMEOUT_SECONDS = 120


def create_bash_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        command = args["command"]
        timeout = args.get("timeout", DEFAULT_TIMEOUT_SECONDS)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return err(f"Command timed out after {timeout}s")

        output = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
        text, truncated = truncate_head(output)
        if truncated:
            text += "\n[output truncated]"

        summary = f"exit code: {proc.returncode}\n{text}"
        if proc.returncode != 0:
            return err(summary)
        return ok(summary)

    return AgentTool(
        name="bash",
        description="Execute a bash command in the working directory. Returns stdout, stderr, and exit code. Output truncated to 2000 lines or 50KB.",
        parameters=PARAMETERS,
        execute=execute,
    )
