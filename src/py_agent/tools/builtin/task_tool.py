from __future__ import annotations

from ...utils.types import AgentTool, AgentToolResult, StreamFn, TextContent
from ._helpers import ok

PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "The task for the sub-agent to carry out, described in full."},
    },
    "required": ["task"],
}


def create_task_tool(cwd: str, model: str, stream_fn: StreamFn, api_key: str | None = None) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult:
        from ...agent import Agent
        from ...utils.system_prompt import build_system_prompt
        from . import create_read_tool, create_grep_tool, create_find_tool, create_ls_tool

        sub_tools = [
            create_read_tool(cwd),
            create_grep_tool(cwd),
            create_find_tool(cwd),
            create_ls_tool(cwd),
        ]
        sub = Agent(
            model=model,
            stream_fn=stream_fn,
            system_prompt=build_system_prompt(cwd, sub_tools),
            tools=sub_tools,
            api_key=api_key,
            max_turns=30,
        )
        messages = await sub.prompt(args["task"])

        final_text = ""
        for msg in reversed(messages):
            if msg.role == "assistant":
                final_text = " ".join(p.text for p in msg.content if isinstance(p, TextContent)).strip()
                if final_text:
                    break

        return ok(final_text or "(sub-agent returned no text)")

    return AgentTool(
        name="task",
        description=(
            "Delegate a self-contained, read-only investigation to a sub-agent (read/grep/find/ls only). "
            "The sub-agent runs its own loop and returns just its final answer, keeping its intermediate "
            "steps out of your context. Use for 'find where X is defined' or 'summarize how Y works'."
        ),
        parameters=PARAMETERS,
        execution_mode="sequential",
        execute=execute,
    )
