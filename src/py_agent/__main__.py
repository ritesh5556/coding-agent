from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from .llm.groq import groq_stream
from .utils.system_prompt import build_system_prompt
from .utils.types import CompactionSettings
from .tools.builtin import create_coding_tools, create_task_tool
from .agent import Agent
from .tui import PyAgentApp

DEFAULT_MODEL = "openai/gpt-oss-120b"


async def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    cwd = os.getcwd()
    tools = create_coding_tools(cwd)
    tools.append(create_task_tool(cwd, DEFAULT_MODEL, groq_stream, api_key))
    system_prompt = build_system_prompt(cwd, tools)

    agent = Agent(
        model=DEFAULT_MODEL,
        stream_fn=groq_stream,
        system_prompt=system_prompt,
        tools=tools,
        api_key=api_key,
        compaction=CompactionSettings(),
    )

    app = PyAgentApp(agent, cwd)
    agent.subscribe(app.on_agent_event)
    await app.run_async()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
