from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from .llm.groq import groq_stream
from .utils.session import load_session
from .utils.session_paths import most_recent_session_path, new_session_path
from .utils.system_prompt import build_system_prompt
from .utils.types import CompactionSettings
from .tools.builtin import create_coding_tools, create_task_tool
from .agent import Agent
from .tui import PyAgentApp

DEFAULT_MODEL = "openai/gpt-oss-120b"


async def main(resume: bool = False) -> None:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    cwd = os.getcwd()
    tools = create_coding_tools(cwd)
    tools.append(create_task_tool(cwd, DEFAULT_MODEL, groq_stream, api_key))
    system_prompt = build_system_prompt(cwd, tools)

    messages = []
    if resume:
        existing = most_recent_session_path(cwd)
        if existing is not None:
            messages = load_session(str(existing))
            session_path = existing
        else:
            session_path = new_session_path(cwd)
    else:
        session_path = new_session_path(cwd)

    agent = Agent(
        model=DEFAULT_MODEL,
        stream_fn=groq_stream,
        system_prompt=system_prompt,
        tools=tools,
        messages=messages,
        api_key=api_key,
        compaction=CompactionSettings(),
    )

    app = PyAgentApp(agent, cwd, session_path)
    agent.subscribe(app.on_agent_event)
    await app.run_async()


def cli() -> None:
    parser = argparse.ArgumentParser(prog="py-agent")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent session for this working directory.",
    )
    args = parser.parse_args()
    asyncio.run(main(resume=args.resume))


if __name__ == "__main__":
    cli()
