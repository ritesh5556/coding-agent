from __future__ import annotations

import asyncio
import os
import sys

from .agent import Agent
from .llm.groq import groq_stream
from .utils.system_prompt import build_system_prompt
from .tools.builtin import create_coding_tools

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def render_event(event) -> None:
    if event.type == "message_update":
        llm_event = event.llm_event
        if llm_event.type == "text_delta":
            print(llm_event.delta, end="", flush=True)
    elif event.type == "message_end" and event.message.role == "assistant":
        print()
    elif event.type == "tool_execution_start":
        print(f"\n[tool] {event.tool_name}({event.args})")
    elif event.type == "tool_execution_end":
        text = event.result.content[0].text if event.result.content else ""
        status = "error" if event.is_error else "done"
        print(f"[tool {status}] {text[:300]}")


async def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    cwd = os.getcwd()
    tools = create_coding_tools(cwd)
    system_prompt = build_system_prompt(cwd, tools)

    agent = Agent(
        model=DEFAULT_MODEL,
        stream_fn=groq_stream,
        system_prompt=system_prompt,
        tools=tools,
        api_key=api_key,
    )
    agent.subscribe(render_event)

    print(f"py-agent ready. cwd={cwd}. Type /quit to exit.")

    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.strip() in ("/quit", "/exit"):
            break
        if not line.strip():
            continue

        try:
            await agent.prompt(line)
        except Exception as exc:
            print(f"\n[error] {exc}", file=sys.stderr)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
