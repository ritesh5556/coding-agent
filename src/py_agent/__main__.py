from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from .agent import Agent
from .llm.groq import groq_stream
from .utils.system_prompt import build_system_prompt
from .utils.session import load_session, save_session
from .utils.types import CompactionSettings
from .tools.builtin import create_coding_tools, create_task_tool

DEFAULT_MODEL = "openai/gpt-oss-120b"


def render_event(event) -> None:
    if event.type == "message_update":
        llm_event = event.llm_event
        if llm_event.type == "text_delta":
            print(llm_event.delta, end="", flush=True)
    elif event.type == "message_end" and event.message.role == "assistant":
        message = event.message
        if getattr(message, "stop_reason", None) == "error":
            detail = message.error_message or (
                message.content[0].text if message.content else "unknown error"
            )
            print(f"\n[error] {detail}", file=sys.stderr)
        else:
            print()
    elif event.type == "tool_execution_start":
        print(f"\n[tool] {event.tool_name}({event.args})")
    elif event.type == "tool_execution_end":
        text = event.result.content[0].text if event.result.content else ""
        status = "error" if event.is_error else "done"
        print(f"[tool {status}] {text[:300]}")


async def _stdin_reader(line_queue: "asyncio.Queue[Optional[str]]") -> None:
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        await line_queue.put(line if line != "" else None)
        if line == "":
            return


def route_streaming_line(agent: Agent, line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None

    if stripped == "/abort":
        agent.abort()
        return "[aborting]"

    if stripped.startswith("/steer "):
        message = stripped[len("/steer "):].strip()
        if not message:
            return None
        agent.steer(message)
        return "[queued steering]"

    if stripped.startswith("/followup "):
        message = stripped[len("/followup "):].strip()
        if not message:
            return None
        agent.follow_up(message)
        return "[queued follow-up]"

    agent.follow_up(stripped)
    return "[queued follow-up]"


async def _run_streaming(
    agent: Agent,
    run_task: "asyncio.Task",
    line_queue: "asyncio.Queue[Optional[str]]",
) -> bool:
    while True:
        get_line = asyncio.ensure_future(line_queue.get())
        done, _ = await asyncio.wait(
            {run_task, get_line}, return_when=asyncio.FIRST_COMPLETED
        )

        if run_task in done:
            get_line.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"\n[error] {exc}", file=sys.stderr)
            return True

        line = get_line.result()
        if line is None:
            agent.abort()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass
            return False

        status = route_streaming_line(agent, line)
        if status is not None:
            print(status)


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
    agent.subscribe(render_event)

    print(
        f"py-agent ready. cwd={cwd}. Commands: /save <file>, /load <file>, /quit. "
        f"While running: type to queue a follow-up, /steer <msg> to inject, /abort to stop."
    )

    line_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    reader_task = asyncio.ensure_future(_stdin_reader(line_queue))

    try:
        while True:
            print("\n> ", end="", flush=True)
            line = await line_queue.get()
            if line is None:
                print()
                break

            stripped = line.strip()
            if stripped in ("/quit", "/exit"):
                break
            if not stripped:
                continue

            if stripped.startswith("/save "):
                target = stripped[len("/save "):].strip()
                try:
                    save_session(target, agent.messages)
                    print(f"[saved {len(agent.messages)} messages to {target}]")
                except Exception as exc:
                    print(f"\n[error] {exc}", file=sys.stderr)
                continue

            if stripped.startswith("/load "):
                target = stripped[len("/load "):].strip()
                try:
                    agent.messages = load_session(target)
                    print(f"[loaded {len(agent.messages)} messages from {target}]")
                except Exception as exc:
                    print(f"\n[error] {exc}", file=sys.stderr)
                continue

            run_task = asyncio.ensure_future(agent.prompt(line))
            await _run_streaming(agent, run_task, line_queue)
    finally:
        reader_task.cancel()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
