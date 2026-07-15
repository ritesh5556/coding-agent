from __future__ import annotations

import asyncio

from conftest import make_assistant_message, make_faux_stream_fn, tool_call

from py_agent.agent_loop import run_agent_loop
from py_agent.utils.types import (
    AgentContext,
    AgentLoopConfig,
    TextContent,
    UserMessage,
)


def _context():
    async def noop_execute(tool_call_id, args, signal, on_update):
        from py_agent.utils.types import AgentToolResult, ToolResultContent

        return AgentToolResult(content=[ToolResultContent(text="ok")])

    from py_agent.utils.types import AgentTool

    bash = AgentTool(
        name="bash",
        description="run",
        parameters={"type": "object"},
        execute=noop_execute,
    )
    return AgentContext(system_prompt="", messages=[], tools=[bash])


async def _drain(messages, context, config):
    events = []
    async for event in run_agent_loop(messages, context, config):
        events.append(event)
    return events


async def test_error_stop_breaks_loop():
    stream_fn = make_faux_stream_fn(
        [make_assistant_message([TextContent(text="boom")], stop_reason="error")]
    )
    config = AgentLoopConfig(model="faux", stream_fn=stream_fn)
    events = await _drain([UserMessage(content=[TextContent(text="hi")])], _context(), config)

    assert stream_fn.calls["count"] == 1
    assert len([e for e in events if e.type == "agent_end"]) == 1


async def test_aborted_stop_breaks_loop():
    stream_fn = make_faux_stream_fn(
        [make_assistant_message([TextContent(text="stopped")], stop_reason="aborted")]
    )
    config = AgentLoopConfig(model="faux", stream_fn=stream_fn)
    events = await _drain([UserMessage(content=[TextContent(text="hi")])], _context(), config)

    assert stream_fn.calls["count"] == 1
    assert len([e for e in events if e.type == "agent_end"]) == 1


async def test_max_turns_guard():
    stream_fn = make_faux_stream_fn(
        [make_assistant_message([tool_call()], stop_reason="tool_use")]
    )
    config = AgentLoopConfig(model="faux", stream_fn=stream_fn, max_turns=3)
    await _drain([UserMessage(content=[TextContent(text="hi")])], _context(), config)

    assert stream_fn.calls["count"] == 3


async def test_abort_signal_between_batches():
    signal = asyncio.Event()

    async def execute_then_abort(tool_call_id, args, signal_arg, on_update):
        from py_agent.utils.types import AgentToolResult, ToolResultContent

        signal.set()
        return AgentToolResult(content=[ToolResultContent(text="ok")])

    from py_agent.utils.types import AgentTool

    bash = AgentTool(
        name="bash",
        description="run",
        parameters={"type": "object"},
        execute=execute_then_abort,
    )
    context = AgentContext(system_prompt="", messages=[], tools=[bash])

    stream_fn = make_faux_stream_fn(
        [make_assistant_message([tool_call()], stop_reason="tool_use")]
    )
    config = AgentLoopConfig(model="faux", stream_fn=stream_fn, signal=signal)
    await _drain([UserMessage(content=[TextContent(text="hi")])], context, config)

    assert stream_fn.calls["count"] == 1
