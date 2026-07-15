from __future__ import annotations

import asyncio

from py_agent.stream import AssistantMessageEventStream
from py_agent.utils.types import (
    EMPTY_USAGE,
    AssistantMessage,
    LlmDoneEvent,
    LlmStartEvent,
    TextContent,
    ToolCall,
)


def make_assistant_message(content, stop_reason="end_turn", model="faux"):
    return AssistantMessage(
        content=content or [TextContent(text="")],
        model=model,
        provider="faux",
        usage=EMPTY_USAGE,
        stop_reason=stop_reason,
    )


def make_faux_stream_fn(responses):
    calls = {"count": 0}

    async def _fill(stream, message):
        stream.push(LlmStartEvent(partial=message))
        stream.push(LlmDoneEvent(reason=message.stop_reason, message=message))
        stream.end(message)

    async def faux(model_id, context, api_key=None):
        idx = min(calls["count"], len(responses) - 1)
        message = responses[idx]
        calls["count"] += 1
        stream = AssistantMessageEventStream()
        asyncio.ensure_future(_fill(stream, message))
        return stream

    faux.calls = calls
    return faux


def tool_call(name="bash", call_id="c1", **arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments)
