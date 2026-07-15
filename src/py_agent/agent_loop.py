from __future__ import annotations

from typing import AsyncGenerator

from .tools.executor import execute_tool_calls
from .utils.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolCall,
    TurnEndEvent,
    TurnStartEvent,
)


async def _run(
    input_messages: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
) -> AsyncGenerator[AgentEvent, None]:
    all_new_messages: list[AgentMessage] = []
    turn_count = 0

    yield AgentStartEvent()

    while True:
        if config.max_turns is not None and turn_count >= config.max_turns:
            break
        turn_count += 1

        turn_messages = list(input_messages)

        yield TurnStartEvent()

        for msg in turn_messages:
            yield MessageStartEvent(message=msg)
            yield MessageEndEvent(message=msg)
            all_new_messages.append(msg)

        llm_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=context.messages + all_new_messages,
            tools=context.tools,
        )
        if config.transform_context is not None:
            update = config.transform_context(llm_context)
            if update.system_prompt is not None:
                llm_context.system_prompt = update.system_prompt
            if update.tools is not None:
                llm_context.tools = update.tools

        stream = await config.stream_fn(config.model, llm_context, api_key=config.api_key)

        final_message: AssistantMessage | None = None
        first_event = True

        async for llm_event in stream:
            if llm_event.type == "start":
                final_message = llm_event.partial
                yield MessageStartEvent(message=llm_event.partial)
                first_event = False
            elif llm_event.type in ("text_delta", "text_end", "toolcall_end"):
                final_message = llm_event.partial
                yield MessageUpdateEvent(message=llm_event.partial, llm_event=llm_event)
            elif llm_event.type == "done":
                final_message = llm_event.message
            elif llm_event.type == "error":
                final_message = llm_event.error
                yield MessageUpdateEvent(message=llm_event.error, llm_event=llm_event)

        if final_message is None:
            final_message = await stream.result()

        if first_event:
            yield MessageStartEvent(message=final_message)

        yield MessageEndEvent(message=final_message)
        all_new_messages.append(final_message)

        if final_message.stop_reason in ("error", "aborted"):
            yield TurnEndEvent(message=final_message, tool_results=[])
            break

        tool_calls = [c for c in final_message.content if isinstance(c, ToolCall)]

        tool_results: list = []
        should_terminate = False

        if tool_calls:
            tool_events: list[AgentEvent] = []

            async def _collect(event: AgentEvent) -> None:
                tool_events.append(event)

            tool_results, should_terminate = await execute_tool_calls(
                tool_calls=tool_calls,
                tools=context.tools,
                emit=_collect,
                before_tool_call=config.before_tool_call,
                after_tool_call=config.after_tool_call,
                mode=config.execution_mode,
                signal=config.signal,
            )

            for event in tool_events:
                yield event

            for tr in tool_results:
                yield MessageStartEvent(message=tr)
                yield MessageEndEvent(message=tr)
                all_new_messages.append(tr)

        yield TurnEndEvent(message=final_message, tool_results=tool_results)

        context = AgentContext(
            system_prompt=context.system_prompt,
            messages=context.messages,
            tools=context.tools,
        )

        if should_terminate:
            break

        if config.signal is not None and config.signal.is_set():
            break

        if tool_calls:
            input_messages = []
            continue

        steering = (
            await config.get_steering_messages() if config.get_steering_messages else []
        )
        if steering:
            input_messages = steering
            continue

        follow_ups = (
            await config.get_follow_up_messages() if config.get_follow_up_messages else []
        )
        if follow_ups:
            input_messages = follow_ups
            continue

        break

    yield AgentEndEvent(messages=all_new_messages)


async def run_agent_loop(
    messages: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
) -> AsyncGenerator[AgentEvent, None]:
    async for event in _run(messages, context, config):
        yield event


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
) -> AsyncGenerator[AgentEvent, None]:
    async for event in _run([], context, config):
        yield event
