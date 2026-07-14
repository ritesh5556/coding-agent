from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

import jsonschema

from ..types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultContent,
    ToolResultMessage,
)

BeforeToolCallHook = Callable[[BeforeToolCallContext], Awaitable[Optional[BeforeToolCallResult]]]
AfterToolCallHook = Callable[[AfterToolCallContext], Awaitable[Optional[AfterToolCallResult]]]
EmitFn = Callable[[AgentEvent], Awaitable[None]]


async def _execute_one(
    tool_call: ToolCall,
    tools: list[AgentTool],
    emit: EmitFn,
    before_tool_call: BeforeToolCallHook | None,
    after_tool_call: AfterToolCallHook | None,
    signal: asyncio.Event | None,
) -> tuple[ToolResultMessage, bool]:
    tool = next((t for t in tools if t.name == tool_call.name), None)

    await emit(ToolExecutionStartEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        args=tool_call.arguments,
    ))

    # Unknown tool — short-circuit with error
    if tool is None:
        result = AgentToolResult(
            content=[ToolResultContent(text=f"Unknown tool: {tool_call.name}")],
            terminate=False,
        )
        await emit(ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=True,
        ))
        return _build_result_message(tool_call.id, result, is_error=True), False

    # ── Phase 1: Prepare ────────────────────────────────────────────────────

    args = tool_call.arguments

    try:
        jsonschema.validate(instance=args, schema=tool.parameters)
    except jsonschema.ValidationError as exc:
        result = AgentToolResult(
            content=[ToolResultContent(text=f"Invalid arguments: {exc.message}")],
        )
        await emit(ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=True,
        ))
        return _build_result_message(tool_call.id, result, is_error=True), False

    short_circuit: AgentToolResult | None = None

    if before_tool_call is not None:
        hook_result = await before_tool_call(BeforeToolCallContext(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=args,
            tool=tool,
        ))
        if hook_result is not None:
            if hook_result.args is not None:
                args = hook_result.args
            if hook_result.result is not None:
                short_circuit = hook_result.result

    # ── Phase 2: Execute ────────────────────────────────────────────────────

    if short_circuit is not None:
        result = short_circuit
    else:
        def on_update(partial_result: object) -> None:
            asyncio.ensure_future(emit(ToolExecutionUpdateEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=args,
                partial_result=partial_result,
            )))

        result = await tool.execute(tool_call.id, args, signal, on_update)

    # ── Phase 3: Finalize ───────────────────────────────────────────────────

    is_error = result.is_error
    should_terminate = result.terminate

    if after_tool_call is not None:
        hook_result = await after_tool_call(AfterToolCallContext(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=args,
            result=result,
            tool=tool,
        ))
        if hook_result is not None:
            if hook_result.result is not None:
                result = hook_result.result
            if hook_result.terminate:
                should_terminate = True

    await emit(ToolExecutionEndEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        result=result,
        is_error=is_error,
    ))

    return _build_result_message(tool_call.id, result, is_error=is_error), should_terminate


def _build_result_message(
    tool_call_id: str,
    result: AgentToolResult,
    is_error: bool,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        content=result.content,
        is_error=is_error,
    )


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    tools: list[AgentTool],
    emit: EmitFn,
    before_tool_call: BeforeToolCallHook | None,
    after_tool_call: AfterToolCallHook | None,
    mode: ToolExecutionMode,
    signal: asyncio.Event | None,
) -> tuple[list[ToolResultMessage], bool]:
    if mode == "parallel":
        pairs = await asyncio.gather(*[
            _execute_one(tc, tools, emit, before_tool_call, after_tool_call, signal)
            for tc in tool_calls
        ])
    else:
        pairs = []
        for tc in tool_calls:
            pairs.append(
                await _execute_one(tc, tools, emit, before_tool_call, after_tool_call, signal)
            )

    results = [msg for msg, _ in pairs]
    should_terminate = any(term for _, term in pairs)
    return results, should_terminate
