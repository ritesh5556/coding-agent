from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Any

from groq import APIError, AsyncGroq

from ..stream import AssistantMessageEventStream
from ..utils.types import (
    EMPTY_USAGE,
    AgentContext,
    AgentMessage,
    AssistantMessage,
    CostBreakdown,
    LlmDoneEvent,
    LlmErrorEvent,
    LlmStartEvent,
    LlmTextDeltaEvent,
    LlmTextEndEvent,
    LlmToolCallEndEvent,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _to_groq_messages(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            text = " ".join(
                part.text for part in msg.content if isinstance(part, TextContent)
            )
            result.append({"role": "user", "content": text})
        elif isinstance(msg, AssistantMessage):
            groq_msg: dict[str, Any] = {"role": "assistant", "content": None}
            text_parts = [c for c in msg.content if isinstance(c, TextContent)]
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
            if text_parts:
                groq_msg["content"] = " ".join(t.text for t in text_parts)
            if tool_calls:
                groq_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            result.append(groq_msg)
        elif isinstance(msg, ToolResultMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": " ".join(c.text for c in msg.content),
                }
            )
    return result


def _to_groq_tools(tools: list) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


_PSEUDO_FUNCTION_CALL_RE = re.compile(
    r"<function[/=]([a-zA-Z0-9_]+)\s*>?\s*(\{.*?\})\s*(?:</function>|>)",
    re.DOTALL,
)


def parse_pseudo_function_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _PSEUDO_FUNCTION_CALL_RE.finditer(text):
        name, raw_args = match.group(1), match.group(2)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            continue
        calls.append(ToolCall(id=f"pseudo-{uuid.uuid4().hex[:8]}", name=name, arguments=args))
    return calls


def _failed_generation_text(exc: BaseException) -> str | None:
    if not isinstance(exc, APIError):
        return None
    body = exc.body
    error_obj = body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error_obj = body["error"]
    if isinstance(error_obj, dict):
        failed_generation = error_obj.get("failed_generation")
        if isinstance(failed_generation, str) and failed_generation.strip():
            return failed_generation
    return None


def recover_failed_generation(exc: BaseException) -> list[ToolCall]:
    text = _failed_generation_text(exc)
    if text is None:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return parse_pseudo_function_calls(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    calls: list[ToolCall] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        args = entry.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(name, str) and name and isinstance(args, dict):
            calls.append(ToolCall(id=f"recovered-{uuid.uuid4().hex[:8]}", name=name, arguments=args))
    return calls


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, APIError):
        body = exc.body
        error_obj = body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            error_obj = body["error"]

        status_code = getattr(exc, "status_code", None)
        label = f"Groq API error ({status_code})" if status_code is not None else "Groq API error"
        parts = [label]

        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if message:
                parts.append(str(message))
            failed_generation = error_obj.get("failed_generation")
            if failed_generation:
                parts.append(f"failed_generation: {failed_generation}")
        else:
            parts.append(exc.message)

        return " — ".join(parts)
    return str(exc)


def _make_partial(model_id: str, text_acc: str, tc_acc: dict[int, dict]) -> AssistantMessage:
    content: list = []
    if text_acc:
        content.append(TextContent(text=text_acc))
    for idx in sorted(tc_acc):
        entry = tc_acc[idx]
        if entry.get("name"):
            content.append(ToolCall(id=entry.get("id", ""), name=entry["name"], arguments={}))
    return AssistantMessage(
        content=content or [TextContent(text="")],
        model=model_id,
        provider="groq",
        usage=EMPTY_USAGE,
        stop_reason="end_turn",
    )


def _build_usage(raw: Any) -> Usage:
    if raw is None:
        return EMPTY_USAGE
    return Usage(
        input=raw.prompt_tokens or 0,
        output=raw.completion_tokens or 0,
        cache_read=0,
        cache_write=0,
        total_tokens=raw.total_tokens or 0,
        cost=CostBreakdown(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


async def _fill_stream(
    stream: AssistantMessageEventStream,
    model_id: str,
    context: AgentContext,
    api_key: str | None,
) -> None:
    text_acc = ""
    tc_acc: dict[int, dict] = {}
    try:
        client = AsyncGroq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        api_messages = _to_groq_messages(context.messages)
        groq_tools = _to_groq_tools(context.tools) if context.tools else None

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "stream": True,
        }
        if context.system_prompt:
            kwargs["messages"] = [{"role": "system", "content": context.system_prompt}] + api_messages
        if groq_tools:
            kwargs["tools"] = groq_tools
            kwargs["tool_choice"] = "auto"

        usage_raw = None
        finish_reason = None
        text_content_index = 0

        partial = _make_partial(model_id, text_acc, tc_acc)
        stream.push(LlmStartEvent(partial=partial))

        response = await client.chat.completions.create(**kwargs)

        async for chunk in response:
            if not chunk.choices:
                if chunk.usage:
                    usage_raw = chunk.usage
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                text_acc += delta.content
                partial = _make_partial(model_id, text_acc, tc_acc)
                stream.push(LlmTextDeltaEvent(
                    content_index=text_content_index,
                    delta=delta.content,
                    partial=partial,
                ))

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {
                            "id": tc_delta.id or "",
                            "name": (tc_delta.function.name or "") if tc_delta.function else "",
                            "arguments": (tc_delta.function.arguments or "") if tc_delta.function else "",
                        }
                    else:
                        if tc_delta.id:
                            tc_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_acc[idx]["arguments"] += tc_delta.function.arguments

        final_content: list = []

        if text_acc:
            partial = _make_partial(model_id, text_acc, tc_acc)
            stream.push(LlmTextEndEvent(
                content_index=text_content_index,
                content=text_acc,
                partial=partial,
            ))
            final_content.append(TextContent(text=text_acc))

        tc_content_index = 1 if text_acc else 0
        for idx in sorted(tc_acc):
            entry = tc_acc[idx]
            try:
                args = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_call = ToolCall(id=entry["id"], name=entry["name"], arguments=args)
            final_content.append(tool_call)
            partial = _make_partial(model_id, text_acc, tc_acc)
            stream.push(LlmToolCallEndEvent(
                content_index=tc_content_index,
                tool_call=tool_call,
                partial=partial,
            ))
            tc_content_index += 1

        stop_reason: StopReason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        if not tc_acc and text_acc and "<function" in text_acc:
            recovered = parse_pseudo_function_calls(text_acc)
            if recovered:
                final_content = [c for c in final_content if not isinstance(c, TextContent)]
                final_content.extend(recovered)
                stop_reason = "tool_use"
            else:
                final_message = AssistantMessage(
                    content=final_content or [TextContent(text="")],
                    model=model_id,
                    provider="groq",
                    usage=_build_usage(usage_raw),
                    stop_reason="error",
                    error_message=(
                        "Model emitted a malformed tool call as text instead of a structured tool_calls "
                        "field, and it could not be recovered. No tool was executed."
                    ),
                )
                stream.push(LlmErrorEvent(reason="error", error=final_message))
                stream.end(final_message)
                return

        final_message = AssistantMessage(
            content=final_content or [TextContent(text="")],
            model=model_id,
            provider="groq",
            usage=_build_usage(usage_raw),
            stop_reason=stop_reason,
        )

        stream.push(LlmDoneEvent(reason=stop_reason, message=final_message))
        stream.end(final_message)

    except Exception as exc:
        recovered = recover_failed_generation(exc)
        if recovered:
            content: list = []
            if text_acc:
                content.append(TextContent(text=text_acc))
            content.extend(recovered)
            final_message = AssistantMessage(
                content=content,
                model=model_id,
                provider="groq",
                usage=EMPTY_USAGE,
                stop_reason="tool_use",
            )
            for i, tool_call in enumerate(recovered):
                stream.push(LlmToolCallEndEvent(
                    content_index=(1 if text_acc else 0) + i,
                    tool_call=tool_call,
                    partial=final_message,
                ))
            stream.push(LlmDoneEvent(reason="tool_use", message=final_message))
            stream.end(final_message)
            return

        detail = _describe_error(exc)
        error_message = AssistantMessage(
            content=[TextContent(text=detail)],
            model=model_id,
            provider="groq",
            usage=EMPTY_USAGE,
            stop_reason="error",
            error_message=detail,
        )
        stream.push(LlmErrorEvent(reason="error", error=error_message))
        stream.fail(exc)


async def groq_stream(
    model_id: str,
    context: AgentContext,
    api_key: str | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    asyncio.ensure_future(_fill_stream(stream, model_id, context, api_key))
    return stream
