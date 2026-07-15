from __future__ import annotations

from ..llm.groq import _to_groq_messages
from .types import (
    AgentContext,
    AgentMessage,
    AssistantMessage,
    CompactionSettings,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

SUMMARY_MARKER = "[CONVERSATION SUMMARY]"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Read the conversation between a user and an AI coding "
    "assistant and produce a structured summary in the exact format specified. Do NOT continue the "
    "conversation or answer any question in it. Output ONLY the structured summary."
)

SUMMARIZATION_PROMPT = """Summarize the conversation so far using EXACTLY this format:

## Goal
The user's overall objective in one or two sentences.

## Constraints & Preferences
Requirements, conventions, and preferences the user stated.

## Progress
### Done
- [x] completed items
### In Progress
- [ ] items currently underway
### Blocked
Anything blocked and why.

## Key Decisions
- **[Decision]**: rationale

## Next Steps
Ordered list of what to do next.

## Critical Context
Exact file paths, function names, and error messages that must be preserved.

Keep each section concise. Preserve exact file paths, function names, and error messages."""


def estimate_tokens(messages: list[AgentMessage]) -> int:
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage) and msg.usage.total_tokens:
            return msg.usage.total_tokens
    chars = 0
    for msg in messages:
        for part in msg.content:
            if isinstance(part, TextContent):
                chars += len(part.text)
            elif isinstance(part, ToolCall):
                chars += len(str(part.arguments)) + len(part.name)
    return chars // 4


def should_compact(messages: list[AgentMessage], settings: CompactionSettings) -> bool:
    return estimate_tokens(messages) > settings.context_window - settings.reserve_tokens


def _is_summary(msg: AgentMessage) -> bool:
    return (
        isinstance(msg, UserMessage)
        and bool(msg.content)
        and isinstance(msg.content[0], TextContent)
        and msg.content[0].text.startswith(SUMMARY_MARKER)
    )


def _estimate_one(msg: AgentMessage) -> int:
    chars = 0
    for part in msg.content:
        if isinstance(part, TextContent):
            chars += len(part.text)
        elif isinstance(part, ToolCall):
            chars += len(str(part.arguments)) + len(part.name)
    return chars // 4


def find_cut_point(messages: list[AgentMessage], keep_recent_tokens: int) -> int:
    acc = 0
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        acc += _estimate_one(messages[i])
        if acc >= keep_recent_tokens:
            cut = i
            break
    while cut < len(messages) and isinstance(messages[cut], ToolResultMessage):
        cut += 1
    return cut


def make_summary_message(summary_text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=f"{SUMMARY_MARKER}\n{summary_text}")])


async def generate_summary(
    head: list[AgentMessage],
    model: str,
    stream_fn,
    api_key: str | None,
) -> str:
    context = AgentContext(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=head + [UserMessage(content=[TextContent(text=SUMMARIZATION_PROMPT)])],
        tools=[],
    )
    stream = await stream_fn(model, context, api_key=api_key)
    async for _ in stream:
        pass
    message = await stream.result()
    return " ".join(p.text for p in message.content if isinstance(p, TextContent)).strip()


async def compact(
    messages: list[AgentMessage],
    model: str,
    stream_fn,
    settings: CompactionSettings,
    api_key: str | None,
) -> list[AgentMessage]:
    cut = find_cut_point(messages, settings.keep_recent_tokens)
    if cut <= 0:
        return messages
    head = messages[:cut]
    tail = messages[cut:]
    if all(_is_summary(m) or isinstance(m, ToolResultMessage) for m in head):
        return messages
    summary_text = await generate_summary(head, model, stream_fn, api_key)
    if not summary_text:
        return messages
    return [make_summary_message(summary_text)] + tail
