from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content types (building blocks inside messages)
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


AssistantContent = Annotated[
    Union[TextContent, ThinkingContent, ToolCall],
    Field(discriminator="type"),
]

UserContent = Annotated[Union[TextContent], Field(discriminator="type")]


# ---------------------------------------------------------------------------
# Usage & cost
# ---------------------------------------------------------------------------

class CostBreakdown(BaseModel):
    input: float
    output: float
    cache_read: float
    cache_write: float
    total: float


class Usage(BaseModel):
    input: int
    output: int
    cache_read: int
    cache_write: int
    total_tokens: int
    cost: CostBreakdown


EMPTY_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=CostBreakdown(input=0, output=0, cache_read=0, cache_write=0, total=0),
)


# ---------------------------------------------------------------------------
# Stop reasons
# ---------------------------------------------------------------------------

StopReason = Literal["end_turn", "tool_use", "max_tokens", "error", "aborted", "stop_sequence"]


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[UserContent]
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent]
    model: str
    provider: str
    usage: Usage
    stop_reason: StopReason
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ToolResultContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: list[ToolResultContent]
    is_error: bool = False
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ---------------------------------------------------------------------------
# LLM streaming events  (emitted during a single assistant turn)
# ---------------------------------------------------------------------------

class LlmStartEvent(BaseModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class LlmTextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class LlmTextEndEvent(BaseModel):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class LlmToolCallEndEvent(BaseModel):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class LlmDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    reason: StopReason
    message: AssistantMessage


class LlmErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    reason: StopReason
    error: AssistantMessage


LlmStreamEvent = Annotated[
    Union[LlmStartEvent, LlmTextDeltaEvent, LlmTextEndEvent, LlmToolCallEndEvent, LlmDoneEvent, LlmErrorEvent],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Agent lifecycle events  (emitted across multiple turns)
# ---------------------------------------------------------------------------

class AgentStartEvent(BaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]


class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage]


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    llm_event: LlmStreamEvent


class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class ToolExecutionUpdateEvent(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    partial_result: Any


class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


AgentEvent = Annotated[
    Union[
        AgentStartEvent,
        AgentEndEvent,
        TurnStartEvent,
        TurnEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        MessageEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        ToolExecutionEndEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------

AgentToolUpdateCallback = Callable[[Any], None]


class AgentToolResult(BaseModel):
    content: list[ToolResultContent]
    details: Optional[Any] = None
    is_error: bool = False
    terminate: bool = False


ToolExecutionMode = Literal["parallel", "sequential"]


class AgentTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    execution_mode: ToolExecutionMode = "parallel"
    execute: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Hook context types
# ---------------------------------------------------------------------------

class BeforeToolCallContext(BaseModel):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    tool: AgentTool


class BeforeToolCallResult(BaseModel):
    args: Optional[dict[str, Any]] = None
    result: Optional[AgentToolResult] = None


class AfterToolCallContext(BaseModel):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    result: AgentToolResult
    tool: AgentTool


class AfterToolCallResult(BaseModel):
    result: Optional[AgentToolResult] = None
    terminate: bool = False


# ---------------------------------------------------------------------------
# Agent loop context and config
# ---------------------------------------------------------------------------

class AgentContext(BaseModel):
    system_prompt: str = ""
    messages: list[AgentMessage]
    tools: list[AgentTool]


class AgentLoopTurnUpdate(BaseModel):
    system_prompt: Optional[str] = None
    tools: Optional[list[AgentTool]] = None


StreamFn = Callable[..., Awaitable["AssistantMessageEventStream"]]  # type: ignore[name-defined]


QueueMode = Literal["all", "one-at-a-time"]

BeforeToolCallHook = Callable[[BeforeToolCallContext], Awaitable[Optional[BeforeToolCallResult]]]
AfterToolCallHook = Callable[[AfterToolCallContext], Awaitable[Optional[AfterToolCallResult]]]
GetMessagesHook = Callable[[], Awaitable[list[AgentMessage]]]
TransformContextHook = Callable[[AgentContext], AgentLoopTurnUpdate]


@dataclass
class AgentLoopConfig:
    model: str
    stream_fn: StreamFn
    api_key: Optional[str] = None
    execution_mode: ToolExecutionMode = "parallel"
    before_tool_call: Optional[BeforeToolCallHook] = None
    after_tool_call: Optional[AfterToolCallHook] = None
    transform_context: Optional[TransformContextHook] = None
    get_steering_messages: Optional[GetMessagesHook] = None
    get_follow_up_messages: Optional[GetMessagesHook] = None
    signal: Optional[Any] = None
