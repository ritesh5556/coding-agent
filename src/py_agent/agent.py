from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Union

from .agent_loop import run_agent_loop
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    QueueMode,
    StreamFn,
    TextContent,
    ToolExecutionMode,
    UserMessage,
)

AgentEventSink = Callable[[AgentEvent], Union[None, Awaitable[None]]]


def _to_user_message(message: Union[str, AgentMessage]) -> AgentMessage:
    if isinstance(message, str):
        return UserMessage(content=[TextContent(text=message)])
    return message


class Agent:
    def __init__(
        self,
        model: str,
        stream_fn: StreamFn,
        system_prompt: str = "",
        tools: Optional[list[AgentTool]] = None,
        messages: Optional[list[AgentMessage]] = None,
        api_key: Optional[str] = None,
        execution_mode: ToolExecutionMode = "parallel",
        steer_mode: QueueMode = "all",
        follow_up_mode: QueueMode = "one-at-a-time",
        before_tool_call=None,
        after_tool_call=None,
        transform_context=None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.tools: list[AgentTool] = tools or []
        self.messages: list[AgentMessage] = messages or []

        self.is_streaming = False
        self.streaming_message: Optional[AgentMessage] = None
        self.pending_tool_calls: set[str] = set()
        self.error_message: Optional[str] = None

        self._stream_fn = stream_fn
        self._api_key = api_key
        self._execution_mode = execution_mode
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._transform_context = transform_context

        self._subscribers: set[AgentEventSink] = set()
        self._steer_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._steer_mode = steer_mode
        self._follow_up_mode = follow_up_mode

        self._task: Optional[asyncio.Task] = None
        self._signal: Optional[asyncio.Event] = None

    # ── Public API ──────────────────────────────────────────────────────────

    async def prompt(self, message: Union[str, AgentMessage]) -> list[AgentMessage]:
        return await self._run_loop([_to_user_message(message)])

    async def continue_(self) -> list[AgentMessage]:
        return await self._run_loop([])

    def steer(self, message: Union[str, AgentMessage]) -> None:
        self._steer_queue.append(_to_user_message(message))

    def follow_up(self, message: Union[str, AgentMessage]) -> None:
        self._follow_up_queue.append(_to_user_message(message))

    def abort(self) -> None:
        if self._signal is not None:
            self._signal.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def wait_for_idle(self) -> None:
        if self._task is not None:
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def subscribe(self, listener: AgentEventSink) -> Callable[[], None]:
        self._subscribers.add(listener)

        def unsubscribe() -> None:
            self._subscribers.discard(listener)

        return unsubscribe

    def reset(self) -> None:
        if self.is_streaming:
            raise RuntimeError("Cannot reset while streaming")
        self.messages = []
        self.error_message = None
        self.streaming_message = None
        self.pending_tool_calls.clear()

    # ── Internal ────────────────────────────────────────────────────────────

    async def _run_loop(self, input_messages: list[AgentMessage]) -> list[AgentMessage]:
        if self.is_streaming:
            raise RuntimeError("Agent is already streaming")

        self._signal = asyncio.Event()
        self.is_streaming = True
        self.error_message = None

        context = AgentContext(
            system_prompt=self.system_prompt,
            messages=list(self.messages),
            tools=self.tools,
        )

        config = AgentLoopConfig(
            model=self.model,
            stream_fn=self._stream_fn,
            api_key=self._api_key,
            execution_mode=self._execution_mode,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
            transform_context=self._transform_context,
            get_steering_messages=lambda: self._drain_queue(self._steer_queue, self._steer_mode),
            get_follow_up_messages=lambda: self._drain_queue(self._follow_up_queue, self._follow_up_mode),
            signal=self._signal,
        )

        self._task = asyncio.current_task()
        new_messages: list[AgentMessage] = []

        try:
            async for event in run_agent_loop(input_messages, context, config):
                self._handle_event(event)
                await self._emit(event)
                if event.type == "agent_end":
                    new_messages = event.messages
            self.messages = self.messages + new_messages
            return new_messages
        except asyncio.CancelledError:
            self.error_message = "Aborted"
            raise
        except Exception as exc:
            self.error_message = str(exc)
            raise
        finally:
            self.is_streaming = False
            self.streaming_message = None
            self._signal = None
            self._task = None

    async def _drain_queue(self, queue: list[AgentMessage], mode: QueueMode) -> list[AgentMessage]:
        if not queue:
            return []
        if mode == "all":
            drained = list(queue)
            queue.clear()
            return drained
        return [queue.pop(0)]

    def _handle_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self.streaming_message = event.message
        elif event.type == "message_update":
            self.streaming_message = event.message
        elif event.type == "message_end":
            self.streaming_message = None
        elif event.type == "tool_execution_start":
            self.pending_tool_calls.add(event.tool_call_id)
        elif event.type == "tool_execution_end":
            self.pending_tool_calls.discard(event.tool_call_id)

    async def _emit(self, event: AgentEvent) -> None:
        for subscriber in list(self._subscribers):
            result = subscriber(event)
            if asyncio.iscoroutine(result):
                await result
