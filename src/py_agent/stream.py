from __future__ import annotations

import asyncio
from typing import AsyncIterator, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._done = asyncio.Event()
        self._result: R | None = None
        self._exception: BaseException | None = None

    def push(self, item: T) -> None:
        self._queue.put_nowait(item)

    def end(self, result: R) -> None:
        self._result = result
        self._queue.put_nowait(None)
        self._done.set()

    def fail(self, exc: BaseException) -> None:
        self._exception = exc
        self._queue.put_nowait(None)
        self._done.set()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def result(self) -> R:
        await self._done.wait()
        if self._exception is not None:
            raise self._exception
        return self._result  # type: ignore[return-value]


from .types import AssistantMessage, LlmStreamEvent  # noqa: E402


class AssistantMessageEventStream(EventStream[LlmStreamEvent, AssistantMessage]):
    pass
