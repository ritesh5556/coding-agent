from __future__ import annotations

import json

from pydantic import TypeAdapter

from .types import AgentMessage

_ADAPTER: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)


def save_session(path: str, messages: list[AgentMessage]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(_ADAPTER.dump_python(msg, mode="json")) + "\n")


def load_session(path: str) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages.append(_ADAPTER.validate_python(json.loads(line)))
    return messages
