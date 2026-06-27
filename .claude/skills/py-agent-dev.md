# py-agent Development Skill

This project is a Python port of the [pi agent architecture](../../../packages/agent/src/) — a lite coding agent using Groq as the LLM provider.

## What This Project Is

A from-scratch Python implementation of a production-grade agent loop, following the exact architecture of pi (TypeScript). The goal is to learn by building: each layer is implemented in isolation, bottom-up.

## Architecture (5 Layers, Bottom Up)

```
Layer 5  agent.py           Stateful Agent class (not yet built)
Layer 4  agent_loop.py      Stateless async generator loop (not yet built)
Layer 3  tools/executor.py  Tool call pipeline — validate/execute/finalize (not yet built)
Layer 2  types.py           All data contracts — COMPLETE
Layer 1  stream.py          EventStream base — COMPLETE
         llm/groq.py        Groq streaming provider — COMPLETE
```

Always read PLAN.md for the full design before touching any file.

## Key Files

- `src/py_agent/types.py` — Every data shape. Edit this first when adding new concepts.
- `src/py_agent/stream.py` — `EventStream[T, R]` and `AssistantMessageEventStream`. Don't overcomplicate.
- `src/py_agent/llm/groq.py` — `groq_stream(model_id, context, api_key)` — sync call, async task fills the stream.
- `PLAN.md` — Full design for all layers including ones not yet built.

## Conventions

- **No comments in code** — name things well instead
- **Pydantic BaseModel** for all types (never plain dataclasses for public API shapes)
- **Discriminated unions** with `Field(discriminator="type")` for event/message types
- **asyncio.ensure_future** to fire-and-forget background tasks inside sync functions
- **async generator** for the agent loop (Layer 4), not a class
- Tool `parameters` is a raw JSON Schema `dict` — passed directly to the Groq API
- Errors go into `AssistantMessage.stop_reason = "error"` + `error_message` field, not raised

## LLM Provider

Groq via the `groq` Python SDK. Default model: `llama-3.3-70b-versatile`.

API key from `GROQ_API_KEY` env var or passed explicitly to `groq_stream()`.

Message format conversion: `_to_groq_messages()` in `llm/groq.py` converts internal `AgentMessage` list to Groq API format. Tool calls from AssistantMessage become `tool_calls` array; ToolResultMessage becomes `role: "tool"`.

## What to Build Next: Layer 3 (tools/executor.py)

Three-phase pipeline for each tool call:
1. **Prepare** — validate args against `tool.parameters` JSON Schema, run `before_tool_call` hook
2. **Execute** — call `await tool.execute(tool_call_id, args, signal, on_update)`
3. **Finalize** — run `after_tool_call` hook, build `ToolResultMessage`

Execution modes: `parallel` (concurrent via asyncio.gather) or `sequential`.

The function signature to implement:
```python
async def execute_tool_calls(
    tool_calls: list[ToolCall],
    tools: list[AgentTool],
    emit: Callable[[AgentEvent], Awaitable[None]],
    before_tool_call,
    after_tool_call,
    mode: ToolExecutionMode,
) -> tuple[list[ToolResultMessage], bool]:
```

## What Layer 4 (agent_loop.py) Looks Like

An async generator. NOT a class. Receives context snapshot + config, yields AgentEvent objects.
See PLAN.md "Layer 4" section for the full loop body pseudocode.

## What Layer 5 (agent.py) Looks Like

A class that owns state and wraps the loop. Matches `agent.ts` exactly.
- `prompt(message)` — new turn
- `steer(message)` — mid-run injection (after current assistant turn)
- `follow_up(message)` — post-completion injection
- `subscribe(listener)` — async event listener
- `abort()` — cancel via asyncio.Event

## Testing Without Groq API

Use a faux stream_fn:
```python
async def faux_stream(model_id, context, api_key=None):
    stream = AssistantMessageEventStream()
    asyncio.ensure_future(_emit_text(stream, "Hello world"))
    return stream
```

## Dependencies

- `groq>=0.11.0` — Groq API client (OpenAI-compatible)
- `pydantic>=2.0.0` — type validation (TypeBox equivalent)
- `pytest` + `pytest-asyncio` — testing

Run: `uv run pytest tests/`
Install: `uv sync`
