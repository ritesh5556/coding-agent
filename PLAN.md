# py-agent — Development Plan

## Goal
Build a lite Python coding agent following the exact same layered architecture as pi (TypeScript).  
Provider: **Groq** (llama-3.3-70b-versatile).

---

## Architecture: Five Layers (Bottom Up)

```
Layer 5  agent.py           Stateful Agent class — owns transcript, queues, events
Layer 4  agent_loop.py      Stateless loop function — LLM → tool calls → repeat
Layer 3  tools/executor.py  Tool execution pipeline — validate, execute, finalize
Layer 2  types.py           All data contracts — messages, events, tools, config
Layer 1  llm/groq.py        LLM client — Groq streaming, returns AssistantMessageEventStream
         stream.py          EventStream base — async queue + result future
```

Each layer only depends on layers below it. The loop has no knowledge of the Agent class. Tools have no knowledge of the loop.

---

## Build Status

| Layer | File | Status |
|-------|------|--------|
| 2 | `src/py_agent/types.py` | ✅ Complete |
| 1 | `src/py_agent/stream.py` | ✅ Complete |
| 1 | `src/py_agent/llm/groq.py` | ✅ Complete |
| 3 | `src/py_agent/tools/executor.py` | ⬜ Next |
| 4 | `src/py_agent/agent_loop.py` | ⬜ Planned |
| 5 | `src/py_agent/agent.py` | ⬜ Planned |
| - | `src/py_agent/tools/builtin/` | ⬜ After core |

---

## Layer 1: LLM Client

**Files:** `stream.py`, `llm/groq.py`

`EventStream[T, R]` is a generic async producer/consumer queue.  
- Producer calls `push(event)` and `end(result)` or `fail(exc)`  
- Consumer iterates with `async for event in stream` and calls `await stream.result()`  

`AssistantMessageEventStream` is `EventStream[LlmStreamEvent, AssistantMessage]`.

`groq_stream(model_id, context, api_key)` — synchronous call that:
1. Creates an empty `AssistantMessageEventStream`
2. Schedules `_fill_stream()` as an asyncio task
3. Returns the stream immediately

`_fill_stream()` does the actual Groq API call, accumulates streaming chunks, pushes typed events, and calls `stream.end(final_message)`.

**Event sequence from one LLM call:**
```
LlmStartEvent
  LlmTextDeltaEvent × N        (per streamed token)
  LlmTextEndEvent
  LlmToolCallEndEvent × M      (per tool call, after arguments accumulated)
LlmDoneEvent
```

---

## Layer 2: Types

**File:** `types.py`

Every data shape in the system. Nothing else defines types — everything imports from here.

### Message types
- `UserMessage` — role=user, list of TextContent
- `AssistantMessage` — role=assistant, list of TextContent | ThinkingContent | ToolCall
- `ToolResultMessage` — role=tool_result, links back to a tool_call_id

### AgentMessage
`Union[UserMessage, AssistantMessage, ToolResultMessage]` — the transcript type

### LLM streaming events
Short-lived events emitted during one assistant turn. The consumer in the agent loop uses these to reconstruct the partial message in real time.

### Agent lifecycle events
Longer-lived events that span the full `agent.prompt()` call. Used by subscribers to drive UI updates.

### AgentTool
```python
class AgentTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema — passed directly to Groq tools format
    execution_mode: "parallel" | "sequential"
    execute: Callable             # async (tool_call_id, params, signal, on_update) -> AgentToolResult
```

### AgentContext
Snapshot passed into the loop: `system_prompt`, `messages`, `tools`

---

## Layer 3: Tool Executor (Next to build)

**File:** `tools/executor.py`

Three-phase pipeline for every tool call:

**Phase 1 — Prepare**
- Validate arguments against `tool.parameters` JSON Schema
- Run `before_tool_call(context)` hook → can override args or short-circuit with a result

**Phase 2 — Execute**
- Call `tool.execute(tool_call_id, args, signal, on_update)`
- `on_update` emits `ToolExecutionUpdateEvent` for streaming tools

**Phase 3 — Finalize**
- Run `after_tool_call(context)` hook → can override result or set `terminate=True`
- Build `ToolResultMessage` from result

**Execution modes** (set per-tool or globally):
- `parallel` — all tool calls from one assistant turn run concurrently
- `sequential` — run one by one in order

```python
async def execute_tool_calls(
    tool_calls: list[ToolCall],
    tools: list[AgentTool],
    emit: Callable[[AgentEvent], Awaitable[None]],
    before_tool_call: BeforeToolCallHook | None,
    after_tool_call: AfterToolCallHook | None,
    mode: ToolExecutionMode,
    signal: asyncio.Event | None,
) -> tuple[list[ToolResultMessage], bool]:  # (results, should_terminate)
    ...
```

---

## Layer 4: Agent Loop (Planned)

**File:** `agent_loop.py`

A **pure async generator function** — no class, no state, just the loop.

```python
async def run_agent_loop(
    messages: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
) -> AsyncGenerator[AgentEvent, None]:
    ...

async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
) -> AsyncGenerator[AgentEvent, None]:
    ...
```

**Loop body for one turn:**
```
emit(TurnStartEvent)
  for each input message:
    emit(MessageStartEvent)
    emit(MessageEndEvent)

  stream = stream_fn(model, context, config)
  emit(MessageStartEvent, partial)
  async for llm_event in stream:
    emit(MessageUpdateEvent, llm_event)
  emit(MessageEndEvent, final_message)

  if final_message has tool_calls:
    tool_results, should_terminate = execute_tool_calls(...)
    for result in tool_results:
      emit(MessageStartEvent)
      emit(MessageEndEvent)

emit(TurnEndEvent)

# Steering: inject mid-run messages
steering = await config.get_steering_messages()
if steering:
    loop with new messages
else:
    # Follow-up: inject only when agent would stop
    follow_ups = await config.get_follow_up_messages()
    if follow_ups:
        loop with follow_ups
    else:
        break

emit(AgentEndEvent)
```

---

## Layer 5: Agent Class (Planned)

**File:** `agent.py`

Stateful wrapper around `run_agent_loop`. Mirrors `agent.ts` exactly.

**State:**
- `messages: list[AgentMessage]` — full transcript
- `tools: list[AgentTool]`
- `system_prompt: str`
- `is_streaming: bool`
- `streaming_message: AgentMessage | None`
- `pending_tool_calls: set[str]`
- `error_message: str | None`

**API:**
```python
await agent.prompt("Hello")           # new conversation turn
await agent.continue_()               # resume from last user/tool message
agent.steer(message)                  # inject after current turn
agent.follow_up(message)              # inject only when agent stops
agent.abort()                         # cancel active run
await agent.wait_for_idle()           # wait for run + all listeners
agent.subscribe(listener)             # async event listener
agent.reset()                         # clear transcript
```

**Queue system:**
- `steering_queue` — drained after each assistant turn
- `follow_up_queue` — drained only when no steering and agent would stop
- Each queue has a mode: `one-at-a-time` or `all`

---

## Groq Models

| Model | Use |
|-------|-----|
| `llama-3.3-70b-versatile` | Default, best tool calling |
| `llama-3.1-8b-instant` | Fast, cheap turns |
| `llama3-groq-70b-8192-tool-use-preview` | Alternative for tool calling |

---

## Key Conventions

- **No comments in source code**
- **Pydantic BaseModel** for all public data types (TypeBox equivalent)
- **JSON Schema dict** for tool parameters (passed directly to Groq API)
- **asyncio** throughout — no threading
- `async for event in stream` is the consumption pattern, not callbacks
- The loop is a generator, not a class — state lives in the Agent wrapper
- Errors are encoded in `AssistantMessage.stop_reason = "error"`, not raised into callers

---

## Testing Plan

```
tests/
  test_types.py          — Pydantic validation, serialization round-trips
  test_stream.py         — EventStream push/drain/result concurrency
  test_groq.py           — Groq integration (requires GROQ_API_KEY)
  test_tools.py          — Executor pipeline with mock tools
  test_agent_loop.py     — Loop with mock stream_fn
  test_agent.py          — Full Agent class with mock loop
```

Faux `stream_fn` pattern for testing without API calls:
```python
async def make_faux_stream(responses: list[str | list[ToolCall]]) -> StreamFn:
    async def faux(model_id, context, api_key=None) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        asyncio.ensure_future(_emit_response(stream, responses.pop(0)))
        return stream
    return faux
```

---

## File Tree

```
py-agent/
├── .claude/
│   └── skills/
│       └── py-agent-dev.md       ← skill for Claude Code sessions
├── src/
│   └── py_agent/
│       ├── __init__.py
│       ├── types.py               ✅ Layer 2 — data model
│       ├── stream.py              ✅ Layer 1 — event stream
│       ├── llm/
│       │   ├── __init__.py
│       │   └── groq.py            ✅ Layer 1 — Groq provider
│       ├── tools/
│       │   ├── __init__.py
│       │   └── executor.py        ⬜ Layer 3 — tool pipeline
│       ├── agent_loop.py          ⬜ Layer 4 — stateless loop
│       └── agent.py               ⬜ Layer 5 — Agent class
├── tests/
├── PLAN.md
├── pyproject.toml
└── .python-version
```
