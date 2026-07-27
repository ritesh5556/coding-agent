# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`py-agent` — a lite coding agent, Python port of the [pi](../packages/agent/src/) agent architecture. Streams LLM responses via Groq, executes tools in an agent loop, exposed as both a CLI (`py-agent`) and an importable library. Python 3.10+ (pinned to 3.12 via `.python-version`); managed with `uv`.

This is a self-contained project with its own git repo, distinct from the TypeScript monorepo one level up (`packages/*`, governed by the root `AGENTS.md`). Rules in that `AGENTS.md` are for the TS code and do NOT apply here.

## Commands

```bash
uv sync                              # install with dev dependencies (.venv)
uv run pytest                        # run all tests (tests/, asyncio_mode=auto)
uv run pytest tests/test_tools.py    # run one test file
uv run pytest tests/test_tools.py::test_grep_finds_matches   # run one test
uv build                             # build sdist + wheel into dist/
GROQ_API_KEY=gsk_... uv run py-agent # run the CLI (needs a Groq key)
```

Tests use a **faux `stream_fn`** (`tests/conftest.py`: `make_faux_stream_fn`, `make_assistant_message`, `tool_call`) — no real Groq calls, no API key needed. Add regression/unit tests against the faux stream, not the live API.

## Architecture

Five layers, each depending only on layers below it. `PLAN.md` and `.claude/skills/py-agent-dev.md` document the original design (note: both predate the current code and describe some layers as "not yet built" — they are all built now; trust the code).

```
Layer 5  agent.py              Agent class — owns transcript, subscribers, steer/follow-up queues, abort signal
Layer 4  agent_loop.py         run_agent_loop() — stateless async generator; LLM turn -> tool calls -> repeat
Layer 3  tools/executor.py     execute_tool_calls() — 3-phase pipeline (validate/before-hook, execute, after-hook)
Layer 2  utils/types.py        every data contract — messages, events, tools, config, hooks
Layer 1  stream.py             EventStream[T,R] async producer/consumer queue + result future
         llm/groq.py           groq_stream() — sync call that spawns a task to fill an AssistantMessageEventStream
```

Data flow: `Agent.prompt()` (L5) builds an `AgentContext` + `AgentLoopConfig` and drives `run_agent_loop()` (L4). The loop calls `config.stream_fn` (L1 `groq_stream`) for each assistant turn, re-emits streamed `LlmStreamEvent`s as unified `AgentEvent`s, and delegates any tool calls to `execute_tool_calls()` (L3). Everything flows as the types in L2.

### Key concepts

- **Two event tiers** (`utils/types.py`): `LlmStreamEvent` (short-lived, one assistant turn: `start`/`text_delta`/`text_end`/`toolcall_end`/`done`/`error`) vs `AgentEvent` (spans a full run: `agent_start`, `turn_start`, `message_start/update/end`, `tool_execution_start/update/end`, `turn_end`, `agent_end`). L4 translates the former into the latter. Subscribers (`agent.subscribe`) consume `AgentEvent`s.
- **The loop is a generator, not a class.** State lives only in the `Agent` wrapper. `Agent` re-runs the loop and appends yielded messages to `self.messages`.
- **Steering vs follow-up queues** (L5 + L4): after each assistant turn the loop drains steering messages (inject mid-run); when the agent would otherwise stop, it drains follow-ups. Modes: `all` (steer default) or `one-at-a-time` (follow-up default).
- **Errors are data, never exceptions to the caller.** A failed LLM turn produces an `AssistantMessage` with `stop_reason="error"` and an `error_message` field; the loop ends the turn on `stop_reason in ("error", "aborted")`. Only `groq_stream`'s `stream.fail(exc)` path surfaces a raw exception through `stream.result()`.
- **Abort** is an `asyncio.Event` (`signal`) threaded through the config into tool execution; `Agent.abort()` sets it and cancels the task.
- **Tool execution modes**: `parallel` (all tool calls from one turn via `asyncio.gather`) or `sequential` (in order). Set per-tool (`AgentTool.execution_mode`) or globally on the Agent. `task` tool is `sequential`.
- **Compaction** (`utils/compaction.py`): when estimated tokens exceed `context_window - reserve_tokens`, the head of the transcript is summarized (via a summarization LLM call) into a single `[CONVERSATION SUMMARY]` user message, preserving `keep_recent_tokens` of tail. Only runs if `CompactionSettings` is passed to `Agent`.

### Groq provider specifics (`llm/groq.py`)

- `groq_stream` returns the stream immediately; `_fill_stream` runs as a fire-and-forget `asyncio.ensure_future` task that does the actual API call and pushes events.
- `_to_groq_messages` converts internal `AgentMessage`s to Groq's OpenAI-compatible format (assistant `tool_calls`, `role: "tool"` results). System prompt is prepended as a `role: "system"` message.
- **Malformed tool-call recovery**: some models emit tool calls as text (`<function=name>{...}`) or in an `APIError.body.failed_generation`. `parse_pseudo_function_calls` and `recover_failed_generation` salvage these into real `ToolCall`s rather than failing the turn.

## Tools

Built-in tools live in `tools/builtin/`, each a `create_*_tool(cwd)` factory returning an `AgentTool` with an async `execute(tool_call_id, args, signal, on_update)` closure. `create_coding_tools(cwd)` bundles read/bash/edit/write/grep/find/ls/todo. The `task` tool (`create_task_tool`) spawns a read-only sub-`Agent` (read/grep/find/ls) to keep an investigation's intermediate steps out of the main context — it is added separately in `__main__.py`, not part of `create_coding_tools`.

- `tools/builtin/_helpers.py`: shared `ok()`/`err()` result builders, `resolve_to_cwd()` path handling, `truncate_head()` (tools cap output at 2000 lines / 50KB).
- Tool `parameters` is a raw JSON Schema `dict`, passed straight to the Groq API and validated with `jsonschema` in the executor before `execute` runs.

## Conventions

- **No comments in source code** — name things well instead.
- **Pydantic `BaseModel`** for all public data shapes; discriminated unions via `Field(discriminator="type")` for message/event types. `AgentLoopConfig` is the one `@dataclass` (holds callables).
- Lazy/local imports inside `tools/builtin/task_tool.py` are intentional — they break the `Agent` <-> tools import cycle. Keep them local.
- Add a new concept to `utils/types.py` first, then wire it through the layers.

## Gotchas

- The default CLI model is `openai/gpt-oss-120b` (`__main__.py`), NOT the `llama-3.3-70b-versatile` mentioned in `README.md`/`PLAN.md`/the dev skill. The docs are out of date on this.
- `main.py` at the project root is a **gitignored, unrelated FastAPI stub** — not part of `py-agent`. The real entry point is `src/py_agent/__main__.py` (`cli()` -> `main()`). Ignore `main.py`.
- CLI slash-commands are `/save <file>`, `/load <file>`, `/quit`, `/exit` (JSONL session persistence via `utils/session.py`).
