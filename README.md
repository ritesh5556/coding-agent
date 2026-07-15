# py-agent

A lite coding agent for your terminal — a Python port of the [pi](https://github.com/badlogic/pi-mono) agent architecture. It streams LLM responses (via Groq), executes tools (`read`, `write`, `edit`, `bash`) in an agent loop, and exposes the whole thing both as a CLI and as a composable Python library.

## Requirements

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

## Installation

From PyPI (once published):

```bash
pip install py-agent
```

From source:

```bash
git clone <repo-url>
cd coding-agent
pip install .
```

As an isolated CLI tool (recommended, via [uv](https://docs.astral.sh/uv/) or pipx):

```bash
uv tool install .
# or
pipx install .
```

## Usage

### CLI

Set your API key and run:

```bash
export GROQ_API_KEY=gsk_...   # Windows PowerShell: $env:GROQ_API_KEY = "gsk_..."
py-agent
```

You get an interactive prompt operating in the current working directory:

```
py-agent ready. cwd=/home/you/project. Type /quit to exit.

> read main.py and fix the bug in parse_args
```

The agent can read, write, and edit files and run shell commands in the directory you launched it from. Exit with `/quit`, `/exit`, or Ctrl-C.

### Library

Everything the CLI uses is importable:

```python
import asyncio
import os

from py_agent import Agent, create_coding_tools, build_system_prompt, groq_stream

async def main():
    cwd = os.getcwd()
    tools = create_coding_tools(cwd)

    agent = Agent(
        model="llama-3.3-70b-versatile",
        stream_fn=groq_stream,
        system_prompt=build_system_prompt(cwd, tools),
        tools=tools,
        api_key=os.environ["GROQ_API_KEY"],
    )
    agent.subscribe(lambda event: print(event.type))

    await agent.prompt("List the files here and summarize the project.")

asyncio.run(main())
```

You can also drop down a layer and drive `run_agent_loop()` directly, supply your own `stream_fn` for a different LLM provider, or pass custom `AgentTool` definitions.

## Architecture

```
Layer 5  Agent class           ── calls ──►  run_agent_loop()
                                                  │
Layer 4  agent_loop._run()  ◄── yields AgentEvent ┘
            │ calls stream_fn ──► Layer 1 groq_stream()  (LlmStreamEvent stream)
            │ calls execute_tool_calls ──► Layer 3 executor  (ToolResultMessage)
            │
Layer 2  types.py  ── all events/messages/config flow as these types ──
```

Layer 4 is the conductor: it pulls from Layer 1 (LLM), delegates to Layer 3 (tools), and re-emits everything as unified `AgentEvent`s for Layer 5 to consume.

## Development

```bash
uv sync          # install with dev dependencies
uv run pytest    # run tests
uv build         # build sdist + wheel into dist/
```

## License

MIT
