from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.suggester import Suggester
from textual.widgets import Footer, Input, RichLog, Static

from .agent import Agent
from .utils.session import load_session, save_session
from .utils.session_paths import new_session_path

IDLE_COMMANDS = {
    "/new": "start a new session",
    "/save": "save transcript to <file>",
    "/load": "load transcript from <file>",
    "/quit": "exit py-agent",
    "/exit": "exit py-agent",
}

STREAMING_COMMANDS = {
    "/steer": "inject <message> at the next turn boundary",
    "/followup": "queue <message> for when the agent would stop",
    "/abort": "stop the current run",
}


class CommandSuggester(Suggester):
    def __init__(self, app: "PyAgentApp") -> None:
        super().__init__(case_sensitive=True)
        self._app = app

    async def get_suggestion(self, value: str) -> Optional[str]:
        if not value.startswith("/") or " " in value:
            return None
        commands = STREAMING_COMMANDS if self._app.agent.is_streaming else IDLE_COMMANDS
        matches = [cmd for cmd in commands if cmd.startswith(value)]
        if len(matches) != 1 or matches[0] == value:
            return None
        return matches[0]


def route_streaming_line(agent: Agent, line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None

    if stripped == "/abort":
        agent.abort()
        return "[aborting]"

    if stripped.startswith("/steer "):
        message = stripped[len("/steer "):].strip()
        if not message:
            return None
        agent.steer(message)
        return "[queued steering]"

    if stripped.startswith("/followup "):
        message = stripped[len("/followup "):].strip()
        if not message:
            return None
        agent.follow_up(message)
        return "[queued follow-up]"

    agent.follow_up(stripped)
    return "[queued follow-up]"


class PyAgentApp(App):
    CSS = """
    #output {
        height: 1fr;
        border: solid $primary;
    }
    #status {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    #hints {
        height: 1;
        background: $panel-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    #prompt {
        dock: bottom;
    }
    """

    def __init__(self, agent: Agent, cwd: str, session_path: Path) -> None:
        super().__init__()
        self.agent = agent
        self.cwd = cwd
        self.session_path = session_path
        self._run_task: Optional[asyncio.Task] = None
        self._assistant_buffer = ""
        self._steer_count = 0
        self._follow_up_count = 0

    def compose(self) -> ComposeResult:
        yield Vertical(
            RichLog(id="output", wrap=True, markup=True, highlight=False),
            Static(self._status_text(), id="status"),
            Static(self._hints_text(""), id="hints"),
            Input(
                id="prompt",
                placeholder="message, or /new /steer /followup /abort /save /load /quit",
                suggester=CommandSuggester(self),
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#output", RichLog)
        log.write(
            f"py-agent ready. cwd={self.cwd}. session={self.session_path}. "
            f"Commands: /new, /save <file>, /load <file>, /quit. "
            f"While running: type to queue a follow-up, /steer <msg> to inject, /abort to stop."
        )
        self.query_one("#prompt", Input).focus()

    def _status_text(self) -> str:
        state = "running…" if self.agent.is_streaming else "idle"
        return f"[{state}] steering:{self._steer_count} follow-up:{self._follow_up_count}"

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())
        self._refresh_hints(self.query_one("#prompt", Input).value)

    def _hints_text(self, value: str) -> str:
        commands = STREAMING_COMMANDS if self.agent.is_streaming else IDLE_COMMANDS
        if value.startswith("/") and " " not in value:
            matches = {cmd: desc for cmd, desc in commands.items() if cmd.startswith(value)}
        else:
            matches = commands
        if not matches:
            return "no matching commands"
        return "  ".join(f"{cmd} ({desc})" for cmd, desc in matches.items())

    def _refresh_hints(self, value: str = "") -> None:
        self.query_one("#hints", Static).update(self._hints_text(value))

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_hints(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value
        self.query_one("#prompt", Input).value = ""

        stripped = line.strip()
        if not stripped:
            return

        log = self.query_one("#output", RichLog)

        if self.agent.is_streaming:
            status = route_streaming_line(self.agent, line)
            if status == "[queued steering]":
                self._steer_count += 1
            elif status == "[queued follow-up]":
                self._follow_up_count += 1
            if status is not None:
                log.write(status)
            self._refresh_status()
            return

        if stripped in ("/quit", "/exit"):
            self.exit()
            return

        if stripped == "/new":
            try:
                self.agent.reset()
            except RuntimeError as exc:
                log.write(f"[error] {exc}")
                return
            self.session_path = new_session_path(self.cwd)
            self._assistant_buffer = ""
            self._steer_count = 0
            self._follow_up_count = 0
            log.clear()
            log.write(f"[new session] {self.session_path}")
            self._refresh_status()
            return

        if stripped.startswith("/save "):
            target = stripped[len("/save "):].strip()
            try:
                save_session(target, self.agent.messages)
                log.write(f"[saved {len(self.agent.messages)} messages to {target}]")
            except Exception as exc:
                log.write(f"[error] {exc}")
            return

        if stripped.startswith("/load "):
            target = stripped[len("/load "):].strip()
            try:
                self.agent.messages = load_session(target)
                log.write(f"[loaded {len(self.agent.messages)} messages from {target}]")
            except Exception as exc:
                log.write(f"[error] {exc}")
            return

        self._steer_count = 0
        self._follow_up_count = 0
        self._run_task = asyncio.ensure_future(self._drive_prompt(line))
        self._refresh_status()

    async def _drive_prompt(self, line: str) -> None:
        log = self.query_one("#output", RichLog)
        try:
            await self.agent.prompt(line)
        except Exception as exc:
            log.write(f"[error] {exc}")
        finally:
            try:
                save_session(str(self.session_path), self.agent.messages)
            except Exception as exc:
                log.write(f"[error] auto-save failed: {exc}")
            self._refresh_status()

    def _flush_assistant_buffer(self) -> None:
        if self._assistant_buffer:
            self.query_one("#output", RichLog).write(self._assistant_buffer)
            self._assistant_buffer = ""

    def on_agent_event(self, event) -> None:
        log = self.query_one("#output", RichLog)

        if event.type == "message_update":
            llm_event = event.llm_event
            if llm_event.type == "text_delta":
                self._assistant_buffer += llm_event.delta
        elif event.type == "message_end" and event.message.role == "assistant":
            message = event.message
            if getattr(message, "stop_reason", None) == "error":
                self._flush_assistant_buffer()
                detail = message.error_message or (
                    message.content[0].text if message.content else "unknown error"
                )
                log.write(f"[error] {detail}")
            else:
                self._flush_assistant_buffer()
        elif event.type == "tool_execution_start":
            self._flush_assistant_buffer()
            log.write(f"[tool] {event.tool_name}({event.args})")
        elif event.type == "tool_execution_end":
            text = event.result.content[0].text if event.result.content else ""
            status = "error" if event.is_error else "done"
            log.write(f"[tool {status}] {text[:300]}")
