"""py-agent — a lite coding agent (Python port of the pi agent architecture)."""

__version__ = "0.1.0"

from .agent import Agent
from .agent_loop import run_agent_loop, run_agent_loop_continue
from .llm.groq import groq_stream
from .stream import AssistantMessageEventStream, EventStream
from .tools.builtin import create_coding_tools
from .tools.executor import execute_tool_calls
from .utils.system_prompt import build_system_prompt
from .utils.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)

__all__ = [
    "__version__",
    "Agent",
    "run_agent_loop",
    "run_agent_loop_continue",
    "groq_stream",
    "EventStream",
    "AssistantMessageEventStream",
    "create_coding_tools",
    "execute_tool_calls",
    "build_system_prompt",
    "AgentContext",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentTool",
    "AgentToolResult",
    "AssistantMessage",
    "ToolResultMessage",
    "UserMessage",
]
