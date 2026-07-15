from ...utils.types import AgentTool
from .bash_tool import create_bash_tool
from .edit_tool import create_edit_tool
from .read_tool import create_read_tool
from .write_tool import create_write_tool


def create_coding_tools(cwd: str) -> list[AgentTool]:
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


__all__ = [
    "create_coding_tools",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
]
