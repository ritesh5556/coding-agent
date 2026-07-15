from ...utils.types import AgentTool
from .bash_tool import create_bash_tool
from .edit_tool import create_edit_tool
from .find_tool import create_find_tool
from .grep_tool import create_grep_tool
from .ls_tool import create_ls_tool
from .read_tool import create_read_tool
from .task_tool import create_task_tool
from .todo_tool import create_todo_tool
from .write_tool import create_write_tool


def create_coding_tools(cwd: str) -> list[AgentTool]:
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
        create_todo_tool(cwd),
    ]


__all__ = [
    "create_coding_tools",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
    "create_todo_tool",
    "create_task_tool",
]
