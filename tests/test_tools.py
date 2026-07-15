from __future__ import annotations

import jsonschema

from py_agent.tools.builtin import (
    create_coding_tools,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
)


def _text(result):
    return "\n".join(c.text for c in result.content)


def _make_tree(tmp_path):
    (tmp_path / "a.py").write_text("import os\nfoo = 1\nFOO = 2\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello foo world\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("bar = foo\n", encoding="utf-8")
    return tmp_path


async def test_grep_finds_matches(tmp_path):
    _make_tree(tmp_path)
    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "foo"}, None, None)
    text = _text(result)
    assert "a.py" in text
    assert "b.txt" in text


async def test_grep_ignore_case(tmp_path):
    _make_tree(tmp_path)
    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "foo", "ignore_case": True}, None, None)
    assert "FOO = 2" in _text(result)


async def test_grep_glob_filter(tmp_path):
    _make_tree(tmp_path)
    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "foo", "glob": "*.py"}, None, None)
    text = _text(result)
    assert "a.py" in text
    assert "b.txt" not in text


async def test_grep_literal(tmp_path):
    _make_tree(tmp_path)
    (tmp_path / "re.txt").write_text("a.b\n", encoding="utf-8")
    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "a.b", "literal": True}, None, None)
    assert "re.txt" in _text(result)


async def test_grep_limit(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("match\n", encoding="utf-8")
    tool = create_grep_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "match", "limit": 3}, None, None)
    lines = [l for l in _text(result).splitlines() if ":" in l]
    assert len(lines) <= 3


async def test_find_glob(tmp_path):
    _make_tree(tmp_path)
    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "**/*.py"}, None, None)
    text = _text(result)
    assert "a.py" in text
    assert "sub/c.py" in text
    assert "b.txt" not in text


async def test_find_no_match(tmp_path):
    _make_tree(tmp_path)
    tool = create_find_tool(str(tmp_path))
    result = await tool.execute("c1", {"pattern": "*.rs"}, None, None)
    assert "No files found" in _text(result)


async def test_ls_lists_with_dir_marker(tmp_path):
    _make_tree(tmp_path)
    tool = create_ls_tool(str(tmp_path))
    result = await tool.execute("c1", {}, None, None)
    text = _text(result)
    assert "a.py" in text
    assert "sub/" in text


async def test_ls_missing_path_errors(tmp_path):
    tool = create_ls_tool(str(tmp_path))
    result = await tool.execute("c1", {"path": "nope"}, None, None)
    assert result.is_error


def test_create_coding_tools_registers_all(tmp_path):
    tools = create_coding_tools(str(tmp_path))
    names = {t.name for t in tools}
    assert names == {"read", "bash", "edit", "write", "grep", "find", "ls", "todo_write"}


def test_tool_params_validate(tmp_path):
    for tool in create_coding_tools(str(tmp_path)):
        jsonschema.Draft7Validator.check_schema(tool.parameters)
