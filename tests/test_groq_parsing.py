from __future__ import annotations

from py_agent.llm.groq import parse_pseudo_function_calls


def test_parses_slash_variant():
    text = '<function/bash {"command": "pip install fastapi"}></function>'
    calls = parse_pseudo_function_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"command": "pip install fastapi"}


def test_parses_equals_variant():
    text = '<function=write>{"path": "main.py", "content": "print(1)"}</function>'
    calls = parse_pseudo_function_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert calls[0].arguments == {"path": "main.py", "content": "print(1)"}


def test_parses_multiple_calls():
    text = (
        '<function/bash {"command": "ls"}></function>\n'
        '<function/read {"path": "a.py"}></function>'
    )
    calls = parse_pseudo_function_calls(text)
    assert [c.name for c in calls] == ["bash", "read"]


def test_no_match_returns_empty():
    text = "Here is a plain explanation with no function calls at all."
    assert parse_pseudo_function_calls(text) == []


def test_does_not_misfire_on_code_block_mentioning_function():
    text = "```python\ndef function_name():\n    pass\n```"
    assert parse_pseudo_function_calls(text) == []


def test_malformed_json_is_skipped():
    text = '<function/bash {not valid json}></function>'
    assert parse_pseudo_function_calls(text) == []
