from __future__ import annotations

import httpx
from groq import APIError, APIStatusError

from py_agent.llm.groq import _describe_error, recover_failed_generation


def _make_status_error(status_code: int, body):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json=body if isinstance(body, dict) else None)
    return APIStatusError(message="Error", response=response, body=body)


def test_extracts_failed_generation_detail():
    body = {
        "error": {
            "message": "Failed to call a function.",
            "failed_generation": 'bash{"command": "pip install fastapi"}',
        }
    }
    exc = _make_status_error(400, body)
    detail = _describe_error(exc)
    assert "Groq API error (400)" in detail
    assert "Failed to call a function." in detail
    assert "failed_generation" in detail
    assert "pip install fastapi" in detail


def test_falls_back_to_message_when_no_body():
    exc = _make_status_error(400, None)
    detail = _describe_error(exc)
    assert "Groq API error (400)" in detail


def test_non_api_error_returns_str():
    detail = _describe_error(ValueError("boom"))
    assert detail == "boom"


def test_extracts_failed_generation_from_midstream_unwrapped_body():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {
        "message": "Failed to call a function.",
        "failed_generation": 'bash{"command": "pip install fastapi"}',
    }
    exc = APIError(message="Failed to call a function.", request=request, body=body)
    detail = _describe_error(exc)
    assert "Groq API error" in detail
    assert "Failed to call a function." in detail
    assert "pip install fastapi" in detail


def _make_api_error(failed_generation):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"message": "Tool call validation failed", "failed_generation": failed_generation}
    return APIError(message="Tool call validation failed", request=request, body=body)


def test_recovers_hallucinated_tool_call():
    exc = _make_api_error('{"name": "repo_browser.print_tree", "arguments": {"path": "", "depth": 2}}')
    calls = recover_failed_generation(exc)
    assert len(calls) == 1
    assert calls[0].name == "repo_browser.print_tree"
    assert calls[0].arguments == {"path": "", "depth": 2}


def test_recovers_string_arguments():
    exc = _make_api_error('{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}')
    calls = recover_failed_generation(exc)
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"command": "ls"}


def test_recovers_pseudo_xml_failed_generation():
    exc = _make_api_error('<function/bash {"command": "ls"}></function>')
    calls = recover_failed_generation(exc)
    assert len(calls) == 1
    assert calls[0].name == "bash"


def test_recover_returns_empty_for_garbage():
    exc = _make_api_error("complete nonsense not parseable")
    assert recover_failed_generation(exc) == []
    assert recover_failed_generation(ValueError("boom")) == []
