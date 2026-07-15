from __future__ import annotations

from conftest import make_assistant_message, make_faux_stream_fn

from py_agent.utils.compaction import (
    SUMMARY_MARKER,
    compact,
    estimate_tokens,
    find_cut_point,
    should_compact,
)
from py_agent.utils.types import (
    EMPTY_USAGE,
    AssistantMessage,
    CompactionSettings,
    CostBreakdown,
    TextContent,
    ToolCall,
    ToolResultContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _user(text):
    return UserMessage(content=[TextContent(text=text)])


def _assistant(text, tokens=0):
    usage = EMPTY_USAGE
    if tokens:
        usage = Usage(
            input=tokens, output=0, cache_read=0, cache_write=0, total_tokens=tokens,
            cost=CostBreakdown(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
    return AssistantMessage(
        content=[TextContent(text=text)], model="faux", provider="faux",
        usage=usage, stop_reason="end_turn",
    )


def test_estimate_tokens_uses_usage():
    messages = [_user("hi"), _assistant("done", tokens=12345)]
    assert estimate_tokens(messages) == 12345


def test_should_compact_threshold():
    settings = CompactionSettings(context_window=1000, reserve_tokens=200)
    below = [_assistant("x", tokens=700)]
    above = [_assistant("x", tokens=900)]
    assert should_compact(below, settings) is False
    assert should_compact(above, settings) is True


def test_cut_point_never_splits_tool_result():
    messages = [
        _user("do it"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="bash", arguments={})],
            model="faux", provider="faux", usage=EMPTY_USAGE, stop_reason="tool_use",
        ),
        ToolResultMessage(tool_call_id="c1", content=[ToolResultContent(text="out")]),
        _user("thanks"),
    ]
    cut = find_cut_point(messages, keep_recent_tokens=1)
    assert not isinstance(messages[cut], ToolResultMessage) or cut == len(messages)


async def test_generate_summary_uses_faux_stream():
    stream_fn = make_faux_stream_fn(
        [make_assistant_message([TextContent(text="## Goal\nBuild a thing")])]
    )
    settings = CompactionSettings(context_window=1000, reserve_tokens=200, keep_recent_tokens=1)
    messages = [_user("old 1"), _assistant("old 2"), _user("recent")]
    result = await compact(messages, "faux", stream_fn, settings, api_key=None)
    assert result[0].content[0].text.startswith(SUMMARY_MARKER)
    assert "Build a thing" in result[0].content[0].text


async def test_compaction_replaces_head_keeps_tail():
    stream_fn = make_faux_stream_fn(
        [make_assistant_message([TextContent(text="summary body")])]
    )
    settings = CompactionSettings(context_window=1000, reserve_tokens=200, keep_recent_tokens=1)
    messages = [_user("head a"), _assistant("head b"), _user("tail keep")]
    result = await compact(messages, "faux", stream_fn, settings, api_key=None)

    assert len(result) < len(messages)
    assert result[0].content[0].text.startswith(SUMMARY_MARKER)
    assert result[-1].content[0].text == "tail keep"

    again = await compact(result, "faux", make_faux_stream_fn(
        [make_assistant_message([TextContent(text="should not run")])]
    ), settings, api_key=None)
    assert again == result
