from __future__ import annotations

import pytest

from src.nebius_client import DeterministicModelClient


@pytest.mark.parametrize(
    ("message", "expected_tool"),
    [
        ("What do you remember about me?", "list_traveler_memories"),
        ("Find museums in Chicago", "find_attractions"),
    ],
)
async def test_deterministic_model_selects_evidence_supported_tool(message, expected_tool):
    turn = await DeterministicModelClient().complete([{"role": "user", "content": message}], [])
    assert [call.name for call in turn.tool_calls] == [expected_tool]


async def test_deterministic_model_asks_for_ambiguous_missing_weather_date():
    turn = await DeterministicModelClient().complete(
        [{"role": "user", "content": "Plan a Chicago day but do not use memory"}], []
    )
    assert turn.tool_calls == []
    assert "what date" in (turn.content or "").lower()


async def test_deterministic_model_marks_explicit_current_weather():
    turn = await DeterministicModelClient().complete(
        [{"role": "user", "content": "What is the current weather in Chicago?"}], []
    )
    assert turn.tool_calls[0].arguments == {"city": "Chicago", "current": True}


async def test_deterministic_model_stops_after_required_tool_error():
    messages = [
        {"role": "user", "content": "Plan a day in Atlantis on 2026-09-05"},
        {
            "role": "tool",
            "name": "get_weather",
            "content": '{"status":"error","error":{"code":"unsupported_city"}}',
        },
    ]
    turn = await DeterministicModelClient().complete(messages, [])
    assert turn.tool_calls == []
    assert "could not complete" in (turn.content or "").lower()
