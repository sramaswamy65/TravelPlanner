from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import Settings
from src.nebius_client import NebiusClient


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        function = SimpleNamespace(
            name="get_weather", arguments='{"city":"Chicago","date":"2026-09-05"}'
        )
        tool_call = SimpleNamespace(id="nebius-call-123", function=function)
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


async def test_nebius_uses_configured_model_and_openai_tool_format():
    settings = Settings(
        nebius_api_key="test-only-key",
        nebius_model="publisher/model-from-env",
        nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
    )
    client = NebiusClient(settings)
    fake = FakeCompletions()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    messages = [{"role": "user", "content": "Weather?"}]
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    turn = await client.complete(messages, tools)
    assert fake.kwargs["model"] == "publisher/model-from-env"
    assert fake.kwargs["messages"] is messages
    assert fake.kwargs["tools"] is tools
    assert fake.kwargs["tool_choice"] == "auto"
    assert turn.tool_calls[0].id == "nebius-call-123"
    assert turn.tool_calls[0].arguments["city"] == "Chicago"
    assert turn.token_usage["total_tokens"] == 14


def test_nebius_never_substitutes_a_missing_model():
    with pytest.raises(ValueError, match="NEBIUS_MODEL"):
        NebiusClient(Settings(nebius_api_key="test-only-key", nebius_model=""))
