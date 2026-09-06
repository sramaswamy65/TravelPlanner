from __future__ import annotations

from src.agent import TravelPlannerAgent
from src.config import Settings
from src.models import AgentRequest, ToolResult
from src.nebius_client import DeterministicModelClient
from tests.test_agent_loop import FakeMCP
from tests.test_tracing import RecordingTracer

REQUEST = "Remember, I am a vegetarian. Can you please look up some good restaurants?"
CONVERSATION = [{"role": "user", "content": "What is the weather in Boston tomorrow?"}]


async def test_compound_request_writes_once_then_searches_inherited_city():
    tracer = RecordingTracer()
    mcp = FakeMCP(
        {
            "save_traveler_memory": ToolResult(
                status="success", data={"memory": {"id": "diet-1", "memory": "Is vegetarian"}}
            ),
            "search_restaurants": ToolResult(
                status="capability_unavailable",
                data={"source": "unavailable live restaurant provider"},
                error={
                    "code": "capability_unavailable",
                    "message": "Live restaurant search is not currently configured",
                },
            ),
        }
    )
    result = await TravelPlannerAgent(
        Settings(app_env="test"),
        DeterministicModelClient(),
        mcp_factory=lambda: mcp,
        tracer=tracer,
    ).run(AgentRequest(message=REQUEST, user_id="alice", conversation=CONVERSATION))
    assert [name for name, _ in mcp.calls] == ["save_traveler_memory", "search_restaurants"]
    assert mcp.calls[0][1] == {
        "user_id": "alice",
        "memory": "Is vegetarian",
        "category": "dietary_preference",
    }
    assert mcp.calls[1][1]["city"] == "Boston"
    assert "remembered that you are vegetarian" in result.answer
    assert "Boston" in result.answer
    assert "Live restaurant search is not currently configured" in result.answer
    assert "no restaurants" not in result.answer.casefold()
    assert result.memories_written == ["diet-1"]
    memory_trace = next(
        event for event in tracer.events if event["name"] == "memory.save_traveler_memory"
    )
    assert memory_trace["outputs"]["data"]["memory"]["id"] == "diet-1"


async def test_failed_memory_write_does_not_block_successful_restaurant_search():
    mcp = FakeMCP(
        {
            "save_traveler_memory": ToolResult(
                status="error", error={"code": "memory_error", "message": "Memory save failed"}
            ),
            "search_restaurants": ToolResult(
                status="success",
                data={
                    "restaurants": [{"name": "Boston Plant Kitchen"}],
                    "source": "local restaurant fixture",
                },
            ),
        }
    )
    result = await TravelPlannerAgent(
        Settings(app_env="test"), DeterministicModelClient(), mcp_factory=lambda: mcp
    ).run(AgentRequest(message=REQUEST, user_id="alice", conversation=CONVERSATION))
    assert [name for name, _ in mcp.calls] == ["save_traveler_memory", "search_restaurants"]
    assert "couldn't save" in result.answer
    assert "Boston Plant Kitchen" in result.answer


async def test_restaurant_request_without_city_asks_clarification_and_calls_no_tool():
    mcp = FakeMCP()
    result = await TravelPlannerAgent(
        Settings(app_env="test"), DeterministicModelClient(), mcp_factory=lambda: mcp
    ).run(AgentRequest(message="Find some restaurants", user_id="alice"))
    assert mcp.calls == []
    assert "which city" in result.answer.casefold()
