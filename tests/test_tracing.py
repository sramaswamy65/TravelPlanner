from __future__ import annotations

from src.agent import TravelPlannerAgent
from src.config import Settings
from src.models import AgentRequest, ModelToolCall, ModelTurn, ToolResult
from src.tracing import TraceManager, sanitize
from tests.test_agent_loop import FakeMCP, ScriptedModel


class RecordingTracer:
    def __init__(self):
        self.events = []
        self.stack = []

    def span(self, name, run_type, inputs, metadata=None):
        tracer = self

        class Span:
            def __enter__(self):
                event = {
                    "name": name,
                    "run_type": run_type,
                    "inputs": inputs,
                    "metadata": metadata,
                    "parent": tracer.stack[-1] if tracer.stack else None,
                }
                tracer.events.append(event)
                tracer.stack.append(name)
                return event

            def __exit__(self, *_):
                tracer.stack.pop()
                return False

        return Span()

    @staticmethod
    def end(run, outputs=None, error=None):
        if run is not None:
            run["outputs"] = outputs
            run["error"] = error


async def test_parent_and_required_child_traces_have_evaluation_metadata():
    tracer = RecordingTracer()
    mcp = FakeMCP(
        {
            "search_traveler_memory": ToolResult(
                status="success", data={"memories": [{"id": "m1", "memory": "I prefer art"}]}
            )
        }
    )
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=[
                    ModelToolCall(
                        id="c1",
                        name="search_traveler_memory",
                        arguments={"user_id": "u", "query": "art"},
                    )
                ]
            ),
            ModelTurn(content="A grounded plan"),
        ]
    )
    request = AgentRequest(
        message="Plan an art day",
        user_id="u",
        case_id="case-1",
        expected_output={"answer": "reference"},
    )
    await TravelPlannerAgent(
        Settings(app_env="test"), model, mcp_factory=lambda: mcp, tracer=tracer
    ).run(request)
    names = [event["name"] for event in tracer.events]
    assert names.count("travel_planner_run") == 1
    assert {
        "nebius_model_call",
        "mcp.search_traveler_memory",
        "memory.search_traveler_memory",
        "final_response_construction",
    } <= set(names)
    assert all(event["metadata"]["case_id"] == "case-1" for event in tracer.events)
    assert all(
        "correctness" in event["metadata"] and "dataset_version" in event["metadata"]
        for event in tracer.events
    )
    assert (
        next(event for event in tracer.events if event["name"] == "mcp.search_traveler_memory")[
            "parent"
        ]
        == "travel_planner_run"
    )
    assert (
        next(event for event in tracer.events if event["name"] == "memory.search_traveler_memory")[
            "parent"
        ]
        == "mcp.search_traveler_memory"
    )


async def test_langsmith_transport_failure_does_not_break_agent(monkeypatch):
    import langsmith

    def broken_trace(*args, **kwargs):
        raise RuntimeError("trace transport failed")

    monkeypatch.setattr(langsmith, "trace", broken_trace)
    tracer = TraceManager(enabled=True, project="test")
    result = await TravelPlannerAgent(
        Settings(app_env="test"),
        ScriptedModel([ModelTurn(content="Still works")]),
        mcp_factory=lambda: FakeMCP(),
        tracer=tracer,
    ).run(AgentRequest(message="Hello", user_id="u"))
    assert result.answer == "Still works"


def test_trace_sanitization_redacts_secret_values_recursively():
    cleaned = sanitize(
        {"message": "my API key is abc123", "token": "tok-value", "nested": ["password: hunter2"]}
    )
    rendered = str(cleaned)
    assert "abc123" not in rendered and "tok-value" not in rendered and "hunter2" not in rendered
    assert rendered.count("[REDACTED]") == 3


def test_trace_manager_builds_client_from_explicit_settings(monkeypatch):
    import langsmith

    captured = {}
    sentinel = object()

    def fake_client(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(langsmith, "Client", fake_client)
    tracer = TraceManager(
        True, "project", api_key="configured-key", endpoint="https://example.invalid"
    )
    assert tracer.client is sentinel
    assert captured == {"api_key": "configured-key", "api_url": "https://example.invalid"}
