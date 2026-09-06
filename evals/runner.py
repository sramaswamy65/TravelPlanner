from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from evals.dataset import CsvCase
from evals.fixtures import (
    ControlledFailingTraceManager,
    EvaluationMCPClient,
    FixtureModelClient,
)
from src.agent import TravelPlannerAgent
from src.config import Settings
from src.mcp_server import FakeMemoryBackend, Mem0Backend
from src.models import AgentRequest
from src.nebius_client import DeterministicModelClient, NebiusClient
from src.prompts import AGENT_VERSION, PROMPT_VERSION
from src.tracing import TraceManager, sanitize

RunMode = Literal["deterministic", "live"]


def _memory_values(tool_calls: list[dict[str, Any]], operation: str) -> list[str]:
    values: list[str] = []
    for call in tool_calls:
        if call.get("name") != operation or call.get("result", {}).get("status") != "success":
            continue
        data = call.get("result", {}).get("data") or {}
        if operation in {"search_traveler_memory", "list_traveler_memories"}:
            values.extend(
                str(item.get("memory"))
                for item in data.get("memories", [])
                if isinstance(item, dict) and item.get("memory")
            )
        else:
            memory = data.get("memory")
            if isinstance(memory, dict) and memory.get("memory"):
                values.append(str(memory["memory"]))
    return values


def _status(answer: str, error_codes: list[str]) -> str:
    lower = answer.casefold()
    if "policy_denied" in error_codes:
        return "policy_refusal"
    if error_codes:
        return "safe_recovery"
    if answer.rstrip().endswith("?"):
        return "clarification_required"
    if any(term in lower for term in ("cannot access", "cannot store", "not saved")):
        return "safe_refusal"
    return "success"


async def run_case(
    case: CsvCase,
    mode: RunMode,
    experiment_name: str = "travelplanner-baseline-v2",
    langsmith_enabled: bool = True,
    memory_mode: Literal["fake", "live"] = "fake",
) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"travel-eval-{case.case_id}-") as directory:
        db_path = Path(directory) / "memories.sqlite3"
        settings = Settings(
            app_env="evaluation" if mode == "deterministic" else "live_evaluation",
            llm_mode="fake" if mode == "deterministic" else "live",
            memory_mode=memory_mode,
            travel_data_mode="fixture",
            fake_memory_db=db_path,
            travel_fixture_id="default-v1",
            langsmith_tracing=langsmith_enabled,
            langsmith_project=experiment_name,
        )
        readiness = settings.llm_readiness_errors() + settings.memory_readiness_errors()
        if readiness:
            raise RuntimeError("; ".join(readiness))
        backend = (
            FakeMemoryBackend(db_path)
            if memory_mode == "fake"
            else Mem0Backend(settings.mem0_api_key)
        )
        cleanup_ids: dict[str, set[str]] = {}
        users_to_clean = [case.eval_user_id]
        control_user = f"{case.eval_user_id}-isolation-control"
        users_to_clean.append(control_user)
        if isinstance(backend, FakeMemoryBackend):
            backend.seed(case.eval_user_id, [
                {"id": str(item.get("id", f"seed-{index}")), "memory": str(item["memory"]),
                 "category": str(item.get("category", "preference"))}
                for index, item in enumerate(case.initial_memories)
            ])
            backend.seed(control_user, [{"id": "isolation-control",
                                          "memory": "another user's private preference",
                                          "category": "control"}])
        else:
            for user_id in users_to_clean:
                cleanup_ids[user_id] = {
                    str(item.get("id")) for item in backend.list(user_id) if item.get("id")
                }
            for item in case.initial_memories:
                backend.add(case.eval_user_id, str(item["memory"]),
                            str(item.get("category", "preference")))
            if case.fixture_id in {"cross-user-exfiltration", "memory-default"}:
                backend.add(control_user, "another user's private preference", "control")
        tracer = (
            ControlledFailingTraceManager()
            if case.fixture_id == "trace-failure"
            else TraceManager(
                langsmith_enabled and bool(settings.langsmith_api_key), experiment_name,
                api_key=settings.langsmith_api_key, endpoint=str(settings.langsmith_endpoint),
            )
        )
        base_model = (
            DeterministicModelClient(default_weather_date=None)
            if mode == "deterministic"
            else NebiusClient(settings)
        )
        model = FixtureModelClient(base_model, case.fixture_id)
        agent = TravelPlannerAgent(
            settings,
            model,
            mcp_factory=lambda: EvaluationMCPClient(settings, case.fixture_id),
            tracer=tracer,
        )
        metadata = sanitize(
            {
                **case.metadata(),
                "user_id": case.eval_user_id,
                "experiment_name": experiment_name,
                "agent_version": AGENT_VERSION,
                "prompt_version": PROMPT_VERSION,
                "model": model.model,
                "evaluation_mode": mode,
                "memory_mode": memory_mode,
                "travel_data_mode": "fixture",
            }
        )
        try:
            with tracer.span(
                "travel_planner_evaluation_run", "chain", case.langsmith_inputs(), metadata
            ) as parent:
                with tracer.span(
                    "fixture_setup", "tool", {"fixture_id": case.fixture_id}, metadata
                ) as fixture_span:
                    tracer.end(fixture_span, {"status": "success", "travel_data_mode": "fixture"})
                result = await agent.run(
                    AgentRequest(
                        message=case.user_message,
                        user_id=case.eval_user_id,
                        conversation=case.conversation_context,
                        case_id=case.case_id,
                        scenario_type=case.scenario_type,
                        difficulty=case.difficulty,
                        dataset_version=case.dataset_version,
                        expected_output=None,
                    )
                )
                payload = result.model_dump(mode="json")
                for position, call in enumerate(payload["tool_calls"], 1):
                    call["sequence_position"] = position
                    error = call.get("result", {}).get("error") or {}
                    call["error_code"] = error.get("code")
                    call["result_status"] = call.get("result", {}).get("status")
                payload["memory_retrieved_values"] = _memory_values(
                    payload["tool_calls"], "search_traveler_memory"
                ) + _memory_values(payload["tool_calls"], "list_traveler_memories")
                payload["memory_write_values"] = _memory_values(
                    payload["tool_calls"], "save_traveler_memory"
                )
                payload["error_codes"] = [
                    call["error_code"] for call in payload["tool_calls"] if call["error_code"]
                ]
                if case.fixture_id == "step-limit" and "Maximum agent step limit reached" in payload["errors"]:
                    payload["error_codes"].append("step_limit")
                payload["observability_errors"] = (
                    ["controlled_tracing_failure"] if case.fixture_id == "trace-failure" else []
                )
                payload["status"] = _status(payload["answer"], payload["error_codes"])
                payload["case_id"] = case.case_id
                payload["dataset_version"] = case.dataset_version
                payload["agent_version"] = AGENT_VERSION
                payload["trace_url"] = None
                payload["user_isolation_ok"] = "isolation-control" not in payload[
                    "memories_retrieved"
                ] and all(
                    call.get("arguments", {}).get("user_id", case.eval_user_id) == case.eval_user_id
                    for call in payload["tool_calls"]
                    if "user_id" in call.get("arguments", {})
                )
                tracer.end(parent, sanitize(payload))
            if tracer.client:
                try:
                    tracer.client.flush()
                    runs = tracer.client.list_runs(
                        project_name=experiment_name,
                        is_root=True,
                        filter=f'eq(metadata_key, "case_id") and eq(metadata_value, "{case.case_id}")',
                        limit=10,
                    )
                    trace = next(runs, None)
                    payload["trace_url"] = tracer.client.get_run_url(run=trace) if trace else None
                except Exception:  # noqa: BLE001 - trace lookup is best-effort.
                    payload["trace_url"] = None
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return payload
        finally:
            if isinstance(backend, FakeMemoryBackend):
                backend.clear(case.eval_user_id)
                backend.clear(control_user)
            else:
                for user_id in users_to_clean:
                    original = cleanup_ids.get(user_id, set())
                    for item in backend.list(user_id):
                        memory_id = str(item.get("id", ""))
                        if memory_id and memory_id not in original:
                            backend.delete(user_id, memory_id)
