from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

JSON_COLUMNS = {
    "conversation_context",
    "initial_memories",
    "expected_tools",
    "expected_tool_arguments",
    "forbidden_tools",
    "expected_retrieved_memories",
    "expected_memory_writes",
    "forbidden_memory_writes",
    "must_include",
    "must_not_include",
}

REAL_TOOLS = {
    "get_weather",
    "find_attractions",
    "search_restaurants",
    "get_attraction_details",
    "search_traveler_memory",
    "save_traveler_memory",
    "list_traveler_memories",
    "delete_traveler_memory",
    "save_itinerary",
}

FIXTURE_IDS = {
    "weather-kathmandu",
    "weather-boston",
    "weather-chicago",
    "weather-san-francisco",
    "attractions-kathmandu",
    "attractions-boston",
    "attractions-chicago",
    "attractions-san-francisco",
    "restaurants-boston",
    "restaurants-chicago",
    "restaurants-san-francisco",
    "restaurants-kathmandu",
    "attraction-details",
    "memory-default",
    "clarification",
    "memory-policy",
    "unsupported-city",
    "invalid-date",
    "empty-attractions",
    "unknown-attraction",
    "unsupported-restaurants",
    "ambiguous-city",
    "closed-attractions",
    "restaurant-limit",
    "outside-range",
    "unsupported-category",
    "weather-timeout",
    "malformed-attractions",
    "weather-error",
    "restaurant-timeout",
    "malformed-details",
    "write-no-approval",
    "unsafe-path",
    "absolute-path",
    "existing-file",
    "missing-memory",
    "memory-timeout",
    "memory-write-error",
    "duplicate-model-calls",
    "step-limit",
    "trace-failure",
    "memory-injection",
    "tool-injection",
    "path-traversal",
    "cross-user-exfiltration",
    "secret-exfiltration",
}

TOOL_ARGUMENTS = {
    "get_weather": {"city", "date", "current"},
    "find_attractions": {"city", "category"},
    "search_restaurants": {"city", "dietary_preference", "limit"},
    "get_attraction_details": {"name"},
    "search_traveler_memory": {"user_id", "query"},
    "save_traveler_memory": {"user_id", "memory", "category"},
    "list_traveler_memories": {"user_id"},
    "delete_traveler_memory": {"user_id", "memory_id"},
    "save_itinerary": {"user_id", "filename", "content", "approved"},
}


class CsvCase(BaseModel):
    case_id: str = Field(min_length=1)
    dataset_version: str
    scenario_type: str
    difficulty: str
    fixture_id: str
    eval_user_id: str = Field(min_length=1)
    llm_mode: str
    memory_mode: str
    travel_data_mode: str
    user_message: str = Field(min_length=1)
    conversation_context: list[dict[str, Any]]
    initial_memories: list[dict[str, Any]]
    expected_tools: list[str]
    tool_sequence_mode: str
    expected_tool_arguments: dict[str, dict[str, Any]]
    forbidden_tools: list[str]
    max_tool_calls: int = Field(ge=0)
    expected_retrieved_memories: list[str]
    expected_memory_writes: list[str]
    forbidden_memory_writes: list[str]
    must_include: list[str]
    must_not_include: list[str]
    expected_status: str = Field(min_length=1)
    expected_error: str
    evaluation_notes: str

    def langsmith_inputs(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "message": self.user_message,
            "user_id": self.eval_user_id,
            "conversation_context": self.conversation_context,
            "initial_memories": self.initial_memories,
            "fixture_id": self.fixture_id,
            "llm_mode": self.llm_mode,
            "memory_mode": self.memory_mode,
            "travel_data_mode": self.travel_data_mode,
        }

    def reference_outputs(self) -> dict[str, Any]:
        keys = (
            "expected_tools",
            "tool_sequence_mode",
            "expected_tool_arguments",
            "forbidden_tools",
            "max_tool_calls",
            "expected_retrieved_memories",
            "expected_memory_writes",
            "forbidden_memory_writes",
            "must_include",
            "must_not_include",
            "expected_status",
            "expected_error",
        )
        return {key: getattr(self, key) for key in keys}

    def metadata(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "case_id",
                "dataset_version",
                "scenario_type",
                "difficulty",
                "fixture_id",
                "evaluation_notes",
            )
        }


def load_csv(path: Path) -> tuple[list[CsvCase], dict[str, Any]]:
    errors: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(CsvCase.model_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)
    cases: list[CsvCase] = []
    for index, row in enumerate(raw_rows, 2):
        case_id = row.get("case_id") or f"row-{index}"
        for column in JSON_COLUMNS:
            try:
                row[column] = json.loads(row[column])
            except json.JSONDecodeError as exc:
                raise ValueError(f"{case_id}: malformed JSON in {column}: {exc}") from exc
        try:
            row["max_tool_calls"] = int(row["max_tool_calls"])
            case = CsvCase.model_validate(row)
        except ValueError as exc:
            raise ValueError(f"{case_id}: {exc}") from exc
        cases.append(case)
    ids = [case.case_id for case in cases]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case IDs: {duplicates}")
    for case in cases:
        if case.dataset_version != "v2":
            errors.append(
                {"case_id": case.case_id, "field": "dataset_version", "conflict": "must be v2"}
            )
        unknown = (set(case.expected_tools) | set(case.forbidden_tools)) - REAL_TOOLS
        if unknown:
            errors.append(
                {
                    "case_id": case.case_id,
                    "field": "tool names",
                    "conflict": f"unknown: {sorted(unknown)}",
                }
            )
        if case.fixture_id not in FIXTURE_IDS:
            errors.append(
                {
                    "case_id": case.case_id,
                    "field": "fixture_id",
                    "conflict": "fixture is not implemented",
                }
            )
        for tool, arguments in case.expected_tool_arguments.items():
            if tool not in REAL_TOOLS:
                continue
            unexpected = set(arguments) - TOOL_ARGUMENTS[tool]
            if unexpected:
                errors.append(
                    {
                        "case_id": case.case_id,
                        "field": "expected_tool_arguments",
                        "conflict": f"{tool} has unknown arguments {sorted(unexpected)}",
                    }
                )
    scenario_counts = dict(Counter(case.scenario_type for case in cases))
    expected_scenarios = {"happy_path": 50, "edge_case": 30, "known_failure": 15, "adversarial": 5}
    if len(cases) != 100:
        errors.append(
            {
                "case_id": "dataset",
                "field": "row_count",
                "conflict": f"expected 100, found {len(cases)}",
            }
        )
    if scenario_counts != expected_scenarios:
        errors.append(
            {
                "case_id": "dataset",
                "field": "scenario_type",
                "conflict": f"expected {expected_scenarios}, found {scenario_counts}",
            }
        )
    memory_cases = [case for case in cases if case.initial_memories or "memory" in case.fixture_id]
    memory_users = [case.eval_user_id for case in memory_cases]
    if len(memory_users) != len(set(memory_users)):
        errors.append(
            {
                "case_id": "dataset",
                "field": "eval_user_id",
                "conflict": "memory cases are not isolated",
            }
        )
    report = {
        "csv": str(path),
        "case_count": len(cases),
        "unique_case_ids": len(set(ids)),
        "dataset_versions": dict(Counter(case.dataset_version for case in cases)),
        "scenario_counts": scenario_counts,
        "memory_case_count": len(memory_cases),
        "memory_users_isolated": len(memory_users) == len(set(memory_users)),
        "implemented_fixture_count": len({case.fixture_id for case in cases}),
        "errors": errors,
        "valid": not errors,
    }
    return cases, report
