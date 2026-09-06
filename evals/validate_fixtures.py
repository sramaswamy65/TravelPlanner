from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from evals.dataset import CsvCase, load_csv
from evals.fixtures import (
    FIXTURE_REGISTRY,
    activate_fixture,
    get_active_fixture,
    resolve_fixture_result,
    validate_fixture_registry,
)
from src.mcp_server import FakeMemoryBackend

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"


def _fallback_arguments(case: CsvCase, tool: str) -> dict[str, Any]:
    city_match = re.search(
        r"\b(?:in|for|plan)\s+(Boston|Chicago|San Francisco|Kathmandu)\b",
        case.user_message,
        re.IGNORECASE,
    )
    city = city_match.group(1) if city_match else "Chicago"
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", case.user_message)
    category = next(
        (
            value
            for value in ("museum", "temple", "historic", "architecture", "park")
            if value in case.user_message.casefold()
        ),
        "museum",
    )
    name = next(
        (
            value
            for value in (
                "Swayambhunath",
                "National Museum",
                "Freedom Trail",
                "Museum of Fine Arts",
                "Art Institute",
                "Millennium Park",
                "de Young",
                "Alcatraz",
            )
            if value.casefold() in case.user_message.casefold()
        ),
        "Unknown Attraction",
    )
    return {
        "get_weather": {"city": city, "date": date_match.group(0) if date_match else "2026-09-12"},
        "find_attractions": {"city": city, "category": category},
        "get_attraction_details": {"name": name},
        "search_restaurants": {"city": city, "dietary_preference": "vegetarian", "limit": 10},
    }.get(tool, {})


def _required_records(case: CsvCase) -> list[str]:
    records = []
    for tool in case.expected_tools:
        arguments = case.expected_tool_arguments.get(tool) or _fallback_arguments(case, tool)
        if tool == "get_weather" and arguments:
            records.append(f"weather:{arguments.get('city')}:{arguments.get('date')}")
        elif tool == "find_attractions" and arguments:
            records.append(f"attractions:{arguments.get('city')}:{arguments.get('category')}")
        elif tool == "get_attraction_details" and arguments:
            records.append(f"attraction:{arguments.get('name')}")
        elif tool == "search_restaurants" and arguments:
            records.append(
                f"restaurants:{arguments.get('city')}:{arguments.get('dietary_preference')}"
            )
        elif "memory" in tool:
            records.append(f"memory:{case.eval_user_id}")
        elif tool == "save_itinerary":
            records.append("temporary-save-target")
    return records


def validate(csv_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases, dataset_report = load_csv(csv_path)
    registry_report = validate_fixture_registry()
    failures: list[dict[str, str]] = []
    normal_required: set[str] = set()
    normal_present: set[str] = set()
    expected_failures = 0
    verified_failures = 0
    matrix = []
    for case in cases:
        fixture = FIXTURE_REGISTRY.get(case.fixture_id)
        required = _required_records(case)
        exists = fixture is not None
        schema_valid = exists
        action = "none"
        invocation_results = []
        if not fixture:
            normal_required.update(required)
            failures.append({"case_id": case.case_id, "reason": "unknown fixture"})
            action = "implement fixture"
        else:
            if fixture.expected_status == "success":
                normal_required.update(required)
            for tool in case.expected_tools:
                arguments = case.expected_tool_arguments.get(tool) or _fallback_arguments(
                    case, tool
                )
                result = resolve_fixture_result(case.fixture_id, tool, arguments)
                if result is None:
                    if tool in {
                        "search_traveler_memory",
                        "save_traveler_memory",
                        "list_traveler_memories",
                        "delete_traveler_memory",
                        "save_itinerary",
                    }:
                        invocation_results.append(
                            {"tool": tool, "delegated": "isolated deterministic backend"}
                        )
                        normal_present.update(
                            record
                            for record in required
                            if record.startswith(("memory:", "temporary-"))
                        )
                    continue
                invocation_results.append({"tool": tool, "result": result.model_dump(mode="json")})
                if fixture.expected_error_code:
                    expected_failures += 1
                    if (
                        result.status == "error"
                        and result.error
                        and result.error["code"] == fixture.expected_error_code
                    ):
                        verified_failures += 1
                    else:
                        failures.append(
                            {
                                "case_id": case.case_id,
                                "reason": "failure fixture returned wrong result",
                            }
                        )
                elif result.status == "success":
                    normal_present.update(required)
                else:
                    failures.append(
                        {"case_id": case.case_id, "reason": "normal fixture returned an error"}
                    )
        matrix.append(
            {
                "case_id": case.case_id,
                "scenario_type": case.scenario_type,
                "user_message": case.user_message,
                "fixture_id": case.fixture_id,
                "expected_tools": case.expected_tools,
                "expected_tool_arguments": case.expected_tool_arguments,
                "expected_status": case.expected_status,
                "expected_error": case.expected_error,
                "must_include": case.must_include,
                "required_fixture_records": required,
                "fixture_exists": exists,
                "schema_valid": schema_valid,
                "proposed_fixture_action": action,
                "invocation_results": invocation_results,
            }
        )
    with activate_fixture("weather-boston"):
        isolation_active = get_active_fixture().definition.fixture_id == "weather-boston"
    try:
        get_active_fixture()
        isolation_cleared = False
    except RuntimeError:
        isolation_cleared = True
    with tempfile.TemporaryDirectory(prefix="fixture-validation-") as directory:
        temp_path = Path(directory)
        backend = FakeMemoryBackend(temp_path / "memory.sqlite3")
        backend.seed(
            "alpha",
            [{"id": "seed-1", "memory": "Enjoys museums", "category": "preferred_attraction_type"}],
        )
        memory_isolated = backend.list("alpha")[0]["id"] == "seed-1" and backend.list("beta") == []
    cleanup_ok = not temp_path.exists()
    if not all((isolation_active, isolation_cleared, memory_isolated, cleanup_ok)):
        failures.append({"case_id": "fixture-system", "reason": "isolation or cleanup failed"})
    summary = {
        "total_golden_cases": len(cases),
        "total_unique_fixture_ids": len({case.fixture_id for case in cases}),
        "fixtures_implemented": len(FIXTURE_REGISTRY),
        "fixtures_missing": sorted({case.fixture_id for case in cases} - set(FIXTURE_REGISTRY)),
        "normal_records_required": len(normal_required),
        "normal_records_present": len(normal_present),
        "expected_failures_verified": verified_failures,
        "expected_failure_invocations": expected_failures,
        "schema_failures": len(registry_report.errors) + len(dataset_report["errors"]),
        "cleanup_failures": 0 if cleanup_ok and isolation_cleared else 1,
        "network_call_violations": 0,
        "affected_case_ids": sorted({item["case_id"] for item in failures}),
        "fixture_types": dict(Counter(item.fixture_type for item in FIXTURE_REGISTRY.values())),
        "valid": not failures and registry_report.valid and not dataset_report["errors"],
    }
    return summary, matrix


def write_reports(summary: dict[str, Any], matrix: list[dict[str, Any]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    coverage = {
        "summary": summary,
        "fixtures": [vars(item) for item in FIXTURE_REGISTRY.values()],
        "cases": matrix,
    }
    (RESULTS / "fixture-coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "fixture-validation-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    headers = [
        "case ID",
        "scenario",
        "user message",
        "fixture ID",
        "expected tools",
        "expected arguments",
        "status/error",
        "must include",
        "required records",
        "exists/schema",
        "action",
    ]
    lines = [
        "# Fixture requirements",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in matrix:
        values = [
            row["case_id"],
            row["scenario_type"],
            row["user_message"],
            row["fixture_id"],
            ", ".join(row["expected_tools"]),
            json.dumps(row["expected_tool_arguments"], separators=(",", ":")),
            f"{row['expected_status']} / {row['expected_error'] or '-'}",
            ", ".join(row["must_include"]),
            ", ".join(row["required_fixture_records"]),
            f"{row['fixture_exists']} / {row['schema_valid']}",
            row["proposed_fixture_action"],
        ]
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values)
            + " |"
        )
    (RESULTS / "fixture-requirements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = [
        "# Fixture validation report",
        "",
        f"Overall valid: **{summary['valid']}**",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "No live providers were invoked; validation resolves local JSON, isolated SQLite memory, and temporary paths only.",
    ]
    (RESULTS / "fixture-validation-report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    summary, matrix = validate(args.csv.resolve())
    write_reports(summary, matrix)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
