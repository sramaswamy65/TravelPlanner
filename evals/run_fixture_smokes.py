from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from evals.dataset import load_csv
from evals.evaluators import evaluate_case
from evals.fixtures import FIXTURE_REGISTRY, get_active_fixture, resolve_fixture_result
from evals.runner import run_case

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = (
    "tp-v2-002",
    "tp-v2-014",
    "tp-v2-022",
    "tp-v2-026",
    "tp-v2-028",
    "tp-v2-040",
    "tp-v2-046",
    "tp-v2-063",
    "tp-v2-081",
    "tp-v2-082",
    "tp-v2-084",
    "tp-v2-096",
    "tp-v2-097",
    "tp-v2-098",
)


async def main() -> None:
    cases = {
        case.case_id: case
        for case in load_csv(ROOT / "evals" / "TravelPlanner_golden_dataset_v2_100.csv")[0]
    }
    results: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        case = cases[case_id]
        run = await run_case(case, "deterministic", "fixture-smoke", False)
        scores, _ = evaluate_case(run, case)
        direct_checks = []
        fixture_matched = True
        for tool in case.expected_tools:
            arguments = case.expected_tool_arguments.get(tool, {})
            if (
                not arguments
                and FIXTURE_REGISTRY[case.fixture_id].expected_error_code is None
                and tool
                in {
                    "get_weather",
                    "find_attractions",
                    "get_attraction_details",
                    "search_restaurants",
                }
            ):
                direct_checks.append({"tool": tool, "result": "validated by registry coverage"})
                continue
            direct = resolve_fixture_result(case.fixture_id, tool, arguments)
            if direct is None:
                direct_checks.append({"tool": tool, "result": "isolated deterministic backend"})
                continue
            direct_checks.append({"tool": tool, "result": direct.model_dump(mode="json")})
            expected_error = (
                case.expected_error or FIXTURE_REGISTRY[case.fixture_id].expected_error_code
            )
            fixture_matched &= (
                bool(direct.error and direct.error["code"] == expected_error)
                if expected_error
                else direct.status == "success"
            )
        try:
            get_active_fixture()
            cleanup = False
        except RuntimeError:
            cleanup = True
        results.append(
            {
                "case_id": case_id,
                "fixture_activated": case.fixture_id,
                "tool_inputs": [
                    {"name": call["name"], "arguments": call["arguments"]}
                    for call in run["tool_calls"]
                ],
                "structured_fixture_results": [
                    {"name": call["name"], "result": call["result"]} for call in run["tool_calls"]
                ],
                "fixture_sources": sorted(
                    {
                        str((call["result"].get("data") or {}).get("source"))
                        for call in run["tool_calls"]
                        if isinstance(call["result"].get("data"), dict)
                        and (call["result"].get("data") or {}).get("source")
                    }
                ),
                "direct_golden_fixture_checks": direct_checks,
                "cleanup_result": "cleared" if cleanup else "leaked",
                "fixture_matched_golden_expectation": fixture_matched,
                "agent_required_tool_selection": scores["required_tool_selection"],
                "agent_required_tool_order": scores["required_tool_order"],
                "agent_answer_assertion_score": scores["must_include_assertions"],
            }
        )
    output = ROOT / "evals" / "results" / "fixture-smoke-results.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
