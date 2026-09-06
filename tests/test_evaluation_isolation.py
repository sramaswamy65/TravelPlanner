from pathlib import Path

import pytest

from evals.dataset import load_csv
from evals.runner import run_case

CSV = Path(__file__).parents[1] / "evals" / "TravelPlanner_golden_dataset_v2_100.csv"


@pytest.mark.asyncio
async def test_sequential_cases_do_not_share_memory():
    cases = load_csv(CSV)[0]
    seeded = next(case for case in cases if case.initial_memories)
    empty = next(
        case for case in cases if not case.initial_memories and case.fixture_id == "memory-default"
    )
    first = await run_case(seeded, "deterministic", langsmith_enabled=False)
    second = await run_case(empty, "deterministic", langsmith_enabled=False)
    assert "isolation-control" not in first["memories_retrieved"]
    assert "isolation-control" not in second["memories_retrieved"]
    assert second["user_isolation_ok"] is True


@pytest.mark.asyncio
async def test_fixture_cleanup_occurs_after_tool_error():
    case = next(case for case in load_csv(CSV)[0] if case.case_id == "tp-v2-081")
    result = await run_case(case, "deterministic", langsmith_enabled=False)
    assert "timeout" in result["error_codes"]
    assert result["status"] == "safe_recovery"
