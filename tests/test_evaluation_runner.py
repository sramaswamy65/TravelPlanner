from argparse import Namespace

import pytest

from evals.dataset import load_csv
from evals.run_evaluation import execute


@pytest.mark.asyncio
async def test_execute_keeps_local_result_when_trace_is_unavailable(monkeypatch):
    case = load_csv(__import__("pathlib").Path("evals/TravelPlanner_golden_dataset_v2_100.csv"))[0][
        0
    ]

    async def fake_run(*args, **kwargs):
        return {
            "answer": "Kathmandu sunny",
            "tool_calls": [],
            "status": "success",
            "error_codes": [],
            "memory_retrieved_values": [],
            "memory_write_values": [],
            "user_isolation_ok": True,
            "latency_ms": 1,
            "step_count": 1,
            "trace_url": None,
        }

    monkeypatch.setattr("evals.run_evaluation.run_case", fake_run)
    args = Namespace(mode="deterministic", experiment_prefix="unit", no_langsmith=True)
    results = await execute([case], args)
    assert len(results) == 1
    assert results[0]["trace_url"] is None


@pytest.mark.asyncio
async def test_deterministic_execution_never_constructs_live_model(monkeypatch):
    case = load_csv(__import__("pathlib").Path("evals/TravelPlanner_golden_dataset_v2_100.csv"))[0][
        0
    ]
    monkeypatch.setattr("evals.runner.NebiusClient", lambda *_: pytest.fail("live model called"))
    from evals.runner import run_case

    result = await run_case(case, "deterministic", langsmith_enabled=False)
    assert result["model"] == "deterministic-fixture-model"
