from __future__ import annotations

import csv
from pathlib import Path

from evals.dataset import load_csv
from evals.evaluators import COMPOSITE_KEYS, evaluate_case, evaluate_composites
from evals.run_evaluation import (
    REVIEW_COLUMNS,
    build_langsmith_evaluators,
    classify_result,
    write_review_csv,
)


def _case():
    return load_csv(Path("evals/TravelPlanner_golden_dataset_v2_100.csv"))[0][0]


def _run(**changes):
    run = {
        "answer": "Kathmandu will be sunny.",
        "status": "success",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Kathmandu", "date": "2026-10-15"}, "result": {"status": "success", "data": {}}}],
        "error_codes": [], "memory_retrieved_values": [], "memory_write_values": [],
        "user_isolation_ok": True, "latency_ms": 1, "step_count": 1,
    }
    run.update(changes)
    return run


def _result(run, scores, details):
    detailed, subchecks = evaluate_case(run, _case())
    return {"case_id": _case().case_id, "scenario_type": _case().scenario_type,
            "actual_behavior": run, "scores": scores, "detailed_scores": detailed,
            "evaluator_details": details, "detailed_subchecks": subchecks,
            "trace_url": "https://smith.example/trace/1", "error_classification": None}


def test_exactly_seven_composites_and_detailed_checks_remain():
    detailed, detailed_items = evaluate_case(_run(), _case())
    scores, details = evaluate_composites(_run(), _case(), detailed)
    assert tuple(scores) == COMPOSITE_KEYS
    assert len(scores) == 7
    assert len(detailed_items) == 18
    assert all("failed_subchecks" in item and "applicable_subcheck_count" in item for item in details)
    assert tuple(item.__name__ for item in build_langsmith_evaluators([_case()])) == COMPOSITE_KEYS


def test_missing_required_data_fails():
    detailed, _ = evaluate_case({}, _case())
    scores, _ = evaluate_composites({}, _case(), detailed)
    assert scores["task_completion"] < 1
    assert scores["tool_trajectory"] < 1
    assert scores["safe_completion"] < 1


def test_review_csv_shape_escaping_trace_and_classification(tmp_path):
    run = _run(answer="Kathmandu is sunny.\nBring water.")
    detailed, _ = evaluate_case(run, _case())
    scores, details = evaluate_composites(run, _case(), detailed)
    success = _result(run, scores, details)
    assert classify_result(success) == "PASS"
    quality = {**success, "scores": {**scores, "grounded_response": 0.5}}
    assert classify_result(quality) == "PARTIAL"
    unsafe = {**success, "scores": {**scores, "safety_compliance": 0.5}}
    assert classify_result(unsafe) == "FAIL"
    path = tmp_path / "review.csv"
    write_review_csv([_case()], [success], path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == REVIEW_COLUMNS
    assert len(rows) == 1 and "\n" in rows[0]["actual_answer"]
    assert rows[0]["trace_url"] == "https://smith.example/trace/1"
    assert "sk-" not in rows[0]["failure_summary"].casefold()
    assert "traceback" not in rows[0]["failure_summary"].casefold()
