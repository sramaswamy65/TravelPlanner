from pathlib import Path

from evals.dataset import load_csv
from evals.evaluators import (
    answer_assertion_score,
    evaluate_case,
    forbidden_write_score,
    tool_trajectory_score,
)


def test_tool_trajectory_exact_match():
    example = {"outputs": {"expected_tool_trajectory": ["memory", "weather"]}}
    assert tool_trajectory_score({"tool_sequence": ["memory", "weather"]}, example)["score"] == 1
    assert tool_trajectory_score({"tool_sequence": ["weather", "memory"]}, example)["score"] == 0


def test_forbidden_write_evaluator():
    example = {"outputs": {"forbidden_memory_writes": ["secret"]}}
    assert forbidden_write_score({"memory_writes": []}, example)["score"] == 1
    assert forbidden_write_score({"memory_writes": ["secret"]}, example)["score"] == 0


def test_answer_assertions_case_insensitive():
    example = {"outputs": {"final_answer_contains": ["Plan", "weather"]}}
    assert answer_assertion_score({"answer": "PLAN based on WEATHER"}, example)["score"] == 1
    assert answer_assertion_score({"answer": "Plan"}, example)["score"] == 0


def test_csv_evaluators_cover_exact_trajectory_and_empty_memory_sets():
    case = load_csv(Path("evals/TravelPlanner_golden_dataset_v2_100.csv"))[0][0]
    run = {
        "answer": "Kathmandu sunny",
        "tool_calls": [
            {
                "name": "get_weather",
                "arguments": {"city": " kathmandu ", "date": "2026-10-15"},
                "result": {"status": "success"},
            }
        ],
        "status": "success",
        "error_codes": [],
        "memory_retrieved_values": [],
        "memory_write_values": [],
        "user_isolation_ok": True,
        "latency_ms": 4,
        "step_count": 2,
    }
    scores, details = evaluate_case(run, case)
    assert scores["required_tool_order"] == 1
    assert scores["tool_argument_correctness"] == 1
    assert scores["memory_retrieval_recall"] == 1
    assert scores["memory_retrieval_precision"] == 1
    assert len(details) == 18
