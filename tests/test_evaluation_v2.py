from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.evaluators import evaluate_case
from evals.judge import RUBRIC, JudgeScores
from evals.schema import EvaluationCase, load_and_validate_dataset
from evals.upload_dataset import DATASET_NAME, upload_cases
from src.config import Settings

DATASET = Path(__file__).parents[1] / "evals" / "golden_dataset.jsonl"


def test_evaluation_environment_forces_fixture_travel_data():
    settings = Settings(app_env="evaluation", travel_data_mode="live")
    assert settings.effective_travel_data_mode == "fixture"


def _case(**output_overrides) -> EvaluationCase:
    outputs = {
        "expected_behavior": "complete",
        "expected_tools": [
            {"name": "get_weather", "arguments": {"city": "Chicago", "date": "2026-09-05"}}
        ],
        "tool_sequence_mode": "minimum",
        "forbidden_tools": [],
        "max_tool_calls": 2,
        "must_include": ["plan"],
        "must_not_include": ["fabricated"],
        "seed_memories": [{"id": "m1", "memory": "relaxed pace", "category": "pace"}],
        "expected_retrieved_memory_ids": ["m1"],
        "expected_applied_memories": ["relaxed pace"],
        "expected_memory_writes": [],
        "forbidden_memory_writes": ["secret"],
        "expected_error_codes": [],
    }
    outputs.update(output_overrides)
    return EvaluationCase.model_validate(
        {
            "inputs": {"message": "Plan Chicago", "user_id": "isolated-user"},
            "outputs": outputs,
            "metadata": {
                "case_id": "unit-v2",
                "dataset_version": "v2",
                "scenario_type": "happy_path",
                "difficulty": "easy",
                "memory_focused": True,
                "fixture_id": "default-v1",
            },
        }
    )


def _run(tools: list[tuple[str, dict]], **overrides):
    run = {
        "answer": "A plan with a relaxed pace",
        "tool_calls": [
            {
                "name": name,
                "arguments": arguments,
                "result": {"status": "success", "data": {}},
            }
            for name, arguments in tools
        ],
        "memories_retrieved": ["m1"],
        "memories_written": [],
        "expected_retrieved_runtime_ids": ["m1"],
        "user_isolation_ok": True,
        "errors": [],
        "latency_ms": 10,
        "step_count": 2,
    }
    run.update(overrides)
    return run


def test_v2_dataset_is_valid_unique_and_isolated():
    cases, report = load_and_validate_dataset(DATASET, "v2")
    assert report.case_count == 12
    assert report.duplicate_case_ids == []
    assert report.isolated_memory_user_ids is True
    assert sum(report.scenario_counts.values()) == 12
    assert sum(report.difficulty_counts.values()) == 12
    assert all(case.outputs.expected_behavior for case in cases)


def test_duplicate_case_ids_are_rejected(tmp_path):
    line = DATASET.read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate case IDs"):
        load_and_validate_dataset(path, "v2")


def test_invalid_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1|line 2"):
        load_and_validate_dataset(path, "v2")


def test_minimum_sequence_allows_relevant_additional_call_and_normalizes_arguments():
    run = _run(
        [
            ("search_traveler_memory", {"query": "preferences"}),
            ("get_weather", {"city": " chicago ", "date": "September 5, 2026"}),
        ]
    )
    scores = evaluate_case(run, _case())
    assert scores["required_tool_order"] == 1
    assert scores["maximum_tool_call_compliance"] == 1
    assert scores["tool_argument_correctness"] == 1


def test_exact_sequence_and_maximum_are_enforced():
    case = _case(tool_sequence_mode="exact", max_tool_calls=1)
    run = _run(
        [
            ("search_traveler_memory", {"query": "preferences"}),
            ("get_weather", {"city": "Chicago", "date": "2026-09-05"}),
        ]
    )
    scores = evaluate_case(run, case)
    assert scores["required_tool_order"] == 0
    assert scores["maximum_tool_call_compliance"] == 0


def test_memory_write_and_user_isolation_scores_are_independent():
    case = _case(expected_memory_writes=["aisle seat"])
    run = _run(
        [("get_weather", {"city": "Chicago", "date": "2026-09-05"})],
        tool_calls=[
            {
                "name": "save_traveler_memory",
                "arguments": {"memory": "I prefer an aisle seat"},
                "result": {"status": "success", "data": {}},
            }
        ],
        user_isolation_ok=False,
    )
    scores = evaluate_case(run, case)
    assert scores["expected_memory_write_accuracy"] == 1
    assert scores["user_isolation_compliance"] == 0


class _FakeLangSmithClient:
    def __init__(self, case: EvaluationCase):
        self.dataset = SimpleNamespace(id="dataset-id")
        self.example = SimpleNamespace(
            id="example-id",
            inputs=case.inputs.model_dump(mode="json"),
            outputs=case.outputs.model_dump(mode="json"),
            metadata=case.metadata.model_dump(mode="json"),
        )
        self.created = self.updated = 0

    def read_dataset(self, *, dataset_name):
        assert dataset_name == DATASET_NAME
        return self.dataset

    def list_examples(self, *, dataset_id):
        assert dataset_id == self.dataset.id
        return [self.example]

    def update_example(self, *args, **kwargs):
        self.updated += 1

    def create_examples(self, *args, **kwargs):
        self.created += 1


def test_langsmith_upload_reuses_unchanged_case():
    case = _case()
    client = _FakeLangSmithClient(case)
    result = upload_cases([case], client=client)
    assert result["created"] is False
    assert result["unchanged"] == 1
    assert client.created == client.updated == 0


def test_judge_rubric_and_calibration_cover_five_dimensions():
    dimensions = {
        "relevance",
        "factual_groundedness",
        "constraint_satisfaction",
        "clarity",
        "no_unsupported_claims",
    }
    assert all(name in RUBRIC for name in dimensions)
    assert dimensions.issubset(JudgeScores.model_fields)
    calibration = Path(__file__).parents[1] / "evals" / "judge_calibration.jsonl"
    rows = [json.loads(line) for line in calibration.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 10
    assert all(dimensions.issubset(row["manual_scores"]) for row in rows)
