from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SeedMemory(BaseModel):
    id: str = Field(min_length=1)
    memory: str = Field(min_length=1)
    category: str = Field(default="preference", min_length=1)


class ExpectedTool(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class CaseInputs(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.@-]+$")
    conversation: list[dict[str, Any]] = Field(default_factory=list)


class CaseOutputs(BaseModel):
    expected_behavior: Literal["complete", "safe_refusal", "error_report"]
    expected_tools: list[ExpectedTool] = Field(default_factory=list)
    tool_sequence_mode: Literal["minimum", "exact"] = "minimum"
    forbidden_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(ge=0, le=20)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    seed_memories: list[SeedMemory] = Field(default_factory=list)
    expected_retrieved_memory_ids: list[str] = Field(default_factory=list)
    expected_applied_memories: list[str] = Field(default_factory=list)
    expected_memory_writes: list[str] = Field(default_factory=list)
    forbidden_memory_writes: list[str] = Field(default_factory=list)
    expected_error_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def expected_outcome_is_actionable(self) -> CaseOutputs:
        if not any(
            (
                self.expected_tools,
                self.forbidden_tools,
                self.must_include,
                self.must_not_include,
                self.expected_memory_writes,
                self.forbidden_memory_writes,
                self.expected_error_codes,
                self.expected_behavior,
            )
        ):
            raise ValueError("case must define an expected outcome or behavior")
        return self


class CaseMetadata(BaseModel):
    case_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    scenario_type: Literal["happy_path", "edge_case", "known_failure", "adversarial"]
    difficulty: Literal["easy", "medium", "hard"]
    memory_focused: bool
    fixture_id: str = Field(min_length=1)


class EvaluationCase(BaseModel):
    inputs: CaseInputs
    outputs: CaseOutputs
    metadata: CaseMetadata


class DatasetValidation(BaseModel):
    dataset_version: str
    case_count: int
    scenario_counts: dict[str, int]
    scenario_percentages: dict[str, float]
    difficulty_counts: dict[str, int]
    difficulty_percentages: dict[str, float]
    memory_case_count: int
    duplicate_case_ids: list[str]
    isolated_memory_user_ids: bool


def load_and_validate_dataset(
    path: Path, dataset_version: str = "v2"
) -> tuple[list[EvaluationCase], DatasetValidation]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Dataset contains no cases")
    wrong_versions = [
        case.metadata.case_id for case in cases if case.metadata.dataset_version != dataset_version
    ]
    if wrong_versions:
        raise ValueError(f"Cases do not match dataset version {dataset_version}: {wrong_versions}")
    ids = [case.metadata.case_id for case in cases]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case IDs: {duplicates}")
    memory_users = [case.inputs.user_id for case in cases if case.metadata.memory_focused]
    isolated = len(memory_users) == len(set(memory_users))
    if not isolated:
        raise ValueError("Memory-focused cases must use isolated user IDs")
    for case in cases:
        seed_ids = [memory.id for memory in case.outputs.seed_memories]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError(f"Duplicate seed memory IDs in {case.metadata.case_id}")
        if not set(case.outputs.expected_retrieved_memory_ids).issubset(seed_ids):
            raise ValueError(
                f"Expected retrieved memories must be seeded in {case.metadata.case_id}"
            )
    scenario = Counter(case.metadata.scenario_type for case in cases)
    difficulty = Counter(case.metadata.difficulty for case in cases)
    total = len(cases)
    report = DatasetValidation(
        dataset_version=dataset_version,
        case_count=total,
        scenario_counts=dict(sorted(scenario.items())),
        scenario_percentages={
            key: round(value * 100 / total, 2) for key, value in sorted(scenario.items())
        },
        difficulty_counts=dict(sorted(difficulty.items())),
        difficulty_percentages={
            key: round(value * 100 / total, 2) for key, value in sorted(difficulty.items())
        },
        memory_case_count=len(memory_users),
        duplicate_case_ids=[],
        isolated_memory_user_ids=isolated,
    )
    return cases, report
