from __future__ import annotations

import csv
import json
from pathlib import Path

from evals.schema import load_and_validate_dataset


def export_csv(source: Path, destination: Path) -> None:
    cases, _ = load_and_validate_dataset(source, "v2")
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case.metadata.case_id,
                "dataset_version": case.metadata.dataset_version,
                "scenario_type": case.metadata.scenario_type,
                "difficulty": case.metadata.difficulty,
                "memory_focused": case.metadata.memory_focused,
                "fixture_id": case.metadata.fixture_id,
                "user_id": case.inputs.user_id,
                "message": case.inputs.message,
                "expected_behavior": case.outputs.expected_behavior,
                "expected_tools": json.dumps(
                    [tool.model_dump(mode="json") for tool in case.outputs.expected_tools],
                    sort_keys=True,
                ),
                "tool_sequence_mode": case.outputs.tool_sequence_mode,
                "forbidden_tools": json.dumps(case.outputs.forbidden_tools),
                "max_tool_calls": case.outputs.max_tool_calls,
                "must_include": json.dumps(case.outputs.must_include),
                "must_not_include": json.dumps(case.outputs.must_not_include),
                "seed_memories": json.dumps(
                    [memory.model_dump(mode="json") for memory in case.outputs.seed_memories],
                    sort_keys=True,
                ),
                "expected_retrieved_memory_ids": json.dumps(
                    case.outputs.expected_retrieved_memory_ids
                ),
                "expected_applied_memories": json.dumps(case.outputs.expected_applied_memories),
                "expected_memory_writes": json.dumps(case.outputs.expected_memory_writes),
                "forbidden_memory_writes": json.dumps(case.outputs.forbidden_memory_writes),
                "expected_error_codes": json.dumps(case.outputs.expected_error_codes),
            }
        )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    export_csv(root / "golden_dataset.jsonl", root / "golden_dataset.csv")
