import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CSV_PATH = PROJECT_ROOT / "evals" / "TravelPlanner_golden_dataset_v2_100.csv"
DATASET_NAME = "memory-travel-agent-golden-v2"

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

REQUIRED_COLUMNS = {
    "case_id",
    "dataset_version",
    "scenario_type",
    "difficulty",
    "fixture_id",
    "eval_user_id",
    "llm_mode",
    "memory_mode",
    "travel_data_mode",
    "user_message",
    "tool_sequence_mode",
    "max_tool_calls",
    "expected_status",
    "expected_error",
    "evaluation_notes",
} | JSON_COLUMNS


def parse_json(value: str, *, case_id: str, column: str):
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in case {case_id!r}, column {column!r}: {exc}") from exc


def load_cases():
    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no evaluation cases")

    case_ids = [row["case_id"] for row in rows]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs were found")

    for row in rows:
        for column in JSON_COLUMNS:
            row[column] = parse_json(row[column], case_id=row["case_id"], column=column)

        row["max_tool_calls"] = int(row["max_tool_calls"])

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Upload the TravelPlanner CSV dataset")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace examples when the named LangSmith dataset is not identical",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = Client()
    cases = load_cases()
    expected_case_ids = {case["case_id"] for case in cases}

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))

    if len(existing) > 1:
        raise RuntimeError(f"Multiple LangSmith datasets are named {DATASET_NAME!r}")

    if existing:
        dataset = existing[0]
        remote_examples = list(client.list_examples(dataset_id=dataset.id))
        remote_case_ids = {
            example.inputs.get("case_id") or (example.metadata or {}).get("case_id")
            for example in remote_examples
        }
        if remote_case_ids == expected_case_ids and len(remote_examples) == len(cases):
            print(f"Dataset is already current: {DATASET_NAME} ({len(cases)} examples)")
            print(f"Dataset URL: {dataset.url}")
            return
        if not args.replace_existing:
            raise RuntimeError(
                f"Dataset {DATASET_NAME!r} contains {len(remote_examples)} different "
                "examples. Re-run with --replace-existing to replace them."
            )
        for example in remote_examples:
            client.delete_example(example.id)
        print(f"Removed {len(remote_examples)} stale examples")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "TravelPlanner v2 golden dataset containing 100 deterministic evaluation cases."
            ),
        )

    inputs = []
    outputs = []
    metadata = []

    for row in cases:
        inputs.append(
            {
                "case_id": row["case_id"],
                "message": row["user_message"],
                "user_id": row["eval_user_id"],
                "conversation_context": row["conversation_context"],
                "initial_memories": row["initial_memories"],
                "fixture_id": row["fixture_id"],
                "llm_mode": row["llm_mode"],
                "memory_mode": row["memory_mode"],
                "travel_data_mode": row["travel_data_mode"],
            }
        )

        outputs.append(
            {
                "expected_tools": row["expected_tools"],
                "tool_sequence_mode": row["tool_sequence_mode"],
                "expected_tool_arguments": (row["expected_tool_arguments"]),
                "forbidden_tools": row["forbidden_tools"],
                "max_tool_calls": row["max_tool_calls"],
                "expected_retrieved_memories": (row["expected_retrieved_memories"]),
                "expected_memory_writes": (row["expected_memory_writes"]),
                "forbidden_memory_writes": (row["forbidden_memory_writes"]),
                "must_include": row["must_include"],
                "must_not_include": row["must_not_include"],
                "expected_status": row["expected_status"],
                "expected_error": row["expected_error"],
            }
        )

        metadata.append(
            {
                "case_id": row["case_id"],
                "dataset_version": row["dataset_version"],
                "scenario_type": row["scenario_type"],
                "difficulty": row["difficulty"],
                "fixture_id": row["fixture_id"],
                "evaluation_notes": row["evaluation_notes"],
            }
        )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )

    print(f"Uploaded {len(cases)} examples")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Dataset URL: {dataset.url}")


if __name__ == "__main__":
    main()
