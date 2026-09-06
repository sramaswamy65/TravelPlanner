from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from evals.dataset import CsvCase, load_csv
from src.config import Settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "evals" / "TravelPlanner_golden_dataset_v2_100.csv"
DEFAULT_DATASET_NAME = "memory-travel-agent-golden-v2-100"
DATASET_NAME = DEFAULT_DATASET_NAME


def upload_cases(
    cases: list[CsvCase],
    dataset_name: str = DEFAULT_DATASET_NAME,
    replace_existing_examples: bool = False,
    client: Client | None = None,
) -> dict[str, Any]:
    settings = Settings()
    if client is None and not settings.langsmith_api_key:
        return {
            "status": "skipped",
            "dataset_name": dataset_name,
            "reason": "LANGSMITH_API_KEY is not configured",
        }
    client = client or Client(
        api_key=settings.langsmith_api_key, api_url=str(settings.langsmith_endpoint)
    )
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        created = False
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="TravelPlanner deterministic golden dataset (100 cases, version v2)",
            metadata={"dataset_version": "v2", "case_count": 100},
        )
        created = True

    def inputs_for(case: Any) -> dict[str, Any]:
        return (
            case.langsmith_inputs()
            if hasattr(case, "langsmith_inputs")
            else case.inputs.model_dump(mode="json")
        )

    def outputs_for(case: Any) -> dict[str, Any]:
        return (
            case.reference_outputs()
            if hasattr(case, "reference_outputs")
            else case.outputs.model_dump(mode="json")
        )

    def metadata_for(case: Any) -> dict[str, Any]:
        return (
            case.metadata()
            if callable(getattr(case, "metadata", None))
            else case.metadata.model_dump(mode="json")
        )

    def case_id_for(case: Any) -> str:
        return case.case_id if hasattr(case, "case_id") else case.metadata.case_id

    existing = {
        str(
            (example.inputs or {}).get("case_id") or (example.metadata or {}).get("case_id")
        ): example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    expected_ids = {case_id_for(case) for case in cases}
    stale = [example for case_id, example in existing.items() if case_id not in expected_ids]
    changed = [
        case_id_for(case)
        for case in cases
        if case_id_for(case) in existing
        and (
            existing[case_id_for(case)].inputs != inputs_for(case)
            or existing[case_id_for(case)].outputs != outputs_for(case)
        )
    ]
    if (stale or changed) and not replace_existing_examples:
        raise RuntimeError(
            f"Dataset contains {len(stale)} stale and {len(changed)} changed examples; "
            "use --replace-existing-examples to replace them"
        )
    for example in stale:
        client.delete_example(example.id)
    for case_id in changed:
        case = next(item for item in cases if case_id_for(item) == case_id)
        client.update_example(
            existing[case_id].id,
            inputs=inputs_for(case),
            outputs=outputs_for(case),
            metadata=metadata_for(case),
            dataset_id=dataset.id,
        )
    new_cases = [case for case in cases if case_id_for(case) not in existing]
    if new_cases:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": inputs_for(case),
                    "outputs": outputs_for(case),
                    "metadata": metadata_for(case),
                }
                for case in new_cases
            ],
            max_concurrency=1,
        )
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "dataset_id": str(dataset.id),
        "dataset_url": getattr(dataset, "url", None),
        "created": created,
        "inserted": len(new_cases),
        "updated": len(changed),
        "deleted": len(stale),
        "unchanged": len(cases) - len(new_cases) - len(changed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--replace-existing-examples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases, report = load_csv(args.csv.resolve())
    print(json.dumps({"validation": report}, indent=2))
    if not report["valid"]:
        raise SystemExit(2)
    if not args.validate_only:
        upload = upload_cases(cases, args.dataset_name, args.replace_existing_examples)
        print(json.dumps({"upload": upload}, indent=2))


if __name__ == "__main__":
    main()
