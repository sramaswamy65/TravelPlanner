from pathlib import Path

from evals.dataset import load_csv
from evals.fixtures import FIXTURE_REGISTRY, validate_fixture_registry


def test_all_100_cases_resolve_to_unique_registry_entries():
    cases = load_csv(Path("evals/TravelPlanner_golden_dataset_v2_100.csv"))[0]
    assert len(cases) == 100
    assert {case.fixture_id for case in cases} <= set(FIXTURE_REGISTRY)
    report = validate_fixture_registry()
    assert report.valid and not report.duplicate_ids
