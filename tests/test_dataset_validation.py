import csv
from pathlib import Path

import pytest

from evals.dataset import load_csv

CSV = Path(__file__).parents[1] / "evals" / "TravelPlanner_golden_dataset_v2_100.csv"


def test_complete_csv_validates_expected_distribution():
    cases, report = load_csv(CSV)
    assert len(cases) == 100
    assert report["valid"] is True
    assert report["scenario_counts"] == {
        "happy_path": 50,
        "edge_case": 30,
        "known_failure": 15,
        "adversarial": 5,
    }


def _rewrite(tmp_path, mutate):
    with CSV.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fields = rows[0].keys()
    mutate(rows)
    target = tmp_path / "cases.csv"
    with target.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = _rewrite(tmp_path, lambda rows: rows[1].update(case_id=rows[0]["case_id"]))
    with pytest.raises(ValueError, match="Duplicate case IDs"):
        load_csv(path)


def test_malformed_json_is_rejected(tmp_path):
    path = _rewrite(tmp_path, lambda rows: rows[0].update(expected_tools="[bad"))
    with pytest.raises(ValueError, match="malformed JSON"):
        load_csv(path)


def test_missing_expected_status_is_rejected(tmp_path):
    path = _rewrite(tmp_path, lambda rows: rows[0].update(expected_status=""))
    with pytest.raises(ValueError, match="expected_status"):
        load_csv(path)
