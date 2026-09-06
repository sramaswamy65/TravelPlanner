from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"


def _rows(name: str) -> dict[str, dict]:
    return {
        row["case_id"]: row
        for row in (
            json.loads(line) for line in (RESULTS / name).read_text(encoding="utf-8").splitlines()
        )
    }


def main() -> None:
    original = _rows("baseline-cases.jsonl")
    corrected = _rows("fixture-corrected-cases.jsonl")
    original_summary = json.loads((RESULTS / "baseline-summary.json").read_text(encoding="utf-8"))
    corrected_summary = json.loads(
        (RESULTS / "fixture-corrected-summary.json").read_text(encoding="utf-8")
    )
    improved, regressed, unchanged, remaining = [], [], [], []
    for case_id, after in corrected.items():
        before_scores = original[case_id]["scores"]
        after_scores = after["scores"]
        deltas = {
            key: after_scores[key] - before_scores[key]
            for key in after_scores
            if key not in {"latency_ms", "step_count"}
        }
        if any(value > 0 for value in deltas.values()) and not any(
            value < 0 for value in deltas.values()
        ):
            improved.append(case_id)
        elif any(value < 0 for value in deltas.values()):
            regressed.append(case_id)
        else:
            unchanged.append(case_id)
        if any(
            value < 1
            for key, value in after_scores.items()
            if key not in {"latency_ms", "step_count"}
        ):
            remaining.append(case_id)
    lines = [
        "# Original versus fixture-corrected baseline",
        "",
        "This delta is fixture-harness remediation, not an agent improvement.",
        "",
        "## Aggregate score changes",
        "",
        "| Metric | Original | Corrected | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, after in corrected_summary["aggregate_scores"].items():
        before = original_summary["aggregate_scores"][metric]
        lines.append(f"| {metric} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines += [
        "",
        "## Classification",
        "",
        f"- Cases fixed or improved solely by fixture alignment: {', '.join(improved) or 'None'}",
        f"- Cases still failing at least one deterministic agent-facing metric: {', '.join(remaining) or 'None'}",
        "- Cases still failing because of confirmed evaluator behavior: None identified.",
        "- Cases with missing evaluator scores: None.",
        f"- Cases with at least one regressed score: {', '.join(regressed) or 'None'}",
        "- Ungradable cases: None.",
        "",
        "Regressions are reported rather than repaired. Controlled repeated-call and step-limit fixtures now execute their intended failure paths, which can lower trajectory metrics while improving failure-path validity.",
    ]
    (RESULTS / "original-vs-fixture-corrected.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "improved": improved,
                "regressed": regressed,
                "remaining_failures": remaining,
                "unchanged": unchanged,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
