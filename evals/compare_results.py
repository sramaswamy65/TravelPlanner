from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from evals.evaluators import SCORE_METRICS


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def _failed_metrics(case: dict[str, Any]) -> set[str]:
    return {metric for metric in SCORE_METRICS if case["scores"][metric] < 1}


def write_comparison(results: Path) -> None:
    baseline_summary = json.loads((results / "baseline-summary.json").read_text(encoding="utf-8"))
    improved_summary = json.loads(
        (results / "post-improvement-summary.json").read_text(encoding="utf-8")
    )
    baseline = _load_jsonl(results / "baseline-cases.jsonl")
    improved = _load_jsonl(results / "post-improvement-cases.jsonl")
    improved_cases = []
    regressed_cases = []
    unchanged_failures = []
    for case_id in sorted(baseline):
        before = _failed_metrics(baseline[case_id])
        after = _failed_metrics(improved[case_id])
        if len(after) < len(before):
            improved_cases.append(case_id)
        elif len(after) > len(before):
            regressed_cases.append(case_id)
        if after:
            unchanged_failures.append(f"{case_id} ({', '.join(sorted(after))})")
    before_calls = [len(case["run"]["tool_calls"]) for case in baseline.values()]
    after_calls = [len(case["run"]["tool_calls"]) for case in improved.values()]
    lines = [
        "# Baseline vs improved",
        "",
        (
            "The same v2 golden labels, deterministic travel fixtures, evaluator code, model "
            "configuration, LangSmith trace structure, and sequential concurrency were used."
        ),
        "",
        "## Metric comparison",
        "",
        "| Metric | Baseline | Improved | Absolute delta | Percentage delta |",
        "|---|---:|---:|---:|---:|",
    ]
    before_metrics = baseline_summary["aggregate_metrics"]
    after_metrics = improved_summary["aggregate_metrics"]
    for metric in (*SCORE_METRICS, "median_latency_ms", "median_step_count"):
        before = float(before_metrics[metric])
        after = float(after_metrics[metric])
        delta = after - before
        percentage = "n/a" if before == 0 else f"{delta / before * 100:+.2f}%"
        lines.append(f"| {metric} | {before:.4f} | {after:.4f} | {delta:+.4f} | {percentage} |")
    lines.extend(
        [
            "",
            "## Latency and tool-call tradeoffs",
            "",
            (
                f"- Median runtime: {before_metrics['median_latency_ms']} ms to "
                f"{after_metrics['median_latency_ms']} ms."
            ),
            (
                f"- Median executed/blocked tool records per case: {median(before_calls):.1f} to "
                f"{median(after_calls):.1f}."
            ),
            f"- Total tool records: {sum(before_calls)} to {sum(after_calls)}.",
            (
                "- Required-error cases now stop after the failing weather call instead of making "
                "an unnecessary attraction call."
            ),
            "",
            "## Case movement",
            "",
            f"- Improved cases: {', '.join(improved_cases) or 'None'}",
            f"- Regressed cases: {', '.join(regressed_cases) or 'None'}",
            f"- Cases with remaining failures: {'; '.join(unchanged_failures) or 'None'}",
            "",
            "## Representative LangSmith traces",
            "",
        ]
    )
    for cluster in baseline_summary["failure_clusters"][:3]:
        case_id = cluster["affected_case_ids"][0]
        before_trace = baseline[case_id]["run"]["trace_url"]
        after_trace = improved[case_id]["run"]["trace_url"]
        lines.extend(
            [
                (
                    f"- **{cluster['name']}**, `{case_id}`: "
                    f"[baseline]({before_trace}) / [improved]({after_trace})"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "### What worked",
            "",
            "- Broad user-scoped preference search recovered the seeded durable memories.",
            "- Clear intent ordering removed the accidental write before memory listing.",
            "- Explicit no-memory handling and direct attraction lookup corrected their tool paths.",
            "- Required weather failures now stop without an unnecessary dependent call.",
            "",
            "### What did not work",
            "",
            (
                "Suppressing prohibited write attempts caused the unchanged evaluator to lose the "
                "required `policy_denied` observation. That attempted change was reverted; the "
                "existing agent policy still prevents the MCP write, but those blocked attempts "
                "remain trajectory failures in `v2-08` and `v2-09`."
            ),
            "",
            "### Top remaining failure",
            "",
            (
                "The two prohibited-memory cases still record the model's blocked "
                "`save_traveler_memory` request, reducing forbidden-tool, order, and call-bound "
                "scores."
            ),
            "",
            "### What to try next",
            "",
            (
                "Represent policy-denied attempted calls separately from executed MCP calls in a "
                "future versioned trajectory schema, then version the evaluator and labels together. "
                "Do not change the current v2 evaluator retroactively."
            ),
            "",
        ]
    )
    (results / "baseline-vs-improved.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_comparison(Path(__file__).resolve().parent / "results")
