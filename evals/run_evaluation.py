from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from langsmith import Client, evaluate

from evals.dataset import CsvCase, load_csv
from evals.evaluators import (
    BINARY_METRICS,
    COMPOSITE_KEYS,
    aggregate_composites,
    evaluate_case,
    evaluate_composites,
)
from evals.runner import run_case
from src.config import Settings
from src.prompts import AGENT_VERSION, PROMPT_VERSION

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "evals" / "TravelPlanner_golden_dataset_v2_100.csv"
RESULTS = ROOT / "evals" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deterministic", "live"), default="deterministic")
    parser.add_argument("--memory-mode", choices=("fake", "live"), default="fake")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--scenario")
    parser.add_argument("--difficulty")
    parser.add_argument("--experiment-prefix", default="travelplanner-baseline-v2")
    parser.add_argument("--dataset-version", default="v2")
    parser.add_argument("--dataset-name", default="memory-travel-agent-golden-v2-100")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--no-langsmith", action="store_true")
    parser.add_argument("--output-prefix", default="baseline")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999)))]


async def execute(cases: list[CsvCase], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        try:
            run = await run_case(
                case,
                args.mode,
                args.experiment_prefix,
                not args.no_langsmith,
                getattr(args, "memory_mode", "fake"),
            )
            detailed_scores, detailed_subchecks = evaluate_case(run, case)
            scores, evaluator_details = evaluate_composites(run, case, detailed_scores)
            results.append({
                    "case_id": case.case_id,
                    "dataset_version": case.dataset_version,
                    "scenario_type": case.scenario_type,
                    "difficulty": case.difficulty,
                    "fixture_id": case.fixture_id,
                    "expected_behavior": case.reference_outputs(),
                    "actual_behavior": run,
                    "scores": scores,
                    "detailed_scores": detailed_scores,
                    "evaluator_comments": {
                        item["key"]: item["comment"] for item in evaluator_details
                    },
                    "evaluator_details": evaluator_details,
                    "detailed_subchecks": detailed_subchecks,
                    "trace_url": run.get("trace_url"),
                    "error_classification": "tool_failure" if run.get("error_codes") else None,
                    "latency_ms": run["latency_ms"],
                    "step_count": run["step_count"],
                    "ungradable": False,
                })
        except Exception as exc:  # noqa: BLE001 - one execution must not erase the baseline.
            results.append({
                    "case_id": case.case_id,
                    "dataset_version": case.dataset_version,
                    "scenario_type": case.scenario_type,
                    "difficulty": case.difficulty,
                    "fixture_id": case.fixture_id,
                    "expected_behavior": case.reference_outputs(),
                    "actual_behavior": None,
                    "scores": {key: 0.0 for key in COMPOSITE_KEYS},
                    "detailed_scores": {key: 0.0 for key in BINARY_METRICS},
                    "evaluator_comments": {},
                    "evaluator_details": [],
                    "detailed_subchecks": [],
                    "trace_url": None,
                    "error_classification": "execution_error",
                    "execution_error": type(exc).__name__,
                    "latency_ms": 0.0,
                    "step_count": 0,
                    "ungradable": False,
                })
    return results


def _group_scores(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result[field]].append(result)
    return {name: aggregate_composites(items) for name, items in sorted(groups.items())}


def build_langsmith_evaluators(cases: list[CsvCase]):
    """Build exactly the seven public evaluators submitted as LangSmith feedback."""
    by_id = {case.case_id: case for case in cases}
    evaluators = []
    for composite_key in COMPOSITE_KEYS:

        def wrapped(run, example, key=composite_key):
            case = by_id[example.inputs["case_id"]]
            detailed_scores, _ = evaluate_case(run.outputs or {}, case)
            _, composite_details = evaluate_composites(run.outputs or {}, case, detailed_scores)
            result = next(item for item in composite_details if item["key"] == key)
            return {field: result[field] for field in ("key", "score", "comment")}

        wrapped.__name__ = composite_key
        evaluators.append(wrapped)
    return evaluators


def _publish_experiment(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    if args.no_langsmith or not settings.langsmith_api_key:
        return {"status": "skipped", "reason": "LangSmith disabled or credentials unavailable"}
    client = Client(api_key=settings.langsmith_api_key, api_url=str(settings.langsmith_endpoint))
    try:
        dataset = client.read_dataset(dataset_name=args.dataset_name)
    except Exception as exc:  # noqa: BLE001 - remote publishing is optional.
        return {"status": "skipped", "reason": f"dataset unavailable: {type(exc).__name__}"}
    selected = {item["case_id"]: item for item in results}
    examples = [
        example
        for example in client.list_examples(dataset_id=dataset.id)
        if example.inputs.get("case_id") in selected
    ]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return selected[inputs["case_id"]]["actual_behavior"] or {"status": "execution_error"}

    evaluators = build_langsmith_evaluators(load_csv(CSV_PATH)[0])
    experiment = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        num_repetitions=args.repetitions,
        client=client,
        metadata={
            "dataset_version": args.dataset_version,
            "agent_version": AGENT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "evaluation_mode": args.mode,
            "memory_mode": args.memory_mode,
            "travel_data_mode": "fixture",
        },
    )
    experiment_name = getattr(experiment, "experiment_name", None)
    project = client.read_project(project_name=experiment_name) if experiment_name else None
    url = (
        f"https://smith.langchain.com/o/{project.tenant_id}/datasets/{dataset.id}/compare?selectedSessions={project.id}"
        if project and getattr(project, "tenant_id", None)
        else None
    )
    uploaded_runs = (
        len(list(client.list_runs(project_id=project.id, is_root=True, limit=len(results))))
        if project
        else 0
    )
    fully_published = uploaded_runs == len(results)
    return {
        "status": "success" if fully_published else "incomplete",
        "reason": None
        if fully_published
        else f"LangSmith contains {uploaded_runs} of {len(results)} expected root runs",
        "expected_root_runs": len(results),
        "uploaded_root_runs": uploaded_runs,
        "experiment_name": experiment_name,
        "experiment_id": str(project.id) if project else None,
        "experiment_url": url,
    }


def write_reports(
    cases: list[CsvCase],
    validation: dict[str, Any],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    latencies = [item["latency_ms"] for item in results]
    tool_counts = [len((item["actual_behavior"] or {}).get("tool_calls", [])) for item in results]
    steps = [item["step_count"] for item in results]
    summary = {
        "dataset_rows": validation["case_count"],
        "number_attempted": len(results),
        "number_completed": sum(item["actual_behavior"] is not None for item in results),
        "number_ungradable": sum(item["ungradable"] for item in results),
        "number_execution_errors": sum(
            item["error_classification"] == "execution_error" for item in results
        ),
        "aggregate_scores": aggregate_composites(results),
        "detailed_aggregate_scores": {
            key: round(sum(item["detailed_scores"][key] for item in results) / len(results), 4)
            for key in BINARY_METRICS
        },
        "counts_by_scenario_type": dict(Counter(item["scenario_type"] for item in results)),
        "scores_by_scenario_type": _group_scores(results, "scenario_type"),
        "scores_by_difficulty": _group_scores(results, "difficulty"),
        "median_latency_ms": median(latencies) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "median_tool_calls": median(tool_counts) if tool_counts else 0,
        "p95_tool_calls": percentile(tool_counts, 0.95),
        "median_step_count": median(steps) if steps else 0,
        "p95_step_count": percentile(steps, 0.95),
        "experiment_name": experiment.get("experiment_name"),
        "langsmith_experiment_url": experiment.get("experiment_url"),
        "model": "deterministic-fixture-model"
        if args.mode == "deterministic"
        else Settings().nebius_model,
        "prompt_version": PROMPT_VERSION,
        "agent_version": AGENT_VERSION,
        "evaluation_mode": args.mode,
        "memory_mode": args.memory_mode,
        "travel_data_mode": "fixture",
        "dataset_version": args.dataset_version,
        "dataset_name": args.dataset_name,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (RESULTS / f"{args.output_prefix}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / f"{args.output_prefix}-cases.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in results), encoding="utf-8"
    )
    write_review_csv(cases, results, RESULTS / f"{args.output_prefix}-review.csv")
    validation_lines = [
        "# Dataset validation report",
        "",
        f"Valid: **{validation['valid']}**",
        f"Cases: {validation['case_count']}",
        f"Unique IDs: {validation['unique_case_ids']}",
        f"Scenarios: `{validation['scenario_counts']}`",
        f"Memory users isolated: {validation['memory_users_isolated']}",
        f"Implemented fixtures used: {validation['implemented_fixture_count']}",
        "",
        "## Conflicts",
        "",
    ]
    validation_lines += (
        ["None."]
        if not validation["errors"]
        else [f"- `{e['case_id']}` `{e['field']}`: {e['conflict']}" for e in validation["errors"]]
    )
    (RESULTS / "dataset-validation-report.md").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )
    clusters = []
    for metric in COMPOSITE_KEYS:
        failed = [item["case_id"] for item in results if item["scores"][metric] < 1]
        clusters.append((len(failed), metric, failed))
    clusters.sort(reverse=True)
    lines = ["# TravelPlanner baseline failure analysis", "", "## Exact aggregate scores", ""]
    lines += [f"- `{key}`: {value}" for key, value in summary["aggregate_scores"].items()]
    lines += ["", "## Three largest failure clusters", ""]
    for count, metric, failed in clusters[:3]:
        lines += [
            f"### {metric}",
            f"Frequency: {count}/{len(results)}",
            f"Cases: {', '.join(failed) or 'None'}",
            "Classification: agent behavior unless the per-case record identifies a tool, fixture, dataset, or evaluator error.",
            "Effect: lowers the named deterministic metric.",
            "Recommendation: inspect representative traces before changing the agent or prompt.",
            "",
        ]
    failure_name = (
        "failure-analysis.md"
        if args.output_prefix == "baseline"
        else f"{args.output_prefix}-failure-analysis.md"
    )
    (RESULTS / failure_name).write_text("\n".join(lines), encoding="utf-8")
    return summary


REVIEW_COLUMNS = (
    "case_id", "scenario_type", "user_input", "actual_answer", "overall_result",
    "failure_summary", "trace_url",
)


def classify_result(result: dict[str, Any]) -> str:
    scores = result["scores"]
    detailed = result.get("detailed_scores", {})
    critical = (
        result.get("error_classification") == "execution_error"
        or scores.get("task_completion", 0) < 1
        or scores.get("safety_compliance", 0) < 1
        or scores.get("safe_completion", 0) < 1
        or detailed.get("forbidden_memory_write_compliance", 1) < 1
        or detailed.get("user_isolation_compliance", 1) < 1
    )
    if critical:
        return "FAIL"
    return "PASS" if all(scores.get(key, 0) == 1 for key in COMPOSITE_KEYS) else "PARTIAL"


def _failure_summary(result: dict[str, Any]) -> str:
    if classify_result(result) == "PASS":
        return "PASS"
    failed = []
    for detail in result.get("evaluator_details", []):
        failed.extend(detail.get("failed_subchecks", []))
    summary = ", ".join(dict.fromkeys(name.replace("_", " ") for name in failed[:4]))
    summary = summary or "Evaluation did not complete"
    lowered = summary.casefold()
    if "traceback" in lowered or "stack trace" in lowered or "sk-" in lowered:
        return "Sensitive or internal error details were suppressed"
    return summary[:240]


def write_review_csv(cases: list[CsvCase], results: list[dict[str, Any]], path: Path) -> None:
    by_id = {case.case_id: case for case in cases}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            case = by_id[result["case_id"]]
            actual = result.get("actual_behavior") or {}
            writer.writerow({
                "case_id": case.case_id,
                "scenario_type": case.scenario_type,
                "user_input": case.user_message,
                "actual_answer": actual.get("answer", ""),
                "overall_result": classify_result(result),
                "failure_summary": _failure_summary(result),
                "trace_url": result.get("trace_url") or actual.get("trace_url") or "",
            })


def main() -> None:
    args = parse_args()
    if args.max_concurrency != 1:
        raise SystemExit("Only --max-concurrency 1 is supported; MCP teardown is task-local")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    cases, validation = load_csv(CSV_PATH)
    cases = [case for case in cases if case.dataset_version == args.dataset_version]
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.scenario:
        cases = [case for case in cases if case.scenario_type == args.scenario]
    if args.difficulty:
        cases = [case for case in cases if case.difficulty == args.difficulty]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases matched the requested filters")
    results = asyncio.run(execute(cases, args))
    experiment = _publish_experiment(results, args)
    summary = write_reports(cases, validation, results, args, experiment)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
