from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import Settings

RUBRIC_VERSION = "itinerary-judge-v1"
RUBRIC = """Score the candidate travel response independently on five dimensions from 1 (poor) to 5 (excellent).
relevance: directly addresses the travel request.
factual_groundedness: uses only facts present in the supplied tool evidence and reference constraints.
constraint_satisfaction: follows dates, place, preferences, safety, and tool restrictions.
clarity: is organized and understandable.
no_unsupported_claims: makes no factual assertion absent from the supplied evidence.
Return JSON only with the five integer fields and a concise rationale. Do not reward style for missing evidence."""


class JudgeScores(BaseModel):
    relevance: int = Field(ge=1, le=5)
    factual_groundedness: int = Field(ge=1, le=5)
    constraint_satisfaction: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    no_unsupported_claims: int = Field(ge=1, le=5)
    rationale: str


async def judge_itinerary(
    candidate: str,
    request: str,
    evidence: list[dict[str, Any]],
    constraints: dict[str, Any],
    settings: Settings | None = None,
) -> JudgeScores:
    settings = settings or Settings()
    if not settings.nebius_api_key or not settings.nebius_model:
        raise RuntimeError("NEBIUS_API_KEY and NEBIUS_MODEL are required for the optional judge")
    client = AsyncOpenAI(api_key=settings.nebius_api_key, base_url=str(settings.nebius_base_url))
    response = await client.chat.completions.create(
        model=settings.nebius_model,
        temperature=0,
        messages=[
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": request,
                        "tool_evidence": evidence,
                        "constraints": constraints,
                        "candidate": candidate,
                    },
                    sort_keys=True,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    return JudgeScores.model_validate_json(content)


async def calibrate(path: Path, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings()
    examples = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(examples) < 10:
        raise ValueError("Judge calibration requires at least 10 manually labeled examples")
    dimensions = (
        "relevance",
        "factual_groundedness",
        "constraint_satisfaction",
        "clarity",
        "no_unsupported_claims",
    )
    comparisons = []
    exact = within_one = total = 0
    disagreements = []
    for example in examples:
        predicted = await judge_itinerary(
            example["candidate"],
            example["request"],
            example["tool_evidence"],
            example["constraints"],
            settings,
        )
        expected = example["manual_scores"]
        differences = {
            dimension: predicted.model_dump()[dimension] - expected[dimension]
            for dimension in dimensions
        }
        exact += sum(delta == 0 for delta in differences.values())
        within_one += sum(abs(delta) <= 1 for delta in differences.values())
        total += len(dimensions)
        disagreement = {key: value for key, value in differences.items() if value != 0}
        if disagreement:
            disagreements.append(
                {"calibration_id": example["calibration_id"], "score_differences": disagreement}
            )
        comparisons.append(
            {
                "calibration_id": example["calibration_id"],
                "manual_scores": expected,
                "judge_scores": predicted.model_dump(),
                "differences": differences,
            }
        )
    return {
        "rubric_version": RUBRIC_VERSION,
        "example_count": len(examples),
        "dimension_rating_count": total,
        "exact_agreement": round(exact / total, 4),
        "within_one_agreement": round(within_one / total, 4),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    source = Path(__file__).with_name("judge_calibration.jsonl")
    destination = Path(__file__).with_name("results") / "judge-calibration.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(calibrate(source))
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in result.items() if key != "comparisons"}, indent=2)
    )
