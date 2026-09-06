from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import Settings

LLM_LABELS = {"Deterministic": "fake", "Live": "live"}
MEMORY_LABELS = {"Fixture": "fake", "Live": "live"}
TRAVEL_LABELS = {"Fixture": "fixture", "Live": "live"}


@dataclass(frozen=True)
class RuntimeContext:
    user_id: str
    llm_mode: Literal["fake", "live"]
    memory_mode: Literal["fake", "live"]
    travel_data_mode: Literal["fixture", "live"]
    generation: int

    @property
    def llm_backend(self) -> str:
        return "Nebius" if self.llm_mode == "live" else "Deterministic model"

    @property
    def memory_backend(self) -> str:
        return "Mem0 Platform" if self.memory_mode == "live" else "Fixture SQLite memory"

    @property
    def travel_backend(self) -> str:
        return (
            "Open-Meteo weather + fixture attractions"
            if self.travel_data_mode == "live"
            else "Fixture travel data"
        )


def runtime_context(
    user_id: str,
    llm_label: str,
    memory_label: str,
    travel_label: str,
    generation: int,
) -> RuntimeContext:
    try:
        return RuntimeContext(
            user_id=user_id,
            llm_mode=LLM_LABELS[llm_label],
            memory_mode=MEMORY_LABELS[memory_label],
            travel_data_mode=TRAVEL_LABELS[travel_label],
            generation=generation,
        )
    except KeyError as exc:
        raise ValueError("Invalid runtime mode selection") from exc


def settings_for_context(base: Settings, context: RuntimeContext) -> Settings:
    return base.model_copy(
        update={
            "app_env": "development",
            "llm_mode": context.llm_mode,
            "memory_mode": context.memory_mode,
            "travel_data_mode": context.travel_data_mode,
        }
    )
