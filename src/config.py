from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    nebius_api_key: str = ""
    nebius_base_url: HttpUrl = "https://api.tokenfactory.nebius.com/v1/"
    nebius_model: str = ""
    mem0_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "TravelPlanner"
    langsmith_endpoint: HttpUrl = "https://api.smith.langchain.com"
    app_env: str = "development"
    llm_mode: Literal["fake", "live"] = "live"
    memory_mode: Literal["fake", "live"] = "live"
    travel_data_mode: Literal["fixture", "live"] = "fixture"
    places_provider: Literal["none"] = "none"
    places_api_key: str = ""
    max_agent_steps: int = Field(default=8, ge=1, le=8)
    mcp_timeout_seconds: float = Field(default=10.0, ge=0.1, le=60)
    fake_memory_db: Path = ROOT_DIR / "output" / "fake_memories.sqlite3"
    travel_fixture_id: str = "default-v1"
    weather_http_timeout_seconds: float = Field(default=5.0, ge=0.1, le=30)
    weather_http_retries: int = Field(default=2, ge=0, le=3)

    @field_validator("nebius_api_key", "mem0_api_key", "langsmith_api_key", "places_api_key")
    @classmethod
    def strip_secrets(cls, value: str) -> str:
        return value.strip()

    @property
    def deterministic(self) -> bool:
        return self.app_env.lower() in {"test", "evaluation", "deterministic"}

    @property
    def effective_llm_mode(self) -> Literal["fake", "live"]:
        return "fake" if self.deterministic else self.llm_mode

    @property
    def effective_memory_mode(self) -> Literal["fake", "live"]:
        return "fake" if self.deterministic else self.memory_mode

    @property
    def effective_travel_data_mode(self) -> Literal["fixture", "live"]:
        return "fixture" if self.deterministic else self.travel_data_mode

    def live_readiness_errors(self) -> list[str]:
        return self.llm_readiness_errors() + self.memory_readiness_errors()

    def llm_readiness_errors(self) -> list[str]:
        errors = []
        if self.effective_llm_mode == "live" and not self.nebius_api_key:
            errors.append("NEBIUS_API_KEY is required in live mode")
        if self.effective_llm_mode == "live" and not self.nebius_model:
            errors.append("NEBIUS_MODEL is required; no model identifier is assumed")
        return errors

    def memory_readiness_errors(self) -> list[str]:
        errors = []
        if self.effective_memory_mode == "live" and not self.mem0_api_key:
            errors.append("MEM0_API_KEY is required in live mode")
        return errors


@lru_cache
def get_settings() -> Settings:
    return Settings()
