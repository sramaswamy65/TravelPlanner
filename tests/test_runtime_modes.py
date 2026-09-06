from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.config import Settings
from src.mcp_client import mcp_environment
from src.nebius_client import DeterministicModelClient, NebiusClient, create_model_client
from src.runtime import runtime_context, settings_for_context


def test_environment_values_are_only_runtime_initial_defaults(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("MEMORY_MODE", "fake")
    monkeypatch.setenv("TRAVEL_DATA_MODE", "fixture")
    base = Settings()
    selected = runtime_context("user", "Deterministic", "Live", "Live", 4)
    runtime = settings_for_context(base, selected)
    assert (base.llm_mode, base.memory_mode, base.travel_data_mode) == ("live", "fake", "fixture")
    assert (runtime.llm_mode, runtime.memory_mode, runtime.travel_data_mode) == (
        "fake",
        "live",
        "live",
    )


def test_selected_modes_are_passed_to_new_mcp_child_environment():
    first = runtime_context("user", "Deterministic", "Fixture", "Fixture", 0)
    second = runtime_context("user", "Deterministic", "Live", "Live", 1)
    base = Settings(app_env="development")
    fixture_environment = mcp_environment(settings_for_context(base, first), {})
    live_environment = mcp_environment(settings_for_context(base, second), {})
    assert fixture_environment["MEMORY_MODE"] == "fake"
    assert fixture_environment["TRAVEL_DATA_MODE"] == "fixture"
    assert live_environment["MEMORY_MODE"] == "live"
    assert live_environment["TRAVEL_DATA_MODE"] == "live"


def test_request_runtime_context_is_immutable():
    context = runtime_context("user", "Deterministic", "Fixture", "Fixture", 7)
    with pytest.raises(FrozenInstanceError):
        context.travel_data_mode = "live"


def test_runtime_context_selects_real_adapter_modes_without_fallback():
    live = settings_for_context(
        Settings(app_env="development"), runtime_context("user", "Live", "Live", "Live", 2)
    )
    fixture = settings_for_context(
        Settings(app_env="development"),
        runtime_context("user", "Deterministic", "Fixture", "Fixture", 3),
    )
    assert (
        live.effective_llm_mode,
        live.effective_memory_mode,
        live.effective_travel_data_mode,
    ) == ("live", "live", "live")
    assert (
        fixture.effective_llm_mode,
        fixture.effective_memory_mode,
        fixture.effective_travel_data_mode,
    ) == ("fake", "fake", "fixture")
    assert isinstance(create_model_client(fixture), DeterministicModelClient)
    assert isinstance(
        create_model_client(
            live.model_copy(update={"nebius_api_key": "test-key", "nebius_model": "test-model"})
        ),
        NebiusClient,
    )


def test_live_llm_factory_rejects_missing_credentials_instead_of_falling_back():
    settings = settings_for_context(
        Settings(app_env="development", nebius_api_key="", nebius_model=""),
        runtime_context("user", "Live", "Fixture", "Fixture", 1),
    )
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        create_model_client(settings)
