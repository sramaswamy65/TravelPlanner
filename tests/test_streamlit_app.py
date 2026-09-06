from __future__ import annotations

import uuid

from streamlit.testing.v1 import AppTest

from src.config import ROOT_DIR, Settings
from src.mcp_server import FakeMemoryBackend


def app_test() -> AppTest:
    return AppTest.from_file(ROOT_DIR / "app.py", default_timeout=30).run()


def test_all_tabs_render_and_chat_history_survives_rerun(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    at = app_test()
    user_id = f"chat-{uuid.uuid4().hex}"
    at.text_input[0].set_value(user_id).run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Chat", "Memories", "About"]
    assert [control.label for control in at.segmented_control] == ["LLM", "Memory", "Weather"]
    assert [control.value for control in at.segmented_control] == [
        "Deterministic",
        "Fixture",
        "Fixture",
    ]
    assert any("Limited fixture data" in caption.value for caption in at.caption)
    at.chat_input[0].set_value("Plan a museum day in Chicago on 2026-09-05").run()
    assert not at.exception
    rendered = [markdown.value for message in at.chat_message for markdown in message.markdown]
    assert "Plan a museum day in Chicago on 2026-09-05" in rendered
    assert any("fixture-grounded" in value for value in rendered)
    at.run()
    rerendered = [markdown.value for message in at.chat_message for markdown in message.markdown]
    assert "Plan a museum day in Chicago on 2026-09-05" in rerendered
    assert any("fixture-grounded" in value for value in rerendered)
    assert any(expander.label == "Agent activity" for expander in at.expander)
    filename = f"approved-{uuid.uuid4().hex}.md"
    at.chat_input[0].set_value(f"Save this itinerary as {filename}").run()
    saved = ROOT_DIR / "output" / "plans" / user_id / filename
    assert saved.exists()
    assert "fixture-grounded" in saved.read_text(encoding="utf-8")
    saved.unlink()


def test_memory_delete_requires_confirmation_and_is_user_scoped(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    settings = Settings(app_env="test")
    backend = FakeMemoryBackend(settings.fake_memory_db)
    user_id = f"ui-{uuid.uuid4().hex}"
    backend.add(user_id, "I prefer art museums", "interest")
    at = app_test()
    at.text_input[0].set_value(user_id).run()
    refresh = next(button for button in at.button if button.label == "Refresh memories")
    refresh.click().run()
    assert any("I prefer art museums" in markdown.value for markdown in at.markdown)
    delete = next(button for button in at.button if button.label == "Delete")
    delete.click().run()
    assert backend.list(user_id)
    assert any(button.label == "Confirm delete" for button in at.button)
    confirm = next(button for button in at.button if button.label == "Confirm delete")
    confirm.click().run()
    assert backend.list(user_id) == []
    assert backend.list("another-user") == []


def test_fixture_sidebar_and_delhi_limit_message_are_explicit(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.setenv("MEMORY_MODE", "fake")
    monkeypatch.setenv("TRAVEL_DATA_MODE", "fixture")
    at = app_test()
    assert [control.value for control in at.segmented_control] == [
        "Deterministic",
        "Fixture",
        "Fixture",
    ]
    at.chat_input[0].set_value("What is the weather in Delhi on 2026-09-05?").run()
    rendered = [markdown.value for message in at.chat_message for markdown in message.markdown]
    assert any(
        "demonstration weather dataset does not contain Delhi" in value for value in rendered
    )


def test_mode_change_persists_history_increments_generation_and_clears_pending_write(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.setenv("MEMORY_MODE", "fake")
    monkeypatch.setenv("TRAVEL_DATA_MODE", "fixture")
    at = app_test()
    at.chat_input[0].set_value("Find museums in Chicago").run()
    assert at.session_state.runtime_generation == 0
    at.session_state.delete_pending = "pending-memory-id"
    at.segmented_control[2].set_value("Live").run()
    assert at.session_state.runtime_generation == 1
    assert at.session_state.delete_pending is None
    assert at.session_state.travel_data_mode == "Live"
    assert at.session_state.runtime_reconnect_pending is True
    rendered = [markdown.value for message in at.chat_message for markdown in message.markdown]
    assert "Find museums in Chicago" in rendered
    assert "Travel data mode changed to Live." in rendered
    at.run()
    assert at.segmented_control[2].value == "Live"
    assert at.session_state.runtime_generation == 1
    at.chat_input[0].set_value("Find art attractions in Chicago").run()
    activity = at.session_state.activities[-1]
    assert activity["runtime"]["generation"] == 1
    assert activity["runtime"]["client_reconnected"] is True
    assert activity["runtime"]["travel_data_backend"].startswith("Open-Meteo weather")
    assert activity["tool_calls"][-1]["result"]["data"]["source"] == "local attraction fixture"


def test_missing_live_credentials_warn_and_never_fall_back(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("MEMORY_MODE", "fake")
    monkeypatch.setenv("TRAVEL_DATA_MODE", "fixture")
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("NEBIUS_MODEL", "")
    at = app_test()
    warnings = [warning.value for warning in at.warning]
    assert any("NEBIUS_API_KEY" in value for value in warnings)
    assert any("NEBIUS_MODEL" in value for value in warnings)
    at.chat_input[0].set_value("Find museums in Chicago").run()
    errors = [error.value for error in at.error]
    assert any("Planner startup failed" in value and "NEBIUS_API_KEY" in value for value in errors)
    rendered = [markdown.value for message in at.chat_message for markdown in message.markdown]
    assert not any("fixture-grounded" in value for value in rendered)


def test_compound_request_reports_both_outcomes_and_rerun_does_not_duplicate(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.setenv("MEMORY_MODE", "fake")
    monkeypatch.setenv("TRAVEL_DATA_MODE", "live")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    user_id = f"compound-{uuid.uuid4().hex}"
    other_user = f"other-{uuid.uuid4().hex}"
    at = app_test()
    at.text_input[0].set_value(user_id).run()
    at.session_state.messages = [
        {"role": "user", "content": "What is the weather in Boston tomorrow?"},
        {"role": "assistant", "content": "A reliable forecast is unavailable."},
    ]
    at.run()
    at.chat_input[0].set_value(
        "Remember, I am a vegetarian. Can you please look up some good restaurants?"
    ).run()
    activity = at.session_state.activities[-1]
    assert [call["name"] for call in activity["tool_calls"]] == [
        "save_traveler_memory",
        "search_restaurants",
    ]
    assert activity["resolved_intents"] == [
        "save_dietary_preference",
        "search_restaurants",
    ]
    assert activity["inherited_city"] == "Boston"
    assert activity["tool_calls"][0]["arguments"]["memory"] == "Is vegetarian"
    assert activity["tool_calls"][0]["result"]["status"] == "success"
    assert activity["tool_calls"][1]["result"]["status"] == "capability_unavailable"
    assert any(status.label == "Completed with limitations" for status in at.status)
    backend = FakeMemoryBackend(Settings().fake_memory_db)
    assert [item["memory"] for item in backend.list(user_id)] == ["Is vegetarian"]
    at.run()
    assert [item["memory"] for item in backend.list(user_id)] == ["Is vegetarian"]
    at.text_input[0].set_value(other_user).run()
    refresh = next(button for button in at.button if button.label == "Refresh memories")
    refresh.click().run()
    assert at.session_state.memories == []
    memory_id = backend.list(user_id)[0]["id"]
    backend.delete(user_id, memory_id)
