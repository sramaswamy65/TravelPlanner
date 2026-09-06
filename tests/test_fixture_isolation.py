import pytest

from evals.fixtures import activate_fixture, get_active_fixture


def test_fixture_cleanup_after_success_and_exception():
    with activate_fixture("weather-boston"):
        assert get_active_fixture().definition.fixture_id == "weather-boston"
    with pytest.raises(RuntimeError, match="No evaluation fixture"):
        get_active_fixture()
    with pytest.raises(ValueError), activate_fixture("weather-kathmandu"):
        raise ValueError("case failed")
    with pytest.raises(RuntimeError, match="No evaluation fixture"):
        get_active_fixture()


def test_unknown_fixture_is_rejected_without_network(monkeypatch):
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("network attempted"))
    with pytest.raises(ValueError, match="Unknown fixture"):
        activate_fixture("not-registered")
