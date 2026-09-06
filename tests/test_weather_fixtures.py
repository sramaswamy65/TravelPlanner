import pytest

from evals.fixtures import resolve_fixture_result


@pytest.mark.parametrize(
    "fixture,city,date,condition",
    [
        ("weather-boston", "Boston", "2026-09-10", "rain"),
        ("weather-kathmandu", "Kathmandu", "2026-10-15", "sunny"),
        ("weather-chicago", "Chicago", "2026-09-12", "windy"),
        ("weather-san-francisco", "San Francisco", "2026-09-14", "cloudy"),
    ],
)
def test_successful_weather_records(fixture, city, date, condition):
    result = resolve_fixture_result(fixture, "get_weather", {"city": city, "date": date})
    assert result.status == "success"
    assert result.data["condition"] == condition
    assert result.data["source"] == "evaluation_fixture"


@pytest.mark.parametrize(
    "fixture,code",
    [
        ("weather-timeout", "timeout"),
        ("weather-error", "provider_error"),
        ("invalid-date", "invalid_date"),
        ("outside-range", "unsupported_date"),
        ("ambiguous-city", "ambiguous_city"),
    ],
)
def test_weather_failure_envelopes(fixture, code):
    result = resolve_fixture_result(fixture, "get_weather", {"city": "Fixture", "date": "bad"})
    assert result.status == "error" and result.error["code"] == code
