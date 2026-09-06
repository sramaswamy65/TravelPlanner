from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import pytest

from src.weather import FORECAST_URL, GEOCODING_URL, OpenMeteoWeather, WeatherError

LOCATION = {
    "results": [
        {
            "name": "Delhi",
            "admin1": "Delhi",
            "country": "India",
            "latitude": 28.6519,
            "longitude": 77.2315,
            "timezone": "Asia/Kolkata",
        }
    ]
}
FORECAST = {
    "daily": {
        "time": ["2026-09-04", "2026-09-05"],
        "weather_code": [2, 61],
        "temperature_2m_max": [31.0, 30.0],
        "temperature_2m_min": [24.0, 23.0],
        "precipitation_probability_max": [20, 65],
    }
}


class StubClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: float):
        self.calls.append((url, params, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(200, json=response, request=httpx.Request("GET", url))


def assert_error(code: str, operation) -> WeatherError:
    with pytest.raises(WeatherError) as caught:
        operation()
    assert caught.value.code == code
    return caught.value


def test_live_weather_geocoding_success_normalizes_fixture_schema():
    client = StubClient([LOCATION, FORECAST])
    result = OpenMeteoWeather(client=client).get_weather("Delhi", "2026-09-05")
    assert result == {
        "city": "Delhi",
        "date": "2026-09-05",
        "condition": "light rain",
        "high_c": 30.0,
        "low_c": 23.0,
        "precipitation_chance": 65,
        "source": "Open-Meteo",
        "timezone": "Asia/Kolkata",
        "latitude": 28.6519,
        "longitude": 77.2315,
    }
    assert [call[0] for call in client.calls] == [GEOCODING_URL, FORECAST_URL]
    assert client.calls[1][1]["timezone"] == "Asia/Kolkata"


def test_current_weather_uses_current_date_in_location_timezone():
    weather = OpenMeteoWeather(
        client=StubClient([LOCATION, FORECAST]),
        now=lambda timezone: datetime(2026, 9, 5, 0, 15, tzinfo=timezone),
    )
    assert weather.get_weather("Delhi", None, current=True)["date"] == "2026-09-05"


def test_ambiguous_city_geocoding_is_structured():
    ambiguous = {
        "results": [
            {**LOCATION["results"][0], "name": "Springfield", "admin1": "Illinois"},
            {**LOCATION["results"][0], "name": "Springfield", "admin1": "Missouri"},
        ]
    }
    error = assert_error(
        "ambiguous_city",
        lambda: OpenMeteoWeather(client=StubClient([ambiguous])).get_weather(
            "Springfield", "2026-09-05"
        ),
    )
    assert "state, region, or country" in error.message


def test_city_qualifier_resolves_ambiguous_geocoding():
    ambiguous = {
        "results": [
            {**LOCATION["results"][0], "name": "Delhi", "country": "India"},
            {**LOCATION["results"][0], "name": "Delhi", "country": "United States"},
        ]
    }
    client = StubClient([ambiguous, FORECAST])
    result = OpenMeteoWeather(client=client).get_weather("Delhi, India", "2026-09-05")
    assert result["source"] == "Open-Meteo"
    assert client.calls[0][1]["name"] == "Delhi"


def test_unknown_city_is_structured():
    assert_error(
        "unknown_city",
        lambda: OpenMeteoWeather(client=StubClient([{"results": []}])).get_weather(
            "Atlantis", "2026-09-05"
        ),
    )


def test_missing_date_asks_for_date_without_network_call():
    client = StubClient([])
    error = assert_error(
        "missing_date", lambda: OpenMeteoWeather(client=client).get_weather("Delhi", None)
    )
    assert "specify a forecast date" in error.message
    assert client.calls == []


def test_date_outside_forecast_range_does_not_fabricate():
    error = assert_error(
        "forecast_unavailable",
        lambda: OpenMeteoWeather(client=StubClient([LOCATION, FORECAST])).get_weather(
            "Delhi", "2027-01-01"
        ),
    )
    assert "reliable forecast" in error.message


def test_http_timeout_is_bounded_and_structured():
    request = httpx.Request("GET", GEOCODING_URL)
    client = StubClient([httpx.ReadTimeout("slow", request=request)] * 3)
    assert_error(
        "timeout",
        lambda: OpenMeteoWeather(client=client, retries=2).get_weather("Delhi", "2026-09-05"),
    )
    assert len(client.calls) == 3


def test_malformed_weather_response_is_structured():
    assert_error(
        "malformed_response",
        lambda: OpenMeteoWeather(
            client=StubClient([LOCATION, {"daily": {"time": ["2026-09-05"]}}])
        ).get_weather("Delhi", "2026-09-05"),
    )
