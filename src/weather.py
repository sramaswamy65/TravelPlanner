from __future__ import annotations

from collections.abc import Callable
from datetime import date as date_type
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CONDITIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}


class WeatherError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OpenMeteoWeather:
    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        retries: int = 2,
        client: httpx.Client | None = None,
        now: Callable[[ZoneInfo], datetime] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.client = client or httpx.Client()
        self.now = now or (lambda timezone: datetime.now(timezone))

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.retries + 1):
            try:
                response = self.client.get(url, params=params, timeout=self.timeout_seconds)
                if response.status_code >= 500 and attempt < self.retries:
                    continue
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise WeatherError(
                        "malformed_response", "Open-Meteo returned malformed JSON."
                    ) from exc
                if not isinstance(payload, dict):
                    raise WeatherError("malformed_response", "Open-Meteo returned malformed JSON.")
                return payload
            except httpx.TimeoutException as exc:
                if attempt == self.retries:
                    raise WeatherError(
                        "timeout", "The live weather provider timed out. Please try again."
                    ) from exc
            except WeatherError:
                raise
            except httpx.HTTPError as exc:
                if attempt == self.retries:
                    raise WeatherError(
                        "provider_error", "The live weather provider is temporarily unavailable."
                    ) from exc
        raise WeatherError(
            "provider_error", "The live weather provider is temporarily unavailable."
        )

    @staticmethod
    def _select_location(city: str, payload: dict[str, Any]) -> dict[str, Any]:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise WeatherError(
                "unknown_city", f"Open-Meteo could not find a location matching {city}."
            )
        valid = [
            item
            for item in results
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("latitude"), (int, float))
            and isinstance(item.get("longitude"), (int, float))
            and isinstance(item.get("timezone"), str)
        ]
        if not valid:
            raise WeatherError(
                "malformed_response", "Open-Meteo returned malformed geocoding data."
            )
        parts = [part.strip().casefold() for part in city.split(",") if part.strip()]
        query = parts[0]
        exact = [item for item in valid if item["name"].casefold() == query]
        if len(parts) > 1:
            exact = [
                item
                for item in exact
                if all(
                    qualifier
                    in " ".join(
                        str(item.get(field, "")).casefold()
                        for field in ("admin1", "admin2", "country", "country_code")
                    )
                    for qualifier in parts[1:]
                )
            ]
        candidates = exact or valid
        if len(candidates) != 1:
            labels = [
                ", ".join(
                    str(value)
                    for value in (item.get("name"), item.get("admin1"), item.get("country"))
                    if value
                )
                for item in candidates[:3]
            ]
            raise WeatherError(
                "ambiguous_city",
                f"{city} matches multiple locations "
                f"({'; '.join(labels)}). Add a state, region, or country.",
            )
        return candidates[0]

    def get_weather(
        self, city: str, requested_date: str | None, *, current: bool = False
    ) -> dict[str, Any]:
        if requested_date is None and not current:
            raise WeatherError(
                "missing_date", "Please specify a forecast date, or ask for current weather."
            )
        geocoded = self._get_json(
            GEOCODING_URL,
            {
            "name": city.split(",", 1)[0].strip(),
                "count": 10,
                "language": "en",
                "format": "json",
            },
        )
        location = self._select_location(city, geocoded)
        try:
            timezone = ZoneInfo(location["timezone"])
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise WeatherError(
                "malformed_response", "Open-Meteo returned an invalid location timezone."
            ) from exc
        if current:
            requested = self.now(timezone).date()
        else:
            try:
                requested = date_type.fromisoformat(str(requested_date))
            except ValueError as exc:
                raise WeatherError(
                    "invalid_arguments", "date must be an ISO date (YYYY-MM-DD)"
                ) from exc
        forecast = self._get_json(
            FORECAST_URL,
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "timezone": location["timezone"],
                "forecast_days": 16,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            },
        )
        daily = forecast.get("daily")
        keys = (
            "time",
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
        )
        if not isinstance(daily, dict) or any(not isinstance(daily.get(key), list) for key in keys):
            raise WeatherError("malformed_response", "Open-Meteo returned malformed forecast data.")
        lengths = {len(daily[key]) for key in keys}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise WeatherError(
                "malformed_response", "Open-Meteo returned inconsistent forecast data."
            )
        target = requested.isoformat()
        if target not in daily["time"]:
            available = sorted(str(value) for value in daily["time"])
            raise WeatherError(
                "forecast_unavailable",
                f"A reliable forecast for {target} is not yet "
                f"available. Open-Meteo currently provides {available[0]} through "
                f"{available[-1]} for this location.",
            )
        index = daily["time"].index(target)
        values = {key: daily[key][index] for key in keys[1:]}
        if (
            not isinstance(values["weather_code"], int)
            or not isinstance(values["temperature_2m_max"], (int, float))
            or not isinstance(values["temperature_2m_min"], (int, float))
            or not isinstance(values["precipitation_probability_max"], (int, float))
        ):
            raise WeatherError(
                "malformed_response", "Open-Meteo returned malformed daily weather values."
            )
        code = values["weather_code"]
        return {
            "city": location["name"],
            "date": target,
            "condition": WMO_CONDITIONS.get(code, f"weather code {code}"),
            "high_c": values["temperature_2m_max"],
            "low_c": values["temperature_2m_min"],
            "precipitation_chance": values["precipitation_probability_max"],
            "source": "Open-Meteo",
            "timezone": location["timezone"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        }
