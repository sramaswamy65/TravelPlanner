from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class PlacesProvider(Protocol):
    source: str

    def search_restaurants(
        self, city: str, dietary_preference: str | None, limit: int
    ) -> list[dict[str, Any]]: ...


class FixturePlacesProvider:
    source = "local restaurant fixture"

    def __init__(self, path: Path) -> None:
        self.path = path

    def search_restaurants(
        self, city: str, dietary_preference: str | None, limit: int
    ) -> list[dict[str, Any]]:
        try:
            items = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Restaurant fixture data is unavailable or malformed") from exc
        if not isinstance(items, list):
            raise TypeError("Restaurant fixture data is unavailable or malformed")
        matches = [
            item
            for item in items
            if isinstance(item, dict) and item.get("city", "").casefold() == city.casefold()
        ]
        if dietary_preference:
            dietary = dietary_preference.casefold()
            matches = [
                item
                for item in matches
                if dietary
                in {str(value).casefold() for value in item.get("dietary_suitability", [])}
                or (dietary == "vegetarian" and item.get("vegetarian_suitable") is True)
            ]
        return matches[:limit]


class UnavailablePlacesProvider:
    source = "unavailable live restaurant provider"

    def search_restaurants(
        self, city: str, dietary_preference: str | None, limit: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Live restaurant search is not currently configured")
