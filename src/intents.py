from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedIntents:
    names: tuple[str, ...]
    city: str | None = None
    inherited_city: bool = False
    memory: str | None = None
    memory_category: str | None = None


def _city_from_text(text: str) -> str | None:
    patterns = (
        r"\b(?:in|for)\s+([A-Z][A-Za-z '-]+?)(?:\s+(?:on|today|tomorrow)\b|[,.?!]|$)",
        r"\b([A-Z][A-Za-z '-]+)\s+weather\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def resolve_intents(message: str, conversation: list[dict[str, Any]]) -> ResolvedIntents:
    lower = message.casefold()
    names: list[str] = []
    memory = None
    category = None
    if "remember" in lower and "vegetarian" in lower:
        names.append("save_dietary_preference")
        memory = "Is vegetarian"
        category = "dietary_preference"
    if "restaurant" in lower:
        names.append("search_restaurants")
    city = _city_from_text(message)
    inherited = False
    if "search_restaurants" in names and city is None:
        for item in reversed(conversation):
            if item.get("role") != "user":
                continue
            city = _city_from_text(str(item.get("content", "")))
            if city:
                inherited = True
                break
    return ResolvedIntents(tuple(names), city, inherited, memory, category)
