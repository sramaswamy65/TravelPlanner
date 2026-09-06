from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import uuid
from datetime import UTC, datetime
from datetime import date as date_type
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from mcp.server.fastmcp import FastMCP

from src.config import ROOT_DIR, get_settings
from src.memory_policy import is_safe_memory_evidence
from src.places import FixturePlacesProvider, UnavailablePlacesProvider
from src.weather import OpenMeteoWeather, WeatherError


def ok(data: dict[str, Any] | list[Any]) -> dict[str, Any]:
    return {"status": "success", "data": data, "error": None}


def err(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "data": None, "error": {"code": code, "message": message}}


def capability_unavailable(message: str, source: str) -> dict[str, Any]:
    return {
        "status": "capability_unavailable",
        "data": {"source": source},
        "error": {"code": "capability_unavailable", "message": message},
    }


class MemoryBackend(Protocol):
    def search(self, user_id: str, query: str) -> list[dict[str, Any]]: ...
    def add(self, user_id: str, memory: str, category: str) -> dict[str, Any]: ...
    def list(self, user_id: str) -> list[dict[str, Any]]: ...
    def delete(self, user_id: str, memory_id: str) -> bool: ...


class FakeMemoryBackend:
    """Deterministic SQLite memory with the same user-scoped CRUD surface as Mem0."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._lock = Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, memory TEXT NOT NULL, category TEXT NOT NULL)"
        )
        self._connection.commit()

    def search(self, user_id: str, query: str) -> list[dict[str, Any]]:
        if query.strip().casefold() in {
            "travel preferences",
            "traveler preferences",
            "preferences",
        }:
            return self.list(user_id)
        words = set(re.findall(r"[a-z]+", query.lower()))
        return [
            m.copy()
            for m in self.list(user_id)
            if not words or words & set(re.findall(r"[a-z]+", m["memory"].lower()))
        ]

    def add(self, user_id: str, memory: str, category: str) -> dict[str, Any]:
        item = {"id": str(uuid.uuid4()), "memory": memory, "category": category, "user_id": user_id}
        with self._lock:
            self._connection.execute(
                "INSERT INTO memories (id, user_id, memory, category) VALUES (?, ?, ?, ?)",
                (item["id"], user_id, memory, category),
            )
            self._connection.commit()
        return item.copy()

    def list(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, memory, category, user_id FROM memories WHERE user_id = ? ORDER BY rowid",
                (user_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE user_id = ? AND id = ?", (user_id, memory_id)
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def seed(self, user_id: str, memories: list[dict[str, str]]) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self._connection.executemany(
                "INSERT INTO memories (id, user_id, memory, category) VALUES (?, ?, ?, ?)",
                [
                    (item["id"], user_id, item["memory"], item.get("category", "preference"))
                    for item in memories
                ],
            )
            self._connection.commit()

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self._connection.commit()


class Mem0Backend:
    confirmation_attempts = 10
    confirmation_interval_seconds = 0.5

    def __init__(self, api_key: str) -> None:
        from mem0 import MemoryClient

        self.client = MemoryClient(api_key=api_key)

    @staticmethod
    def _items(response: Any) -> list[dict[str, Any]]:
        if isinstance(response, dict):
            response = response.get("results", response.get("memories", []))
        return [dict(item) for item in response or [] if isinstance(item, dict)]

    def search(self, user_id: str, query: str) -> list[dict[str, Any]]:
        return self._items(self.client.search(query, version="v2", filters={"user_id": user_id}))

    def add(self, user_id: str, memory: str, category: str) -> dict[str, Any]:
        result = self.client.add(
            [{"role": "user", "content": memory}],
            user_id=user_id,
            metadata={"category": category},
            infer=False,
            output_format="v1.1",
        )
        items = self._items(result)
        confirmed = next((item for item in items if item.get("id")), None)
        if confirmed:
            return confirmed
        for _ in range(self.confirmation_attempts):
            time.sleep(self.confirmation_interval_seconds)
            confirmed = next(
                (
                    item
                    for item in self.list(user_id)
                    if item.get("id")
                    and str(item.get("memory", "")).strip().casefold() == memory.strip().casefold()
                ),
                None,
            )
            if confirmed:
                return confirmed
        raise RuntimeError("Mem0 did not confirm the memory write")

    def list(self, user_id: str) -> list[dict[str, Any]]:
        return self._items(self.client.get_all(version="v2", filters={"user_id": user_id}))

    def delete(self, user_id: str, memory_id: str) -> bool:
        if not any(str(m.get("id")) == memory_id for m in self.list(user_id)):
            return False
        self.client.delete(memory_id)
        return True


class UnavailableMemoryBackend:
    def _fail(self):
        raise RuntimeError("Live memory is not configured")

    def search(self, user_id: str, query: str) -> list[dict[str, Any]]:
        return self._fail()

    def add(self, user_id: str, memory: str, category: str) -> dict[str, Any]:
        return self._fail()

    def list(self, user_id: str) -> list[dict[str, Any]]:
        return self._fail()

    def delete(self, user_id: str, memory_id: str) -> bool:
        return self._fail()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Fixture data is unavailable or malformed: {path.name}") from exc


def _valid_text(value: str, field: str, maximum: int = 200) -> str:
    value = value.strip()
    if not value or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} must contain 1-{maximum} printable characters")
    return value


def _valid_content(value: str, maximum: int = 50_000) -> str:
    value = value.strip()
    if not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"content must contain 1-{maximum} characters and no null bytes")
    return value


def create_server(
    memory_backend: MemoryBackend | None = None, root_dir: Path = ROOT_DIR
) -> FastMCP:
    settings = get_settings()
    if settings.travel_fixture_id != "default-v1":
        raise ValueError(f"Unsupported travel fixture: {settings.travel_fixture_id}")
    backend = memory_backend
    if backend is None:
        if settings.effective_memory_mode == "fake":
            backend = FakeMemoryBackend(settings.fake_memory_db)
        elif settings.mem0_api_key:
            backend = Mem0Backend(settings.mem0_api_key)
        else:
            backend = UnavailableMemoryBackend()
    memory_source = (
        "fixture SQLite memory" if isinstance(backend, FakeMemoryBackend) else "Mem0 Platform"
    )
    live_weather = OpenMeteoWeather(
        timeout_seconds=settings.weather_http_timeout_seconds,
        retries=settings.weather_http_retries,
    )
    server = FastMCP("TravelPlanner Tools", json_response=True)
    restaurant_provider = (
        FixturePlacesProvider(root_dir / "data" / "restaurant_fixtures.json")
        if settings.effective_travel_data_mode == "fixture"
        else UnavailablePlacesProvider()
    )

    @server.tool(
        description=(
            "Get weather for an ISO date. Omit date only when the user explicitly asks for "
            "current weather, and then set current=true. If the request has no date and is not "
            "current, ask the user for a date before calling this tool."
        )
    )
    def get_weather(city: str, date: str | None = None, current: bool = False) -> dict[str, Any]:
        try:
            city = _valid_text(city, "city", 100)
            if settings.effective_travel_data_mode == "live":
                return ok(live_weather.get_weather(city, date, current=current))
            if date is None and not current:
                return err(
                    "missing_date", "Please specify a forecast date, or ask for current weather."
                )
            requested = datetime.now(UTC).date() if current else date_type.fromisoformat(str(date))
            fixtures = _load_json(root_dir / "data" / "weather_fixtures.json")
            city_data = fixtures.get(city.casefold())
            if city_data == "SIMULATED_TIMEOUT":
                return err("timeout", "Weather provider timed out; no weather result is available")
            if not isinstance(city_data, dict):
                return err(
                    "unsupported_city",
                    f"The demonstration weather dataset does not contain {city}.",
                )
            forecast = city_data.get(requested.isoformat()) or city_data.get("default")
            if not isinstance(forecast, dict):
                return err(
                    "date_unavailable",
                    f"The demonstration weather dataset has no entry for {requested.isoformat()} in {city}.",
                )
            normalized = dict(forecast)
            normalized.pop("status", None)
            normalized["city"] = city
            normalized["date"] = requested.isoformat()
            return ok(normalized)
        except WeatherError as exc:
            return err(exc.code, exc.message)
        except (TypeError, ValueError) as exc:
            return err("invalid_arguments", str(exc))

    @server.tool(
        description=(
            "Find deterministic local attractions by city and category. Use directly for requests "
            "such as 'find museums'; weather and memory are not prerequisites."
        )
    )
    def find_attractions(city: str, category: str) -> dict[str, Any]:
        try:
            city = _valid_text(city, "city", 100)
            category = _valid_text(category, "category", 80)
            attractions = _load_json(root_dir / "data" / "attractions.json")
            supported = sorted({str(item.get("city")) for item in attractions if item.get("city")})
            if city.casefold() not in {item.casefold() for item in supported}:
                return err(
                    "unsupported_city",
                    f"The local attraction dataset does not contain {city}. "
                    f"It currently supports only: {', '.join(supported)}.",
                )
            supported_categories = sorted(
                {
                    str(category)
                    for item in attractions
                    for category in item.get("categories", [item.get("category")])
                    if category
                }
            )
            if category.casefold() != "any" and category.casefold() not in {
                item.casefold() for item in supported_categories
            }:
                return err(
                    "unsupported_category",
                    f"The attraction dataset does not support category {category}. "
                    f"It supports only: {', '.join(supported_categories)}.",
                )
            matches = [
                item
                for item in attractions
                if item.get("city", "").casefold() == city.casefold()
                and (
                    category.casefold() == "any"
                    or category.casefold()
                    in {
                        str(value).casefold()
                        for value in item.get("categories", [item.get("category")])
                        if value
                    }
                )
            ]
            return ok(
                {
                    "city": city,
                    "category": category,
                    "attractions": matches,
                    "source": "local attraction fixture",
                }
            )
        except ValueError as exc:
            return err("invalid_arguments", str(exc))

    @server.tool(
        description=(
            "Search the configured restaurant provider. Use this tool, not find_attractions, "
            "for restaurant, dining, or food-place requests. Results are matches, not rankings."
        )
    )
    def search_restaurants(
        city: str, dietary_preference: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        try:
            city = _valid_text(city, "city", 100)
            if dietary_preference is not None:
                dietary_preference = _valid_text(dietary_preference, "dietary_preference", 100)
            if not 1 <= limit <= 20:
                return err("invalid_arguments", "limit must be between 1 and 20")
            try:
                matches = restaurant_provider.search_restaurants(city, dietary_preference, limit)
            except NotImplementedError:
                return capability_unavailable(
                    "Live restaurant search is not currently configured, and fixture results "
                    "were not used in live mode.",
                    restaurant_provider.source,
                )
            supported = sorted(
                {
                    str(item.get("city"))
                    for item in _load_json(root_dir / "data" / "restaurant_fixtures.json")
                    if isinstance(item, dict) and item.get("city")
                }
            )
            if city.casefold() not in {item.casefold() for item in supported}:
                return err(
                    "unsupported_city",
                    f"The restaurant fixture dataset does not contain {city}. "
                    f"It currently supports only: {', '.join(supported)}.",
                )
            return ok(
                {
                    "city": city,
                    "dietary_preference": dietary_preference,
                    "restaurants": matches,
                    "source": restaurant_provider.source,
                }
            )
        except (TypeError, ValueError) as exc:
            return err("invalid_arguments", str(exc))

    @server.tool(description="Get details for one deterministic local attraction.")
    def get_attraction_details(name: str) -> dict[str, Any]:
        try:
            name = _valid_text(name, "name", 150)
            attractions = _load_json(root_dir / "data" / "attractions.json")
            match = next(
                (
                    item
                    for item in attractions
                    if item.get("name", "").casefold() == name.casefold()
                ),
                None,
            )
            if not match:
                return err("not_found", f"Attraction not found: {name}")
            if match.get("closed") or all(
                value == "closed" for value in match.get("opening_hours", {}).values()
            ):
                return err("closed_at_requested_time", f"{name} is closed in the fixture data")
            return ok({**match, "source": "local attraction fixture"})
        except ValueError as exc:
            return err("invalid_arguments", str(exc))

    @server.tool(
        description=(
            "Search durable traveler preferences scoped to one user ID. Use query "
            "'travel preferences' to retrieve all durable preferences relevant to trip planning."
        )
    )
    def search_traveler_memory(user_id: str, query: str) -> dict[str, Any]:
        try:
            memories = backend.search(
                _valid_text(user_id, "user_id", 80),
                _valid_text(query, "query", 500),
            )
            safe_memories = [
                memory
                for memory in memories
                if is_safe_memory_evidence(str(memory.get("memory", "")))
            ]
            return ok({"memories": safe_memories, "source": memory_source})
        except ValueError as exc:
            return err("invalid_arguments", str(exc))
        except Exception:  # noqa: BLE001 - backend errors must not leak provider details.
            return err("memory_error", "Memory search failed")

    @server.tool(
        description=(
            "Save an explicitly authorized durable traveler preference for one user. Do not call "
            "for temporary trip facts, forecasts, credentials, secrets, or inferred sensitive facts."
        )
    )
    def save_traveler_memory(user_id: str, memory: str, category: str) -> dict[str, Any]:
        try:
            item = backend.add(
                _valid_text(user_id, "user_id", 80),
                _valid_text(memory, "memory", 500),
                _valid_text(category, "category", 50),
            )
            return ok({"memory": item, "source": memory_source})
        except ValueError as exc:
            return err("invalid_arguments", str(exc))
        except Exception:  # noqa: BLE001 - backend errors must not leak provider details.
            return err("memory_error", "Memory save failed")

    @server.tool(
        description=(
            "List durable traveler memories scoped to one user ID. Use when the user asks what is "
            "remembered; this is read-only and must not be replaced by a save call."
        )
    )
    def list_traveler_memories(user_id: str) -> dict[str, Any]:
        try:
            return ok(
                {
                    "memories": backend.list(_valid_text(user_id, "user_id", 80)),
                    "source": memory_source,
                }
            )
        except ValueError as exc:
            return err("invalid_arguments", str(exc))
        except Exception:  # noqa: BLE001 - backend errors must not leak provider details.
            return err("memory_error", "Memory listing failed")

    @server.tool(description="Delete one memory owned by a user after explicit authorization.")
    def delete_traveler_memory(user_id: str, memory_id: str) -> dict[str, Any]:
        try:
            deleted = backend.delete(
                _valid_text(user_id, "user_id", 80), _valid_text(memory_id, "memory_id", 100)
            )
            return (
                ok({"deleted": deleted, "memory_id": memory_id, "source": memory_source})
                if deleted
                else err("not_found", "Memory was not found for this user")
            )
        except ValueError as exc:
            return err("invalid_arguments", str(exc))
        except Exception:  # noqa: BLE001 - backend errors must not leak provider details.
            return err("memory_error", "Memory deletion failed")

    @server.tool(
        description="Save an approved Markdown itinerary beneath the user's plans directory."
    )
    def save_itinerary(user_id: str, filename: str, content: str, approved: bool) -> dict[str, Any]:
        if not approved:
            return err("approval_required", "Explicit approval is required before saving")
        try:
            user_id = _valid_text(user_id, "user_id", 80)
            filename = _valid_text(filename, "filename", 100)
            if Path(filename).is_absolute() or Path(filename).name != filename or ".." in filename:
                return err("unsafe_path", "Filename must be a simple relative filename")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.md", filename):
                return err("unsafe_filename", "Filename must be a safe .md filename")
            safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
            directory = (root_dir / "output" / "plans" / safe_user).resolve()
            base = (root_dir / "output" / "plans").resolve()
            if base not in directory.parents:
                return err("unsafe_path", "Resolved output path is unsafe")
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / filename
            if destination.exists():
                return err("already_exists", "Plan already exists; choose a new filename")
            destination.write_text(_valid_content(content), encoding="utf-8")
            return ok(
                {
                    "path": str(destination.relative_to(root_dir)),
                    "bytes": destination.stat().st_size,
                }
            )
        except (OSError, ValueError):
            return err("write_error", "The itinerary could not be safely written")

    return server


if __name__ == "__main__":
    try:
        create_server().run(transport="stdio")
    except Exception as exc:  # noqa: BLE001 - top-level protocol boundary reports only exception type.
        print(f"MCP server failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
