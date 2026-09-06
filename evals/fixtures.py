from __future__ import annotations

import json
import re
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

from src.config import ROOT_DIR, Settings
from src.mcp_client import TravelMCPClient
from src.mcp_server import Mem0Backend
from src.models import ModelToolCall, ModelTurn, ToolResult

FixtureType = Literal[
    "normal data fixture",
    "empty-result fixture",
    "unsupported-city fixture",
    "unsupported-category fixture",
    "invalid-date case",
    "ambiguous-city fixture",
    "provider-error fixture",
    "timeout fixture",
    "malformed-data fixture",
    "closed-attraction fixture",
    "approval-required fixture",
    "unsafe-path fixture",
    "existing-file fixture",
    "missing-memory fixture",
    "memory-provider failure fixture",
    "repeated-call fixture",
    "step-limit fixture",
    "tracing-failure fixture",
    "prompt-injection fixture",
    "cross-user isolation fixture",
    "secret-handling fixture",
    "clarification-only fixture",
]


@dataclass(frozen=True)
class FixtureDefinition:
    fixture_id: str
    fixture_type: FixtureType
    description: str
    supported_tools: tuple[str, ...]
    data_source: str
    injected_behavior: str | None = None
    expected_status: str = "success"
    expected_error_code: str | None = None
    latency_ms: int = 0
    cleanup_behavior: str = "ContextVar reset and temporary case directory removal"


@dataclass(frozen=True)
class FixtureValidationReport:
    valid: bool
    fixture_count: int
    duplicate_ids: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass
class FixtureContext:
    definition: FixtureDefinition
    _token: Token[FixtureDefinition | None] | None = None

    def __enter__(self) -> Self:
        self._token = _ACTIVE_FIXTURE.set(self.definition)
        return self

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _ACTIVE_FIXTURE.reset(self._token)
            self._token = None


def _definition(
    fixture_id: str,
    fixture_type: FixtureType,
    tools: tuple[str, ...] = (),
    error: str | None = None,
    behavior: str | None = None,
    source: str = "evaluation registry",
) -> FixtureDefinition:
    return FixtureDefinition(
        fixture_id=fixture_id,
        fixture_type=fixture_type,
        description=f"Deterministic {fixture_type} for {fixture_id}.",
        supported_tools=tools,
        data_source=source,
        injected_behavior=behavior,
        expected_status="error" if error else "success",
        expected_error_code=error,
    )


_DEFINITIONS = [
    *[
        _definition(
            f"weather-{city}",
            "normal data fixture",
            ("get_weather",),
            source="data/weather_fixtures.json",
        )
        for city in ("kathmandu", "boston", "chicago", "san-francisco")
    ],
    *[
        _definition(
            f"attractions-{city}",
            "normal data fixture",
            ("find_attractions",),
            source="data/attractions.json",
        )
        for city in ("kathmandu", "boston", "chicago", "san-francisco")
    ],
    *[
        _definition(
            f"restaurants-{city}",
            "normal data fixture",
            ("search_restaurants",),
            source="data/restaurant_fixtures.json",
        )
        for city in ("boston", "chicago", "san-francisco", "kathmandu")
    ],
    _definition(
        "attraction-details",
        "normal data fixture",
        ("get_attraction_details",),
        source="data/attractions.json",
    ),
    _definition(
        "memory-default",
        "normal data fixture",
        (
            "search_traveler_memory",
            "save_traveler_memory",
            "list_traveler_memories",
            "delete_traveler_memory",
        ),
        source="isolated fake-memory SQLite",
    ),
    _definition("clarification", "clarification-only fixture"),
    _definition(
        "memory-policy",
        "normal data fixture",
        ("save_traveler_memory", "list_traveler_memories"),
        source="isolated fake-memory SQLite",
    ),
    _definition(
        "unsupported-city",
        "unsupported-city fixture",
        ("get_weather", "find_attractions", "search_restaurants"),
        "unsupported_city",
    ),
    _definition("invalid-date", "invalid-date case", ("get_weather",), "invalid_date"),
    _definition(
        "empty-attractions", "empty-result fixture", ("find_attractions",), "empty_results"
    ),
    _definition(
        "unknown-attraction", "empty-result fixture", ("get_attraction_details",), "not_found"
    ),
    _definition(
        "unsupported-restaurants",
        "unsupported-city fixture",
        ("search_restaurants",),
        "unsupported_city",
    ),
    _definition("ambiguous-city", "ambiguous-city fixture", ("get_weather",), "ambiguous_city"),
    _definition(
        "closed-attractions",
        "closed-attraction fixture",
        ("get_attraction_details",),
        "closed_at_requested_time",
    ),
    _definition(
        "restaurant-limit",
        "normal data fixture",
        ("search_restaurants",),
        behavior="Clamp requested limit to 10",
        source="data/restaurant_fixtures.json",
    ),
    _definition("outside-range", "invalid-date case", ("get_weather",), "unsupported_date"),
    _definition(
        "unsupported-category",
        "unsupported-category fixture",
        ("find_attractions",),
        "unsupported_category",
    ),
    _definition("weather-timeout", "timeout fixture", ("get_weather",), "timeout"),
    _definition(
        "malformed-attractions",
        "malformed-data fixture",
        ("find_attractions",),
        "malformed_fixture",
    ),
    _definition("weather-error", "provider-error fixture", ("get_weather",), "provider_error"),
    _definition("restaurant-timeout", "timeout fixture", ("search_restaurants",), "timeout"),
    _definition(
        "malformed-details",
        "malformed-data fixture",
        ("get_attraction_details",),
        "malformed_fixture",
    ),
    _definition(
        "write-no-approval", "approval-required fixture", ("save_itinerary",), "approval_required"
    ),
    _definition("unsafe-path", "unsafe-path fixture", ("save_itinerary",), "unsafe_path"),
    _definition("absolute-path", "unsafe-path fixture", ("save_itinerary",), "unsafe_path"),
    _definition(
        "existing-file", "existing-file fixture", ("save_itinerary",), "overwrite_not_approved"
    ),
    _definition(
        "missing-memory", "missing-memory fixture", ("delete_traveler_memory",), "not_found"
    ),
    _definition(
        "memory-timeout", "memory-provider failure fixture", ("list_traveler_memories",), "timeout"
    ),
    _definition(
        "memory-write-error",
        "memory-provider failure fixture",
        ("save_traveler_memory",),
        "provider_error",
    ),
    _definition(
        "duplicate-model-calls", "repeated-call fixture", ("get_weather",), "repeated_call"
    ),
    _definition(
        "step-limit", "step-limit fixture", ("get_weather", "find_attractions"), "step_limit"
    ),
    _definition(
        "trace-failure",
        "tracing-failure fixture",
        ("get_weather", "find_attractions"),
        behavior="Trace adapter failure is contained",
    ),
    _definition(
        "memory-injection",
        "prompt-injection fixture",
        ("search_traveler_memory", "get_weather", "find_attractions"),
        source="isolated fake-memory SQLite",
    ),
    _definition(
        "tool-injection",
        "prompt-injection fixture",
        ("find_attractions",),
        source="data/attractions.json",
    ),
    _definition("path-traversal", "unsafe-path fixture", ("save_itinerary",), "unsafe_path"),
    _definition("cross-user-exfiltration", "cross-user isolation fixture"),
    _definition("secret-exfiltration", "secret-handling fixture"),
]

FIXTURE_REGISTRY = MappingProxyType({item.fixture_id: item for item in _DEFINITIONS})
_ACTIVE_FIXTURE: ContextVar[FixtureDefinition | None] = ContextVar("travel_fixture", default=None)


def activate_fixture(fixture_id: str) -> FixtureContext:
    try:
        return FixtureContext(FIXTURE_REGISTRY[fixture_id])
    except KeyError as exc:
        raise ValueError(f"Unknown fixture: {fixture_id}") from exc


def get_active_fixture() -> FixtureContext:
    definition = _ACTIVE_FIXTURE.get()
    if definition is None:
        raise RuntimeError("No evaluation fixture is active")
    return FixtureContext(definition)


def clear_fixture() -> None:
    _ACTIVE_FIXTURE.set(None)


def validate_fixture_registry() -> FixtureValidationReport:
    ids = [item.fixture_id for item in _DEFINITIONS]
    duplicates = tuple(sorted({item for item in ids if ids.count(item) > 1}))
    errors = tuple(
        f"{item.fixture_id}: error fixture lacks an error code"
        for item in _DEFINITIONS
        if item.expected_status == "error" and not item.expected_error_code
    )
    return FixtureValidationReport(not duplicates and not errors, len(ids), duplicates, errors)


def _load(path: str) -> Any:
    return json.loads((ROOT_DIR / path).read_text(encoding="utf-8"))


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(status="error", data=None, error={"code": code, "message": message})


def save_itinerary_fixture(
    root: Path,
    user_id: str,
    filename: str,
    content: str,
    approved: bool,
    overwrite_approved: bool = False,
) -> ToolResult:
    """Exercise save validation beneath a caller-owned temporary directory."""
    if not approved:
        return _error("approval_required", ERROR_MESSAGES["approval_required"])
    candidate = Path(filename)
    if candidate.is_absolute() or candidate.name != filename or ".." in filename:
        return _error("unsafe_path", ERROR_MESSAGES["unsafe_path"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.md", filename):
        return _error("unsafe_path", ERROR_MESSAGES["unsafe_path"])
    safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
    directory = (root / safe_user).resolve()
    if root.resolve() not in directory.parents:
        return _error("unsafe_path", ERROR_MESSAGES["unsafe_path"])
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.exists() and not overwrite_approved:
        return _error("overwrite_not_approved", ERROR_MESSAGES["overwrite_not_approved"])
    destination.write_text(content, encoding="utf-8")
    return ToolResult(
        status="success",
        data={
            "path": f"{safe_user}/{filename}",
            "bytes": destination.stat().st_size,
            "source": "evaluation_fixture",
        },
    )


ERROR_MESSAGES = {
    "unsupported_city": "The deterministic fixture does not support that city.",
    "unsupported_category": "The deterministic fixture does not support that category.",
    "unsupported_date": "A reliable fixture forecast is not available for that date.",
    "invalid_date": "The date must use valid ISO YYYY-MM-DD format.",
    "ambiguous_city": "The city is ambiguous; provide a country or region.",
    "empty_results": "The deterministic fixture returned no results.",
    "not_found": "The requested fixture record was not found.",
    "closed_at_requested_time": "The attraction is closed at the requested time.",
    "timeout": "The deterministic provider timed out immediately.",
    "provider_error": "The deterministic provider returned a controlled error.",
    "malformed_fixture": "The deterministic fixture response is malformed.",
    "approval_required": "Explicit approval is required before saving.",
    "unsafe_path": "Unsafe filename: use a simple relative Markdown filename.",
    "overwrite_not_approved": "The itinerary already exists and overwrite was not approved.",
    "repeated_call": "A repeated identical tool call was detected.",
    "step_limit": "The deterministic model reached the configured step limit.",
}


class EvaluationMCPClient:
    """Fixture-only tool adapter; it never contacts a live travel provider."""

    def __init__(self, settings: Settings, fixture_id: str) -> None:
        self.inner = TravelMCPClient(settings)
        self.direct_memory = (
            Mem0Backend(settings.mem0_api_key) if settings.effective_memory_mode == "live" else None
        )
        self.fixture_id = fixture_id
        self.fixture_context: FixtureContext | None = None

    async def __aenter__(self) -> Self:
        self.fixture_context = activate_fixture(self.fixture_id)
        self.fixture_context.__enter__()
        try:
            if self.direct_memory is None:
                await self.inner.__aenter__()
            return self
        except BaseException:
            self.fixture_context.__exit__(None, None, None)
            raise

    async def __aexit__(self, *exc: object) -> None:
        try:
            if self.direct_memory is None:
                await self.inner.__aexit__(*exc)
        finally:
            if self.fixture_context:
                self.fixture_context.__exit__(*exc)

    async def list_tools(self) -> list[dict[str, Any]]:
        if self.direct_memory is not None:
            return _evaluation_tool_schemas()
        return await self.inner.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = resolve_fixture_result(self.fixture_id, name, arguments)
        if result is not None:
            return result
        if self.direct_memory is not None:
            return _call_live_memory_tool(self.direct_memory, name, arguments)
        return await self.inner.call_tool(name, arguments)


def _evaluation_tool_schemas() -> list[dict[str, Any]]:
    arguments = {
        "get_weather": {"city": "string", "date": "string"},
        "find_attractions": {"city": "string", "category": "string"},
        "search_restaurants": {"city": "string", "dietary_preference": "string", "limit": "integer"},
        "get_attraction_details": {"name": "string"},
        "search_traveler_memory": {"user_id": "string", "query": "string"},
        "save_traveler_memory": {"user_id": "string", "memory": "string", "category": "string"},
        "list_traveler_memories": {"user_id": "string"},
        "delete_traveler_memory": {"user_id": "string", "memory_id": "string"},
        "save_itinerary": {"user_id": "string", "filename": "string", "content": "string", "approved": "boolean"},
    }
    required = {
        "get_weather": ["city", "date"], "find_attractions": ["city", "category"],
        "search_restaurants": ["city"], "get_attraction_details": ["name"],
        "search_traveler_memory": ["user_id", "query"],
        "save_traveler_memory": ["user_id", "memory", "category"],
        "list_traveler_memories": ["user_id"],
        "delete_traveler_memory": ["user_id", "memory_id"],
        "save_itinerary": ["user_id", "filename", "content", "approved"],
    }
    return [{"type": "function", "function": {"name": name,
             "description": f"TravelPlanner {name} tool.",
             "parameters": {"type": "object", "properties": {
                 key: {"type": kind} for key, kind in fields.items()},
                 "required": required[name], "additionalProperties": False}}}
            for name, fields in arguments.items()]


def _call_live_memory_tool(
    backend: Mem0Backend, name: str, arguments: dict[str, Any]
) -> ToolResult:
    try:
        user_id = str(arguments.get("user_id", ""))
        if name == "search_traveler_memory":
            memories = backend.search(user_id, str(arguments.get("query", "")))
            return ToolResult(status="success", data={"memories": memories, "source": "Mem0 Platform"})
        if name == "list_traveler_memories":
            return ToolResult(status="success", data={"memories": backend.list(user_id), "source": "Mem0 Platform"})
        if name == "save_traveler_memory":
            memory = backend.add(user_id, str(arguments.get("memory", "")),
                                 str(arguments.get("category", "preference")))
            return ToolResult(status="success", data={"memory": memory, "source": "Mem0 Platform"})
        if name == "delete_traveler_memory":
            deleted = backend.delete(user_id, str(arguments.get("memory_id", "")))
            return ToolResult(status="success", data={"deleted": deleted, "source": "Mem0 Platform"})
        if name == "save_itinerary" and arguments.get("approved"):
            return ToolResult(status="success", data={"saved": True, "source": "evaluation_fixture"})
        if name == "save_itinerary":
            return _error("approval_required", ERROR_MESSAGES["approval_required"])
        return _error("invalid_arguments", "The evaluation tool request is unsupported.")
    except Exception:  # noqa: BLE001 - providers become safe structured tool failures.
        return _error("provider_error", "The live memory provider returned an error.")


def resolve_fixture_result(
    fixture_id: str, name: str, arguments: dict[str, Any]
) -> ToolResult | None:
    fixture = FIXTURE_REGISTRY[fixture_id]
    if fixture.expected_error_code and name in fixture.supported_tools:
        return _error(fixture.expected_error_code, ERROR_MESSAGES[fixture.expected_error_code])
    if name == "get_weather" and fixture.expected_status == "success":
        city, requested = str(arguments.get("city", "")), str(arguments.get("date", ""))
        record = _load("data/weather_fixtures.json").get(city.casefold(), {}).get(requested)
        if isinstance(record, dict):
            record = {key: value for key, value in record.items() if key != "status"}
        return (
            ToolResult(status="success", data=record)
            if isinstance(record, dict)
            else _error("unsupported_date", ERROR_MESSAGES["unsupported_date"])
        )
    if name == "find_attractions" and fixture.expected_status == "success":
        city, category = str(arguments.get("city", "")), str(arguments.get("category", ""))
        records = [
            item
            for item in _load("data/attractions.json")
            if item["city"].casefold() == city.casefold()
            and (
                category.casefold() == "any"
                or category.casefold() in {value.casefold() for value in item["categories"]}
            )
        ]
        if fixture_id == "tool-injection":
            records = [item for item in records if "UNTRUSTED FIXTURE TEXT" in item["description"]]
        return ToolResult(
            status="success",
            data={
                "city": city,
                "category": category,
                "attractions": records,
                "source": "evaluation_fixture",
            },
        )
    if name == "get_attraction_details" and fixture.expected_status == "success":
        requested = str(arguments.get("name", ""))
        normalized = requested.casefold().replace("the ", "")
        record = next(
            (
                item
                for item in _load("data/attractions.json")
                if normalized in item["name"].casefold() or item["name"].casefold() in normalized
            ),
            None,
        )
        return (
            ToolResult(status="success", data=record)
            if record
            else _error("not_found", ERROR_MESSAGES["not_found"])
        )
    if name == "search_restaurants" and fixture.expected_status == "success":
        city = str(arguments.get("city", ""))
        dietary = arguments.get("dietary_preference")
        limit = min(max(int(arguments.get("limit", 5)), 1), 10)
        records = [
            item
            for item in _load("data/restaurant_fixtures.json")
            if item["city"].casefold() == city.casefold()
            and (
                not dietary
                or str(dietary).casefold()
                in {value.casefold() for value in item["dietary_suitability"]}
            )
        ][:limit]
        return ToolResult(
            status="success",
            data={
                "city": city,
                "dietary_preference": dietary,
                "requested_limit": arguments.get("limit", 5),
                "applied_limit": limit,
                "restaurants": records,
                "source": "evaluation_fixture",
            },
        )
    return None


class FixtureModelClient:
    """Injects only model-loop failure behavior; normal decisions remain delegated."""

    def __init__(self, delegate: Any, fixture_id: str) -> None:
        self.delegate = delegate
        self.fixture_id = fixture_id
        self.model = delegate.model
        self.turn = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.turn += 1
        if self.fixture_id == "duplicate-model-calls" and any(
            item.get("role") == "tool" and item.get("name") == "get_weather" for item in messages
        ):
            return ModelTurn(
                tool_calls=[
                    ModelToolCall(
                        id=str(uuid.uuid4()),
                        name="get_weather",
                        arguments={"city": "Chicago", "date": "2026-09-12"},
                    )
                ]
            )
        if self.fixture_id == "step-limit":
            return ModelTurn(
                tool_calls=[
                    ModelToolCall(
                        id=str(uuid.uuid4()),
                        name="get_weather",
                        arguments={
                            "city": "San Francisco",
                            "date": f"2026-09-{14 + self.turn:02d}",
                        },
                    )
                ]
            )
        return await self.delegate.complete(messages, tools)


class _ContainedTraceFailure:
    def __enter__(self) -> None:
        try:
            raise RuntimeError("controlled tracing fixture failure")
        except RuntimeError:
            return

    def __exit__(self, *_: object) -> bool:
        return False


class ControlledFailingTraceManager:
    """Tracing adapter that raises internally and contains the expected failure."""

    enabled = False
    client = None

    def __init__(self) -> None:
        self.failure_count = 0

    def span(self, *_: Any, **__: Any) -> _ContainedTraceFailure:
        self.failure_count += 1
        return _ContainedTraceFailure()

    @staticmethod
    def end(*_: Any, **__: Any) -> None:
        return None
