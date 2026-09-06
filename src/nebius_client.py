from __future__ import annotations

import json
import re
import uuid
from typing import Any, Protocol

from openai import AsyncOpenAI, BadRequestError

from src.config import Settings
from src.intents import resolve_intents
from src.memory_policy import should_search_memory
from src.models import ModelToolCall, ModelTurn


class ModelClient(Protocol):
    model: str

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...


class NebiusClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.nebius_api_key or not settings.nebius_model:
            raise ValueError("NEBIUS_API_KEY and NEBIUS_MODEL are required in live mode")
        self.model = settings.nebius_model
        self.client = AsyncOpenAI(
            api_key=settings.nebius_api_key, base_url=str(settings.nebius_base_url)
        )

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
        except BadRequestError as exc:
            message = str(exc).lower()
            if "tool" in message or "function" in message:
                raise RuntimeError(
                    f"Configured NEBIUS_MODEL '{self.model}' does not support the required tool/function calling interface"
                ) from exc
            raise
        choice = response.choices[0].message
        calls = []
        for call in choice.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {"__invalid_json__": call.function.arguments}
            calls.append(ModelToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = response.usage
        return ModelTurn(
            content=choice.content,
            tool_calls=calls,
            token_usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            },
        )


class DeterministicModelClient:
    """Small scripted model substitute for offline demos/tests; it never answers external facts."""

    model = "deterministic-fixture-model"

    def __init__(self, default_weather_date: str | None = None) -> None:
        self.default_weather_date = default_weather_date

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), ""
        )
        user_index = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=0,
        )
        resolved = resolve_intents(user, messages[:user_index])
        called = [m.get("name") for m in messages if m.get("role") == "tool"]
        lower = user.lower()
        city_match = re.search(r"\b(?:in|for)\s+([A-Za-z ,'-]+?)(?:\s+on\s+|[.?!]|$)", user)
        city = city_match.group(1).strip() if city_match else "Chicago"
        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", user)
        date = date_match.group(0) if date_match else self.default_weather_date
        current_weather = "current weather" in lower or "weather right now" in lower
        category = next(
            (c for c in ("museum", "outdoor", "food", "history", "art") if c in lower), "any"
        )
        call: tuple[str, dict[str, Any]] | None = None
        observations = [m.get("content", "") for m in messages if m.get("role") == "tool"]

        def observation(name: str) -> dict[str, Any] | None:
            for item in reversed(messages):
                if item.get("role") == "tool" and item.get("name") == name:
                    try:
                        payload = json.loads(str(item.get("content", "{}")))
                        return payload if isinstance(payload, dict) else None
                    except json.JSONDecodeError:
                        return None
            return None

        save_match = re.search(
            r"\bsave (?:this|the) itinerary(?: as ([A-Za-z0-9_.-]+\.md))?", lower
        )
        memory_listing = any(
            phrase in lower
            for phrase in (
                "what do you remember",
                "remember about me",
                "list my memories",
                "my memories",
            )
        )
        if "save_dietary_preference" in resolved.names and "save_traveler_memory" not in called:
            return ModelTurn(
                tool_calls=[
                    ModelToolCall(
                        id=str(uuid.uuid4()),
                        name="save_traveler_memory",
                        arguments={
                            "user_id": "__USER_ID__",
                            "memory": "Is vegetarian",
                            "category": "dietary_preference",
                        },
                    )
                ]
            )
        if "search_restaurants" in resolved.names:
            if resolved.city is None:
                saved = observation("save_traveler_memory")
                prefix = (
                    "I've remembered that you are vegetarian. "
                    if saved and saved.get("status") == "success"
                    else ""
                )
                return ModelTurn(content=f"{prefix}Which city should I search for restaurants in?")
            if "search_restaurants" not in called:
                return ModelTurn(
                    tool_calls=[
                        ModelToolCall(
                            id=str(uuid.uuid4()),
                            name="search_restaurants",
                            arguments={
                                "city": resolved.city,
                                "dietary_preference": (
                                    "vegetarian" if "vegetarian" in lower else None
                                ),
                                "limit": 5,
                            },
                        )
                    ]
                )
            saved = observation("save_traveler_memory")
            restaurants = observation("search_restaurants") or {}
            outcomes: list[str] = []
            if "save_dietary_preference" in resolved.names:
                if saved and saved.get("status") == "success":
                    outcomes.append("I've remembered that you are vegetarian.")
                else:
                    detail = ((saved or {}).get("error") or {}).get(
                        "message", "The preference could not be saved"
                    )
                    outcomes.append(f"I couldn't save the vegetarian preference: {detail}.")
            context = "Based on our conversation, " if resolved.inherited_city else ""
            if restaurants.get("status") == "success":
                matches = (restaurants.get("data") or {}).get("restaurants", [])
                if matches:
                    names = ", ".join(str(item.get("name")) for item in matches if item.get("name"))
                    outcomes.append(
                        f"{context}these matching restaurants are available in "
                        f"{resolved.city}: {names}."
                    )
                else:
                    outcomes.append(
                        f"{context}a valid restaurant search found no matching fixture "
                        f"results in {resolved.city}."
                    )
            else:
                detail = (restaurants.get("error") or {}).get("message", "Restaurant search failed")
                outcomes.append(
                    f"{context}you appear to be looking for restaurants in {resolved.city}. "
                    f"{detail}"
                )
            return ModelTurn(content=" ".join(outcomes))
        if "save_dietary_preference" in resolved.names and "save_traveler_memory" in called:
            saved = observation("save_traveler_memory") or {}
            if saved.get("status") == "success":
                return ModelTurn(content="I've remembered that you are vegetarian.")
            detail = (saved.get("error") or {}).get("message", "The preference could not be saved")
            return ModelTurn(content=f"I couldn't save the vegetarian preference: {detail}.")
        if any(
            '"status":"error"' in str(item).replace(" ", "")
            or '"status":"capability_unavailable"' in str(item).replace(" ", "")
            for item in observations
        ):
            detail = "A required tool returned an error."
            for item in reversed(observations):
                try:
                    payload = json.loads(str(item))
                    message = (payload.get("error") or {}).get("message")
                    if message:
                        detail = str(message)
                        break
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue
            return ModelTurn(
                content=(
                    f"I could not complete the plan. {detail} No replacement facts were invented."
                )
            )
        if save_match and "save_itinerary" not in called:
            prior_plan = next(
                (
                    str(m.get("content", ""))
                    for m in reversed(messages[:-1])
                    if m.get("role") == "assistant"
                ),
                "Approved itinerary",
            )
            call = (
                "save_itinerary",
                {
                    "user_id": "__USER_ID__",
                    "filename": save_match.group(1) or "travel-day.md",
                    "content": prior_plan,
                    "approved": True,
                },
            )
        elif (memory_listing or "forget" in lower) and "list_traveler_memories" not in called:
            call = ("list_traveler_memories", {"user_id": "__USER_ID__"})
        elif "remember" in lower and not memory_listing and "save_traveler_memory" not in called:
            call = (
                "save_traveler_memory",
                {"user_id": "__USER_ID__", "memory": user, "category": "preference"},
            )
        elif "forget" in lower and "delete_traveler_memory" not in called:
            for message in reversed(messages):
                if (
                    message.get("role") == "tool"
                    and message.get("name") == "list_traveler_memories"
                ):
                    try:
                        payload = json.loads(message.get("content", "{}"))
                        memories = (payload.get("data") or {}).get("memories", [])
                        query = lower.split("forget", 1)[-1].strip(" .")
                        match = next(
                            (
                                m
                                for m in memories
                                if query and query in str(m.get("memory", "")).lower()
                            ),
                            None,
                        )
                        if match:
                            call = (
                                "delete_traveler_memory",
                                {"user_id": "__USER_ID__", "memory_id": str(match["id"])},
                            )
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass
                    break
        elif (
            should_search_memory(user)
            and any(x in lower for x in ("plan", "itinerary", "things to do"))
            and "search_traveler_memory" not in called
        ):
            call = (
                "search_traveler_memory",
                {"user_id": "__USER_ID__", "query": "travel preferences"},
            )
        elif (
            any(x in lower for x in ("plan", "itinerary", "weather"))
            and "get_weather" not in called
        ):
            if date:
                call = ("get_weather", {"city": city, "date": date})
            elif current_weather:
                call = ("get_weather", {"city": city, "current": True})
            else:
                return ModelTurn(content="What date should I use for the weather forecast?")
        elif (
            any(
                x in lower
                for x in ("plan", "itinerary", "attraction", "things to do", "find", "museum")
            )
            and "find_attractions" not in called
        ):
            call = ("find_attractions", {"city": city, "category": category})
        if call:
            return ModelTurn(
                tool_calls=[ModelToolCall(id=str(uuid.uuid4()), name=call[0], arguments=call[1])]
            )
        subject = f"{category} " if category != "any" else ""
        grounding = (
            "live-weather-grounded"
            if any('"source":"Open-Meteo"' in str(item).replace(" ", "") for item in observations)
            else "fixture-grounded"
        )
        return ModelTurn(
            content=(
                f"Here is a {grounding} {subject}day plan based on the weather, attractions, "
                "and relevant preferences returned by the local tools. Review opening details "
                "before departure."
            )
        )


def create_model_client(settings: Settings) -> ModelClient:
    if settings.effective_llm_mode == "live":
        return NebiusClient(settings)
    return DeterministicModelClient()
