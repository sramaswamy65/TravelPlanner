from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from pydantic import ValidationError

from src.config import Settings
from src.intents import resolve_intents
from src.mcp_client import TravelMCPClient
from src.memory_policy import (
    evaluate_memory_write,
    is_forget_request,
    is_safe_memory_evidence,
    should_search_memory,
)
from src.models import AgentRequest, AgentResult, ToolCallRecord, ToolResult
from src.nebius_client import ModelClient
from src.prompts import AGENT_VERSION, PROMPT_VERSION, SYSTEM_PROMPT
from src.tracing import TraceManager, sanitize


class TravelPlannerAgent:
    def __init__(
        self,
        settings: Settings,
        model_client: ModelClient,
        mcp_factory: Callable[[], AbstractAsyncContextManager[TravelMCPClient]] | None = None,
        tracer: TraceManager | None = None,
    ) -> None:
        self.settings = settings
        self.model = model_client
        self.mcp_factory = mcp_factory or (lambda: TravelMCPClient(settings))
        self.tracer = tracer or TraceManager(
            settings.langsmith_tracing and bool(settings.langsmith_api_key),
            settings.langsmith_project,
            api_key=settings.langsmith_api_key,
            endpoint=str(settings.langsmith_endpoint),
        )

    def _authorize(self, request: AgentRequest, name: str, args: dict[str, Any]) -> str | None:
        lower = request.message.lower()
        if name in {"search_traveler_memory", "list_traveler_memories"} and not (
            should_search_memory(request.message) or "remember" in lower or is_forget_request(request.message)
        ):
            return "Memory access was suppressed because preferences are not relevant"
        if name == "save_traveler_memory":
            decision = evaluate_memory_write(request.message)
            if not decision.eligible:
                return decision.reason
        if name == "delete_traveler_memory" and not is_forget_request(request.message):
            return "Explicit forget/delete authorization is required"
        if name == "save_itinerary":
            approved = any(phrase in lower for phrase in ("approve saving", "save this itinerary", "save the itinerary"))
            has_prior_itinerary = any(
                message.get("role") == "assistant" and message.get("content")
                for message in request.conversation
            )
            if not approved or not args.get("approved") or not has_prior_itinerary:
                return "Explicit approval is required before saving an itinerary"
        return None

    async def run(self, request: AgentRequest) -> AgentResult:
        started = time.perf_counter()
        metadata = {
            "case_id": request.case_id,
            "user_id": request.user_id,
            "scenario_type": request.scenario_type,
            "difficulty": request.difficulty,
            "dataset_version": request.dataset_version,
            "run_name": "travel_planner_run",
            "agent_version": AGENT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": self.model.model,
            "environment": self.settings.app_env,
            "expected_output": request.expected_output,
            "predicted_output": None,
            "tool_sequence": [],
            "memory_ids_retrieved": [],
            "memory_ids_written": [],
            "correctness": None,
            "errors": [],
            "latency": None,
            "token_usage": {},
        }
        records: list[ToolCallRecord] = []
        errors: list[str] = []
        retrieved: list[str] = []
        retrieved_preferences: list[str] = []
        written: list[str] = []
        usage: dict[str, int] = {}
        answer = ""
        steps = 0
        seen: set[str] = set()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *request.conversation,
            {"role": "user", "content": request.message},
        ]
        with self.tracer.span("travel_planner_run", "chain", {"message": request.message}, metadata) as parent:
            try:
                async with self.mcp_factory() as mcp:
                    tools = await mcp.list_tools()
                    for steps in range(1, self.settings.max_agent_steps + 1):
                        with self.tracer.span("nebius_model_call", "llm", {"messages": messages, "tools": [t["function"]["name"] for t in tools]}, metadata) as span:
                            turn = await self.model.complete(messages, tools)
                            self.tracer.end(span, turn.model_dump())
                        for key, value in turn.token_usage.items():
                            usage[key] = usage.get(key, 0) + value
                        if not turn.tool_calls:
                            answer = (turn.content or "").strip() or "The model returned no answer."
                            break
                        messages.append({
                            "role": "assistant",
                            "content": turn.content,
                            "tool_calls": [
                                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                for c in turn.tool_calls
                            ],
                        })
                        for call in turn.tool_calls:
                            args = dict(call.arguments)
                            if "user_id" in args:
                                args["user_id"] = request.user_id
                            if call.name == "save_traveler_memory":
                                resolved = resolve_intents(request.message, request.conversation)
                                if resolved.memory:
                                    args["memory"] = resolved.memory
                                    args["category"] = resolved.memory_category
                                else:
                                    args["memory"] = request.message
                            if call.name == "save_itinerary":
                                prior_itinerary = next(
                                    (
                                        str(message.get("content", ""))
                                        for message in reversed(request.conversation)
                                        if message.get("role") == "assistant" and message.get("content")
                                    ),
                                    "",
                                )
                                if prior_itinerary:
                                    args["content"] = prior_itinerary
                            signature = json.dumps([call.name, args], sort_keys=True, separators=(",", ":"))
                            if signature in seen:
                                result = ToolResult(status="error", error={"code": "repeated_call", "message": "Repeated identical tool call prevented"})
                            elif "__invalid_json__" in args:
                                result = ToolResult(status="error", error={"code": "invalid_arguments", "message": "Tool arguments were not valid JSON"})
                            else:
                                seen.add(signature)
                                denial = self._authorize(request, call.name, args)
                                if denial:
                                    result = ToolResult(status="error", error={"code": "policy_denied", "message": denial})
                                else:
                                    tool_started = time.perf_counter()
                                    with self.tracer.span(f"mcp.{call.name}", "tool", {"arguments": sanitize(args)}, metadata) as tool_span:
                                        memory_operation = call.name in {
                                            "search_traveler_memory", "save_traveler_memory",
                                            "list_traveler_memories", "delete_traveler_memory",
                                        }
                                        memory_type = "retriever" if call.name == "search_traveler_memory" else "tool"
                                        memory_span = self.tracer.span(
                                            f"memory.{call.name}", memory_type, {"user_id": request.user_id}, metadata
                                        ) if memory_operation else None
                                        try:
                                            if memory_span:
                                                with memory_span as child_span:
                                                    result = await mcp.call_tool(call.name, args)
                                                    self.tracer.end(child_span, result.model_dump())
                                            else:
                                                result = await mcp.call_tool(call.name, args)
                                        except Exception:  # noqa: BLE001 - external MCP failures become structured observations.
                                            result = ToolResult(status="error", error={"code": "mcp_failure", "message": "MCP call failed"})
                                        self.tracer.end(tool_span, result.model_dump())
                                    latency = (time.perf_counter() - tool_started) * 1000
                                    record = ToolCallRecord(tool_call_id=call.id, name=call.name, arguments=sanitize(args), result=result, latency_ms=latency)
                                    records.append(record)
                                    if result.status == "success" and isinstance(result.data, dict):
                                        memories = result.data.get("memories", [])
                                        if call.name in {"search_traveler_memory", "list_traveler_memories"}:
                                            retrieved.extend(str(m.get("id")) for m in memories if isinstance(m, dict) and m.get("id"))
                                            retrieved_preferences.extend(
                                                str(m.get("memory")) for m in memories
                                                if isinstance(m, dict)
                                                and m.get("memory")
                                                and is_safe_memory_evidence(str(m["memory"]))
                                            )
                                        memory = result.data.get("memory")
                                        if call.name == "save_traveler_memory" and isinstance(memory, dict) and memory.get("id"):
                                            written.append(str(memory["id"]))
                            if signature in seen and not any(r.tool_call_id == call.id for r in records):
                                records.append(ToolCallRecord(tool_call_id=call.id, name=call.name, arguments=sanitize(args), result=result, latency_ms=0))
                            if result.status != "success":
                                errors.append(result.error["message"] if result.error else "Unknown tool error")
                            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result.model_dump_json()})
                    else:
                        answer = "I stopped after reaching the eight-step safety limit. The request was not completed."
                        errors.append("Maximum agent step limit reached")
            except (ValidationError, RuntimeError, OSError) as exc:
                answer = "The planner could not complete the request because a required service failed."
                errors.append(str(sanitize(str(exc))))
            latency = (time.perf_counter() - started) * 1000
            if retrieved_preferences and "saved preference" not in answer.lower():
                disclosed = "; ".join(dict.fromkeys(str(sanitize(item)) for item in retrieved_preferences))
                answer = f"{answer}\n\nSaved preferences consulted: {disclosed}."
            with self.tracer.span("final_response_construction", "chain", {"draft": answer, "errors": errors}, metadata) as final_span:
                result = AgentResult(
                    answer=answer,
                    tool_calls=records,
                    memories_retrieved=list(dict.fromkeys(retrieved)),
                    memories_written=list(dict.fromkeys(written)),
                    step_count=steps,
                    errors=errors,
                    latency_ms=latency,
                    model=self.model.model,
                    prompt_version=PROMPT_VERSION,
                    token_usage=usage,
                )
                self.tracer.end(final_span, {"answer": result.answer})
            self.tracer.end(parent, {
                "predicted_output": result.answer,
                "tool_sequence": [record.name for record in records],
                "memory_ids_retrieved": result.memories_retrieved,
                "memory_ids_written": result.memories_written,
                "errors": result.errors,
                "latency": result.latency_ms,
                "token_usage": usage,
            })
            return result
