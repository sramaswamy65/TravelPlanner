from __future__ import annotations

import asyncio

import streamlit as st

from src.agent import TravelPlannerAgent
from src.config import Settings
from src.intents import resolve_intents
from src.mcp_client import TravelMCPClient
from src.models import AgentRequest
from src.nebius_client import create_model_client
from src.runtime import runtime_context, settings_for_context
from src.tracing import sanitize

st.set_page_config(page_title="Travel Day Planner", page_icon=":material/travel_explore:", layout="wide")


def run_async(awaitable):
    return asyncio.run(awaitable)


def init_state(base: Settings) -> None:
    defaults = {
        "messages": [],
        "activities": [],
        "delete_pending": None,
        "pending_request": None,
        "memories": None,
        "memory_owner": None,
        "runtime_generation": 0,
        "runtime_reconnect_pending": False,
        "llm_mode": "Live" if base.effective_llm_mode == "live" else "Deterministic",
        "memory_mode": "Live" if base.effective_memory_mode == "live" else "Fixture",
        "travel_data_mode": "Live" if base.effective_travel_data_mode == "live" else "Fixture",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def mode_changed(label: str, key: str) -> None:
    selected = st.session_state[key]
    st.session_state.runtime_generation += 1
    st.session_state.runtime_reconnect_pending = True
    st.session_state.delete_pending = None
    st.session_state.pending_request = None
    st.session_state.memories = None
    st.session_state.memory_owner = None
    st.session_state.messages.append(
        {"role": "assistant", "content": f"{label} changed to {selected}."}
    )


def operation_status(activity: dict) -> str:
    calls = {call["name"]: call["result"]["status"] for call in activity["tool_calls"]}
    save = calls.get("save_traveler_memory")
    restaurants = calls.get("search_restaurants")
    if save == "success" and restaurants == "success":
        return "Restaurant search complete"
    if save == "success" and restaurants:
        return "Completed with limitations"
    if save and save != "success" and restaurants == "success":
        return "Completed with limitations"
    if save and save != "success":
        return "Unable to save preference"
    if save == "success":
        return "Preference saved"
    if restaurants == "success":
        return "Restaurant search complete"
    if restaurants:
        return "Completed with limitations"
    return "Itinerary complete"


async def call_memory_tool(settings: Settings, name: str, arguments: dict):
    async with TravelMCPClient(settings) as client:
        return await client.call_tool(name, arguments)


base_settings = Settings()
init_state(base_settings)

st.title("Travel Day Planner")
st.caption("One agent using Nebius inference, local MCP tools, Mem0 traveler preferences, and LangSmith traces.")

with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value="demo-traveler", max_chars=80)
    st.subheader("Operating modes")
    st.segmented_control(
        "LLM",
        ["Deterministic", "Live"],
        key="llm_mode",
        on_change=mode_changed,
        args=("LLM mode", "llm_mode"),
    )
    st.segmented_control(
        "Memory",
        ["Fixture", "Live"],
        key="memory_mode",
        on_change=mode_changed,
        args=("Memory mode", "memory_mode"),
    )
    st.segmented_control(
        "Weather",
        ["Fixture", "Live"],
        key="travel_data_mode",
        on_change=mode_changed,
        args=("Travel data mode", "travel_data_mode"),
    )
    st.caption("Attractions\n\nLimited fixture data")
    context = runtime_context(
        user_id,
        st.session_state.llm_mode,
        st.session_state.memory_mode,
        st.session_state.travel_data_mode,
        st.session_state.runtime_generation,
    )
    settings = settings_for_context(base_settings, context)
    if context.travel_data_mode == "live":
        st.info("Weather: Live; Attractions: Limited fixture data.")
    st.caption(f"Weather: {st.session_state.travel_data_mode}")
    st.caption("Attractions: Fixture")
    st.caption(
        "Restaurants: Fixture"
        if context.travel_data_mode == "fixture"
        else "Restaurants: Live/Unavailable"
    )
    st.caption(f"Memory: {st.session_state.memory_mode}")
    for warning in settings.llm_readiness_errors() + settings.memory_readiness_errors():
        st.warning(warning)
    st.metric("Nebius model", settings.nebius_model or "Not configured")
    tracing_on = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    st.write("LangSmith tracing", "On" if tracing_on else "Off")
    memory_backend = (
        "Fake memory backend"
        if context.memory_mode == "fake"
        else ("Configured" if settings.mem0_api_key else "Not configured")
    )
    st.write("Mem0", memory_backend)
    if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.messages = []
        st.session_state.activities = []
        st.rerun()

chat_tab, memories_tab, about_tab = st.tabs(["Chat", "Memories", "About"])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input(
        "Plan a day, manage a preference, or ask what I remember", submit_mode="disable"
    )
    if prompt:
        request_context = context
        request_settings = settings_for_context(base_settings, request_context)
        conversation = list(st.session_state.messages[-12:])
        resolved = resolve_intents(prompt, conversation)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                readiness = request_settings.llm_readiness_errors()
                if readiness:
                    raise ValueError("; ".join(readiness))
                model = create_model_client(request_settings)
                agent = TravelPlannerAgent(request_settings, model)
                with st.status("Planning with the agent and local MCP tools...", expanded=True) as status:
                    result = run_async(agent.run(AgentRequest(message=prompt, user_id=user_id, conversation=conversation)))
                    activity = result.model_dump()
                    status.update(label=operation_status(activity), state="complete")
                    for call in result.tool_calls:
                        st.write(f"Called {call.name}: {call.result.status}")
                st.markdown(result.answer)
                activity["resolved_intents"] = list(resolved.names)
                activity["inherited_city"] = resolved.city if resolved.inherited_city else None
                activity["intent_outcomes"] = [
                    {
                        "intent": name,
                        "status": next(
                            (
                                call["result"]["status"]
                                for call in activity["tool_calls"]
                                if call["name"]
                                == (
                                    "save_traveler_memory"
                                    if name == "save_dietary_preference"
                                    else name
                                )
                            ),
                            "clarification_required",
                        ),
                    }
                    for name in resolved.names
                ]
                activity["runtime"] = {
                    "llm_backend": request_context.llm_backend,
                    "memory_backend": request_context.memory_backend,
                    "travel_data_backend": request_context.travel_backend,
                    "generation": request_context.generation,
                    "client_reconnected": bool(st.session_state.runtime_reconnect_pending),
                }
                st.session_state.activities.append(activity)
                st.session_state.runtime_reconnect_pending = False
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except Exception as exc:  # noqa: BLE001 - UI boundary must contain SDK failures.
                message = f"Planner startup failed: {sanitize(str(exc))}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
    if st.session_state.activities:
        with st.expander("Agent activity"):
            activity = st.session_state.activities[-1]
            runtime = activity.get("runtime", {})
            st.write("Resolved intents:", activity.get("resolved_intents", []))
            if activity.get("inherited_city"):
                st.write("Inherited city context:", activity["inherited_city"])
            st.write(
                f"LLM: {runtime.get('llm_backend', 'Unknown')} | "
                f"Memory: {runtime.get('memory_backend', 'Unknown')} | "
                f"Travel data: {runtime.get('travel_data_backend', 'Unknown')}"
            )
            st.write(
                f"Runtime generation: {runtime.get('generation', 0)} | "
                f"Client reconnected: {'Yes' if runtime.get('client_reconnected') else 'No'}"
            )
            st.write(f"Steps: {activity['step_count']} | Latency: {activity['latency_ms']:.1f} ms")
            st.caption("Each entry is a tool the agent selected and called through the local MCP server.")
            for call in activity["tool_calls"]:
                st.markdown(f"**{call['name']}**")
                data = call["result"].get("data")
                source = data.get("source") if isinstance(data, dict) else None
                st.caption(f"Source: {source or 'No source returned (error)'}")
                st.json({"arguments": call["arguments"], "result": call["result"]}, expanded=False)
            if activity["errors"]:
                st.warning("\n".join(activity["errors"]))
            st.write("Final intent outcomes:", activity.get("intent_outcomes", []))

with memories_tab:
    st.subheader("Traveler memories")
    st.caption("Memory operations are sent through the local MCP server and always scoped to the sidebar user ID.")
    if st.button("Refresh memories", icon=":material/refresh:"):
        try:
            response = run_async(call_memory_tool(settings, "list_traveler_memories", {"user_id": user_id}))
            st.session_state.memories = response.data.get("memories", []) if response.status == "success" and isinstance(response.data, dict) else []
            st.session_state.memory_owner = user_id
        except Exception:  # noqa: BLE001 - UI boundary returns a secret-safe message.
            st.session_state.memories = None
            st.error("The memory service is unavailable. No credentials or memory content were exposed.")
    if context.memory_mode == "fake":
        st.info("Fake memory mode uses a local user-scoped SQLite test store.")
    try:
        if context.memory_mode == "live" and not settings.mem0_api_key:
            st.warning("Set MEM0_API_KEY to list persistent memories.")
        elif st.session_state.memories is not None and st.session_state.memory_owner == user_id:
            memories = st.session_state.memories
            if not memories:
                st.write("No memories saved for this user.")
            for memory in memories:
                cols = st.columns([5, 1])
                cols[0].write(memory.get("memory", ""))
                if cols[1].button("Delete", key=f"delete_{memory.get('id')}"):
                    st.session_state.delete_pending = memory.get("id")
                if st.session_state.delete_pending == memory.get("id"):
                    st.warning("Delete this memory? This action requires confirmation.")
                    if st.button("Confirm delete", key=f"confirm_{memory.get('id')}", type="primary"):
                        run_async(call_memory_tool(settings, "delete_traveler_memory", {"user_id": user_id, "memory_id": memory.get("id")}))
                        st.session_state.delete_pending = None
                        st.session_state.memories = None
                        st.session_state.memory_owner = None
                        st.rerun()
        else:
            st.write("Select Refresh memories to load memories for this user.")
    except Exception:  # noqa: BLE001 - UI boundary returns a secret-safe message.
        st.error("The memory service is unavailable. No credentials or memory content were exposed.")

with about_tab:
    st.subheader("Architecture")
    st.write("A single reason-act-observe agent discovers and invokes eight tools through one local MCP stdio server. Weather uses fixtures or Open-Meteo according to TRAVEL_DATA_MODE. Attractions always use the limited local fixture dataset. Memory and LLM operation are controlled independently, and LangSmith tracing is optional and failure-tolerant.")
    st.warning("Analysis and planning only. Confirm hours, accessibility, transport, and weather with authoritative sources before travel.")
    st.info("Saving is two-step: first generate a plan, then send a new message such as 'Save this itinerary as chicago-day.md'. The second message is the explicit approval.")
