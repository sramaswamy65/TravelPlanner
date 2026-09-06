from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Self

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError

from src.config import ROOT_DIR, Settings
from src.models import ToolResult


def mcp_environment(settings: Settings, base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(base) if base is not None else os.environ.copy()
    environment["APP_ENV"] = "test" if settings.deterministic else settings.app_env
    environment["LLM_MODE"] = settings.effective_llm_mode
    environment["MEMORY_MODE"] = settings.effective_memory_mode
    environment["TRAVEL_DATA_MODE"] = settings.effective_travel_data_mode
    environment["FAKE_MEMORY_DB"] = str(settings.fake_memory_db)
    environment["TRAVEL_FIXTURE_ID"] = settings.travel_fixture_id
    environment["WEATHER_HTTP_TIMEOUT_SECONDS"] = str(settings.weather_http_timeout_seconds)
    environment["WEATHER_HTTP_RETRIES"] = str(settings.weather_http_retries)
    return environment


class TravelMCPClient:
    """Official MCP stdio client. The agent never imports server tool handlers."""

    def __init__(self, settings: Settings, root_dir: Path = ROOT_DIR) -> None:
        self.settings = settings
        self.root_dir = root_dir
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> Self:
        self._stack = AsyncExitStack()
        environment = mcp_environment(self.settings)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server"],
            env=environment,
            cwd=str(self.root_dir),
        )
        reader, writer = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(reader, writer))
        with anyio.fail_after(self.settings.mcp_timeout_seconds):
            await self.session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack:
            await self._stack.aclose()

    def _require_session(self) -> ClientSession:
        if self.session is None:
            raise RuntimeError("MCP client is not connected")
        return self.session

    async def list_tools(self) -> list[dict[str, Any]]:
        with anyio.fail_after(self.settings.mcp_timeout_seconds):
            result = await self._require_session().list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            with anyio.fail_after(self.settings.mcp_timeout_seconds):
                result = await self._require_session().call_tool(name, arguments)
        except TimeoutError:
            return ToolResult(
                status="error", error={"code": "timeout", "message": "MCP tool timed out"}
            )
        if result.isError:
            return ToolResult(
                status="error",
                error={"code": "protocol_error", "message": "MCP tool returned an error"},
            )
        payload: Any = getattr(result, "structuredContent", None)
        if payload is None and result.content:
            text = getattr(result.content[0], "text", "")
            try:
                payload = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                return ToolResult(
                    status="error",
                    error={"code": "malformed_response", "message": "MCP returned malformed JSON"},
                )
        try:
            return ToolResult.model_validate(payload)
        except ValidationError:
            return ToolResult(
                status="error",
                error={
                    "code": "malformed_response",
                    "message": "MCP response did not match the tool result schema",
                },
            )
