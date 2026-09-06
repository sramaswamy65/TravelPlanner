from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ToolResult(BaseModel):
    status: Literal["success", "error", "capability_unavailable"]
    data: dict[str, Any] | list[Any] | None = None
    error: dict[str, str] | None = None


class ToolCallRecord(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: ToolResult
    latency_ms: float = Field(ge=0)


class AgentResult(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    memories_retrieved: list[str] = Field(default_factory=list)
    memories_written: list[str] = Field(default_factory=list)
    step_count: int = Field(ge=0, le=8)
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    model: str
    prompt_version: str
    token_usage: dict[str, int] = Field(default_factory=dict)


class ModelToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ModelTurn(BaseModel):
    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    id: str
    memory: str
    category: str = "preference"
    user_id: str


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.@-]+$")
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    case_id: str | None = None
    scenario_type: str = "interactive"
    difficulty: str = "standard"
    dataset_version: str = "v2"
    expected_output: dict[str, Any] | None = None

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        return value.strip()
