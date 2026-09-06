from __future__ import annotations

import re
from types import TracebackType
from typing import Any

SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|bearer|password|secret|token)\b"
    r"\s*(?:is\s+|[:=]\s*)?(?:bearer\s+)?([^\s,;\"}]+)"
)


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET_PATTERN.sub(r"\1=[REDACTED]", value)
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(s in k.lower() for s in ("key", "token", "secret", "authorization")) else sanitize(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


class TraceManager:
    def __init__(
        self,
        enabled: bool,
        project: str,
        api_key: str = "",
        endpoint: str = "https://api.smith.langchain.com",
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.client: Any = None
        if enabled and api_key:
            try:
                from langsmith import Client

                self.client = Client(api_key=api_key, api_url=endpoint)
            except Exception:  # noqa: BLE001 - invalid tracing config must not break the app.
                self.enabled = False

    def span(self, name: str, run_type: str, inputs: dict[str, Any], metadata: dict[str, Any] | None = None) -> SafeSpan:
        return SafeSpan(
            self.enabled,
            self.project,
            name,
            run_type,
            sanitize(inputs),
            sanitize(metadata or {}),
            self.client,
        )

    @staticmethod
    def end(run: Any, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        if run is None:
            return
        try:
            run.end(outputs=sanitize(outputs or {}), error=sanitize(error) if error else None)
        except Exception:  # noqa: BLE001
            return  # Trace delivery is intentionally best-effort.


class SafeSpan:
    def __init__(
        self,
        enabled: bool,
        project: str,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
        client: Any = None,
    ) -> None:
        self.enabled, self.project, self.name = enabled, project, name
        self.run_type, self.inputs, self.metadata = run_type, inputs, metadata
        self.client = client
        self.context: Any = None
        self.run: Any = None

    def __enter__(self) -> Any:
        if not self.enabled:
            return None
        try:
            from langsmith import trace
            self.context = trace(
                name=self.name,
                run_type=self.run_type,
                inputs=self.inputs,
                metadata=self.metadata,
                project_name=self.project,
                client=self.client,
            )
            self.run = self.context.__enter__()
            return self.run
        except Exception:  # noqa: BLE001 - tracing must never break application work.
            self.context = None
            return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc, traceback)
            except Exception:  # noqa: BLE001
                return False  # Trace delivery is intentionally best-effort.
        return False
