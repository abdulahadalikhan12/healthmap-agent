"""MLflow 3 tracing on Databricks.

Set tracking URI to `databricks` so runs land in the workspace. If
`DATABRICKS_HOST`/`TOKEN` are set we also export them so MLflow picks them up.

Usage (in orchestrator):
    from backend.databricks.mlflow3 import configure, span

    configure()
    with span("01_parsed_query", inputs={"query": q}) as s:
        s.set_outputs(parsed.model_dump())
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from backend.config import settings


_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    import mlflow

    if settings.databricks_host:
        os.environ.setdefault("DATABRICKS_HOST", settings.databricks_host)
    if settings.databricks_token:
        os.environ.setdefault("DATABRICKS_TOKEN", settings.databricks_token)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(
        f"/Shared/{settings.mlflow_experiment_name}"
        if not settings.mlflow_experiment_name.startswith("/")
        else settings.mlflow_experiment_name
    )
    _configured = True


class _Span:
    """Small convenience wrapper so callers don't have to remember the MLflow
    tracing API. Works on MLflow 3+ (`mlflow.start_span`). Falls back to a
    plain context manager on older versions."""

    def __init__(self, name: str, inputs: dict[str, Any] | None):
        self.name = name
        self.inputs = inputs or {}
        self._cm = None
        self._span = None

    def __enter__(self):
        import mlflow

        start_span = getattr(mlflow, "start_span", None)
        if start_span is not None:
            self._cm = start_span(name=self.name, attributes=self.inputs)
            self._span = self._cm.__enter__()
        return self

    def set_outputs(self, outputs: Any) -> None:
        if self._span is not None and hasattr(self._span, "set_outputs"):
            self._span.set_outputs(outputs)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None and hasattr(self._span, "set_attribute"):
            self._span.set_attribute(key, value)

    def __exit__(self, exc_type, exc, tb):
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc, tb)
        return False


@contextmanager
def span(name: str, inputs: dict[str, Any] | None = None) -> Iterator[_Span]:
    configure()
    s = _Span(name, inputs)
    with s as live:
        yield live
