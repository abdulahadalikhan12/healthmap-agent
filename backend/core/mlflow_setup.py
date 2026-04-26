"""MLflow setup helpers.

Two modes:

* **Local** (`USE_DATABRICKS=0`): tracking URI from `settings.mlflow_tracking_uri`
  (defaults to `./mlruns`). Useful for dev runs.
* **Databricks MLflow 3** (`USE_DATABRICKS=1`): tracking URI set to
  `databricks`; experiment defaults to `/Shared/<experiment_name>`. Enables
  GenAI-style traces in the workspace's MLflow Experiments UI.

We log one MLflow run per `/query` call. The orchestrator wraps everything
in `mlflow.start_run(...)` and uses helpers here to log stages. When MLflow 3
is installed we also emit `start_span`-based traces for each stage so the
Tracing tab shows the tree directly.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow

from backend.config import settings


_initialized = False


def init() -> None:
    global _initialized
    if _initialized:
        return
    if settings.use_databricks:
        if settings.databricks_host:
            os.environ.setdefault("DATABRICKS_HOST", settings.databricks_host)
        if settings.databricks_token:
            os.environ.setdefault("DATABRICKS_TOKEN", settings.databricks_token)
        mlflow.set_tracking_uri("databricks")
        exp = settings.mlflow_experiment_name
        if not exp.startswith("/"):
            exp = f"/Shared/{exp}"
        mlflow.set_experiment(exp)
    else:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
    _initialized = True


@contextmanager
def stage_span(name: str, inputs: dict[str, Any] | None = None) -> Iterator[Any]:
    """Context manager that opens an MLflow 3 span when available (Tracing UI
    on Databricks). Becomes a no-op on older MLflow installs."""
    start_span = getattr(mlflow, "start_span", None)
    if start_span is None:
        yield None
        return
    try:
        with start_span(name=name, attributes=inputs or {}) as s:
            yield s
    except Exception:
        yield None


@contextmanager
def query_run(query: str) -> Iterator[None]:
    init()
    with mlflow.start_run(run_name=f"query: {query[:60]}"):
        mlflow.log_param("query", query)
        yield


def log_step(name: str, payload: Any) -> None:
    """Log a JSON-serialisable payload as an artifact under stage name."""
    try:
        text = json.dumps(payload, default=str, indent=2)
    except TypeError:
        text = str(payload)
    mlflow.log_text(text, f"steps/{name}.json")


def log_metric(name: str, value: float) -> None:
    try:
        mlflow.log_metric(name, float(value))
    except Exception:
        pass


def log_genai_style_trace(
    *,
    query: str,
    steps: list[str],
    result_summary: dict,
) -> None:
    """Structured trace for MLflow UI (span-like tree) + compatibility hooks.

    MLflow 2.14+ exposes experiment traces; we always persist a full JSON
    artifact so runs remain inspectable even on older tracking servers.
    """
    tree = {
        "type": "agent_run",
        "query": query[:500],
        "spans": [{"name": f"step_{i+1}", "output": s} for i, s in enumerate(steps)],
        "result": result_summary,
    }
    log_step("agent_traceability_tree", tree)
    # Per-step text files (readable in MLflow Artifacts, similar to span outputs)
    try:
        for i, s in enumerate(steps):
            mlflow.log_text(s, f"traces/span_{i+1:02d}.txt")
    except Exception:
        pass
