"""Thin accessors for Databricks SDK + Vector Search + Model Serving clients.

Each helper is cached so we don't re-authenticate on every request.
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import settings


@lru_cache(maxsize=1)
def workspace_client():
    """Databricks WorkspaceClient. Uses DATABRICKS_HOST / DATABRICKS_TOKEN env
    vars (which we also expose via `settings`)."""
    from databricks.sdk import WorkspaceClient

    kwargs: dict[str, str] = {}
    if settings.databricks_host:
        kwargs["host"] = settings.databricks_host
    if settings.databricks_token:
        kwargs["token"] = settings.databricks_token
    return WorkspaceClient(**kwargs) if kwargs else WorkspaceClient()


@lru_cache(maxsize=1)
def vector_search_client():
    """Mosaic AI Vector Search client."""
    from databricks.vector_search.client import VectorSearchClient

    return VectorSearchClient(
        workspace_url=settings.databricks_host or None,
        personal_access_token=settings.databricks_token or None,
        disable_notice=True,
    )


def serving_client():
    """Serving endpoints client (chat / embeddings via Model Serving)."""
    return workspace_client().serving_endpoints
