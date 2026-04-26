"""Mosaic AI Vector Search wrapper used by `retrieval_agent` when
`USE_DATABRICKS=1`. Returns a pandas DataFrame shaped like the FAISS meta
frame so the rest of the pipeline stays unchanged.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.config import settings
from backend.databricks.client import vector_search_client


# Columns we try to pull back from the index (all optional; Vector Search
# returns whatever exists in the source Delta table).
_DEFAULT_COLUMNS: list[str] = [
    "facility_id",
    "name",
    "state",
    "district",
    "pin",
    "rural",
    "latitude",
    "longitude",
    "facility_type",
    "phone",
    "email",
    "notes",
]


def similarity_search(
    query_text: str,
    *,
    k: int = 20,
    filters: dict[str, Any] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Query the Vector Search Delta Sync index using `query_text` (server-side
    embedding via the index's embedding_source column). Filters use Vector
    Search filter syntax, e.g. {"state": "Bihar"} or {"rural": True}.
    """
    client = vector_search_client()
    index = client.get_index(
        endpoint_name=settings.dbx_vs_endpoint,
        index_name=settings.dbx_full_vs_index,
    )
    res = index.similarity_search(
        query_text=query_text,
        columns=list(columns or _DEFAULT_COLUMNS),
        num_results=int(k),
        filters=filters or None,
    )
    data = (res or {}).get("result", {}).get("data_array") or []
    cols_meta = (res or {}).get("manifest", {}).get("columns") or []
    col_names = [c.get("name") for c in cols_meta] or (columns or _DEFAULT_COLUMNS)
    df = pd.DataFrame(data, columns=col_names)
    # Vector Search appends a score column named `score` after the requested columns.
    if "score" in df.columns:
        df = df.rename(columns={"score": "_vs_score"})
    return df
