"""Agent 2 — Retrieval.

Two back-ends behind the same `retrieve(...)` signature:

* **Local / dev** (`USE_DATABRICKS=0`): hybrid FAISS top-K + structured filters
  on the on-disk meta parquet.
* **Databricks** (`USE_DATABRICKS=1`): Mosaic AI Vector Search (Delta Sync
  index) over the facilities Delta table, with VS filters for state/district
  /rural.

Both paths return the same DataFrame columns so the orchestrator is
agnostic.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from backend.config import settings
from backend.core.schemas import ParsedQuery


# --------------------------------------------------------------------------- #
# Databricks (Mosaic AI Vector Search) path
# --------------------------------------------------------------------------- #
def _vs_filters(parsed: ParsedQuery) -> dict[str, Any]:
    """Convert parsed query into Vector Search filter map."""
    f: dict[str, Any] = {}
    if parsed.state:
        # VS filter syntax supports case-sensitive exact match; our Delta
        # stores `state` title-cased so we normalize lightly.
        f["state"] = parsed.state.strip().title()
    if parsed.district:
        f["district"] = parsed.district.strip().title()
    if parsed.rural is True:
        f["rural"] = True
    return f


def _retrieve_dbx(parsed: ParsedQuery, query_text: str, *, top_k: int) -> pd.DataFrame:
    from backend.databricks.vector_search import similarity_search

    filters = _vs_filters(parsed)
    df = similarity_search(query_text, k=top_k * 4, filters=filters)

    # If strict filter returned nothing, relax: try without rural, then without state.
    if df.empty and "rural" in filters:
        relaxed = dict(filters)
        relaxed.pop("rural", None)
        df = similarity_search(query_text, k=top_k * 4, filters=relaxed)
    if df.empty and "state" in filters:
        df = similarity_search(query_text, k=top_k * 4)

    if df.empty:
        return df

    if "_vs_score" in df.columns:
        df = df.rename(columns={"_vs_score": "_score"})
    return df.head(top_k).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Local FAISS path (unchanged behavior)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_index() -> tuple[Any, pd.DataFrame]:
    import faiss  # type: ignore

    index = faiss.read_index(str(settings.index_path))
    meta = pd.read_parquet(settings.index_meta_path)
    return index, meta


def _structured_filter(meta: pd.DataFrame, parsed: ParsedQuery) -> pd.Series:
    mask = pd.Series(True, index=meta.index)
    if parsed.state and "state" in meta.columns:
        mask &= meta["state"].fillna("").str.lower().str.contains(parsed.state.lower())
    if parsed.district and "district" in meta.columns:
        mask &= meta["district"].fillna("").str.lower().str.contains(parsed.district.lower())
    if parsed.rural is True and "rural" in meta.columns:
        mask &= meta["rural"].fillna(False).astype(bool)
    return mask


def _retrieve_faiss(parsed: ParsedQuery, query_text: str, *, top_k: int) -> pd.DataFrame:
    import faiss  # type: ignore
    import numpy as np

    from backend.core.llm import embed

    index, meta = _load_index()

    qvec = np.array(embed([query_text])[0], dtype="float32")[None, :]
    faiss.normalize_L2(qvec)
    scores, idx = index.search(qvec, top_k * 8)
    candidates = meta.iloc[idx[0]].copy()
    candidates["_score"] = scores[0]

    mask = _structured_filter(candidates, parsed)
    filtered = candidates[mask]

    if parsed.state and len(filtered) > 0:
        return filtered.head(top_k).reset_index(drop=True)
    if len(filtered) < top_k:
        filtered = candidates
    return filtered.head(top_k).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def retrieve(parsed: ParsedQuery, query_text: str, *, top_k: int = 10) -> pd.DataFrame:
    if settings.use_databricks:
        return _retrieve_dbx(parsed, query_text, top_k=top_k)
    return _retrieve_faiss(parsed, query_text, top_k=top_k)
