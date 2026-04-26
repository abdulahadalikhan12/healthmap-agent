"""Agent 3 — Capability Extraction.

Used both:
- Offline (batch) by `pipeline.batch_extract` over the full dataset.
- Online as a fallback if a candidate row hasn't been pre-extracted yet.

We persist results to `data/extracted/capabilities.parquet` keyed by
`facility_id`, so the runtime path is just a lookup most of the time.
Live-extracted rows are also appended to the cache so subsequent queries
benefit.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import pandas as pd

from backend.config import settings
from backend.core.llm import chat_json
from backend.core.prompts import EXTRACT_PROMPT
from backend.core.schemas import Capabilities, Evidence


_cache_lock = threading.Lock()
_in_memory_cache: dict[str, tuple[Capabilities, Evidence]] = {}


_TRISTATE_VALUES = {"yes", "no", "uncertain"}
_DOCTOR_VALUES = {"full-time", "part-time", "unknown"}


def _normalize_tristate(v: object) -> str:
    """Coerce any LLM-emitted value to a valid TriState. Default 'uncertain'."""
    if not isinstance(v, str):
        return "uncertain"
    s = v.strip().lower()
    if s in _TRISTATE_VALUES:
        return s
    # Common LLM drift: "unknown", "n/a", "none" -> "uncertain"
    return "uncertain"


def _normalize_doctor(v: object) -> str:
    if not isinstance(v, str):
        return "unknown"
    s = v.strip().lower()
    if s in _DOCTOR_VALUES:
        return s
    return "unknown"


def extract_one(notes: str) -> tuple[Capabilities, Evidence]:
    """Extract capabilities + evidence from a single hospital's notes.

    Robust: never raises. On any LLM/parse failure, returns the all-uncertain
    `Capabilities()` default and empty `Evidence()`.
    """
    if not notes or not notes.strip():
        return Capabilities(), Evidence()

    try:
        raw: dict[str, Any] = chat_json(
            EXTRACT_PROMPT.replace("{notes}", notes[:6000]),
            max_tokens=900,  # generous enough for the JSON to never truncate
        )
    except Exception:
        return Capabilities(), Evidence()

    cap = Capabilities(
        has_icu=_normalize_tristate(raw.get("has_icu")),
        has_emergency=_normalize_tristate(raw.get("has_emergency")),
        has_surgery=_normalize_tristate(raw.get("has_surgery")),
        has_anesthesiologist=_normalize_tristate(raw.get("has_anesthesiologist")),
        has_oxygen=_normalize_tristate(raw.get("has_oxygen")),
        doctor_type=_normalize_doctor(raw.get("doctor_type")),
    )
    evidence_raw = raw.get("evidence") or {}
    if not isinstance(evidence_raw, dict):
        evidence_raw = {}
    ev = Evidence(**{
        k: (evidence_raw.get(k) if isinstance(evidence_raw.get(k), str) else None)
        for k in Evidence.model_fields
    })
    return cap, ev


@lru_cache(maxsize=1)
def _load_extractions() -> pd.DataFrame | None:
    """Load pre-computed extractions.

    * Databricks path: read the `facility_extractions` Delta table once via
      the SQL warehouse and keep it in memory for the lifetime of the
      process. Requires `DBX_WAREHOUSE_ID` to be set.
    * Local path: read the parquet cache written by `batch_extract`.
    """
    if settings.use_databricks and settings.dbx_warehouse_id:
        try:
            from backend.databricks.sql import run_query

            return run_query(
                f"SELECT * FROM {settings.dbx_full_extractions_table}"
            )
        except Exception:
            # Fall through to parquet cache if warehouse query fails.
            pass
    if not settings.extractions_path.exists():
        return None
    return pd.read_parquet(settings.extractions_path)


def lookup(facility_id: str) -> tuple[Capabilities, Evidence] | None:
    """Look up a pre-extracted row. Returns None if not yet processed."""
    fid = str(facility_id)
    if fid in _in_memory_cache:
        return _in_memory_cache[fid]
    df = _load_extractions()
    if df is None:
        return None
    row = df[df["facility_id"].astype(str) == fid]
    if row.empty:
        return None
    r = row.iloc[0]
    cap = Capabilities(
        has_icu=_normalize_tristate(r.get("has_icu")),
        has_emergency=_normalize_tristate(r.get("has_emergency")),
        has_surgery=_normalize_tristate(r.get("has_surgery")),
        has_anesthesiologist=_normalize_tristate(r.get("has_anesthesiologist")),
        has_oxygen=_normalize_tristate(r.get("has_oxygen")),
        doctor_type=_normalize_doctor(r.get("doctor_type")),
    )
    ev = Evidence(
        icu=(r.get("ev_icu") or None) or None,
        emergency=(r.get("ev_emergency") or None) or None,
        surgery=(r.get("ev_surgery") or None) or None,
        anesthesiologist=(r.get("ev_anesthesiologist") or None) or None,
        oxygen=(r.get("ev_oxygen") or None) or None,
        doctor_type=(r.get("ev_doctor_type") or None) or None,
    )
    out = (cap, ev)
    _in_memory_cache[fid] = out
    return out


def cache_extraction(facility_id: str, row_meta: dict, cap: Capabilities, ev: Evidence) -> None:
    """Append a freshly-extracted row to the on-disk cache and memory cache.

    Used by the orchestrator after a live (cache-miss) extraction so future
    queries don't pay the LLM cost again.
    """
    fid = str(facility_id)
    _in_memory_cache[fid] = (cap, ev)
    new_row = {
        "facility_id": fid,
        "name": row_meta.get("name"),
        "state": row_meta.get("state"),
        "district": row_meta.get("district"),
        "pin": row_meta.get("pin"),
        "rural": row_meta.get("rural"),
        "facility_type": row_meta.get("facility_type"),
        "has_icu": cap.has_icu,
        "has_emergency": cap.has_emergency,
        "has_surgery": cap.has_surgery,
        "has_anesthesiologist": cap.has_anesthesiologist,
        "has_oxygen": cap.has_oxygen,
        "doctor_type": cap.doctor_type,
        "ev_icu": ev.icu or "",
        "ev_emergency": ev.emergency or "",
        "ev_surgery": ev.surgery or "",
        "ev_anesthesiologist": ev.anesthesiologist or "",
        "ev_oxygen": ev.oxygen or "",
        "ev_doctor_type": ev.doctor_type or "",
    }
    with _cache_lock:
        try:
            existing = (
                pd.read_parquet(settings.extractions_path)
                if settings.extractions_path.exists()
                else pd.DataFrame()
            )
            out = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            settings.extractions_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(settings.extractions_path)
            # Invalidate the lru-cache so next reload sees the new row.
            _load_extractions.cache_clear()
        except Exception:
            pass
