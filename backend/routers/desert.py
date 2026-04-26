"""GET /desert-map route — aggregates capability gaps.

Two execution modes:

* **Local** (`USE_DATABRICKS=0`): read local parquet extractions.
* **Databricks** (`USE_DATABRICKS=1` + warehouse): run Spark SQL directly on
  the Delta tables in Unity Catalog. No parquet required.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from backend.config import settings
from backend.core.schemas import (
    DesertGap,
    DesertMapResponse,
    PinDesertGap,
    PinDesertMapResponse,
)


router = APIRouter()

_CAP_COLS = ("has_icu", "has_emergency", "has_surgery",
             "has_anesthesiologist", "has_oxygen")


def _load_extractions_df() -> pd.DataFrame:
    """Read from Delta via SQL warehouse on Databricks, else parquet."""
    if settings.use_databricks and settings.dbx_warehouse_id:
        from backend.databricks.sql import run_query

        return run_query(
            f"SELECT * FROM {settings.dbx_full_extractions_table}"
        )
    if not settings.extractions_path.exists():
        raise HTTPException(503, "Extractions not built yet.")
    return pd.read_parquet(settings.extractions_path)


def _load_facilities_df() -> pd.DataFrame:
    if settings.use_databricks and settings.dbx_warehouse_id:
        from backend.databricks.sql import run_query

        return run_query(
            f"SELECT facility_id, pin, latitude, longitude, state "
            f"FROM {settings.dbx_full_table}"
        )
    if not settings.processed_path.exists():
        raise HTTPException(503, "Processed hospitals not built yet.")
    return pd.read_parquet(settings.processed_path)


@router.get("/desert-map", response_model=DesertMapResponse)
def desert_map(
    min_total: int = Query(5, ge=0, description="Hide groups with fewer than this many facilities."),
    capability: str | None = Query(None, description="Optional filter: 'icu', 'surgery', etc."),
) -> DesertMapResponse:
    """Aggregated view of capability gaps by state."""
    df = _load_extractions_df()
    if "state" not in df.columns:
        raise HTTPException(500, "Extractions missing `state` column.")

    gaps: list[DesertGap] = []
    for state, sub in df.groupby("state"):
        total = int(len(sub))
        if total < min_total:
            continue
        for col in _CAP_COLS:
            if col not in sub.columns:
                continue
            cap_name = col.replace("has_", "")
            if capability and cap_name != capability.lower():
                continue
            missing = int(((sub[col] == "no") | (sub[col] == "uncertain")).sum())
            gaps.append(
                DesertGap(
                    state=str(state),
                    capability=cap_name,
                    missing_or_uncertain=missing,
                    total=total,
                    gap_ratio=round(missing / total, 3),
                )
            )
    gaps.sort(key=lambda g: g.gap_ratio, reverse=True)
    return DesertMapResponse(gaps=gaps)


@router.get("/desert-map/pins", response_model=PinDesertMapResponse)
def desert_map_pins(
    min_per_pin: int = Query(1, ge=1, description="Hide PINs with fewer than this many facilities."),
    capability: str = Query("icu", description="Capability: icu, emergency, surgery, anesthesiologist, oxygen"),
    top: int = Query(40, ge=1, le=200, description="Return up to this many highest-risk PINs."),
) -> PinDesertMapResponse:
    """PIN-level medical desert / crisis zones for dynamic mapping."""
    ex = _load_extractions_df()
    pr = _load_facilities_df()

    cap = capability.lower()
    if not cap.startswith("has_"):
        cap = f"has_{cap}"
    key = cap
    if key not in ex.columns:
        raise HTTPException(400, f"Unknown capability / column: {key}")

    merged = ex.merge(
        pr[["facility_id", "pin", "latitude", "longitude", "state"]],
        on="facility_id",
        how="left",
        suffixes=("", "_src"),
    )
    merged["pin"] = merged["pin"].fillna("").astype(str).str.strip()
    merged = merged[merged["pin"].str.len() > 0]

    cap_label = key.replace("has_", "", 1) if key.startswith("has_") else key
    zones: list[PinDesertGap] = []
    for pin, sub in merged.groupby("pin"):
        pin_s = str(pin).strip()
        if not pin_s:
            continue
        n = int(len(sub))
        if n < min_per_pin:
            continue
        miss = int(((sub[key] == "no") | (sub[key] == "uncertain")).sum())
        risk = round(miss / n, 3) if n else 0.0
        state_col = "state_src" if "state_src" in sub.columns else "state"
        st_series = sub[state_col].dropna() if state_col in sub.columns else pd.Series(dtype=object)
        stv: str | None
        if len(st_series):
            s0 = st_series.iloc[0]
            stv = None if s0 is None or (isinstance(s0, float) and pd.isna(s0)) else str(s0)
        else:
            stv = None
        lat = sub["latitude"].dropna() if "latitude" in sub.columns else pd.Series(dtype=float)
        lng = sub["longitude"].dropna() if "longitude" in sub.columns else pd.Series(dtype=float)
        c_lat = float(lat.astype(float).mean()) if len(lat) else None
        c_lng = float(lng.astype(float).mean()) if len(lng) else None
        zones.append(
            PinDesertGap(
                pin=pin_s,
                state=stv,
                capability=cap_label,
                total=n,
                missing_or_uncertain=miss,
                risk=risk,
                centroid_lat=c_lat,
                centroid_lng=c_lng,
            )
        )
    zones.sort(key=lambda z: (z.risk, z.missing_or_uncertain), reverse=True)
    return PinDesertMapResponse(zones=zones[:top])
