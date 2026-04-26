"""Orchestrates the 7 agents end-to-end and logs everything to MLflow."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from backend.agents import (
    extraction_agent,
    query_agent,
    reasoning_agent,
    retrieval_agent,
    trace_agent,
    trust_agent,
    validator_agent,
)
from backend.config import settings
from backend.core import mlflow_setup
from backend.core.schemas import (
    Capabilities,
    Evidence,
    HospitalMeta,
    HospitalResult,
    Location,
    QueryResponse,
    Trace,
)


def _g(row, key, default=None):
    """Safe pandas-Series.get-or-getattr wrapper."""
    try:
        v = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
    except Exception:
        v = default
    if v is None:
        return None
    # pandas can give us numpy NaN that isn't None.
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    return v


def _row_location(row) -> Location:
    return Location(
        state=_g(row, "state"),
        district=_g(row, "district"),
        pin=str(_g(row, "pin")) if _g(row, "pin") is not None else None,
        rural=bool(_g(row, "rural")) if _g(row, "rural") is not None else None,
        latitude=float(_g(row, "latitude")) if _g(row, "latitude") is not None else None,
        longitude=float(_g(row, "longitude")) if _g(row, "longitude") is not None else None,
    )


def _row_meta(row) -> HospitalMeta:
    return HospitalMeta(facility_type=_g(row, "facility_type"))


def _clean_contact(val) -> str | None:
    """Strip pandas/Excel junk so we never send 'nan' strings to the client."""
    if val is None:
        return None
    try:
        import math

        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>", "nat"):
        return None
    return s


def _row_dict(row) -> dict:
    """Convert a pandas Series-ish row to a plain dict."""
    try:
        return row.to_dict()
    except AttributeError:
        return dict(row)


def run(query: str, *, top_k: int = 5, retrieve_k: int = 15) -> QueryResponse:
    """End-to-end agent pipeline.

    Retrieves a wider net (`retrieve_k`), validates + trust-scores ALL of
    them, then ranks by `(0.6 * capability_match + 0.4 * trust_score)` so a
    well-verified hospital outranks a hospital that merely claims more.
    Only the top `top_k` get the (LLM-generated) reasoning sentence.
    """
    steps: list[str] = []

    with mlflow_setup.query_run(query):
        # 1. Query understanding.
        t0 = time.time()
        with mlflow_setup.stage_span("01_query_agent", {"query": query}) as s:
            parsed = query_agent.parse_query(query)
            if s is not None and hasattr(s, "set_outputs"):
                s.set_outputs(parsed.model_dump())
        mlflow_setup.log_step("01_parsed_query", parsed.model_dump())
        mlflow_setup.log_metric("query_parse_seconds", time.time() - t0)
        steps.append(f"Parsed query - required: {parsed.required_capabilities}")

        # 2. Retrieval (wider net).
        t0 = time.time()
        with mlflow_setup.stage_span(
            "02_retrieval_agent",
            {"k": retrieve_k, "state": parsed.state, "rural": parsed.rural,
             "backend": "mosaic" if settings.use_databricks else "faiss"},
        ):
            candidates = retrieval_agent.retrieve(parsed, query, top_k=retrieve_k)
        mlflow_setup.log_step(
            "02_retrieved",
            candidates[["facility_id", "name"]].to_dict(orient="records")
            if "facility_id" in candidates.columns else candidates.head().to_dict(),
        )
        mlflow_setup.log_metric("retrieval_seconds", time.time() - t0)
        steps.append(f"Retrieved {len(candidates)} candidate hospitals.")

        # 3. Extraction (lookup-first; cache-miss fallback runs in parallel).
        caps: list[Capabilities] = [None] * len(candidates)  # type: ignore
        evs: list[Evidence] = [None] * len(candidates)  # type: ignore
        cache_misses: list[int] = []
        for i, (_, row) in enumerate(candidates.iterrows()):
            looked = extraction_agent.lookup(str(_g(row, "facility_id", "")))
            if looked is None:
                cache_misses.append(i)
            else:
                caps[i], evs[i] = looked

        if cache_misses:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=min(8, len(cache_misses))) as ex:
                live = list(ex.map(
                    lambda i: extraction_agent.extract_one(
                        _g(candidates.iloc[i], "notes", "") or ""
                    ),
                    cache_misses,
                ))
            for i, (cap, ev) in zip(cache_misses, live):
                caps[i] = cap
                evs[i] = ev
                # Persist to cache so future queries skip the LLM.
                row = candidates.iloc[i]
                extraction_agent.cache_extraction(
                    str(_g(row, "facility_id", "")),
                    {
                        "name": _g(row, "name"),
                        "state": _g(row, "state"),
                        "district": _g(row, "district"),
                        "pin": _g(row, "pin"),
                        "rural": _g(row, "rural"),
                        "facility_type": _g(row, "facility_type"),
                    },
                    cap, ev,
                )
            mlflow_setup.log_metric("live_extract_seconds", time.time() - t0)

        mlflow_setup.log_step("03_extractions", [c.model_dump() for c in caps])
        steps.append(
            f"Extracted capabilities for {len(candidates)} candidates "
            f"({len(cache_misses)} live, {len(candidates) - len(cache_misses)} from cache)."
        )

        # 4. Validate + trust-score every candidate (cheap: rule-based by default).
        with mlflow_setup.stage_span("04_validator_agent", {"n": len(caps)}):
            validator_results = [validator_agent.validate(c, parsed, use_llm=False) for c in caps]
        with mlflow_setup.stage_span("04b_trust_agent", {"n": len(caps)}):
            trust_results = [
                trust_agent.score(c, e, v) for c, e, v in zip(caps, evs, validator_results)
            ]
        mlflow_setup.log_step(
            "04_validator_trust",
            [
                {
                    "facility_id": str(_g(candidates.iloc[i], "facility_id", f"row-{i}")),
                    "issues": [iss.model_dump() for iss in validator_results[i].issues],
                    "trust": trust_results[i].model_dump(),
                }
                for i in range(len(candidates))
            ],
        )

        # 5. Combined ranking: capability match × trust.
        cap_scores = [reasoning_agent.score_hospital(c, parsed) for c in caps]
        combined = [
            (i, 0.6 * cap_scores[i] + 0.4 * trust_results[i].trust_score)
            for i in range(len(candidates))
        ]
        combined.sort(key=lambda x: x[1], reverse=True)
        mlflow_setup.log_step("05_ranking", combined)
        steps.append(
            f"Ranked by (0.6*match + 0.4*trust); top combined score = {combined[0][1]:.2f}"
        )

        # 6. Top-K result rendering with LLM reasoning text (parallel calls).
        top = combined[:top_k]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=min(8, len(top))) as ex:
            reasoning_texts = list(ex.map(
                lambda pair: trace_agent.explain_hospital(
                    name=str(_g(candidates.iloc[pair[0]], "name", "Unknown")),
                    location=_row_location(candidates.iloc[pair[0]]).model_dump(),
                    cap=caps[pair[0]],
                    validator=validator_results[pair[0]],
                    parsed=parsed,
                ),
                top,
            ))
        mlflow_setup.log_metric("reasoning_seconds", time.time() - t0)

        results: list[HospitalResult] = []
        validator_findings = []
        for (idx, combined_score), reasoning in zip(top, reasoning_texts):
            row = candidates.iloc[idx]
            cap = caps[idx]
            ev = evs[idx]
            v = validator_results[idx]
            trust = trust_results[idx]
            location = _row_location(row)
            meta = _row_meta(row)

            evidence_dict = {
                k: getattr(ev, k) for k in ("icu", "emergency", "surgery",
                                            "anesthesiologist", "oxygen", "doctor_type")
                if getattr(ev, k)
            }

            fid = str(_g(row, "facility_id", f"row-{idx}"))
            results.append(
                HospitalResult(
                    facility_id=fid,
                    name=str(_g(row, "name", "Unknown")),
                    location=location,
                    meta=meta,
                    capabilities=cap,
                    trust_score=trust.trust_score,
                    flags=trust.flags,
                    evidence=evidence_dict,
                    reasoning=reasoning,
                    phone=_clean_contact(_g(row, "phone")),
                    email=_clean_contact(_g(row, "email")),
                    trust_breakdown=trust.breakdown,
                )
            )
            validator_findings.append({
                "facility_id": fid,
                "combined_score": round(combined_score, 3),
                "issues": [i.model_dump() for i in v.issues],
                "trust": trust.breakdown.model_dump(),
            })

        mlflow_setup.log_step("06_results", [r.model_dump() for r in results])

        trace = Trace(
            parsed_query=parsed,
            retrieved_ids=[r.facility_id for r in results],
            validator_findings=validator_findings,
            trust_breakdown={r.facility_id: r.trust_score for r in results},
            steps=steps,
        )
        mlflow_setup.log_step("07_trace", trace.model_dump())
        mlflow_setup.log_genai_style_trace(
            query=query,
            steps=steps,
            result_summary={
                "n_results": len(results),
                "facility_ids": [r.facility_id for r in results],
            },
        )

    return QueryResponse(results=results, trace=trace)
