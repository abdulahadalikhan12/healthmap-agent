"""Run SQL against a Databricks SQL warehouse via the SDK (no extra connector
dep). Used to load pre-computed extractions into the online API so the
orchestrator doesn't pay per-query LLM cost for capabilities.
"""
from __future__ import annotations

import pandas as pd

from backend.config import settings
from backend.databricks.client import workspace_client


def run_query(sql: str, *, limit_rows: int = 20000) -> pd.DataFrame:
    """Execute `sql` against `settings.dbx_warehouse_id` and return a
    pandas DataFrame. Uses the SDK's Statement Execution API, so no
    `databricks-sql-connector` dependency is required."""
    if not settings.dbx_warehouse_id:
        raise RuntimeError(
            "DBX_WAREHOUSE_ID is not set; cannot run SQL against Unity Catalog."
        )
    w = workspace_client()
    r = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=settings.dbx_warehouse_id,
        wait_timeout="30s",
        row_limit=limit_rows,
    )
    # Poll briefly if still pending.
    status = getattr(r, "status", None)
    state = getattr(status, "state", None)
    if str(state).upper() in ("PENDING", "RUNNING"):
        import time
        for _ in range(20):
            r = w.statement_execution.get_statement(statement_id=r.statement_id)
            state = getattr(getattr(r, "status", None), "state", None)
            if str(state).upper() not in ("PENDING", "RUNNING"):
                break
            time.sleep(1)

    manifest = getattr(r, "manifest", None)
    result = getattr(r, "result", None)
    if manifest is None or result is None:
        return pd.DataFrame()
    cols = [c.name for c in (manifest.schema.columns or [])]
    rows = getattr(result, "data_array", None) or []
    return pd.DataFrame(rows, columns=cols)
