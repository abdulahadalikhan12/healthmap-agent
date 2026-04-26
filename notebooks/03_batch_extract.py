# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Batch capability extraction via FM Serving
# MAGIC
# MAGIC Runs the Trust Scorer's extraction prompt over every row in the
# MAGIC `facilities` Delta table using an Agent Bricks / FM Serving endpoint
# MAGIC (default: `databricks-meta-llama-3-1-70b-instruct`). Writes results to
# MAGIC `facility_extractions` Delta so the online API can look up answers by
# MAGIC `facility_id` without paying per-query LLM cost.
# MAGIC
# MAGIC Re-run whenever source rows change.

# COMMAND ----------
# MAGIC %pip install -q databricks-sdk mlflow>=3.0
# dbutils.library.restartPython()

# COMMAND ----------
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
from databricks.sdk import WorkspaceClient

dbutils.widgets.text("catalog", "healthmap")
dbutils.widgets.text("schema", "facilities")
dbutils.widgets.text("source_table", "facilities")
dbutils.widgets.text("target_table", "facility_extractions")
dbutils.widgets.text("fm_endpoint", "databricks-meta-llama-3-1-70b-instruct")
dbutils.widgets.text("limit", "0")  # 0 = all rows; set e.g. 200 for a smoke run
dbutils.widgets.text("experiment", "/Shared/healthmap-agent")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SRC = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('source_table')}"
DST = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('target_table')}"
FM = dbutils.widgets.get("fm_endpoint")
LIMIT = int(dbutils.widgets.get("limit"))
EXPERIMENT = dbutils.widgets.get("experiment")

mlflow.set_experiment(EXPERIMENT)

w = WorkspaceClient()

# COMMAND ----------
EXTRACT_PROMPT_TEMPLATE = """You extract structured medical capabilities from a hospital's free-form notes.

Be STRICT and CONSERVATIVE:
- If a capability is not explicitly mentioned → "uncertain".
- Do NOT infer. If they say "general medicine", do NOT mark surgery as yes.
- If they say "ICU available" → has_icu = "yes".
- If they say "no ICU" / "ICU under construction" → has_icu = "no".

Output ONLY valid JSON with this exact shape:
{
  "has_icu": "yes"|"no"|"uncertain",
  "has_emergency": "yes"|"no"|"uncertain",
  "has_surgery": "yes"|"no"|"uncertain",
  "has_anesthesiologist": "yes"|"no"|"uncertain",
  "has_oxygen": "yes"|"no"|"uncertain",
  "doctor_type": "full-time"|"part-time"|"unknown",
  "evidence": {
    "icu": string, "emergency": string, "surgery": string,
    "anesthesiologist": string, "oxygen": string, "doctor_type": string
  }
}

Each evidence value MUST be a verbatim sentence copied from the notes, or "" if uncertain.

NOTES:
\"\"\"
{notes}
\"\"\"

JSON:"""


def extract_one(notes: str) -> dict:
    if not notes or not notes.strip():
        return {}
    prompt = EXTRACT_PROMPT_TEMPLATE.replace("{notes}", notes[:6000])
    try:
        resp = w.serving_endpoints.query(
            name=FM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=900,
            extra_params={"response_format": {"type": "json_object"}},
        )
        choices = getattr(resp, "choices", None) or resp.get("choices", [])
        first = choices[0]
        msg = getattr(first, "message", None) or first.get("message", {})
        content = getattr(msg, "content", None) or msg.get("content", "")
        return json.loads(content)
    except Exception:
        return {}


def normalize(obj: dict) -> dict:
    tri = {"yes", "no", "uncertain"}
    doc = {"full-time", "part-time", "unknown"}

    def _t(v):
        return v if isinstance(v, str) and v.lower() in tri else "uncertain"

    def _d(v):
        return v if isinstance(v, str) and v.lower() in doc else "unknown"

    ev = obj.get("evidence") or {}
    if not isinstance(ev, dict):
        ev = {}
    return {
        "has_icu": _t(obj.get("has_icu")),
        "has_emergency": _t(obj.get("has_emergency")),
        "has_surgery": _t(obj.get("has_surgery")),
        "has_anesthesiologist": _t(obj.get("has_anesthesiologist")),
        "has_oxygen": _t(obj.get("has_oxygen")),
        "doctor_type": _d(obj.get("doctor_type")),
        "ev_icu": str(ev.get("icu") or ""),
        "ev_emergency": str(ev.get("emergency") or ""),
        "ev_surgery": str(ev.get("surgery") or ""),
        "ev_anesthesiologist": str(ev.get("anesthesiologist") or ""),
        "ev_oxygen": str(ev.get("oxygen") or ""),
        "ev_doctor_type": str(ev.get("doctor_type") or ""),
    }

# COMMAND ----------
src_df = spark.table(SRC).select("facility_id", "state", "district", "pin", "rural",
                                 "facility_type", "notes")
if LIMIT > 0:
    src_df = src_df.limit(LIMIT)
rows = [r.asDict() for r in src_df.collect()]
print(f"Rows to extract: {len(rows):,}")

# COMMAND ----------
import pandas as pd

BATCH = 24  # serving throughput; adjust based on endpoint concurrency
results: list[dict] = []

with mlflow.start_run(run_name=f"batch_extract ({len(rows)} rows)") as run:
    mlflow.log_params({"endpoint": FM, "rows": len(rows), "source": SRC, "target": DST})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=BATCH) as ex:
        futs = {ex.submit(extract_one, r.get("notes") or ""): r for r in rows}
        for i, fut in enumerate(as_completed(futs)):
            r = futs[fut]
            obj = fut.result()
            merged = {
                "facility_id": r["facility_id"],
                "state": r.get("state"), "district": r.get("district"),
                "pin": r.get("pin"), "rural": r.get("rural"),
                "facility_type": r.get("facility_type"),
                **normalize(obj),
            }
            results.append(merged)
            if (i + 1) % 500 == 0:
                print(f"... {i + 1}/{len(rows)} ({time.time() - t0:.1f}s)")
    dt = time.time() - t0
    mlflow.log_metric("extract_seconds", dt)
    mlflow.log_metric("rows_processed", len(results))
    print(f"Done in {dt:.1f}s")

df = pd.DataFrame(results)
display(df.head())

# COMMAND ----------
sdf = spark.createDataFrame(df)
(
    sdf.write.mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(DST)
)
print("Wrote", DST)
display(spark.table(DST).limit(5))
