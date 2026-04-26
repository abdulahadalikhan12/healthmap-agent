# Running healthmap-agent on Databricks

This guide walks through the **Databricks-native** path using Agent Bricks /
Model Serving, Mosaic AI Vector Search, and MLflow 3. The local FAISS +
OpenAI path still works (flip `USE_DATABRICKS` to swap).

## What runs where

| Layer | On Databricks | On Hugging Face / local |
|-------|---------------|-------------------------|
| Delta tables (`facilities`, `facility_extractions`) | **Unity Catalog** | — |
| Vector search (notes → top-K) | **Mosaic AI Vector Search** (Delta Sync index) | FAISS file |
| Chat model (extraction, validator, reasoning) | **Model Serving** endpoint (`databricks-meta-llama-3-1-70b-instruct` or a custom Agent Brick) | OpenAI API |
| Embeddings | **Model Serving** endpoint (`databricks-bge-large-en`) | OpenAI `text-embedding-3-small` |
| Per-query tracing | **MLflow 3** on Databricks (`Experiments → Tracing` tab) | Local `./mlruns` folder |
| FastAPI server | Hugging Face Space (unchanged) — calls Databricks endpoints over HTTPS | Uvicorn locally |

The backend is the same FastAPI app; we just flip to the Databricks-backed
implementations via env vars.

## One-time setup

1. **Create** a Databricks workspace with **Unity Catalog** enabled.
2. **Generate** a PAT (or service-principal token) with `WORKSPACE_ACCESS`.
3. **Upload** `dataset/VF_Hackathon_Dataset_India_Large.xlsx` to a UC Volume:
   ```
   /Volumes/healthmap/facilities/raw/VF_Hackathon_Dataset_India_Large.xlsx
   ```
4. **Import** the three notebooks in `notebooks/` into your workspace.

### Run order (in the workspace)

1. `01_ingest_to_uc.py` — creates the `facilities` Delta table with
   `phone` / `email` / `notes` / lat / lng / `pin`. Prints how many rows have
   contacts.
2. `02_create_vs_endpoint_and_index.py` — creates the `healthmap-vs`
   Vector Search endpoint and the `facilities_vs_index` Delta Sync index.
3. `03_batch_extract.py` — runs the extraction prompt over all rows via
   Model Serving, writes `facility_extractions` Delta (trust-scorer cache).

Every notebook is idempotent: rerun any time data changes.

## Local / Hugging Face backend config

In `.env` (or HF Space **Settings → Variables and secrets**):

```env
USE_DATABRICKS=1
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=dapi-...
DBX_CATALOG=healthmap
DBX_SCHEMA=facilities
DBX_TABLE=facilities
DBX_EXTRACTIONS_TABLE=facility_extractions
DBX_VS_ENDPOINT=healthmap-vs
DBX_VS_INDEX=facilities_vs_index
DBX_FM_ENDPOINT=databricks-meta-llama-3-1-70b-instruct
DBX_EMBED_ENDPOINT=databricks-bge-large-en
DBX_WAREHOUSE_ID=<sql-warehouse-id>
MLFLOW_EXPERIMENT_NAME=/Shared/healthmap-agent
```

**Dependencies** (install alongside `requirements.txt`):
```bash
pip install -r requirements.txt -r requirements-databricks.txt
```

## Verification

After notebooks finish and env vars are set:

```bash
uvicorn backend.app:app --reload --port 8000

# Retrieval via Mosaic + Model Serving + MLflow 3 tracing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find rural Bihar facility for emergency appendectomy with part-time doctors"}'

# Crisis map driven by Delta aggregates (warehouse SQL)
curl "http://localhost:8000/desert-map/pins?capability=icu&top=10"
```

In the Databricks workspace → **Experiments → /Shared/healthmap-agent →
Tracing**, you'll see a run per query with spans for each agent step.

## Stretch-goal mapping

| Brief requirement | Where it lives now |
|-------------------|--------------------|
| Massive unstructured extraction | `03_batch_extract.py` over Delta + Model Serving |
| Multi-attribute reasoning | `query_agent` + `retrieval_agent` (Mosaic VS) + `reasoning_agent` |
| Trust Scorer | `validator_agent` + `trust_agent` with 4-factor breakdown |
| Row-level citations | `extraction_agent` keeps verbatim sentences per capability |
| MLflow 3 tracing | `mlflow_setup.stage_span(...)` spans + per-run artifacts |
| Self-correction / Validator | `validator_agent` (rules + optional Tavily standards) |
| Dynamic crisis mapping | `/desert-map/pins` backed by UC SQL + frontend crisis page |
