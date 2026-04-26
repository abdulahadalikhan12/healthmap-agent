# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Create Mosaic AI Vector Search endpoint + Delta Sync index
# MAGIC
# MAGIC Creates (or reuses) a standard Vector Search endpoint and a **Delta Sync
# MAGIC index** over the facilities table from notebook 01. Embeddings are
# MAGIC computed automatically by Databricks using the `databricks-bge-large-en`
# MAGIC Foundation Model Serving endpoint.
# MAGIC
# MAGIC Re-run this notebook any time you want to recreate the index.

# COMMAND ----------
# MAGIC %pip install -q databricks-vectorsearch databricks-sdk
# dbutils.library.restartPython()  # uncomment if install changed env

# COMMAND ----------
import time

from databricks.vector_search.client import VectorSearchClient

dbutils.widgets.text("catalog", "healthmap")
dbutils.widgets.text("schema", "facilities")
dbutils.widgets.text("table", "facilities")
dbutils.widgets.text("endpoint_name", "healthmap-vs")
dbutils.widgets.text("index_name", "facilities_vs_index")
dbutils.widgets.text("embed_endpoint", "databricks-bge-large-en")
dbutils.widgets.text("primary_key", "facility_id")
dbutils.widgets.text("text_column", "notes")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE = dbutils.widgets.get("table")
ENDPOINT = dbutils.widgets.get("endpoint_name")
INDEX_SHORT = dbutils.widgets.get("index_name")
EMBED = dbutils.widgets.get("embed_endpoint")
PK = dbutils.widgets.get("primary_key")
TEXT_COL = dbutils.widgets.get("text_column")

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"
INDEX_FQ = f"{CATALOG}.{SCHEMA}.{INDEX_SHORT}"
print("Endpoint       :", ENDPOINT)
print("Source table   :", SOURCE_TABLE)
print("Index          :", INDEX_FQ)
print("Embedding model:", EMBED)

client = VectorSearchClient(disable_notice=True)

# COMMAND ----------
# MAGIC %md ## Endpoint

# COMMAND ----------
try:
    existing = client.list_endpoints().get("endpoints", [])
    names = [e.get("name") for e in existing]
    if ENDPOINT not in names:
        client.create_endpoint(name=ENDPOINT, endpoint_type="STANDARD")
        print("Creating endpoint", ENDPOINT)
    else:
        print("Endpoint already exists:", ENDPOINT)
except Exception as e:
    print("list/create endpoint failed, continuing:", e)

# Wait for ONLINE before creating the index.
for attempt in range(60):
    try:
        ep = client.get_endpoint(ENDPOINT)
        state = ep.get("endpoint_status", {}).get("state") or ep.get("state")
        print(f"[{attempt:02d}] endpoint state:", state)
        if str(state).upper() in ("ONLINE", "READY", "PROVISIONED"):
            break
    except Exception as e:
        print("waiting:", e)
    time.sleep(15)

# COMMAND ----------
# MAGIC %md ## Delta Sync index

# COMMAND ----------
indexes = []
try:
    indexes = client.list_indexes(name=ENDPOINT).get("vector_indexes", [])
except Exception:
    pass
existing_index_names = [i.get("name") for i in indexes]
if INDEX_FQ in existing_index_names:
    print("Index already exists, deleting to recreate for parity:", INDEX_FQ)
    try:
        client.delete_index(endpoint_name=ENDPOINT, index_name=INDEX_FQ)
        time.sleep(5)
    except Exception as e:
        print("delete failed (continuing):", e)

idx = client.create_delta_sync_index(
    endpoint_name=ENDPOINT,
    index_name=INDEX_FQ,
    source_table_name=SOURCE_TABLE,
    primary_key=PK,
    embedding_source_column=TEXT_COL,
    embedding_model_endpoint_name=EMBED,
    pipeline_type="TRIGGERED",  # run on demand; switch to CONTINUOUS for live sync
)
print("Index creation triggered.")

# COMMAND ----------
# Wait for the index to be ready.
for attempt in range(80):
    try:
        info = client.get_index(
            endpoint_name=ENDPOINT, index_name=INDEX_FQ
        ).describe()
        status = info.get("status", {})
        ready = bool(status.get("ready"))
        msg = status.get("message") or status.get("detailed_state", "")
        print(f"[{attempt:02d}] ready={ready} state={msg}")
        if ready:
            break
    except Exception as e:
        print("waiting for index:", e)
    time.sleep(20)

# COMMAND ----------
# MAGIC %md ## Smoke query
# MAGIC
# MAGIC Confirms the index answers similarity queries.

# COMMAND ----------
idx = client.get_index(endpoint_name=ENDPOINT, index_name=INDEX_FQ)
res = idx.similarity_search(
    query_text="rural Bihar hospital with ICU and anesthesiology",
    columns=["facility_id", "name", "state", "district", "pin", "rural",
             "phone", "email", "notes"],
    num_results=5,
)
display(res.get("result", {}).get("data_array", []))
