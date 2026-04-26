# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest VF India 10k facilities into Unity Catalog
# MAGIC
# MAGIC Reads the raw XLSX dataset, applies the same canonicalization used by the
# MAGIC local pipeline (phone/email normalization, rural heuristic, merged notes),
# MAGIC and writes a Delta table to `<catalog>.<schema>.<table>`.
# MAGIC
# MAGIC **Run once.** Re-run to refresh when the source XLSX changes.

# COMMAND ----------
# MAGIC %pip install -q openpyxl pandas pyarrow
# dbutils.library.restartPython()  # uncomment if the install changes the env

# COMMAND ----------
import json
import ast
import pandas as pd

# --- Parameters (override via notebook widgets for reruns) ---
dbutils.widgets.text("catalog", "healthmap")
dbutils.widgets.text("schema", "facilities")
dbutils.widgets.text("table", "facilities")
dbutils.widgets.text(
    "source_path",
    # A DBFS / UC Volume path. Upload the XLSX to either:
    #   /Volumes/<catalog>/<schema>/raw/VF_Hackathon_Dataset_India_Large.xlsx
    # or /FileStore/tables/VF_Hackathon_Dataset_India_Large.xlsx
    "/Volumes/healthmap/facilities/raw/VF_Hackathon_Dataset_India_Large.xlsx",
)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE = dbutils.widgets.get("table")
SOURCE_PATH = dbutils.widgets.get("source_path")
FQ_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print("Target table:", FQ_TABLE)
print("Source path :", SOURCE_PATH)

# COMMAND ----------
_URBAN_CITIES = frozenset({
    "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad",
    "surat", "jaipur", "lucknow", "kanpur", "nagpur", "visakhapatnam",
    "indore", "thane", "bhopal", "patna", "vadodara", "ghaziabad", "ludhiana",
    "coimbatore", "agra", "madurai", "nashik", "faridabad", "meerut",
    "rajkot", "kalyan", "vasai", "vijayawada", "jabalpur", "mysore",
    "mysuru", "gwalior", "aurangabad", "ranchi", "howrah", "jodhpur",
    "raipur", "kota", "guwahati", "chandigarh", "dehradun", "noida",
    "gurgaon", "gurugram", "amritsar", "allahabad", "prayagraj", "varanasi",
    "srinagar", "navi mumbai", "ulhasnagar", "tiruchirappalli", "trichy",
    "salem", "warangal", "kochi", "cochin", "thiruvananthapuram",
    "trivandrum", "kozhikode", "calicut", "thrissur",
})


def _format_list_field(val, label):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() == "nan" or s in ("[]", "['']", "[\"\"]"):
        return ""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed if str(x).strip()]
            return f"{label}: " + "; ".join(items) + "." if items else ""
    except (ValueError, SyntaxError):
        pass
    return f"{label}: {s}."


def _build_notes(row):
    parts = []
    desc = str(row.get("description") or "").strip()
    if desc and desc.lower() != "nan":
        parts.append(f"Description: {desc}")
    for col, label in [
        ("specialties", "Specialties"),
        ("procedure", "Procedures"),
        ("equipment", "Equipment"),
        ("capability", "Capabilities listed"),
    ]:
        parts.append(_format_list_field(row.get(col), label))
    for col, label in [("facilityTypeId", "Facility type"), ("operatorTypeId", "Operator type")]:
        v = row.get(col)
        if v and str(v).strip().lower() != "nan":
            parts.append(f"{label}: {v}.")
    return " ".join(p for p in parts if p)


def _is_rural(city):
    if city is None or (isinstance(city, float) and pd.isna(city)):
        return None
    return str(city).strip().lower() not in _URBAN_CITIES


def _clean_official_phone(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return None
    s = str(n)
    if len(s) < 10:
        return None
    if s.startswith("91") and len(s) >= 12:
        return "+" + s
    if len(s) == 10:
        return "+91" + s
    return "+" + s


def _normalize_phone_token(s):
    s = s.strip()
    if not s:
        return None
    if s.startswith("+"):
        return s
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) == 10:
        return "+91" + digits
    if len(digits) >= 10:
        return "+" + digits
    return None


def _first_from_phone_numbers(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, tuple)) and val:
        return _normalize_phone_token(str(val[0]).strip())
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            return _normalize_phone_token(str(parsed[0]).strip())
    except (ValueError, SyntaxError):
        pass
    return _normalize_phone_token(s)


def _row_phone(row):
    p = _clean_official_phone(row.get("officialPhone"))
    if p:
        return p
    return _first_from_phone_numbers(row.get("phone_numbers"))


def _clean_email(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan" or "@" not in s:
        return None
    return s

# COMMAND ----------
raw = pd.read_excel(SOURCE_PATH)
print("Raw rows:", len(raw))
raw.columns = [c.strip() for c in raw.columns]

# Some exports use slightly different column names; map common aliases.
_aliases = {
    "address_stateOrRegion": ["state", "address_state"],
    "address_city": ["city"],
    "address_zipOrPostcode": ["zip", "pin", "address_zip"],
    "facilityTypeId": ["facility_type_id"],
    "operatorTypeId": ["operator_type_id"],
    "officialPhone": ["official_phone"],
    "phone_numbers": ["phones"],
}
for canon, alts in _aliases.items():
    if canon not in raw.columns:
        for alt in alts:
            if alt in raw.columns:
                raw[canon] = raw[alt]
                break

out = pd.DataFrame()
out["facility_id"] = [f"vf-{i:05d}" for i in range(len(raw))]
out["name"] = raw.get("name", "").astype(str).str.strip()
out["state"] = raw.get("address_stateOrRegion", "").astype(str).str.strip()
out["district"] = raw.get("address_city", "").astype(str).str.strip()
out["pin"] = raw.get("address_zipOrPostcode", "").astype(str).str.strip().replace({"nan": None})
out["rural"] = raw.get("address_city").apply(_is_rural) if "address_city" in raw.columns else None
out["latitude"] = raw.get("latitude")
out["longitude"] = raw.get("longitude")
out["facility_type"] = raw.get("facilityTypeId", "").astype(str).str.strip().replace({"nan": None})
out["phone"] = raw.apply(_row_phone, axis=1)
out["email"] = raw["email"].map(_clean_email) if "email" in raw.columns else None
out["notes"] = raw.apply(_build_notes, axis=1)

display(out.head(5))
print("Rows with phone:", out["phone"].notna().sum())
print("Rows with email:", out["email"].notna().sum())

# COMMAND ----------
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, DoubleType, LongType,
)

schema_struct = StructType([
    StructField("facility_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("state", StringType(), True),
    StructField("district", StringType(), True),
    StructField("pin", StringType(), True),
    StructField("rural", BooleanType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("facility_type", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("email", StringType(), True),
    StructField("notes", StringType(), True),
])

# Cast to exact types for safety before writing to Delta.
for col, t in (("latitude", float), ("longitude", float)):
    try:
        out[col] = out[col].astype(float)
    except Exception:
        out[col] = None
out["rural"] = out["rural"].astype("boolean") if "boolean" in str(out["rural"].dtype) else out["rural"]

sdf = spark.createDataFrame(out, schema=schema_struct)
sdf = sdf.dropDuplicates(["facility_id"])

# Required for Vector Search Delta Sync index — enable CDF.
spark.sql(
    f"CREATE TABLE IF NOT EXISTS {FQ_TABLE} ("
    "facility_id STRING NOT NULL, "
    "name STRING, state STRING, district STRING, pin STRING, rural BOOLEAN, "
    "latitude DOUBLE, longitude DOUBLE, facility_type STRING, "
    "phone STRING, email STRING, notes STRING"
    ") TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FQ_TABLE)

# Add a primary key constraint so Vector Search can key on facility_id.
try:
    spark.sql(f"ALTER TABLE {FQ_TABLE} ALTER COLUMN facility_id SET NOT NULL")
    spark.sql(
        f"ALTER TABLE {FQ_TABLE} ADD CONSTRAINT pk_facility PRIMARY KEY (facility_id)"
    )
except Exception as e:
    print("PK step skipped (likely already set):", e)

print("Wrote", FQ_TABLE)
display(spark.table(FQ_TABLE).limit(5))
