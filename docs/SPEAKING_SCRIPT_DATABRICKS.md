# Speaking script — Databricks + Mosaic Vector Search stack

Use this for demos, judging, or a short video voice-over. It matches the **Databricks-native** design: **Unity Catalog + Delta**, **Mosaic AI Vector Search**, **Model Serving** (foundation models or **Agent Bricks**), **SQL warehouse** analytics, **MLflow 3** tracing, with **FastAPI** still fronting the user-facing API.

---

## 90 seconds (elevator)

> “We built **Healthmap Agent** on the **Databricks Data Intelligence Platform** for about ten thousand Indian healthcare facilities. Raw records live in **Unity Catalog** as **Delta** tables — one canonical row per facility with merged **notes**, plus **phone** and **email** when the source has them.
>
> **Retrieval** uses **Mosaic AI Vector Search**: a **Delta Sync** index over those notes, with embeddings served by a **Databricks foundation model endpoint** — so similarity search stays fast and stays in sync when the table updates.
>
> **Reasoning and extraction** run through **Databricks Model Serving** — today a hosted Llama-class endpoint; the same hook fits a custom **Agent Brick** when we productize it. We batch-extract capabilities once into a second Delta table so online queries are mostly lookups, not repeated LLM cost.
>
> **Trust** still matters: a **validator** checks medical consistency — surgery without anesthesiology gets flagged — and we can ground rules with web standards via Tavily. Every user query logs a full chain in **MLflow 3** on Databricks — **Experiments → Tracing** — so you can see each agent step, not just a final answer.
>
> The **public API** stays a small **FastAPI** service — we deploy it on **Hugging Face Spaces** for the hackathon — but the data plane and model plane are **Databricks-native**. **PIN-level crisis views** aggregate straight from Delta through the **SQL warehouse**, so the map reflects the same extractions the agent uses.”

---

## About 3 minutes (standard demo)

> “I’ll walk through **what runs on Databricks** and **why**.
>
> **Data.** We start from the Virtue Foundation spreadsheet — messy free text: equipment, specialties, hours, claims about emergency care. Notebook one **canonicalizes** that into **Delta** under Unity Catalog: stable IDs, geography, **PIN**, and we preserve **contact fields** wherever the dataset actually has them. That table is the **system of record** for what we show in the product.
>
> **Vector search.** Notebook two stands up **Mosaic AI Vector Search**: a managed endpoint plus a **Delta Sync** index on the `notes` column. Databricks keeps embeddings aligned with the table — so we’re not maintaining a separate FAISS file or wondering if the index is stale. At query time the API asks Mosaic for **semantic neighbors** and then applies the same **structured filters** we used before — state, district, rural — so ‘Bihar’ doesn’t silently become ‘Maharashtra’.
>
> **Models.** Chat completions — query parsing, capability extraction, short reasoning — go through **Model Serving**. We default to a **hosted foundation model** endpoint for speed; the architecture is the same if we swap in an **Agent Brick** that wraps tools and policies. Embeddings for anything we still compute client-side also go through a **Databricks embedding endpoint** — one stack, one auth model, one bill.
>
> **Batch intelligence.** The heavy extraction pass runs as a **batch job in the workspace**: we write **`facility_extractions`** Delta — ICU, emergency, surgery, anesthesia, oxygen, staffing — with **verbatim evidence sentences** pulled from the notes. That table feeds the **trust scorer** and keeps latency low when users search live.
>
> **Governance and observability.** Unity Catalog means the tables are **discoverable and permissioned**. **MLflow 3** in the workspace records **runs and traces** per query — you can open the **Tracing** tab and see spans for parse, retrieval, validation, trust — not just a blob of JSON in a log file.
>
> **Crisis mapping.** Our **PIN desert** endpoint doesn’t guess: it **joins** extractions to facilities in SQL over the warehouse, groups by PIN, and returns risk as the fraction of facilities that are **no** or **uncertain** on a chosen capability. The frontend paints that on a map of India.
>
> **Edge API.** We still expose **FastAPI** — in the hackathon that’s on **Hugging Face** — with a PAT to Databricks. That’s deliberate: judges hit a stable HTTPS URL while the serious data and models stay on the platform you asked us to use.”

---

## If a judge asks about Genie (15-second add-on)

> “**Genie** in our build is the natural place for **analyst-facing** exploration — ‘show me states with the worst ICU gaps’ — and for **orchestrating** the notebook pipeline: ingest, index health, extraction backlog. The **online** query path stays in FastAPI so we keep **sub-second** API contracts; Genie complements that for ops and storytelling, not a replacement for the serving layer.”

---

## If a judge asks “why not everything inside Databricks Apps?”

> “We **could** collapse the API into Databricks; for the hackathon we kept **FastAPI on Hugging Face** so the frontend team has a **familiar** `POST /query` URL and CORS is trivial. The **important** part — catalog, vectors, models, traces — already **lives on Databricks**. Migrating the last hop is mostly networking and IAM, not a redesign.”

---

## Closing line (optional)

> “The outcome is the same product promise — **trust-scored, evidence-backed** facility discovery for India — but the **substrate** is what the brief asked for: **Databricks** for data and intelligence, **Mosaic** for vectors, **MLflow 3** for agent observability, and **Delta** as the single source of truth.”

---

## Speaker notes (non-verbal)

- **Screen 1:** Databricks Catalog → `healthmap.facilities` + `facility_extractions`.
- **Screen 2:** Vector Search → endpoint **online**, index **ready**, sample similarity query.
- **Screen 3:** MLflow experiment → one run → **Tracing** tab with spans.
- **Screen 4:** Live `POST /query` from Swagger or frontend; show **evidence** + **trust breakdown** in JSON.
- **Screen 5:** Crisis map or `/desert-map/pins` JSON with `zones` array.

Trim any paragraph if you run over time; the **90-second** block is the minimum viable pitch.
