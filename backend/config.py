"""Centralized settings loaded from .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    # Optional override for OpenAI-compatible providers (Groq, Together, etc.)
    openai_base_url: str | None = None

    tavily_api_key: str = ""

    data_raw_path: str = "dataset/VF_Hackathon_Dataset_India_Large.xlsx"
    data_dir: str = "data"

    extraction_sample_size: int = 1000

    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "healthmap-agent"

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- Databricks stack (Mosaic AI Vector Search + Agent Bricks / FM Serving) ---
    # Flip the orchestrator to the Databricks-hosted path. When False we use
    # FAISS + OpenAI (old local path) for dev / fallback.
    use_databricks: bool = False
    databricks_host: str = ""             # e.g. https://<workspace>.cloud.databricks.com
    databricks_token: str = ""            # PAT or service principal token
    dbx_catalog: str = "healthmap"
    dbx_schema: str = "facilities"
    dbx_table: str = "facilities"          # Delta table with canonical rows
    dbx_extractions_table: str = "facility_extractions"  # capability rows
    dbx_vs_endpoint: str = "healthmap-vs"
    dbx_vs_index: str = "facilities_vs_index"  # short name; full = catalog.schema.index
    dbx_vs_primary_key: str = "facility_id"
    dbx_vs_text_column: str = "notes"
    dbx_fm_endpoint: str = "databricks-meta-llama-3-1-70b-instruct"
    dbx_embed_endpoint: str = "databricks-bge-large-en"
    dbx_warehouse_id: str = ""             # SQL warehouse id for UC SQL queries

    @property
    def dbx_full_table(self) -> str:
        return f"{self.dbx_catalog}.{self.dbx_schema}.{self.dbx_table}"

    @property
    def dbx_full_extractions_table(self) -> str:
        return f"{self.dbx_catalog}.{self.dbx_schema}.{self.dbx_extractions_table}"

    @property
    def dbx_full_vs_index(self) -> str:
        return f"{self.dbx_catalog}.{self.dbx_schema}.{self.dbx_vs_index}"

    @property
    def data_root(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.data_raw_path

    @property
    def processed_path(self) -> Path:
        return self.data_root / "processed" / "hospitals.parquet"

    @property
    def index_path(self) -> Path:
        return self.data_root / "index" / "faiss.index"

    @property
    def index_meta_path(self) -> Path:
        return self.data_root / "index" / "faiss_meta.parquet"

    @property
    def extractions_path(self) -> Path:
        return self.data_root / "extracted" / "capabilities.parquet"

    @property
    def tavily_cache_dir(self) -> Path:
        return self.data_root / "tavily_cache"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
