"""Databricks-backed implementations of retrieval, LLM, tracing.

Only imported when `settings.use_databricks` is True. The module keeps the
heavy deps (databricks-sdk, databricks-vectorsearch) lazy so the old FAISS /
OpenAI path still works without them installed.
"""
