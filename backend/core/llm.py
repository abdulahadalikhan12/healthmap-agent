"""LLM client. Dispatches to OpenAI (local dev) or Databricks Model Serving
(production, when `USE_DATABRICKS=1`). API is identical either way so
everything upstream of this module is provider-agnostic.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from backend.config import settings


_client: OpenAI | None = None


def _use_dbx() -> bool:
    return bool(settings.use_databricks)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def chat_json(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> dict[str, Any]:
    if _use_dbx():
        from backend.databricks.llm import chat_json as _dbx_chat_json

        return _dbx_chat_json(
            prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
    """Run a chat completion that must return JSON.

    We force `response_format=json_object` so the model returns parseable
    JSON. Caller is responsible for shape validation (Pydantic).

    Robust: returns an empty dict on parse failure (e.g. truncated output)
    rather than raising. Callers must handle missing keys gracefully.
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model or settings.openai_llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only valid JSON. No prose."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Strip optional code fences and try again.
    cleaned = content.strip().lstrip("```json").lstrip("```").rstrip("```")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Last resort: try to recover the leading well-formed prefix
    # (truncated outputs often only break in the trailing 'evidence' block).
    for cutoff in range(len(cleaned), 0, -1):
        prefix = cleaned[:cutoff].rstrip().rstrip(",")
        # Try closing with a brace.
        candidate = prefix + "}" if not prefix.endswith("}") else prefix
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def chat_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> str:
    """Plain text completion (used by ranking + trace simplification)."""
    if _use_dbx():
        from backend.databricks.llm import chat_text as _dbx_chat_text

        return _dbx_chat_text(
            prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
    client = get_client()
    resp = client.chat.completions.create(
        model=model or settings.openai_llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def embed(texts: list[str], *, model: str | None = None, batch_size: int = 256) -> list[list[float]]:
    """Embed a list of texts. Batches automatically."""
    if _use_dbx():
        from backend.databricks.llm import embed as _dbx_embed

        return _dbx_embed(texts, model=model, batch_size=min(batch_size, 96))
    client = get_client()
    out: list[list[float]] = []
    model_name = model or settings.openai_embed_model
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return out
