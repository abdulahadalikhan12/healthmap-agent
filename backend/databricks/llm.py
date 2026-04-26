"""Chat + embeddings via Databricks Model Serving endpoints.

Drop-in for `backend.core.llm.chat_json` / `chat_text` / `embed`. The
orchestrator calls these through `core.llm`, which dispatches here when
`settings.use_databricks` is True.
"""
from __future__ import annotations

import json
from typing import Any

from backend.config import settings
from backend.databricks.client import serving_client


def _extract_message_content(resp: Any) -> str:
    choices: list[Any] = getattr(resp, "choices", None) or resp.get("choices", [])  # type: ignore[union-attr]
    if not choices:
        return ""
    first = choices[0]
    msg = getattr(first, "message", None) or first.get("message", {})
    content = getattr(msg, "content", None) or msg.get("content", "")
    return str(content or "")


def chat_json(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> dict[str, Any]:
    """Run a chat completion that must return JSON. Mirrors the OpenAI path so
    callers (extraction_agent, query_agent, validator_agent, trust_agent) need
    no changes."""
    endpoint = model or settings.dbx_fm_endpoint
    client = serving_client()
    try:
        resp = client.query(
            name=endpoint,
            messages=[
                {"role": "system", "content": "Return only valid JSON. No prose."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params={"response_format": {"type": "json_object"}},
        )
    except Exception:
        return {}
    content = _extract_message_content(resp) or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip().lstrip("```json").lstrip("```").rstrip("```")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        for cutoff in range(len(cleaned), 0, -1):
            prefix = cleaned[:cutoff].rstrip().rstrip(",")
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
    endpoint = model or settings.dbx_fm_endpoint
    client = serving_client()
    try:
        resp = client.query(
            name=endpoint,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        return ""
    return _extract_message_content(resp).strip()


def embed(texts: list[str], *, model: str | None = None, batch_size: int = 96) -> list[list[float]]:
    if not texts:
        return []
    endpoint = model or settings.dbx_embed_endpoint
    client = serving_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            resp = client.query(name=endpoint, input=chunk)
        except Exception:
            out.extend([[] for _ in chunk])
            continue
        data = getattr(resp, "data", None) or resp.get("data", [])  # type: ignore[union-attr]
        for row in data:
            emb = getattr(row, "embedding", None) or row.get("embedding")
            out.append(list(emb or []))
    return out
