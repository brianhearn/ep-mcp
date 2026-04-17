"""Ollama local embedding provider."""

from __future__ import annotations

import asyncio
import logging

import httpx

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_BATCH_SIZE = 32          # texts per /api/embed call (safe for CPU)
_REQUEST_TIMEOUT = 120.0  # seconds per batch request


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local Ollama embedding provider.

    Uses the Ollama /api/embed endpoint in sequential batches of 32 texts.
    Zero network overhead vs. remote APIs — typical query latency on a
    2-vCPU VM is 30-150ms vs ~3s for Gemini.

    The model must be pulled before use: ``ollama pull nomic-embed-text``
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim_map: dict[str, int] = {
            "nomic-embed-text": 768,
            "nomic-embed-text:latest": 768,
            "mxbai-embed-large": 1024,
            "mxbai-embed-large:latest": 1024,
            "bge-m3": 1024,
            "all-minilm": 384,
        }
        self._dimension: int | None = self._dim_map.get(model)

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                f"Unknown dimension for Ollama model '{self._model}'. "
                "Add it to the _dim_map or set embedding.dimension in config."
            )
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in sequential batches via Ollama /api/embed."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            for batch_idx in range(total_batches):
                batch = texts[batch_idx * _BATCH_SIZE : (batch_idx + 1) * _BATCH_SIZE]
                if total_batches > 1:
                    logger.info(
                        "Ollama embed batch %d/%d (%d texts)",
                        batch_idx + 1, total_batches, len(batch),
                    )
                try:
                    resp = await client.post(
                        f"{self._base_url}/api/embed",
                        json={"model": self._model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and self._dimension is None:
                        self._dimension = len(embeddings[0])
                    all_embeddings.extend(embeddings)
                except httpx.ConnectError as e:
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self._base_url}. "
                        "Is Ollama running? Try: systemctl start ollama"
                    ) from e
                except httpx.ReadTimeout as e:
                    raise RuntimeError(
                        f"Ollama embed timed out on batch {batch_idx+1}/{total_batches}. "
                        f"Try reducing batch size or increasing timeout."
                    ) from e

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query via /api/embeddings (fast path)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": query},
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = data["embedding"]
                if self._dimension is None:
                    self._dimension = len(embedding)
                return embedding
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Try: systemctl start ollama"
                ) from e
