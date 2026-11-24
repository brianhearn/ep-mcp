"""Standard OpenAI and OpenAI-compatible (Ollama, LM Studio, vLLM) embedding provider."""

from __future__ import annotations

import logging
import os

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the OpenAI Embeddings API or any compatible base_url."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        output_dimensionality: int | None = None,
        batch_size: int = 64,
    ):
        from openai import AsyncOpenAI

        self._model = model
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            if self._base_url:
                # Local OpenAI-compatible servers (e.g. Ollama) often accept any placeholder key
                resolved_key = "local-no-key"
            else:
                raise ValueError(
                    "OPENAI_API_KEY is required when embedding.provider='openai' "
                    "(or set openai_base_url for a local OpenAI-compatible endpoint)."
                )
        self._output_dimensionality = output_dimensionality
        self._dim = output_dimensionality or _DEFAULT_DIMS.get(model, 1536)
        self._batch_size = batch_size
        self._client = AsyncOpenAI(api_key=resolved_key, base_url=self._base_url)

    @property
    def model_name(self) -> str:
        prefix = "openai-compat" if self._base_url else "openai"
        if self._output_dimensionality:
            return f"{prefix}/{self._model}@{self._output_dimensionality}d"
        return f"{prefix}/{self._model}"

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            kwargs: dict = {"model": self._model, "input": batch}
            if self._output_dimensionality and self._model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self._output_dimensionality
            resp = await self._client.embeddings.create(**kwargs)
            ordered = sorted(resp.data, key=lambda item: item.index)
            for item in ordered:
                vec = list(item.embedding)
                if len(vec) != self._dim:
                    self._dim = len(vec)
                all_vectors.append(vec)
        return all_vectors
