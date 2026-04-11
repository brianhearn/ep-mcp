"""Tests for server configuration."""

import tempfile
from pathlib import Path

import pytest

from ep_mcp.config import ServerConfig, load_config


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_config(tmp_dir: Path, content: str) -> Path:
    p = tmp_dir / "config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


class TestLoadConfig:
    def test_valid_config(self, tmp_dir):
        path = _write_config(tmp_dir, """
server:
  host: "0.0.0.0"
  port: 8100

packs:
  - slug: "ezt-designer"
    path: "/data/packs/ezt-designer"
    api_keys: ["key_abc123"]

embedding:
  provider: "gemini"
  model: "gemini-embedding-001"

retrieval:
  vector_weight: 0.7
  text_weight: 0.3
  mmr_enabled: true
  mmr_lambda: 0.7
""")
        config = load_config(path)
        assert config.host == "0.0.0.0"
        assert config.port == 8100
        assert len(config.packs) == 1
        assert config.packs[0].slug == "ezt-designer"
        assert config.packs[0].api_keys == ["key_abc123"]
        assert config.embedding.provider == "gemini"
        assert config.retrieval.vector_weight == 0.7
        assert config.retrieval.mmr_lambda == 0.7

    def test_defaults(self, tmp_dir):
        path = _write_config(tmp_dir, """
packs:
  - slug: "test"
    path: "/tmp/test"
""")
        config = load_config(path)
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.embedding.provider == "gemini"
        assert config.retrieval.vector_weight == 0.7
        assert config.retrieval.min_score == 0.35
        assert config.retrieval.default_max_results == 10

    def test_multiple_packs(self, tmp_dir):
        path = _write_config(tmp_dir, """
packs:
  - slug: "pack-a"
    path: "/data/a"
    api_keys: ["key1"]
  - slug: "pack-b"
    path: "/data/b"
    api_keys: ["key2", "key3"]
""")
        config = load_config(path)
        assert len(config.packs) == 2
        assert config.packs[1].api_keys == ["key2", "key3"]

    def test_azure_embedding(self, tmp_dir):
        path = _write_config(tmp_dir, """
packs:
  - slug: "test"
    path: "/tmp/test"
embedding:
  provider: "azure_openai"
  model: "text-embedding-3-small"
  azure_endpoint: "https://myresource.openai.azure.com"
  azure_deployment: "text-embedding-3-small"
""")
        config = load_config(path)
        assert config.embedding.provider == "azure_openai"
        assert config.embedding.azure_endpoint == "https://myresource.openai.azure.com"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_dir):
        path = _write_config(tmp_dir, "key: [unclosed\n  - nested: {bad")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(path)

    def test_non_mapping(self, tmp_dir):
        path = _write_config(tmp_dir, "- just a list")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_config(path)
