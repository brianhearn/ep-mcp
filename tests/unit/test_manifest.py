"""Tests for manifest.yaml parsing."""

import tempfile
from pathlib import Path

import pytest

from ep_mcp.pack.manifest import ManifestError, parse_manifest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_manifest(tmp_dir: Path, content: str) -> Path:
    p = tmp_dir / "manifest.yaml"
    p.write_text(content, encoding="utf-8")
    return p


class TestParseManifest:
    def test_valid_minimal(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: test-pack
name: Test Pack
type: product
""")
        m = parse_manifest(path)
        assert m.slug == "test-pack"
        assert m.name == "Test Pack"
        assert m.type == "product"
        assert m.version == "1.0.0"  # default
        assert m.entry_point == "overview.md"  # default

    def test_valid_full(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: ezt-designer
name: EasyTerritory Designer
type: product
version: "2.0.0"
description: Territory planning tool
entry_point: overview.md
schema_version: "3.1"
context:
  always:
    - overview.md
    - concepts/core.md
  on_demand:
    - faq/general.md
freshness:
  refresh_cycle: monthly
  coverage_pct: 100
  last_full_review: "2026-04-10"
""")
        m = parse_manifest(path)
        assert m.slug == "ezt-designer"
        assert m.type == "product"
        assert m.version == "2.0.0"
        assert m.schema_version == "3.1"
        assert m.context.always == ["overview.md", "concepts/core.md"]
        assert m.context.on_demand == ["faq/general.md"]
        assert m.freshness.coverage_pct == 100
        assert m.freshness.last_full_review == "2026-04-10"
        assert isinstance(m.raw, dict)

    def test_missing_file(self, tmp_dir):
        with pytest.raises(ManifestError, match="not found"):
            parse_manifest(tmp_dir / "manifest.yaml")

    def test_missing_slug(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
name: Test
type: product
""")
        with pytest.raises(ManifestError, match="slug"):
            parse_manifest(path)

    def test_missing_name(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: test
type: product
""")
        with pytest.raises(ManifestError, match="name"):
            parse_manifest(path)

    def test_missing_type(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: test
name: Test
""")
        with pytest.raises(ManifestError, match="type"):
            parse_manifest(path)

    def test_invalid_type(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: test
name: Test
type: invalid
""")
        with pytest.raises(ManifestError, match="Invalid pack type"):
            parse_manifest(path)

    def test_invalid_yaml(self, tmp_dir):
        path = _write_manifest(tmp_dir, "key: [unclosed\n  - nested: {bad")
        with pytest.raises(ManifestError, match="Invalid YAML"):
            parse_manifest(path)

    def test_non_mapping(self, tmp_dir):
        path = _write_manifest(tmp_dir, "- just a list")
        with pytest.raises(ManifestError, match="must be a YAML mapping"):
            parse_manifest(path)

    def test_person_type(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: brian
name: Brian GPT
type: person
""")
        m = parse_manifest(path)
        assert m.type == "person"

    def test_process_type(self, tmp_dir):
        path = _write_manifest(tmp_dir, """
slug: solar-install
name: Solar Install
type: process
""")
        m = parse_manifest(path)
        assert m.type == "process"

    def test_context_single_string(self, tmp_dir):
        """Context tiers should handle single string as well as list."""
        path = _write_manifest(tmp_dir, """
slug: test
name: Test
type: product
context:
  always: overview.md
""")
        m = parse_manifest(path)
        assert m.context.always == ["overview.md"]
