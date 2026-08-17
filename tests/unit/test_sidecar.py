"""Tests for RFC-004 chunk sidecar loading and sidecar-driven chunking."""

from pathlib import Path

import pytest
import yaml

from ep_mcp.index.chunker import chunk_file
from ep_mcp.pack.sidecar import load_sidecar, sidecar_path_for_md, span_sha256


FIXTURE_MD = """---
title: Sample Workflow
type: workflow
id: demo/workflows/sample
content_hash: sha256:abc
---
<!-- context -->

# Sample Workflow

Opening paragraph for retrieval anchor.

## First Section

Section one content with enough words to matter.

## Second Section

Section two content continues here.
"""


@pytest.fixture
def sidecar_fixture(tmp_path: Path):
    md_path = tmp_path / "workflows" / "sample.md"
    md_path.parent.mkdir(parents=True)
    md_path.write_text(FIXTURE_MD, encoding="utf-8")

    body_start = FIXTURE_MD.index("# Sample Workflow") + 0
    lines = FIXTURE_MD.splitlines()
    first_section_line = next(i for i, line in enumerate(lines, start=1) if line.startswith("## First"))
    second_section_line = next(i for i, line in enumerate(lines, start=1) if line.startswith("## Second"))

    sidecar_data = {
        "schema_version": "1.0",
        "source_id": "demo/workflows/sample",
        "content_hash": f"sha256:{span_sha256(FIXTURE_MD.split('---', 2)[2].lstrip())}",
        "generated_by": "test-fixture",
        "chunks": [
            {
                "chunk_id": "sample--opening",
                "chunk_order": 0,
                "section": None,
                "line_range": [10, first_section_line - 1],
            },
            {
                "chunk_id": "sample--first-section",
                "chunk_order": 1,
                "section": "First Section",
                "line_range": [first_section_line, second_section_line - 1],
                "context_prefix": "From Sample Workflow, section First Section.",
            },
            {
                "chunk_id": "sample--second-section",
                "chunk_order": 2,
                "section": "Second Section",
                "line_range": [second_section_line, len(lines)],
            },
        ],
    }
    sidecar_path = md_path.with_suffix(".chunks.yaml")
    sidecar_path.write_text(yaml.safe_dump(sidecar_data, sort_keys=False), encoding="utf-8")
    return md_path, sidecar_path


class TestSidecarLoader:
    def test_find_default_sidecar_path(self, sidecar_fixture):
        md_path, sidecar_path = sidecar_fixture
        found = sidecar_path_for_md(md_path.parent.parent, "workflows/sample.md")
        assert found == sidecar_path

    def test_load_sidecar(self, sidecar_fixture):
        _, sidecar_path = sidecar_fixture
        sidecar = load_sidecar(sidecar_path)
        assert sidecar is not None
        assert sidecar.source_id == "demo/workflows/sample"
        assert len(sidecar.chunks) == 3
        assert sidecar.chunks[0].line_range[0] == 10


class TestSidecarChunking:
    def test_sidecar_boundaries_are_authoritative(self, sidecar_fixture):
        md_path, sidecar_path = sidecar_fixture
        raw = md_path.read_text(encoding="utf-8")
        content = raw.split("---", 2)[2].lstrip()
        sidecar = load_sidecar(sidecar_path)
        chunks = chunk_file(
            "workflows/sample.md",
            content,
            "Sample Workflow",
            raw_content=raw,
            sidecar=sidecar,
        )

        assert len(chunks) == 3
        assert chunks[0].section_slug == "opening"
        assert chunks[1].section_slug == "first-section"
        assert chunks[1].line_range == sidecar.chunks[1].line_range
        assert "Section one content" in chunks[1].content
        assert chunks[1].span_hash == span_sha256(chunks[1].content)
        assert chunks[1].indexed_content is not None
        assert chunks[1].indexed_content.startswith("From Sample Workflow")
        assert "Section one content" in chunks[1].indexed_content
        assert not chunks[1].content.startswith("From Sample Workflow")
        assert chunks[0].indexed_content is None

    def test_prefix_does_not_change_span_hash(self, sidecar_fixture):
        md_path, sidecar_path = sidecar_fixture
        raw = md_path.read_text(encoding="utf-8")
        content = raw.split("---", 2)[2].lstrip()
        sidecar = load_sidecar(sidecar_path)
        chunks = chunk_file(
            "workflows/sample.md",
            content,
            "Sample Workflow",
            raw_content=raw,
            sidecar=sidecar,
        )
        assert chunks[1].span_hash == span_sha256(chunks[1].content)
        assert chunks[1].text_for_index() != chunks[1].content
