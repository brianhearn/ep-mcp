"""Tests for EP-native chunking."""

import pytest

from ep_mcp.index.chunker import Chunk, chunk_file, estimate_tokens


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_simple(self):
        text = "hello world foo bar baz"
        tokens = estimate_tokens(text)
        assert tokens == int(5 * 1.3)  # 5 words × 1.3

    def test_multiline(self):
        text = "line one\nline two\nline three"
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestChunkFile:
    def test_small_file_single_chunk(self):
        content = "# Topic\n\nShort content about a topic."
        chunks = chunk_file("test.md", content, "Topic")
        assert len(chunks) == 1
        assert chunks[0].file_path == "test.md"
        assert chunks[0].chunk_index == 0
        assert chunks[0].content == content
        assert chunks[0].title == "Topic"

    def test_file_under_threshold(self):
        # ~50 words, well under 1000 tokens
        content = "# Overview\n\n" + ("This is a word. " * 50)
        chunks = chunk_file("overview.md", content, "Overview", max_tokens=1000)
        assert len(chunks) == 1

    def test_oversized_splits_at_headings(self):
        # Build content with multiple ## sections that exceeds threshold
        sections = []
        for i in range(10):
            sections.append(f"## Section {i}\n\n" + ("Word " * 100))
        content = "# Big Document\n\nIntro paragraph.\n\n" + "\n\n".join(sections)
        
        chunks = chunk_file("big.md", content, "Big Document", max_tokens=200)
        assert len(chunks) > 1
        # First chunk should have the intro
        # 0.7.0: title/heading-path prefix lives in indexed_content only;
        # content and span_hash stay the original span (brief 1.3).
        for chunk in chunks[1:]:
            assert chunk.content.startswith("## Section")
            assert chunk.text_for_index().startswith("# Big Document\n\n## Section")

    def test_oversized_no_headings_single_chunk(self):
        # Long content with no headings — can't split, returns single chunk
        content = "Word " * 1000
        chunks = chunk_file("flat.md", content, "Flat", max_tokens=200)
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        sections = []
        for i in range(5):
            sections.append(f"## Section {i}\n\n" + ("Word " * 100))
        content = "\n\n".join(sections)

        chunks = chunk_file("multi.md", content, "Multi", max_tokens=100)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_content(self):
        chunks = chunk_file("empty.md", "", "Empty")
        assert len(chunks) == 1
        assert chunks[0].token_count == 0

    def test_preserves_file_path(self):
        chunks = chunk_file("concepts/topic-a.md", "# Topic A\n\nContent.", "Topic A")
        assert chunks[0].file_path == "concepts/topic-a.md"


class TestSplitSpans:
    """0.7.0: split chunks keep original spans, line ranges and heading paths."""

    def _doc(self):
        body = (
            "# Guide\n\nIntro text.\n\n## Setup\n\n" + "alpha " * 120 + "\n\n"
            "### Ports\n\n" + "beta " * 120 + "\n\n## Usage\n\n" + "gamma " * 120 + "\n"
        )
        raw = "---\ntitle: Guide\ntype: concept\n---\n" + body
        return raw, body

    def test_line_range_and_span(self):
        from ep_mcp.pack.sidecar import extract_line_range_text, span_sha256
        raw, body = self._doc()
        chunks = chunk_file("g.md", body, "Guide", max_tokens=100, raw_content=raw)
        assert len(chunks) == 4
        for ch in chunks:
            assert ch.line_range is not None
            span = extract_line_range_text(raw, ch.line_range).strip("\n")
            assert span.strip() == ch.content
            assert ch.span_hash == span_sha256(ch.content)

    def test_heading_path(self):
        raw, body = self._doc()
        chunks = chunk_file("g.md", body, "Guide", max_tokens=100, raw_content=raw)
        paths = [c.heading_path for c in chunks]
        assert paths == [["Guide"], ["Guide", "Setup"], ["Guide", "Setup", "Ports"], ["Guide", "Usage"]]
        assert chunks[2].indexed_content.startswith("# Guide\n\n### Ports")
        assert chunks[0].indexed_content is None
        full = chunk_file("g.md", body, "Guide", max_tokens=100, raw_content=raw, prefix_heading_path=True)
        assert full[2].indexed_content.startswith("# Guide > Setup > Ports\n\n### Ports")
        assert full[2].content == chunks[2].content

    def test_heading_path_ignores_code_fences(self):
        from ep_mcp.index.chunker import heading_path_at
        raw = "# T\n\n## A\n\n```\n## not a heading\n```\n\ntext\n"
        assert heading_path_at(raw, 9, "T") == ["T", "A"]
