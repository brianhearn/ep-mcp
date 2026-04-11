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
        # Subsequent chunks should have title prefix
        for chunk in chunks[1:]:
            assert chunk.content.startswith("# Big Document")

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
