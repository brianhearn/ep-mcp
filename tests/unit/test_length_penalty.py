"""Tests for apply_length_penalty in scorer.py."""

import pytest
from ep_mcp.retrieval.scorer import apply_length_penalty


def _chunk(content: str) -> dict:
    return {"content": content}


def test_short_chunk_is_penalized():
    chunks = {1: _chunk("short")}  # 5 chars, well below default 80
    scores = {1: 1.0}
    result = apply_length_penalty(scores, chunks, short_threshold=80, short_penalty=0.15)
    assert result[1] == pytest.approx(0.85, rel=1e-6)


def test_long_chunk_is_not_penalized():
    long_text = "x" * 200
    chunks = {1: _chunk(long_text)}
    scores = {1: 1.0}
    result = apply_length_penalty(scores, chunks, short_threshold=80, short_penalty=0.15)
    assert result[1] == pytest.approx(1.0, rel=1e-6)


def test_exactly_at_threshold_is_not_penalized():
    text = "x" * 80  # exactly at threshold, not below
    chunks = {1: _chunk(text)}
    scores = {1: 0.9}
    result = apply_length_penalty(scores, chunks, short_threshold=80, short_penalty=0.15)
    assert result[1] == pytest.approx(0.9, rel=1e-6)


def test_one_below_threshold_is_penalized():
    text = "x" * 79
    chunks = {1: _chunk(text)}
    scores = {1: 0.9}
    result = apply_length_penalty(scores, chunks, short_threshold=80, short_penalty=0.15)
    assert result[1] == pytest.approx(0.9 * 0.85, rel=1e-6)


def test_mixed_chunks():
    chunks = {
        1: _chunk("tiny"),         # short → penalized
        2: _chunk("x" * 150),     # long → not penalized
        3: _chunk("x" * 50),      # short → penalized
    }
    scores = {1: 1.0, 2: 0.8, 3: 0.6}
    result = apply_length_penalty(scores, chunks, short_threshold=80, short_penalty=0.15)
    assert result[1] == pytest.approx(0.85, rel=1e-6)
    assert result[2] == pytest.approx(0.80, rel=1e-6)
    assert result[3] == pytest.approx(0.51, rel=1e-6)


def test_missing_chunk_id_is_skipped():
    chunks = {}  # no matching chunks
    scores = {99: 1.0}
    result = apply_length_penalty(scores, chunks, short_threshold=80, short_penalty=0.15)
    assert result[99] == pytest.approx(1.0, rel=1e-6)  # unchanged


def test_empty_scores():
    result = apply_length_penalty({}, {})
    assert result == {}


def test_custom_threshold_and_penalty():
    chunks = {1: _chunk("x" * 30)}  # 30 chars
    scores = {1: 1.0}
    # threshold=50, penalty=0.20 → 30 < 50, so penalized
    result = apply_length_penalty(scores, chunks, short_threshold=50, short_penalty=0.20)
    assert result[1] == pytest.approx(0.80, rel=1e-6)
