"""Golden retrieval eval.

Default unit run: only the offline fixture self-test runs (fake embedder).
Real packs: pass ``--golden <file|dir>`` (or EP_GOLDEN) and optionally
``--packs-root`` (or EP_PACKS_ROOT, default /workspace/expert-packs).
Thresholds: EP_EVAL_MIN="recall@5=0.8,mrr=0.6,must_not_violations=0".
Clock: EP_EVAL_TODAY=2026-10-20 pins freshness evaluation.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.eval.golden_eval import (
    aggregate,
    check_thresholds,
    find_pack_dir,
    format_report,
    load_golden,
    run_pack,
    score_query,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_score_query_metrics():
    q = {"id": "x", "expected": ["a.md", "b.md"], "must_not": ["z.md"]}
    row = score_query(q, ["c.md", "a.md", "z.md", "b.md"])
    assert row["rank"] == 2 and row["rr"] == 0.5
    assert row["r@1"] == 0 and row["r@3"] == 1 and row["full@3"] == 0.5 and row["full@5"] == 1
    assert row["must_not_hit"] == ["z.md"]
    oos = score_query({"id": "o", "expected": [], "accept": ["faq.md"]}, ["faq.md"])
    assert oos["oos"] and oos["oos_clean"]
    agg = aggregate([row, oos])
    assert agg["mrr"] == 0.5 and agg["must_not_violations"] == 1 and agg["oos_clean"] == 1.0


def test_fixture_pack_offline(monkeypatch):
    monkeypatch.setenv("EP_EVAL_PROVIDER", "fake")
    golden = load_golden(FIXTURES / "golden-fixture.yaml")[0]
    rep = asyncio.run(run_pack(golden, str(FIXTURES / "golden-fixture"), today="2026-10-20"))
    assert rep["overall"]["answerable"] == 3
    assert rep["overall"]["recall@3"] == 1.0
    assert not check_thresholds(rep, {"recall@3": 1.0})
    # 3-file pack: quota.md is always in the top-5 for fx-002, so the guard fires
    assert rep["overall"]["must_not_violations"] == 1
    assert check_thresholds(rep, {"must_not_violations": 0})
    assert "golden-fixture" in format_report([rep])


def _golden_params(config):
    src = config.getoption("--golden") or os.environ.get("EP_GOLDEN")
    return load_golden(src) if src else []


def pytest_generate_tests(metafunc):
    if "golden_set" in metafunc.fixturenames:
        sets = _golden_params(metafunc.config)
        metafunc.parametrize("golden_set", sets or [None], ids=[g["pack"] for g in sets] or ["none"])


@pytest.mark.golden
def test_golden_pack(golden_set, request):
    if golden_set is None:
        pytest.skip("No golden set: pass --golden or EP_GOLDEN")
    roots = [request.config.getoption("--packs-root") or os.environ.get("EP_PACKS_ROOT")
             or "/workspace/expert-packs"]
    pack_dir = find_pack_dir(golden_set["pack"], roots)
    if not pack_dir:
        pytest.skip(f"pack {golden_set['pack']} not found under {roots}")
    if os.environ.get("EP_EVAL_PROVIDER", "").lower() != "fake" and not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    rep = asyncio.run(run_pack(golden_set, pack_dir, today=os.environ.get("EP_EVAL_TODAY")))
    print("\n" + format_report([rep]))
    mins = os.environ.get("EP_EVAL_MIN", "")
    thresholds = {k: float(v) for k, v in (p.split("=", 1) for p in mins.split(",") if "=" in p)}
    fails = check_thresholds(rep, thresholds)
    assert not fails, fails
