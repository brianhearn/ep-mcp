"""Golden retrieval eval for ep-mcp.

Golden file format (one per pack)::

    pack: <slug>
    queries:
      - id: <slug>-001
        query: "..."
        category: direct | multi-hop | subsection | out-of-scope | volatile   # optional
        expected: [paths best-first]   # [] = out-of-scope, no answer atom expected
        must_not: [paths]              # optional; violation if any appears in top-5
        accept: [paths]                # optional; allowed redirect atoms for out-of-scope
        notes: "..."

Metrics (file-level; results are deduped by source_file in rank order):
  recall@k   — share of answerable queries with ANY expected atom in top-k
  full@k     — mean share of expected atoms present in top-k (multi-atom queries)
  mrr        — mean reciprocal rank of the first expected atom (0 if absent from top-10)
  must_not   — count of queries with a must_not atom in top-5
  oos_clean  — share of out-of-scope queries whose top-3 holds only `accept` atoms (or nothing)

Usable both from pytest (tests/eval/test_golden.py) and as a CLI against any
ep-mcp checkout (so baseline and branch can be compared with the same harness)::

    PYTHONPATH=<ep-mcp checkout> python tests/eval/golden_eval.py \
        --golden /path/golden --packs-root /path/packs --today 2026-10-20 --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

KS = (1, 3, 5)


def load_golden(path: str | Path) -> list[dict]:
    p = Path(path)
    files = sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    out = []
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict) or "pack" not in data:
            continue
        data["_file"] = str(f)
        data.setdefault("queries", [])
        out.append(data)
    return out


def score_query(q: dict, ranked: list[str]) -> dict:
    expected = list(q.get("expected") or [])
    must_not = set(q.get("must_not") or [])
    accept = set(q.get("accept") or [])
    row = {"id": q.get("id"), "category": q.get("category") or "uncategorized",
           "query": q.get("query"), "top": ranked[:5], "expected": expected}
    row["must_not_hit"] = sorted(must_not & set(ranked[:5]))
    if not expected:
        row["oos"] = True
        row["oos_clean"] = all(r in accept for r in ranked[:3])
        return row
    row["oos"] = False
    rank = next((i + 1 for i, r in enumerate(ranked[:10]) if r in expected), None)
    row["rank"] = rank
    row["rr"] = 1.0 / rank if rank else 0.0
    for k in KS:
        top = set(ranked[:k])
        row[f"r@{k}"] = 1.0 if top & set(expected) else 0.0
        row[f"full@{k}"] = len(top & set(expected)) / len(expected)
    return row


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if not r["oos"]]
    oos = [r for r in rows if r["oos"]]
    agg: dict = {"n": len(rows), "answerable": len(ans), "out_of_scope": len(oos)}
    for k in KS:
        agg[f"recall@{k}"] = round(sum(r[f"r@{k}"] for r in ans) / len(ans), 4) if ans else None
        agg[f"full@{k}"] = round(sum(r[f"full@{k}"] for r in ans) / len(ans), 4) if ans else None
    agg["mrr"] = round(sum(r["rr"] for r in ans) / len(ans), 4) if ans else None
    agg["must_not_violations"] = sum(1 for r in rows if r["must_not_hit"])
    agg["oos_clean"] = round(sum(r["oos_clean"] for r in oos) / len(oos), 4) if oos else None
    return agg


def by_category(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    return {c: aggregate(g) for c, g in sorted(groups.items())}


def _make_provider(dimension_hint: int | None = None):
    """Gemini by default; EP_EVAL_PROVIDER=fake gives a deterministic offline provider."""
    if os.environ.get("EP_EVAL_PROVIDER", "").lower() == "fake":
        return _FakeProvider()
    from ep_mcp.config import EmbeddingConfig, ServerConfig
    from ep_mcp.server import create_embedding_provider

    return create_embedding_provider(ServerConfig(embedding=EmbeddingConfig()))


class _FakeProvider:
    """Hashing bag-of-words embedder (offline fixture runs only)."""

    model_name = "fake-hash-64"
    dimension = 64

    def _vec(self, text: str) -> list[float]:
        import hashlib
        import math
        v = [0.0] * self.dimension
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        return self._vec(text)


async def run_pack(golden: dict, pack_dir: str, today: str | None = None,
                   max_results: int = 10, seed_index: bool = True) -> dict:
    from ep_mcp.config import RetrievalConfig
    from ep_mcp.index.manager import IndexManager
    from ep_mcp.index.sqlite_store import SQLiteStore
    from ep_mcp.pack.loader import load_pack
    from ep_mcp.retrieval.engine import RetrievalEngine
    from ep_mcp.retrieval.graph_helpers import GraphLookup
    from ep_mcp.tools.ep_search import ep_search

    provider = _make_provider()
    with tempfile.TemporaryDirectory(prefix="ep-eval-") as tmp:
        # Seed with the pack's existing index so its embedding cache is reused
        # (never written back: the pack directory is not modified).
        existing = Path(pack_dir) / ".ep-mcp" / "index.db"
        if seed_index and existing.exists() and getattr(provider, "model_name", "") != "fake-hash-64":
            shutil.copy(existing, Path(tmp) / "index.db")
        pack = load_pack(pack_dir, index_dir_override=tmp)
        store = SQLiteStore(pack.index_path, embedding_dimension=provider.dimension)
        store.open()
        try:
            stats = await IndexManager(pack, store, provider).build_index()
            cfg_kwargs = {}
            if today and "freshness_today" in RetrievalConfig.model_fields:
                cfg_kwargs["freshness_today"] = today
            engine = RetrievalEngine(pack, store, provider, RetrievalConfig(**cfg_kwargs),
                                     GraphLookup.from_pack(pack))
            rows = []
            for q in golden["queries"]:
                hits = await ep_search(engine, q["query"], max_results=max_results)
                ranked: list[str] = []
                for h in hits:
                    if h.get("requires_expanded"):
                        continue  # appended dependencies are not ranked hits
                    if h["source_file"] not in ranked:
                        ranked.append(h["source_file"])
                row = score_query(q, ranked)
                row["heading_paths"] = [h.get("heading_path") for h in hits[:3]]
                rows.append(row)
        finally:
            store.close()
    import ep_mcp
    return {
        "pack": golden["pack"],
        "ep_mcp_version": getattr(ep_mcp, "__version__", "?"),
        "index": repr(stats),
        "overall": aggregate(rows),
        "by_category": by_category(rows),
        "rows": rows,
    }


def find_pack_dir(slug: str, roots: list[str]) -> str | None:
    for root in roots:
        cand = Path(root) / slug
        if (cand / "manifest.yaml").exists():
            return str(cand)
    return None


def check_thresholds(report: dict, thresholds: dict) -> list[str]:
    fails = []
    for key, minimum in (thresholds or {}).items():
        val = report["overall"].get(key)
        if key == "must_not_violations":
            if val is not None and val > minimum:
                fails.append(f"{report['pack']}: {key}={val} > {minimum}")
        elif val is not None and val < minimum:
            fails.append(f"{report['pack']}: {key}={val} < {minimum}")
    return fails


def format_report(reports: list[dict]) -> str:
    cols = ["recall@1", "recall@3", "recall@5", "mrr", "full@5", "must_not_violations", "oos_clean"]
    lines = [f"{'pack/category':32} {'n':>3} " + " ".join(f"{c[:9]:>9}" for c in cols)]

    def fmt(name, agg):
        vals = [agg.get(c) for c in cols]
        return f"{name:32} {agg['n']:>3} " + " ".join(
            f"{'-' if v is None else (f'{v:.3f}' if isinstance(v, float) else v):>9}" for v in vals)

    for rep in reports:
        lines.append(fmt(f"{rep['pack']} (v{rep['ep_mcp_version']})", rep["overall"]))
        for cat, agg in rep["by_category"].items():
            lines.append(fmt(f"  {cat}", agg))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", required=True)
    ap.add_argument("--packs-root", action="append", default=[])
    ap.add_argument("--today", default=None, help="Pin freshness clock (YYYY-MM-DD)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--min", action="append", default=[], help="threshold, e.g. recall@5=0.8")
    args = ap.parse_args(argv)
    roots = args.packs_root or ["/workspace/expert-packs"]
    thresholds = {k: float(v) for k, v in (m.split("=", 1) for m in args.min)}
    reports, fails = [], []
    for g in load_golden(args.golden):
        pd = find_pack_dir(g["pack"], roots)
        if not pd:
            print(f"skip {g['pack']}: pack not found under {roots}", file=sys.stderr)
            continue
        rep = asyncio.run(run_pack(g, pd, today=args.today))
        reports.append(rep)
        fails += check_thresholds(rep, thresholds)
    print(format_report(reports))
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    for f in fails:
        print("THRESHOLD FAIL:", f, file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
