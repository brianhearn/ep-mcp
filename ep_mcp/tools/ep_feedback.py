"""ep_feedback MCP tool — append-only gap/staleness backlog for pack maintainers (M4).

Never modifies pack Markdown files directly. Appends structured JSONL records
to a local feedback queue for human / hydration review.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..pack.models import Pack

logger = logging.getLogger(__name__)

VALID_KINDS = frozenset({"missing", "stale", "incorrect", "other"})


def ep_feedback(
    pack: Pack,
    kind: str,
    detail: str,
    *,
    path: str | None = None,
    id: str | None = None,
    reason: str | None = None,
    user_goal: str | None = None,
    feedback_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Append a structured feedback item to the pack's local JSONL hydration queue."""
    norm_kind = (kind or "").strip().lower()
    if norm_kind not in VALID_KINDS:
        return {
            "error": f"Invalid kind {kind!r}. Must be one of: {', '.join(sorted(VALID_KINDS))}",
        }

    detail_clean = (detail or "").strip()
    if not detail_clean:
        return {"error": "'detail' must be a non-empty description of the gap or issue."}

    if feedback_dir is not None:
        queue_path = Path(feedback_dir).resolve() / f"{pack.slug}.jsonl"
    else:
        # Default alongside index inside .ep-mcp/ so pack content stays untouched
        base = Path(pack.index_path).parent if pack.index_path else Path(pack.pack_dir) / ".ep-mcp"
        queue_path = base / "feedback.jsonl"

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(timezone.utc).isoformat()
    record = {
        "recorded_at": recorded_at,
        "pack": pack.slug,
        "pack_version": pack.version,
        "kind": norm_kind,
        "path": path,
        "id": id,
        "detail": detail_clean,
        "reason": reason,
        "user_goal": user_goal,
    }
    with open(queue_path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("ep_feedback recorded | pack=%s kind=%s path=%r", pack.slug, norm_kind, path)
    return {
        "status": "recorded",
        "pack": pack.slug,
        "pack_version": pack.version,
        "kind": norm_kind,
        "recorded_at": recorded_at,
        "queue_file": str(queue_path),
    }
