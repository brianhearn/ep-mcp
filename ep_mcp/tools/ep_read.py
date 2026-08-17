"""ep_read MCP tool — load a whole atom by path or provenance id."""

from __future__ import annotations

import logging

from ..pack.models import Pack, PackFile

logger = logging.getLogger(__name__)


def ep_read(
    pack: Pack,
    path: str | None = None,
    id: str | None = None,
    reconstruct: bool = False,
) -> dict:
    """Read a whole ExpertPack atom.

    Args:
        pack: Loaded pack
        path: Pack-relative file path (forward slashes)
        id: Provenance id (e.g. pack/concepts/topic)
        reconstruct: Include raw markdown and provenance envelope

    Returns:
        Atom payload, or an error dict if the file cannot be resolved.
    """
    if not path and not id:
        return {"error": "Provide path or id"}

    pack_file = None
    if path:
        pack_file = _resolve_by_path(pack, path)
    if pack_file is None and id:
        pack_file = _resolve_by_id(pack, id)

    if pack_file is None:
        logger.info("ep_read miss | pack=%s path=%r id=%r", pack.slug, path, id)
        return {"error": "File not found in pack", "path": path, "id": id}

    payload = {
        "path": pack_file.path,
        "title": pack_file.title,
        "type": pack_file.type,
        "tags": pack_file.tags,
        "content": pack_file.content,
        "id": pack_file.provenance.id,
        "content_hash": pack_file.provenance.content_hash,
        "verified_at": pack_file.provenance.verified_at,
        "verified_by": pack_file.provenance.verified_by,
        "confidence": pack_file.provenance.confidence,
        "retrieval_strategy": pack_file.retrieval_strategy,
        "concept_scope": pack_file.concept_scope,
        "requires": pack_file.requires,
        "activation": _activation_dict(pack_file),
    }
    if reconstruct:
        payload["original_markdown"] = pack_file.raw_content
        payload["original_span"] = pack_file.raw_content
        payload["provenance_block"] = {
            "id": pack_file.provenance.id,
            "source_file": pack_file.path,
            "content_hash": pack_file.provenance.content_hash,
            "verified_at": pack_file.provenance.verified_at,
            "verified_by": pack_file.provenance.verified_by,
            "confidence": pack_file.provenance.confidence,
        }

    logger.info(
        "ep_read | pack=%s path=%s reconstruct=%s",
        pack.slug, pack_file.path, reconstruct,
    )
    return payload


def _resolve_by_path(pack: Pack, path: str) -> PackFile | None:
    key = path.replace("\\", "/").lstrip("/")
    if key in pack.files:
        return pack.files[key]
    lower = key.lower()
    for fp, f in pack.files.items():
        if fp.lower() == lower:
            return f
    return None


def _resolve_by_id(pack: Pack, prov_id: str) -> PackFile | None:
    for f in pack.files.values():
        if f.provenance.id == prov_id:
            return f
    return None


def _activation_dict(pack_file: PackFile) -> dict | None:
    if pack_file.activation is None:
        return None
    return {
        "tools": pack_file.activation.tools,
        "constraints": pack_file.activation.constraints,
        "next": pack_file.activation.next,
    }
