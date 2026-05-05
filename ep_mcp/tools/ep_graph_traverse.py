"""ep_graph_traverse MCP tool implementation."""

from __future__ import annotations

import logging
from collections import deque

from ..pack.models import Pack, PackGraph
from ..retrieval.graph_helpers import GraphLookup, GraphNodeInfo

logger = logging.getLogger(__name__)


def _node_to_dict(node_id: str, info: GraphNodeInfo | None, file_path: str | None) -> dict:
    """Serialize graph node metadata for tool output."""
    return {
        "id": node_id,
        "file": file_path,
        "title": info.title if info else None,
        "type": info.type if info else None,
        "kind": info.kind if info else None,
        "aliases": info.aliases if info else [],
        "status": info.status if info else None,
        "is_file_backed": bool(file_path),
    }


def ep_graph_traverse(
    pack: Pack,
    graph_lookup: GraphLookup | None,
    file_path: str,
    depth: int = 1,
    edge_kinds: list[str] | None = None,
) -> dict:
    """Traverse the knowledge graph from a starting file path.

    Performs BFS up to `depth` hops from the starting node, returning
    connected nodes with their metadata and edge information.

    Args:
        pack: Loaded Pack with graph data.
        graph_lookup: GraphLookup for file_path ↔ node_id mapping.
        file_path: Starting file path (e.g. 'concepts/auto-build.md').
        depth: Number of hops to follow (clamped to 1-3).
        edge_kinds: Filter by edge types (wikilink, related, context).
                   If None, follows all edge types.

    Returns:
        Dict with start_node info, connected nodes, and traversal stats.
    """
    # Clamp depth to 1-3
    depth = max(1, min(3, depth))

    # Handle missing graph or lookup
    if pack.graph is None or graph_lookup is None:
        return {
            "start_node": {"file": file_path, "title": None, "type": None},
            "connected": [],
            "total_edges_traversed": 0,
        }

    graph = pack.graph

    # Resolve start node. Accept either a pack-relative file path or a raw
    # graph node id (needed for ontology_entity nodes, which have no file).
    start_node_id = graph_lookup.file_to_node_id.get(file_path)
    if start_node_id is None and file_path in graph_lookup.node_info:
        start_node_id = file_path
    if start_node_id is None:
        return {
            "start_node": {"id": file_path, "file": file_path, "title": None, "type": None},
            "connected": [],
            "total_edges_traversed": 0,
        }

    start_info = graph_lookup.node_info.get(start_node_id)
    start_node_dict = _node_to_dict(start_node_id, start_info, file_path if start_info is None else start_info.file)

    # BFS traversal
    visited: set[str] = {start_node_id}
    connected: list[dict] = []
    total_edges_traversed = 0

    # Queue items: (node_id, current_depth)
    queue: deque[tuple[str, int]] = deque([(start_node_id, 0)])

    while queue:
        current_id, current_depth = queue.popleft()

        if current_depth >= depth:
            continue

        # Find neighbors at this level
        for edge in graph.edges:
            if edge_kinds and edge.kind not in edge_kinds:
                continue

            neighbor_id: str | None = None
            if edge.source == current_id:
                neighbor_id = edge.target
            elif edge.target == current_id:
                neighbor_id = edge.source

            if neighbor_id is None:
                continue

            total_edges_traversed += 1

            if neighbor_id in visited:
                continue

            visited.add(neighbor_id)

            # Resolve neighbor info. Ontology/entity nodes may have no source file;
            # keep their node id and metadata instead of dropping them.
            neighbor_info = graph_lookup.node_info.get(neighbor_id)
            node_dict = _node_to_dict(
                neighbor_id,
                neighbor_info,
                graph_lookup.node_id_to_file.get(neighbor_id),
            )
            node_dict["edge_kind"] = edge.kind
            node_dict["depth"] = current_depth + 1
            connected.append(node_dict)

            queue.append((neighbor_id, current_depth + 1))

    logger.info(
        "ep_graph_traverse | pack=%s file=%s depth=%d edge_kinds=%s "
        "connected=%d edges_traversed=%d",
        pack.slug, file_path, depth, edge_kinds,
        len(connected), total_edges_traversed,
    )

    return {
        "start_node": start_node_dict,
        "connected": connected,
        "total_edges_traversed": total_edges_traversed,
    }
