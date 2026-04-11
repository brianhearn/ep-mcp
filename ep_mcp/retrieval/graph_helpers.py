"""Graph helper utilities for mapping between file paths and node IDs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..pack.models import Pack, PackGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphNodeInfo:
    """Metadata about a single graph node."""

    id: str
    title: str | None = None
    type: str | None = None
    file: str | None = None


@dataclass
class GraphLookup:
    """Bi-directional lookup between file paths and graph node IDs.

    Built from the raw _graph.yaml to preserve node metadata that
    PackGraph.nodes doesn't retain (title, type, file fields).
    """

    file_to_node_id: dict[str, str] = field(default_factory=dict)
    node_id_to_file: dict[str, str] = field(default_factory=dict)
    node_info: dict[str, GraphNodeInfo] = field(default_factory=dict)

    @classmethod
    def from_pack(cls, pack: Pack) -> GraphLookup | None:
        """Build a GraphLookup from a pack's _graph.yaml.

        Args:
            pack: Loaded Pack with pack_dir set.

        Returns:
            GraphLookup if _graph.yaml exists and has node data, else None.
        """
        if not pack.pack_dir:
            return None

        graph_path = Path(pack.pack_dir) / "_graph.yaml"
        if not graph_path.exists():
            return None

        try:
            raw = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            logger.warning("Failed to read _graph.yaml for graph lookup")
            return None

        if not isinstance(raw, dict):
            return None

        nodes_raw = raw.get("nodes", [])
        if not isinstance(nodes_raw, list):
            return None

        lookup = cls()
        for node in nodes_raw:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            file_path = node.get("file")
            if not node_id:
                continue

            info = GraphNodeInfo(
                id=node_id,
                title=node.get("title"),
                type=node.get("type"),
                file=file_path,
            )
            lookup.node_info[node_id] = info

            if file_path:
                lookup.file_to_node_id[file_path] = node_id
                lookup.node_id_to_file[node_id] = file_path

        logger.debug(
            "Built graph lookup: %d nodes, %d file mappings",
            len(lookup.node_info),
            len(lookup.file_to_node_id),
        )
        return lookup

    @classmethod
    def from_raw_nodes(cls, nodes_raw: list[dict]) -> GraphLookup:
        """Build a GraphLookup from a list of raw node dicts.

        Useful for testing without needing a YAML file on disk.

        Args:
            nodes_raw: List of node dicts with id, title, type, file keys.

        Returns:
            GraphLookup instance.
        """
        lookup = cls()
        for node in nodes_raw:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            file_path = node.get("file")
            if not node_id:
                continue

            info = GraphNodeInfo(
                id=node_id,
                title=node.get("title"),
                type=node.get("type"),
                file=file_path,
            )
            lookup.node_info[node_id] = info

            if file_path:
                lookup.file_to_node_id[file_path] = node_id
                lookup.node_id_to_file[node_id] = file_path

        return lookup

    def get_neighbor_file_paths(
        self,
        file_path: str,
        graph: PackGraph,
        edge_kinds: list[str] | None = None,
    ) -> list[str]:
        """Get file paths of neighbors for a given file path.

        Converts file_path → node_id, calls graph.neighbors(),
        then converts neighbor node_ids back to file_paths.

        Args:
            file_path: Source file path (e.g. 'concepts/auto-build.md').
            graph: The PackGraph with edges.
            edge_kinds: Optional filter on edge kinds.

        Returns:
            List of neighbor file paths.
        """
        node_id = self.file_to_node_id.get(file_path)
        if not node_id:
            return []

        neighbor_ids = graph.neighbors(node_id, edge_kinds)
        result = []
        for nid in neighbor_ids:
            fp = self.node_id_to_file.get(nid)
            if fp:
                result.append(fp)
        return result
