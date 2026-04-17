"""Optional file watcher for live pack reindex (dev/authoring mode).

Watches pack source directories for .md file changes and triggers a debounced
incremental reindex via IndexManager. Designed for Obsidian authoring workflows
where pack files are edited live and you want the index to stay current without
restarting the server.

Usage (dev mode only — not recommended for production):
    Enable via config: server.dev_mode_watch: true
    or via env: EP_MCP_DEV_WATCH=1

Requires watchdog: pip install watchdog
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..index.manager import IndexManager
    from ..pack.models import Pack

logger = logging.getLogger(__name__)

# Default debounce delay (seconds) — waits this long after the last change before reindexing
DEFAULT_DEBOUNCE_SECONDS = 2.0


class PackFileWatcher:
    """Watches a pack's source directory and triggers debounced incremental reindex.

    Uses watchdog for filesystem events. The debounce prevents redundant reindex
    runs when an editor saves multiple files in rapid succession (e.g. Obsidian
    sync or a bulk rename).

    Architecture:
    - watchdog runs in a background thread (synchronous)
    - Filesystem events set a flag + schedule a debounced coroutine via the asyncio loop
    - The reindex itself runs on the asyncio event loop (IndexManager is async)
    """

    def __init__(
        self,
        pack: "Pack",
        index_manager: "IndexManager",
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ):
        self.pack = pack
        self.index_manager = index_manager
        self.loop = loop
        self.debounce_seconds = debounce_seconds
        self._observer = None
        self._debounce_handle: asyncio.TimerHandle | None = None
        self._lock = threading.Lock()
        self._reindex_pending = False

    def start(self) -> bool:
        """Start watching the pack directory. Returns False if watchdog is not installed."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning(
                "watchdog not installed — file watching unavailable. "
                "Run: pip install watchdog"
            )
            return False

        pack_path = Path(self.pack.manifest.pack_path)
        if not pack_path.exists():
            logger.warning(
                "Pack path does not exist — skipping file watch: %s", pack_path
            )
            return False

        handler = _MarkdownEventHandler(self._on_change)
        observer = Observer()
        observer.schedule(handler, str(pack_path), recursive=True)
        observer.start()
        self._observer = observer

        logger.info(
            "File watcher started for pack '%s' at %s (debounce=%.1fs)",
            self.pack.slug, pack_path, self.debounce_seconds,
        )
        return True

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("File watcher stopped for pack '%s'", self.pack.slug)

    def _on_change(self, path: str) -> None:
        """Called from watchdog thread when a .md file changes."""
        logger.debug("File change detected: %s", path)
        with self._lock:
            self._reindex_pending = True
        # Schedule debounced reindex on the asyncio loop (thread-safe)
        self.loop.call_soon_threadsafe(self._schedule_debounced_reindex)

    def _schedule_debounced_reindex(self) -> None:
        """Cancel any pending reindex timer and schedule a new one."""
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
        self._debounce_handle = self.loop.call_later(
            self.debounce_seconds,
            lambda: asyncio.ensure_future(self._run_reindex(), loop=self.loop),
        )

    async def _run_reindex(self) -> None:
        """Run the incremental reindex on the asyncio event loop."""
        with self._lock:
            if not self._reindex_pending:
                return
            self._reindex_pending = False

        logger.info(
            "File watcher: triggering incremental reindex for pack '%s'",
            self.pack.slug,
        )
        try:
            # Reload pack files from disk before reindexing
            from ..pack.loader import load_pack
            updated_pack = load_pack(self.pack.manifest.pack_path)
            self.index_manager.pack = updated_pack
            stats = await self.index_manager.build_index()
            logger.info(
                "File watcher: reindex complete for '%s' — %s",
                self.pack.slug, stats,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "File watcher: reindex failed for pack '%s': %s",
                self.pack.slug, exc,
            )


class _MarkdownEventHandler:
    """Watchdog event handler that fires on .md file changes."""

    def __init__(self, callback):
        self._callback = callback

    def dispatch(self, event) -> None:
        """Called by watchdog for every filesystem event."""
        if event.is_directory:
            return
        src = getattr(event, "src_path", "")
        dest = getattr(event, "dest_path", "")
        for path in (src, dest):
            if path and path.endswith(".md"):
                self._callback(path)
                return


def start_watchers(
    pack_instances: list,
    loop: asyncio.AbstractEventLoop,
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
) -> list[PackFileWatcher]:
    """Start file watchers for all pack instances.

    Args:
        pack_instances: List of PackInstance objects (pack + store + engine + mcp).
        loop: The running asyncio event loop.
        debounce_seconds: Debounce delay before triggering reindex.

    Returns:
        List of started PackFileWatcher objects (only those that started successfully).
    """
    watchers = []
    for instance in pack_instances:
        watcher = PackFileWatcher(
            pack=instance.pack,
            index_manager=instance.index_manager,
            loop=loop,
            debounce_seconds=debounce_seconds,
        )
        if watcher.start():
            watchers.append(watcher)
    return watchers
