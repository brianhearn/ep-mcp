"""API key authentication middleware (Phase 1)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class APIKeyAuth:
    """Phase 1 authentication: per-pack API key validation.

    Keys can be configured via:
    1. Pack config: api_keys list
    2. Environment variable: EP_MCP_KEY_{SLUG_UPPER}
    """

    def __init__(self, pack_keys: dict[str, set[str]] | None = None):
        """Initialize with pack → keys mapping.

        Args:
            pack_keys: {pack_slug: set_of_valid_keys}
        """
        self._pack_keys: dict[str, set[str]] = pack_keys or {}

        # Also check environment variables
        for slug in list(self._pack_keys.keys()):
            env_key = os.environ.get(f"EP_MCP_KEY_{slug.upper().replace('-', '_')}")
            if env_key:
                self._pack_keys[slug].add(env_key)

    def add_pack_keys(self, slug: str, keys: list[str]) -> None:
        """Register API keys for a pack."""
        if slug not in self._pack_keys:
            self._pack_keys[slug] = set()
        self._pack_keys[slug].update(keys)

        # Check env var too
        env_key = os.environ.get(f"EP_MCP_KEY_{slug.upper().replace('-', '_')}")
        if env_key:
            self._pack_keys[slug].add(env_key)

    def authenticate(self, auth_header: str, pack_slug: str) -> bool:
        """Validate API key for the requested pack.

        Args:
            auth_header: Full Authorization header value
            pack_slug: Which pack is being accessed

        Returns:
            True if authenticated, False otherwise
        """
        if not auth_header.startswith("Bearer "):
            return False

        key = auth_header[7:]
        valid_keys = self._pack_keys.get(pack_slug, set())

        if not valid_keys:
            # No keys configured = open access (for dev/testing)
            logger.warning("No API keys configured for pack '%s' — allowing open access", pack_slug)
            return True

        return key in valid_keys
