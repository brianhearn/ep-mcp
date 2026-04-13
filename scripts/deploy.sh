#!/usr/bin/env bash
# deploy.sh — Deploy EP MCP source to ExpertPack droplet
#
# Usage:
#   ./scripts/deploy.sh                  # deploy current working tree
#   ./scripts/deploy.sh --restart-only   # just restart the service (no file copy)
#
# Why not pip install?
#   pip install . rewrites the entrypoint shebang to the build machine's venv path.
#   On the droplet the venv is at /opt/ep-mcp/.venv — a path that doesn't exist on EasyBot.
#   Copying source files directly avoids touching the entrypoint or installed metadata.

set -euo pipefail

REMOTE_HOST="root@165.245.136.51"
REMOTE_SRC="/opt/ep-mcp/ep_mcp"
REMOTE_SITE_PKG="/opt/ep-mcp/.venv/lib/python3.12/site-packages/ep_mcp"
LOCAL_SRC="$(cd "$(dirname "$0")/.." && pwd)/ep_mcp"

RESTART_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--restart-only" ]] && RESTART_ONLY=true
done

if [[ "$RESTART_ONLY" == "false" ]]; then
  echo "==> Syncing ep_mcp/ to /opt/ep-mcp/ep_mcp/ ..."
  rsync -az --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    "$LOCAL_SRC/" "${REMOTE_HOST}:${REMOTE_SRC}/"

  echo "==> Syncing ep_mcp/ to site-packages ..."
  rsync -az --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    "$LOCAL_SRC/" "${REMOTE_HOST}:${REMOTE_SITE_PKG}/"

  echo "==> Clearing __pycache__ on remote ..."
  ssh "$REMOTE_HOST" "find ${REMOTE_SRC} ${REMOTE_SITE_PKG} -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true"
fi

echo "==> Restarting ep-mcp service ..."
ssh "$REMOTE_HOST" "systemctl restart ep-mcp"

echo "==> Waiting for service to stabilize ..."
sleep 2
ssh "$REMOTE_HOST" "systemctl is-active ep-mcp && journalctl -u ep-mcp -n 10 --no-pager"

echo ""
echo "✅ Deploy complete."
