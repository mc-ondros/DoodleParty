#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-3000}
DEMO_MODE=${DEMO_MODE:-0}
# Optional Basic Auth for /admin & admin APIs
ADMIN_USER=${ADMIN_USER:-admin}
ADMIN_PASSWORD=${ADMIN_PASSWORD:-unihack2025}

echo "Starting Express QuickDraw server on http://$HOST:$PORT..."
echo "Admin auth: user='$ADMIN_USER' password set? $([ -n "$ADMIN_PASSWORD" ] && echo yes || echo no)"
HOST=$HOST PORT=$PORT DEMO_MODE=$DEMO_MODE ADMIN_USER=$ADMIN_USER ADMIN_PASSWORD=$ADMIN_PASSWORD node express-server.cjs
