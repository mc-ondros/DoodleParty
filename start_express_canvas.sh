#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-3000}
DEMO_MODE=${DEMO_MODE:-0}

echo "Starting Express QuickDraw server on http://$HOST:$PORT..."
HOST=$HOST PORT=$PORT DEMO_MODE=$DEMO_MODE node express-server.js
