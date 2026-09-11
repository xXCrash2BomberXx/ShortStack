#!/usr/bin/env bash
# Launches the LLM Stack Control Panel regardless of the current working
# directory. Assumes this script lives in the same folder as app.py, which
# in turn should sit next to your docker-compose.yaml (or have COMPOSE_DIR
# pointed at it).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python3" app.py "$@"
