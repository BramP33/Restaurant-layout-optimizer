#!/usr/bin/env bash
# Start de Zaalplanner lokaal op http://localhost:8765
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn app.server.main:app --host 127.0.0.1 --port 8765 "$@"
