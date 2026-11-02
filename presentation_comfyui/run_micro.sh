#!/usr/bin/env bash
# Linux/macOS frontend for the cross-platform launcher.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
exec "$PY" scripts/bootstrap.py micro "$@"
