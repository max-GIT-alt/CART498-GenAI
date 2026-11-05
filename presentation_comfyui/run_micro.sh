#!/usr/bin/env bash
# Linux/macOS frontend for the cross-platform launcher (scripts/bootstrap.py).
# The launcher itself needs only Python 3.9+; it finds ComfyUI's own venv,
# checks it is complete, and starts ComfyUI with it.
set -euo pipefail
cd "$(dirname "$0")"

# DEMO_PYTHON (or the older PYTHON) picks the interpreter for the launcher.
if [ -n "${DEMO_PYTHON:-${PYTHON:-}}" ]; then
  candidates=("${DEMO_PYTHON:-$PYTHON}")
else
  candidates=(python3 python)
fi
PY=""
for candidate in "${candidates[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "error: run_micro.sh needs Python 3.9 or newer (tried: ${candidates[*]})." >&2
  echo "       Install it, or set DEMO_PYTHON=/path/to/python3." >&2
  exit 1
fi
exec "$PY" scripts/bootstrap.py micro "$@"
