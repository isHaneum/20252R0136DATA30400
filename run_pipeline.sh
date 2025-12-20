#!/usr/bin/env bash
set -euo pipefail

# Legacy entrypoint.
# The project is now driven by `run.sh` + `run_pipeline.py`.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exec bash "$SCRIPT_DIR/run.sh" "$@"
