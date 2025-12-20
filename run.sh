#!/usr/bin/env bash
set -euo pipefail

# One-command runner:
# - creates/activates venv
# - pip install -r requirements.txt
# - runs python run_pipeline.py (passes through all args)
#
# Examples:
#   bash run.sh --student-id 2021320045
#   bash run.sh --student-id 2021320045 --use-llm

# Disable HF fast-transfer to avoid missing hf_transfer dependency in minimal environments
export HF_HUB_ENABLE_HF_TRANSFER=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  python -m venv "$VENV"
fi

# Activate venv (Linux/macOS vs Windows Git-Bash)
if [ -f "$VENV/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
elif [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

python run_pipeline.py "$@"
