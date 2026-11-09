#!/usr/bin/env bash
# Runs once after the container is created. The venv itself is built into the
# image (/opt/venv); this only re-syncs it in case requirements.txt changed.
set -euo pipefail

cd "$(dirname "$0")/.."

pip install -q -r requirements.txt

mkdir -p data results
touch results/.gitkeep

echo
echo "Done. Python: $(command -v python)  |  Data dir: ${MTAH_DATA_DIR:-data}"
echo "Optional deep-learning stack (video / ball detection): pip install -r requirements-dl.txt"
