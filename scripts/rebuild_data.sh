#!/bin/sh
# Rebuild everything under data/processed from data/raw.
#
#   sh scripts/rebuild_data.sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON=./.venv/bin/python
"$PYTHON" -m scentvec.data
"$PYTHON" -m scentvec.chem
