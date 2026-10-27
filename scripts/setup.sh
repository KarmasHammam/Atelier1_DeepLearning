#!/usr/bin/env bash
# Installation locale : aucun paquet n'est ajouté au Python du système.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python -m src.data
echo "Installation terminée. Activation : source .venv/bin/activate"
