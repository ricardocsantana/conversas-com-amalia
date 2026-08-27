#!/usr/bin/env bash
# One-shot setup for "Conversas com Amália" — a European Portuguese (pt-PT)
# voice assistant built on huggingface/speech-to-speech, teex/amalia (via
# Ollama), and the kokoro_tts_eu_pt voice.
#
# Creates/recreates a conda env named "amalia" on Python 3.11 (required: the
# Darwin-only deps this project pulls in, e.g. misaki/mlx, have no wheels for
# newer Python yet) and installs this repo editable with the kokoro-eu-pt
# extra. Also pulls the teex/amalia model via Ollama if Ollama is installed.
#
# Usage: bash scripts/amalia_setup.sh

set -euo pipefail

ENV_NAME="amalia"
PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

echo "==> Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

ENV_PY="$(conda run -n "${ENV_NAME}" which python)"
ENV_PIP="$(conda run -n "${ENV_NAME}" which pip)"

echo "==> Installing this repo (editable) with the kokoro-eu-pt extra"
"${ENV_PIP}" install -e "${REPO_ROOT}[kokoro-eu-pt]"

if command -v ollama >/dev/null 2>&1; then
  echo "==> Ollama found. Pulling teex/amalia (this can take a few minutes)"
  ollama pull teex/amalia
else
  echo "==> Ollama not found on PATH."
  echo "    Install it from https://ollama.com/download, then run:"
  echo "      ollama pull teex/amalia"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  bash scripts/amalia_run.sh"
