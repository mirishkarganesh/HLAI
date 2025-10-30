#!/usr/bin/env bash
set -euo pipefail

# Configuration via environment variables or defaults
URL=${URL:-"https://huggingface.co/papers/date/2025-09-30"}
LIMIT=${LIMIT:-}
STYLE=${STYLE:-"concise"}
BEST_ONLY=${BEST_ONLY:-"false"}
QUESTION=${QUESTION:-"What are the main contributions?"}
DEBUG=${DEBUG:-"false"}

# Absolute project directory
PROJECT_DIR="/media/ganesh/Data1/Projects/highlevel"
SCRIPT_DIR="$PROJECT_DIR"

# Use the specified conda env's python directly (no activation)
CONDA_ENV_PATH="/media/ganesh/Data1/Projects/highlevel/envs/tryrag"
PYTHON_BIN="$CONDA_ENV_PATH/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  # Fallback: try conda run if available
  if command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="conda run -p $CONDA_ENV_PATH python"
  else
    echo "[ERROR] Could not find python at $CONDA_ENV_PATH/bin/python and conda is not available." >&2
    exit 1
  fi
fi

if [ "$DEBUG" = "true" ]; then
  set -x
  if [[ "$PYTHON_BIN" == conda* ]]; then
    echo "Using Python via: $PYTHON_BIN"
    $PYTHON_BIN --version
  else
    echo "Using Python: $PYTHON_BIN"
    "$PYTHON_BIN" --version
  fi
fi

# Dependency sanity check (no installs here by design)
missing=()
for mod in requests bs4 pdfplumber sentence_transformers numpy sklearn transformers accelerate torch rich pydantic feedparser wikipedia; do
  if [[ "$PYTHON_BIN" == conda* ]]; then
    $PYTHON_BIN -c "import ${mod}" 2>/dev/null || missing+=("$mod")
  else
    "$PYTHON_BIN" -c "import ${mod}" 2>/dev/null || missing+=("$mod")
  fi
done
if [ ${#missing[@]} -ne 0 ]; then
  echo "[ERROR] Missing Python modules in the selected environment: ${missing[*]}" >&2
  echo "Please install once into $CONDA_ENV_PATH (e.g., pip install -r $SCRIPT_DIR/requirements.txt) and rerun." >&2
  exit 2
fi

# Build argument list
ARGS=(--url "$URL" --style "$STYLE" --question "$QUESTION")
if [ -n "${LIMIT}" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ "$BEST_ONLY" = "true" ]; then
  ARGS+=(--best_only)
fi

if [[ "$PYTHON_BIN" == conda* ]]; then
  $PYTHON_BIN "$SCRIPT_DIR/run_chat.py" "${ARGS[@]}"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/run_chat.py" "${ARGS[@]}"
fi
