#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! python3 - <<'PY' >/dev/null 2>&1
import PySide6
PY
then
  python3 -m pip install --user -e "$ROOT_DIR"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "${XDG_STATE_HOME:-$HOME/.local/state}/npu-audio-enhancer"
exec python3 -m npu_audio_enhancer.desktop "$@" \
  >> "${XDG_STATE_HOME:-$HOME/.local/state}/npu-audio-enhancer/desktop.log" 2>&1
