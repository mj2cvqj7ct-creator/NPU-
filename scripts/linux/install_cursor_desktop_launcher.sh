#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_PATH="$ROOT_DIR/assets/npu-audio-enhancer.svg"
DESKTOP_FILE="$APPS_DIR/npu-audio-enhancer.desktop"
DESKTOP_COPY="$DESKTOP_DIR/NPU Audio Enhancer.desktop"

mkdir -p "$DESKTOP_DIR" "$APPS_DIR"

cat > "$DESKTOP_FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Name=NPU Audio Enhancer
Comment=Cursor desktop app for realtime NPU audio enhancement controls
Exec=$ROOT_DIR/scripts/linux/run_desktop_app.sh
Icon=$ICON_PATH
Terminal=false
Categories=AudioVideo;Audio;
StartupNotify=true
DESKTOP

chmod +x "$DESKTOP_FILE"
cp "$DESKTOP_FILE" "$DESKTOP_COPY"
chmod +x "$DESKTOP_COPY"

echo "Installed launcher: $DESKTOP_COPY"
echo "Application entry: $DESKTOP_FILE"
