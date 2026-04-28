#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT_DIR/dist/windows"
mkdir -p "$OUT_DIR"

x86_64-w64-mingw32-g++ \
  -std=c++17 \
  -O2 \
  -static \
  -municode \
  -mwindows \
  "$ROOT_DIR/native/windows/main.cpp" \
  -o "$OUT_DIR/NPUStreamingMusicEnhancer.exe" \
  -lgdi32 \
  -lmsimg32 \
  -lcomctl32

x86_64-w64-mingw32-g++ \
  -std=c++17 \
  -O2 \
  -static \
  -municode \
  -mwindows \
  "$ROOT_DIR/native/windows/installer.cpp" \
  -o "$OUT_DIR/NPUStreamingMusicEnhancerInstaller.exe" \
  -lole32 \
  -luuid \
  -lshell32

echo "Built $OUT_DIR/NPUStreamingMusicEnhancer.exe"
echo "Built $OUT_DIR/NPUStreamingMusicEnhancerInstaller.exe"
