# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

A flat Python 3.12+ project with five CLI/GUI tools (no monorepo, no framework). All source and test files live at the repository root. See `README.md` for usage examples.

| Tool | Purpose |
|---|---|
| `guardian_blacklist.py` | Local firewall blacklist CLI for suspicious IPs |
| `audio_lossless_assistant.py` | Lossless audio codec assessment and preservation planning |
| `npu_audio_enhancement_assistant.py` | AI/NPU-assisted audio enhancement planning |
| `windows_ldac_assistant.py` | Windows LDAC Bluetooth readiness diagnostics |
| `audio_desktop_app.py` | Tkinter GUI integrating the three audio tools |

### Dependencies

- **Required:** Python 3.12+ stdlib only (zero third-party packages for core functionality).
- **Optional:** `onnxruntime` (NPU detection in `npu_audio_enhancement_assistant.py`; graceful CPU fallback if absent), `python3-tk` (for `audio_desktop_app.py` GUI).

### Commands

- **Lint:** `ruff check .` — one pre-existing F401 warning in `npu_audio_enhancement_assistant.py` (unused `platform` import).
- **Test:** `python3 -m unittest discover -s . -p 'test_*.py' -v` — runs all 35 tests.
- **Run CLIs:** Each tool is a standalone script, e.g. `python3 guardian_blacklist.py list`. See `README.md` for full CLI usage.

### Gotchas

- `guardian_blacklist.py add` only accepts globally routable IPv4/IPv6 addresses. Private, loopback, reserved (including 203.0.113.0/24 documentation range), and multicast IPs are rejected.
- Blacklist data defaults to `~/.local/share/guardian-blacklist/blacklist.json`. Override with `GUARDIAN_BLACKLIST_HOME` env var or `--data-dir` flag. Tests use temp directories automatically.
- The `audio_desktop_app.py` GUI requires a display server (X11/Wayland). In headless Cloud VMs, tests pass without a display since they exercise the logic layer, not the Tkinter event loop.
- `windows_ldac_assistant.py` works in diagnostic mode on non-Windows (DPAPI is replaced by a plaintext test protector).
