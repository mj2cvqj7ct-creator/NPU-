# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a pure-Python (3.10+) CLI toolkit with four modules and a Tkinter GUI. It has **zero external dependencies** — everything runs on the Python standard library alone.

| Module | Entry point | Purpose |
|---|---|---|
| Guardian Blacklist | `guardian_blacklist.py` | Local firewall blacklist CLI for suspicious IPs |
| Audio Lossless Assistant | `audio_lossless_assistant.py` | Lossless codec assessment and preservation plans |
| NPU Audio Enhancement | `npu_audio_enhancement_assistant.py` | AI/NPU-accelerated audio enhancement plans |
| Windows LDAC Assistant | `windows_ldac_assistant.py` | LDAC Bluetooth codec diagnosis for Windows |
| Desktop GUI | `audio_desktop_app.py` | Tkinter tabbed interface (requires `python3-tk`) |

### Running tests

```bash
python3 -m unittest discover -v --start-directory . --pattern 'test_*.py'
```

All 35 tests use only the Python stdlib and run in under 1 second. No external services, databases, or Docker are needed.

### Linting

No linter is configured in the repo. `ruff` is installed in the VM update script for basic lint checking:

```bash
python3 -m ruff check .
```

There is one pre-existing lint warning (`F401` unused import in `npu_audio_enhancement_assistant.py`).

### Running the CLI applications

All CLIs are standalone scripts. Use `--data-dir <tmpdir>` (on `guardian_blacklist.py`) or `--dry-run` (on `windows_ldac_assistant.py`) to avoid side effects. See `README.md` for full usage examples.

### Non-obvious caveats

- **IP validation**: `guardian_blacklist.py` rejects private, loopback, reserved (including TEST-NET `192.0.2.x`, `198.51.100.x`, `203.0.113.x`), and multicast addresses. Use genuinely routable public IPs for testing (e.g. `185.220.101.42`).
- **Windows-only features**: DPAPI encryption and registry startup in `windows_ldac_assistant.py` only work on Windows. Tests use a `PlaintextTestProtector` and `--dry-run` to work cross-platform.
- **Tkinter GUI**: `audio_desktop_app.py` requires `python3-tk`, which is not installed by default in the Cloud VM. The GUI tests do not require it (they test logic helpers only).
- **`--data-dir` placement**: For `guardian_blacklist.py`, `--data-dir` must come **before** the subcommand (e.g. `python3 guardian_blacklist.py --data-dir /tmp/demo add ...`).
