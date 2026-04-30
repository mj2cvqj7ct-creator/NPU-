# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Python 3 project with **no third-party dependencies** (standard library only). There are no `requirements.txt`, `pyproject.toml`, or package manager config files.

### Services overview

| Module | Type | Run command |
|---|---|---|
| `guardian_blacklist.py` | CLI | `python3 guardian_blacklist.py <subcommand>` |
| `audio_lossless_assistant.py` | CLI | `python3 audio_lossless_assistant.py <subcommand>` |
| `npu_audio_enhancement_assistant.py` | CLI | `python3 npu_audio_enhancement_assistant.py <subcommand>` |
| `windows_ldac_assistant.py` | CLI | `python3 windows_ldac_assistant.py <subcommand>` |
| `audio_desktop_app.py` | Tkinter GUI | `DISPLAY=:1 python3 audio_desktop_app.py` |

### Lint / Test / Run

- **Lint:** `ruff check .` (ruff is pre-installed at `~/.local/bin/ruff`)
- **Tests:** `python3 -m unittest discover -v` (35 tests, all use `unittest` from stdlib)
- **GUI app:** requires `python3-tk` system package (`sudo apt-get install -y python3-tk`); set `DISPLAY=:1` when running headless

### Gotchas

- `guardian_blacklist.py add` rejects private/loopback/reserved/multicast IPs. Use a routable public IP for testing (e.g. `185.220.101.1`).
- `203.0.113.0/24` (TEST-NET-3) is treated as reserved and will be rejected.
- The `windows_ldac_assistant.py` DPAPI features only work on Windows; on Linux it uses a plaintext test protector. Pass `--system Windows` to `status` to simulate Windows diagnostics.
- `npu_audio_enhancement_assistant.py` gracefully falls back to CPU if `onnxruntime` is not installed.
- All test files use `unittest` (not pytest). `python3 -m pytest` also works but is not required.
- README is in Japanese. See the `## 使い方` section for CLI usage examples.
