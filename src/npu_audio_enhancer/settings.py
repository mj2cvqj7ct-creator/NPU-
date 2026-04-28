from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


APP_ID = "npu-audio-enhancer"


@dataclass(frozen=True)
class AppSettings:
    spotify: bool = True
    apple_music: bool = True
    youtube_music: bool = True
    profile: str = "holographic-vocal-stage"
    latency_path: str = "ASIO XMOS USB DAC - extreme low latency"
    active_on_start: bool = False
    recommendation_tick: int = 0
    autostart_enabled: bool = True


def config_dir() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME")
    if root:
        return Path(root) / APP_ID
    return Path.home() / ".config" / APP_ID


def settings_path() -> Path:
    return config_dir() / "settings.json"


def autostart_dir() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME")
    if root:
        return Path(root) / "autostart"
    return Path.home() / ".config" / "autostart"


def load_settings(path: str | Path | None = None) -> AppSettings:
    path = Path(path) if path is not None else settings_path()
    if not path.exists():
        return AppSettings()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppSettings()

    defaults = asdict(AppSettings())
    defaults.update({key: data[key] for key in defaults.keys() & data.keys()})
    return AppSettings(**defaults)


def save_settings(settings: AppSettings, path: str | Path | None = None) -> None:
    path = Path(path) if path is not None else settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_autostart_entry(root_dir: str | Path, enabled: bool = True) -> Path:
    path = autostart_dir() / f"{APP_ID}.desktop"
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root_dir).resolve()
    hidden = "false" if enabled else "true"
    path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=NPU Audio Enhancer",
                "Comment=Start realtime NPU audio enhancement when the desktop session starts",
                f"Exec={root / 'scripts/linux/run_desktop_app.sh'} --start-minimized",
                f"Icon={root / 'assets/npu-audio-enhancer.svg'}",
                "Terminal=false",
                "Categories=AudioVideo;Audio;",
                "StartupNotify=false",
                f"Hidden={hidden}",
                "X-GNOME-Autostart-enabled=true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path
